# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run tracking context manager.

This module provides the primary interface for tracking quantum experiments,
including parameter logging, metric recording, and artifact management.

The main entry points are:

- :func:`track` - Create a tracking context (recommended)
- :class:`Run` - Context manager class for experiment runs
- :func:`wrap_backend` - Convenience function for backend wrapping

Examples
--------
Basic usage with context manager:

>>> from devqubit_engine.tracking.run import track
>>> with track(project="bell_state") as run:
...     run.log_param("shots", 1000)
...     run.log_param("optimization_level", 3)
...     # ... execute quantum circuit ...
...     run.log_metric("fidelity", 0.95)

Using the wrap pattern for automatic artifact logging:

>>> from devqubit_engine.tracking.run import track, wrap_backend
>>> from qiskit_aer import AerSimulator
>>> with track(project="bell_state") as run:
...     backend = wrap_backend(run, AerSimulator())
...     job = backend.run(circuit, shots=1000)
...     counts = job.result().get_counts()
"""

from __future__ import annotations

import logging
import threading
import traceback as _tb
import weakref
from pathlib import Path
from typing import Any, Sequence

from devqubit_engine.config import Config, get_config
from devqubit_engine.schema.validation import validate_run_record
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.types import (
    ArtifactRef,
    ObjectStoreProtocol,
    RegistryProtocol,
)
from devqubit_engine.tracking.fingerprints import (
    compute_fingerprints_from_envelope,
    compute_fingerprints_from_envelopes,
)
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.uec.errors import EnvelopeValidationError
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.utils.common import (
    generate_ulid,
    is_manual_run_record,
    sha256_bytes,
    utc_now_iso,
)
from devqubit_engine.utils.env import capture_environment, capture_git_provenance
from devqubit_engine.utils.qasm3 import coerce_openqasm3_sources
from devqubit_engine.utils.serialization import json_dumps, to_jsonable


logger = logging.getLogger(__name__)

# Maximum artifact size in bytes (20 MB default)
MAX_ARTIFACT_BYTES: int = 20 * 1024 * 1024

# Internal auto-flush interval.  Chosen to balance crash-safety (short
# enough that a kill loses at most a few seconds of data) against I/O
# overhead (long enough to batch many points per write).
_FLUSH_INTERVAL_SECONDS: float = 10.0


class _FlushWorker:
    """
    Daemon thread that periodically flushes run state to the registry.

    Mirrors the approach used by W&B's internal sync thread: a background
    daemon wakes every ``interval`` seconds and persists buffered data so
    the user never has to think about crash-safety or flush timing.

    Parameters
    ----------
    run : Run
        The run to flush.  Stored as a weak reference so the worker
        does not prevent garbage collection if the run is abandoned.
    interval : float
        Seconds between flush cycles.
    """

    def __init__(self, run: Run, interval: float = _FLUSH_INTERVAL_SECONDS) -> None:
        self._run_ref: weakref.ref[Run] = weakref.ref(run)
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="devqubit-flush",
        )
        self._thread.start()

    def _loop(self) -> None:
        """Wake every *interval* seconds and flush, until stopped."""
        while not self._stop.wait(self._interval):
            run = self._run_ref()
            if run is None:
                break
            try:
                run.flush()
            except Exception:
                # Logged inside flush(); nothing else to do.
                pass

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop.set()
        self._thread.join(timeout=5.0)


class Run:
    """
    Context manager for tracking a quantum experiment run.

    Provides methods for logging parameters, metrics, tags, and artifacts
    during experiment execution. Automatically captures environment and
    git provenance on entry, and finalizes the run record on exit.

    Parameters
    ----------
    project : str
        Project name for organizing runs.
    adapter : str, optional
        Adapter name. Auto-detected when using :meth:`wrap`.
        Default is "manual".
    run_name : str, optional
        Human-readable run name for display.
    store : ObjectStoreProtocol, optional
        Object store for artifacts. Created from config if not provided.
    registry : RegistryProtocol, optional
        Run registry for metadata. Created from config if not provided.
    config : Config, optional
        Configuration object. Uses global config if not provided.
    capture_env : bool, optional
        Whether to capture environment on start. Default is True.
    capture_git : bool, optional
        Whether to capture git provenance on start. Default is True.
    group_id : str, optional
        Group/experiment identifier for grouping related runs
        (e.g., parameter sweeps, benchmark suites).
    group_name : str, optional
        Human-readable group name.
    parent_run_id : str, optional
        Parent run ID for lineage tracking (e.g., rerun-from-baseline).

    Attributes
    ----------
    run_id : str
        Unique run identifier (ULID).
    status : str
        Current run status.
    store : ObjectStoreProtocol
        Object store for artifacts.
    registry : RegistryProtocol
        Run registry for metadata.
    record : dict
        Raw run record dictionary.
    """

    def __init__(
        self,
        project: str,
        adapter: str = "manual",
        run_name: str | None = None,
        store: ObjectStoreProtocol | None = None,
        registry: RegistryProtocol | None = None,
        config: Config | None = None,
        capture_env: bool = True,
        capture_git: bool = True,
        group_id: str | None = None,
        group_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self._lock = threading.Lock()

        self._run_id = generate_ulid()
        self._project = project
        self._adapter = adapter
        self._run_name = run_name
        self._artifacts: list[ArtifactRef] = []
        self._pending_tracked_jobs: list[Any] = []

        # Metric-series buffer: accumulated points awaiting flush.
        self._metric_buffer: list[dict[str, Any]] = []

        # Background flush worker — started in __enter__, stopped in
        # _finalize.  None until context manager entry.
        self._flush_worker: _FlushWorker | None = None

        # Get config and backends
        cfg = config or get_config()
        self._store = store or create_store(config=cfg)
        self._registry = registry or create_registry(config=cfg)
        self._config = cfg

        # Initialize record structure
        self.record: dict[str, Any] = {
            "schema": "devqubit.run/1.0",
            "run_id": self._run_id,
            "created_at": utc_now_iso(),
            "project": {"name": project},
            "adapter": adapter,
            "info": {"status": "RUNNING"},
            "data": {"params": {}, "metrics": {}, "tags": {}},
            "artifacts": [],
        }

        if run_name:
            self.record["info"]["run_name"] = run_name

        # Group/lineage support
        if group_id:
            self.record["group_id"] = str(group_id)
        if group_name:
            self.record["group_name"] = str(group_name)
        if parent_run_id:
            self.record["parent_run_id"] = str(parent_run_id)

        # Capture environment and provenance
        if capture_env:
            self.record["environment"] = capture_environment()

        should_capture_git = capture_git and cfg.capture_git
        if should_capture_git:
            git_info = capture_git_provenance()
            if git_info:
                self.record.setdefault("provenance", {})["git"] = {
                    k: v for k, v in git_info.items() if v is not None
                }

        logger.info(
            "Run started: run_id=%s, project=%s, adapter=%s",
            self._run_id,
            project,
            adapter,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Get the unique run identifier."""
        return self._run_id

    @property
    def run_name(self) -> str | None:
        """Get the human-readable run name."""
        return self._run_name

    @property
    def status(self) -> str:
        """Get the current run status."""
        return self.record.get("info", {}).get("status", "RUNNING")

    @property
    def store(self) -> ObjectStoreProtocol:
        """Get the object store for artifacts."""
        return self._store

    @property
    def registry(self) -> RegistryProtocol:
        """Get the run registry."""
        return self._registry

    # -----------------------------------------------------------------------
    # Logging methods
    # -----------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter value.

        Parameters are experimental configuration values that should
        remain constant during the run.

        Parameters
        ----------
        key : str
            Parameter name. Must not contain dots (they conflict with
            the query parser's nested field resolution).
        value : Any
            Parameter value. Will be converted to JSON-serializable form.

        Raises
        ------
        ValueError
            If key contains a dot character.
        """
        if "." in key:
            raise ValueError(
                f"Param key cannot contain dots: {key!r}. "
                f"Use underscores or nested dicts instead."
            )
        jsonable_value = to_jsonable(value)
        with self._lock:
            self.record["data"]["params"][key] = jsonable_value
        logger.debug("Logged param: %s=%r", key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log multiple parameters at once.

        Parameters
        ----------
        params : dict
            Dictionary of parameter name-value pairs.
        """
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Log a metric value.

        When ``step`` is provided the value is appended to the metric's
        time-series history **and** the scalar summary
        (``data.metrics[key]``) is updated to the latest value so that
        run listings / comparisons always reflect the most recent state.

        Parameters
        ----------
        key : str
            Metric name (e.g. ``"train/loss"``, ``"qnn/fidelity"``).
        value : float
            Metric value (will be converted to float).
        step : int, optional
            Step number for time series tracking. If provided, the metric
            is stored as a time series point **and** the scalar summary is
            updated. If ``None``, stores only as a scalar summary.

        Raises
        ------
        TypeError
            If step is not an integer.
        ValueError
            If step is negative.
        """
        value_f = float(value)

        if step is not None:
            if not isinstance(step, int):
                raise TypeError(
                    f"step must be an int or None, got {type(step).__name__}"
                )
            if step < 0:
                raise ValueError(f"step must be non-negative, got {step}")

            ts = utc_now_iso()
            point = {"value": value_f, "step": step, "timestamp": ts}

            with self._lock:
                # Summary always reflects the latest value — used for
                # run listings, comparisons, and sorting.
                self.record["data"]["metrics"][key] = value_f

                # Buffer for incremental flush to metric_points table.
                # NOT stored in record["data"]["metric_series"] during
                # the run to keep the record lean for periodic flushes.
                # metric_series is reconstructed at finalize time.
                self._metric_buffer.append(
                    {"run_id": self._run_id, "key": key, **point}
                )

            logger.debug("Logged metric series: %s[%d]=%f", key, step, value_f)
        else:
            with self._lock:
                self.record["data"]["metrics"][key] = value_f
            logger.debug("Logged metric: %s=%f", key, value_f)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Log multiple metrics at once.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric name-value pairs.
        """
        for key, value in metrics.items():
            self.log_metric(key, value)

    def flush(self) -> None:
        """
        Persist the current run state and buffered metric history.

        Called automatically by the background flush worker every few
        seconds while the run is active.  Can also be called manually
        for immediate persistence (e.g. before a known-risky operation).

        Thread-safe: acquires the internal lock only briefly to swap
        out the buffer.
        """
        with self._lock:
            buf = self._metric_buffer
            self._metric_buffer = []

        # Flush buffered metric points
        if buf and hasattr(self._registry, "save_metric_points"):
            try:
                self._registry.save_metric_points(buf)
            except Exception:
                logger.exception("Failed to flush metric points buffer")
                # Re-queue so data is not lost
                with self._lock:
                    self._metric_buffer = buf + self._metric_buffer

        # Persist run record (summary + status)
        try:
            self._registry.save(self.record)
        except Exception:
            logger.exception("Failed to flush run record")

        logger.debug("Flushed run state: run_id=%s", self._run_id)

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a string tag.

        Tags are string key-value pairs for categorization and filtering.

        Parameters
        ----------
        key : str
            Tag name.
        value : str
            Tag value (will be converted to string).
        """
        str_value = str(value)
        with self._lock:
            self.record["data"]["tags"][key] = str_value
        logger.debug("Set tag: %s=%s", key, value)

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set multiple tags at once.

        Parameters
        ----------
        tags : dict
            Dictionary of tag name-value pairs.
        """
        for key, value in tags.items():
            self.set_tag(key, value)

    # -----------------------------------------------------------------------
    # Artifact logging methods
    # -----------------------------------------------------------------------

    def log_text(
        self,
        name: str,
        text: str,
        kind: str = "text.note",
        role: str = "artifact",
        encoding: str = "utf-8",
        meta: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """
        Log a plain-text artifact.

        Parameters
        ----------
        name : str
            Artifact name.
        text : str
            Text content.
        kind : str, optional
            Artifact type identifier. Default is "text.note".
        role : str, optional
            Logical role. Default is "artifact".
        encoding : str, optional
            Text encoding. Default is "utf-8".
        meta : dict, optional
            Additional metadata.

        Returns
        -------
        ArtifactRef
            Reference to the stored artifact.
        """
        meta_out: dict[str, Any] = {"name": name, "filename": name}
        if meta:
            meta_out.update(meta)

        data = text.encode(encoding)
        return self.log_bytes(
            kind=kind,
            data=data,
            media_type=f"text/plain; charset={encoding}",
            role=role,
            meta=meta_out,
        )

    def log_bytes(
        self,
        kind: str,
        data: bytes,
        media_type: str,
        role: str = "artifact",
        meta: dict[str, Any] | None = None,
        *,
        max_bytes: int | None = None,
        truncate: bool = False,
    ) -> ArtifactRef:
        """
        Log a binary artifact.

        Parameters
        ----------
        kind : str
            Artifact type identifier (e.g., "qiskit.qpy.circuits").
        data : bytes
            Binary content.
        media_type : str
            MIME type (e.g., "application/x-qpy").
        role : str, optional
            Logical role. Default is "artifact".
        meta : dict, optional
            Additional metadata.
        max_bytes : int, optional
            Maximum allowed size in bytes. Defaults to ``MAX_ARTIFACT_BYTES``.
        truncate : bool, optional
            If True and data exceeds max_bytes, truncate data.

        Returns
        -------
        ArtifactRef
            Reference to the stored artifact.

        Raises
        ------
        ValueError
            If data exceeds max_bytes and truncate is False.
        """
        limit = max_bytes if max_bytes is not None else MAX_ARTIFACT_BYTES
        size = len(data)

        if limit > 0 and size > limit:
            if truncate:
                full_digest = sha256_bytes(data)
                data = data[:limit]
                meta = dict(meta) if meta else {}
                meta["truncated"] = True
                meta["original_size"] = size
                meta["original_digest"] = full_digest
                logger.warning(
                    "Artifact truncated: kind=%s, original_size=%d, limit=%d",
                    kind,
                    size,
                    limit,
                )
            else:
                raise ValueError(
                    f"Artifact size ({size} bytes) exceeds limit ({limit} bytes). "
                    f"Set truncate=True to allow truncation or increase max_bytes."
                )

        digest = self._store.put_bytes(data)
        ref = ArtifactRef(
            kind=kind,
            digest=digest,
            media_type=media_type,
            role=role,
            meta=meta,
        )
        with self._lock:
            self._artifacts.append(ref)
        logger.debug(
            "Logged artifact: kind=%s, role=%s, digest=%s...", kind, role, digest[:24]
        )
        return ref

    def log_json(
        self,
        name: str,
        obj: Any,
        role: str = "artifact",
        kind: str | None = None,
    ) -> ArtifactRef:
        """
        Log a JSON artifact.

        Parameters
        ----------
        name : str
            Artifact name.
        obj : Any
            Object to serialize as JSON.
        role : str, optional
            Logical role. Default is "artifact".
        kind : str, optional
            Artifact type identifier. Defaults to "json.{name}".

        Returns
        -------
        ArtifactRef
            Reference to the stored artifact.
        """
        data = json_dumps(obj, normalize_floats=True).encode("utf-8")
        return self.log_bytes(
            kind=kind or f"json.{name}",
            data=data,
            media_type="application/json",
            role=role,
            meta={"name": name},
        )

    def log_envelope(self, envelope: ExecutionEnvelope) -> bool:
        """
        Validate and log execution envelope.

        This is the canonical validation function that all adapters shall use.
        For adapter runs, invalid envelope raises EnvelopeValidationError.
        For manual runs, invalid envelope is logged but execution continues.

        Parameters
        ----------
        envelope : ExecutionEnvelope
            Completed envelope to validate and log.

        Returns
        -------
        bool
            True if envelope was valid, False otherwise.

        Raises
        ------
        EnvelopeValidationError
            If adapter run produces invalid envelope (strict enforcement).
        """
        validation = envelope.validate_schema()

        is_adapter_run = (
            envelope.producer.adapter
            and envelope.producer.adapter != "manual"
            and envelope.producer.adapter != ""
        )

        if validation.ok:
            self.log_json(
                name="execution_envelope",
                obj=envelope.to_dict(),
                role="envelope",
                kind="devqubit.envelope.json",
            )
            logger.debug("Logged valid execution envelope")
        else:
            if is_adapter_run:
                error_details = [str(e) for e in validation.errors]
                raise EnvelopeValidationError(
                    adapter=envelope.producer.adapter,
                    errors=error_details,
                )

            logger.warning(
                "Envelope validation failed (manual run, continuing): %d errors",
                validation.error_count,
            )

            self.log_json(
                name="envelope_validation_error",
                obj={
                    "errors": [str(e) for e in validation.errors],
                    "error_count": validation.error_count,
                },
                role="config",
                kind="devqubit.envelope.validation_error.json",
            )

            with self._lock:
                self.record["envelope_validation_error"] = {
                    "errors": [str(e) for e in validation.errors],
                    "count": validation.error_count,
                }

            self.log_json(
                name="execution_envelope_invalid",
                obj=envelope.to_dict(),
                role="envelope",
                kind="devqubit.envelope.invalid.json",
            )

        return validation.valid

    def log_file(
        self,
        path: str | Path,
        kind: str,
        role: str = "artifact",
        media_type: str | None = None,
    ) -> ArtifactRef:
        """
        Log a file as an artifact.

        Parameters
        ----------
        path : str or Path
            Path to the file.
        kind : str
            Artifact type identifier.
        role : str, optional
            Logical role. Default is "artifact".
        media_type : str, optional
            MIME type. Defaults to "application/octet-stream".

        Returns
        -------
        ArtifactRef
            Reference to the stored artifact.
        """
        path = Path(path)
        data = path.read_bytes()
        return self.log_bytes(
            kind=kind,
            data=data,
            media_type=media_type or "application/octet-stream",
            role=role,
            meta={"filename": path.name},
        )

    def log_openqasm3(
        self,
        source: str | Sequence[str] | Sequence[dict[str, Any]] | dict[str, str],
        *,
        name: str = "program",
        role: str = "program",
        anchor: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Log OpenQASM 3 program(s).

        Supports both single-circuit and multi-circuit runs.

        Parameters
        ----------
        source : str, sequence, or dict
            OpenQASM3 input(s). Accepts single string, list of strings,
            list of dicts with "name"/"source" keys, or dict mapping names.
        name : str, optional
            Logical name for the source. Default is "program".
        role : str, optional
            Artifact role. Default is "program".
        anchor : bool, optional
            Write stable pointers under ``record["program"]``. Default is True.
        meta : dict, optional
            Extra metadata for artifacts.

        Returns
        -------
        dict
            Result with keys: ``items`` (list) and ``ref`` (single-circuit).
        """
        items_in = coerce_openqasm3_sources(source, default_name=name)

        out_items: list[dict[str, Any]] = []
        meta_base: dict[str, Any] = {"name": name}
        if meta:
            meta_base.update(meta)

        for item in items_in:
            prog_name = item["name"]
            prog_source = item["source"]
            prog_index = int(item["index"])

            meta_item = {
                **meta_base,
                "program_name": prog_name,
                "program_index": prog_index,
            }

            ref = self.log_bytes(
                kind="source.openqasm3",
                data=prog_source.encode("utf-8"),
                media_type="application/openqasm",
                role=role,
                meta={**meta_item, "qasm_version": "3.0"},
            )

            out_items.append(
                {
                    "name": prog_name,
                    "index": prog_index,
                    "ref": ref,
                }
            )

        if anchor:
            with self._lock:
                prog = self.record.setdefault("program", {})
                oq3_list = prog.setdefault("openqasm3", [])

                if not isinstance(oq3_list, list):
                    raise TypeError("record['program']['openqasm3'] must be a list")

                for item in out_items:
                    ref = item["ref"]
                    entry: dict[str, Any] = {
                        "name": item["name"],
                        "index": int(item["index"]),
                        "kind": ref.kind,
                        "digest": ref.digest,
                        "media_type": ref.media_type,
                        "role": ref.role,
                    }
                    oq3_list.append(entry)

        result: dict[str, Any] = {"items": out_items}
        if len(out_items) == 1:
            result["ref"] = out_items[0]["ref"]

        logger.debug("Logged %d OpenQASM3 program(s)", len(out_items))
        return result

    # -----------------------------------------------------------------------
    # Executor wrapping
    # -----------------------------------------------------------------------

    def wrap(self, executor: Any, **kwargs: Any) -> Any:
        """
        Wrap an executor (backend/device) for automatic tracking.

        Parameters
        ----------
        executor : Any
            SDK executor (e.g., Qiskit backend, PennyLane device).
        **kwargs : Any
            Adapter-specific options.

        Returns
        -------
        Any
            Wrapped executor with the same interface as the original.

        Raises
        ------
        ValueError
            If no adapter supports the given executor type.
        """
        from devqubit_engine.adapters import resolve_adapter

        adapter = resolve_adapter(executor)

        with self._lock:
            self.record["adapter"] = adapter.name
            self._adapter = adapter.name

            desc = adapter.describe_executor(executor)
            self.record["backend"] = desc

        logger.debug("Wrapped executor with adapter: %s", adapter.name)
        return adapter.wrap_executor(executor, self, **kwargs)

    # -----------------------------------------------------------------------
    # Status management
    # -----------------------------------------------------------------------

    def fail(
        self,
        error: BaseException | None = None,
        *,
        exc_type: type[BaseException] | None = None,
        exc_tb: Any = None,
        status: str = "FAILED",
    ) -> None:
        """
        Mark the run as failed and record exception details.

        Parameters
        ----------
        error : BaseException, optional
            Exception that caused the failure.
        exc_type : type, optional
            Exception type for traceback formatting.
        exc_tb : Any, optional
            Traceback object for formatting.
        status : str, optional
            Status to set. Default is "FAILED".
        """
        with self._lock:
            self.record["info"]["status"] = status
            self.record["info"]["ended_at"] = utc_now_iso()

        if error is None:
            logger.info("Run marked as %s: %s", status, self._run_id)
            return

        etype = exc_type or type(error)
        tb = exc_tb if exc_tb is not None else getattr(error, "__traceback__", None)
        formatted = "".join(_tb.format_exception(etype, error, tb))

        with self._lock:
            self.record.setdefault("errors", []).append(
                {
                    "type": etype.__name__,
                    "message": str(error),
                    "traceback": formatted,
                }
            )

        logger.warning(
            "Run %s: %s - %s: %s",
            status,
            self._run_id,
            etype.__name__,
            str(error),
        )

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _has_valid_envelope(self) -> bool:
        """Check if a valid envelope artifact exists."""
        with self._lock:
            for artifact in self._artifacts:
                if (
                    artifact.role == "envelope"
                    and artifact.kind == "devqubit.envelope.json"
                ):
                    return True
        return False

    def _load_all_envelopes(self) -> list[ExecutionEnvelope]:
        """
        Load all existing envelope artifacts.

        Returns
        -------
        list of ExecutionEnvelope
            All loaded envelopes (may be empty).
        """
        from devqubit_engine.storage.artifacts.io import load_artifact_json

        envelopes: list[ExecutionEnvelope] = []

        with self._lock:
            envelope_artifacts = [
                a
                for a in self._artifacts
                if a.role == "envelope" and a.kind == "devqubit.envelope.json"
            ]

        for artifact in envelope_artifacts:
            try:
                envelope_data = load_artifact_json(artifact, self._store)
                if isinstance(envelope_data, dict):
                    envelope = ExecutionEnvelope.from_dict(envelope_data)
                    envelopes.append(envelope)
            except Exception as e:
                logger.warning("Failed to load envelope: %s", e)

        return envelopes

    def _build_final_envelopes(self) -> list[ExecutionEnvelope]:
        """
        Build or load final envelopes for finalization.

        For adapter runs: loads all existing envelopes from artifacts.
        For manual runs: synthesizes single envelope from run record.

        Returns
        -------
        list of ExecutionEnvelope
            Final envelopes. Empty list if adapter run has no envelope (error).
        """
        from devqubit_engine.uec.api.synthesize import synthesize_envelope

        is_manual = is_manual_run_record(self.record)

        if self._has_valid_envelope():
            # Adapter run with existing envelope(s) - load all
            return self._load_all_envelopes()

        if not is_manual:
            # Adapter run without envelope - this is an error
            adapter = self.record.get("adapter", "unknown")
            with self._lock:
                self.record["info"]["status"] = "FAILED"
                self.record.setdefault("errors", []).append(
                    {
                        "type": "MissingExecutionEnvelope",
                        "message": (
                            f"Adapter run (adapter={adapter}) completed without "
                            f"creating execution envelope. This is an adapter "
                            f"integration error - adapters must create envelopes."
                        ),
                        "adapter": adapter,
                    }
                )

            logger.error(
                "Adapter run '%s' (adapter=%s) completing without envelope. "
                "Run marked as FAILED.",
                self._run_id,
                adapter,
            )
            return []

        # Manual run - synthesize single envelope
        with self._lock:
            record_copy = dict(self.record)
            artifacts_copy = list(self._artifacts)

        try:
            temp_record = RunRecord(record=record_copy, artifacts=artifacts_copy)
            envelope = synthesize_envelope(temp_record, self._store)
            envelope.metadata["auto_generated"] = True
            logger.debug("Synthesized envelope for manual run %s", self._run_id)
            return [envelope]
        except Exception as e:
            logger.warning("Failed to synthesize envelope: %s", e)
            return []

    def _compute_fingerprints_from_envelopes(
        self,
        envelopes: list[ExecutionEnvelope],
    ) -> dict[str, str]:
        """
        Compute fingerprints from one or more envelopes.

        Parameters
        ----------
        envelopes : list of ExecutionEnvelope
            Envelopes to compute fingerprints from.

        Returns
        -------
        dict
            Computed fingerprints.
        """
        if not envelopes:
            return {}

        if len(envelopes) == 1:
            return compute_fingerprints_from_envelope(envelopes[0])

        # Multi-envelope: aggregate fingerprints
        return compute_fingerprints_from_envelopes(envelopes)

    def _enrich_envelopes(
        self,
        envelopes: list[ExecutionEnvelope],
        fingerprints: dict[str, str],
    ) -> None:
        """
        Enrich all envelopes with tracker namespace and fingerprints.

        Parameters
        ----------
        envelopes : list of ExecutionEnvelope
            Envelopes to enrich (modified in-place).
        fingerprints : dict
            Fingerprints to add.
        """
        from devqubit_engine.uec.api.synthesize import (
            add_fingerprints_to_envelope,
            enrich_envelope_with_tracker,
        )

        with self._lock:
            record_copy = dict(self.record)

        for envelope in envelopes:
            if "tracker" not in envelope.metadata:
                enrich_envelope_with_tracker(envelope, record_copy, fingerprints)
            else:
                add_fingerprints_to_envelope(envelope, fingerprints)

    def _finalize_pending_tracked_jobs(self) -> None:
        """Finalize any tracked jobs that never had .result() called."""
        if not self._pending_tracked_jobs:
            return

        logger.debug(
            "Finalizing %d pending tracked job(s) for run '%s'",
            len(self._pending_tracked_jobs),
            self._run_id,
        )

        for job in self._pending_tracked_jobs:
            try:
                if hasattr(job, "finalize_as_pending"):
                    job.finalize_as_pending()
            except Exception as e:
                logger.debug("Failed to finalize pending job: %s", e)

        self._pending_tracked_jobs.clear()

    def _persist_enriched_envelopes(
        self,
        envelopes: list[ExecutionEnvelope],
    ) -> None:
        """
        Write enriched envelopes to the object store and update artifact refs.

        For adapter runs (envelopes loaded from existing artifacts), the
        corresponding artifact references are replaced in-place, avoiding a
        delete/re-add cycle.  For manual runs (synthesized envelopes with no
        prior artifact), new references are appended.

        Parameters
        ----------
        envelopes : list of ExecutionEnvelope
            Enriched envelopes ready to persist.
        """
        # Serialize all envelopes and write to object store
        new_refs: list[ArtifactRef] = []
        for envelope in envelopes:
            data = json_dumps(
                envelope.to_dict(),
                normalize_floats=True,
            ).encode("utf-8")
            digest = self._store.put_bytes(data)
            ref = ArtifactRef(
                kind="devqubit.envelope.json",
                digest=digest,
                media_type="application/json",
                role="envelope",
                meta={"name": "execution_envelope"},
            )
            new_refs.append(ref)

        with self._lock:
            # Collect indices of existing envelope artifacts
            old_indices = [
                idx
                for idx, a in enumerate(self._artifacts)
                if a.role == "envelope" and a.kind == "devqubit.envelope.json"
            ]

            # Replace existing refs in-place where possible
            for i, ref in enumerate(new_refs):
                if i < len(old_indices):
                    self._artifacts[old_indices[i]] = ref
                else:
                    self._artifacts.append(ref)

            # Remove leftover old refs if there were more old than new
            if len(old_indices) > len(new_refs):
                extra = old_indices[len(new_refs) :]
                for idx in reversed(extra):
                    self._artifacts.pop(idx)

        for ref in new_refs:
            logger.debug(
                "Persisted enriched envelope: digest=%s...",
                ref.digest[:24],
            )

    def _finalize(self, success: bool = True) -> None:
        """
        Finalize the run record and persist it.

        Envelope lifecycle:
        1. Build or load envelope(s) (single source of truth)
        2. Compute fingerprints from all envelopes
        3. Enrich all envelopes with tracker namespace + fingerprints
        4. Persist enriched envelopes (in-place artifact ref update)
        5. Flush remaining metric buffer
        6. Save to registry

        Parameters
        ----------
        success : bool, optional
            If True and status is "RUNNING", set to "FINISHED".
        """
        # Update status
        with self._lock:
            if success and self.record["info"]["status"] == "RUNNING":
                self.record["info"]["status"] = "FINISHED"
                self.record["info"]["ended_at"] = utc_now_iso()

        # Stop background flush worker before the final synchronous flush
        # to avoid racing with it.
        if self._flush_worker is not None:
            self._flush_worker.stop()
            self._flush_worker = None

        # Flush any remaining metric buffer before final save
        self.flush()

        # Finalize pending tracked jobs
        self._finalize_pending_tracked_jobs()

        # Build or load all envelopes
        envelopes = self._build_final_envelopes()

        fingerprints: dict[str, str] = {}
        if envelopes:
            # Compute fingerprints from all envelopes
            fingerprints = self._compute_fingerprints_from_envelopes(envelopes)

            # Enrich all envelopes with tracker namespace + fingerprints
            self._enrich_envelopes(envelopes, fingerprints)

            # Write enriched envelopes, updating artifact refs in-place
            self._persist_enriched_envelopes(envelopes)

        # Update record with fingerprints
        with self._lock:
            self.record["fingerprints"] = fingerprints
            self.record["artifacts"] = [a.to_dict() for a in self._artifacts]

            # Reconstruct metric_series from the metric_points table
            # so the finalized record JSON is complete (backward compat,
            # export, remote backends). During the run this was kept
            # out of the record to avoid O(n) JSON rewrites on flush.
            if hasattr(self._registry, "load_metric_series"):
                series = self._registry.load_metric_series(self._run_id)
                if series:
                    self.record["data"]["metric_series"] = series

            run_record = RunRecord(record=self.record, artifacts=list(self._artifacts))

        # Validate if enabled
        if self._config.validate:
            validate_run_record(run_record.to_dict())
            logger.debug("Run record validated successfully")

        # Save to registry
        self._registry.save(run_record.to_dict())

        # Release registry resources
        if hasattr(self._registry, "close"):
            self._registry.close()

        logger.info(
            "Run finalized: run_id=%s, status=%s, artifacts=%d, envelopes=%d",
            self._run_id,
            self.record["info"]["status"],
            len(self._artifacts),
            len(envelopes),
        )

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    def __enter__(self) -> Run:
        """
        Enter the run context and persist the initial record.

        The record is saved with ``status=RUNNING`` so that long-running
        hardware jobs are immediately visible in the registry and the
        frontend while the experiment is in progress. A background
        flush worker is started to periodically persist metric data.
        """
        self._registry.save(self.record)
        self._flush_worker = _FlushWorker(self, interval=_FLUSH_INTERVAL_SECONDS)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the run context, handling any exceptions."""
        if exc_type is not None:
            if exc_type is KeyboardInterrupt:
                status = "KILLED"
                error = (
                    exc_val
                    if isinstance(exc_val, BaseException)
                    else KeyboardInterrupt()
                )
            else:
                status = "FAILED"
                error = (
                    exc_val
                    if isinstance(exc_val, BaseException)
                    else Exception(str(exc_val))
                )

            self.fail(error, exc_type=exc_type, exc_tb=exc_tb, status=status)

            try:
                self._finalize(success=False)
            except Exception as finalize_error:
                with self._lock:
                    self.record.setdefault("errors", []).append(
                        {
                            "type": type(finalize_error).__name__,
                            "message": f"Finalization error: {finalize_error}",
                            "traceback": _tb.format_exc(),
                        }
                    )
                logger.exception("Error during run finalization")

            return False

        try:
            self._finalize(success=True)
        except Exception as finalize_error:
            with self._lock:
                self.record["info"]["status"] = "PARTIAL"
                self.record["info"]["finalization_error"] = str(finalize_error)
                self.record.setdefault("errors", []).append(
                    {
                        "type": type(finalize_error).__name__,
                        "message": f"Finalization error: {finalize_error}",
                        "traceback": _tb.format_exc(),
                    }
                )
            logger.exception("Error during run finalization (success path)")

            try:
                self._registry.save(self.record)
            except Exception:
                logger.exception("Failed to save run record after finalization error")

            raise finalize_error

        return False

    def __repr__(self) -> str:
        """Return a string representation of the run."""
        return (
            f"Run(run_id={self._run_id!r}, project={self._project!r}, "
            f"adapter={self._adapter!r}, status={self.status!r})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def track(
    project: str,
    adapter: str = "manual",
    run_name: str | None = None,
    store: ObjectStoreProtocol | None = None,
    registry: RegistryProtocol | None = None,
    config: Config | None = None,
    capture_env: bool = True,
    capture_git: bool = True,
    group_id: str | None = None,
    group_name: str | None = None,
    parent_run_id: str | None = None,
) -> Run:
    """
    Create a tracking context for a quantum experiment.

    This is the recommended entry point for tracking experiments.

    Parameters
    ----------
    project : str
        Project name for organizing runs.
    adapter : str, optional
        Adapter name. Auto-detected when using ``wrap()``. Default is "manual".
    run_name : str, optional
        Human-readable run name.
    store : ObjectStoreProtocol, optional
        Object store for artifacts.
    registry : RegistryProtocol, optional
        Run registry.
    config : Config, optional
        Configuration object.
    capture_env : bool, optional
        Capture environment on start. Default is True.
    capture_git : bool, optional
        Capture git provenance on start. Default is True.
    group_id : str, optional
        Group identifier for related runs.
    group_name : str, optional
        Human-readable group name.
    parent_run_id : str, optional
        Parent run ID for lineage tracking.

    Returns
    -------
    Run
        Run context manager.

    Examples
    --------
    >>> with track(project="bell_state") as run:
    ...     run.log_param("shots", 1000)
    ...     run.log_metric("fidelity", 0.95)
    """
    return Run(
        project=project,
        adapter=adapter,
        run_name=run_name,
        store=store,
        registry=registry,
        config=config,
        capture_env=capture_env,
        capture_git=capture_git,
        group_id=group_id,
        group_name=group_name,
        parent_run_id=parent_run_id,
    )


def wrap_backend(run: Run, backend: Any, **kwargs: Any) -> Any:
    """
    Wrap a quantum backend for automatic artifact tracking.

    Convenience function equivalent to ``run.wrap(backend, **kwargs)``.

    Parameters
    ----------
    run : Run
        Active experiment run from :func:`track`.
    backend : Any
        Quantum backend or device instance.
    **kwargs : Any
        Adapter-specific options.

    Returns
    -------
    Any
        Wrapped backend with the same interface as the original.

    Examples
    --------
    >>> with track(project="bell") as run:
    ...     backend = wrap_backend(run, AerSimulator())
    ...     job = backend.run(qc, shots=1000)
    """
    return run.wrap(backend, **kwargs)
