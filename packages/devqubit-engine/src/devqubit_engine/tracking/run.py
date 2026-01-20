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
    compute_fingerprints,
    compute_fingerprints_from_envelopes,
)
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.uec.errors import EnvelopeValidationError, MissingEnvelopeError
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.utils.common import sha256_digest, utc_now_iso
from devqubit_engine.utils.env import capture_environment, capture_git_provenance
from devqubit_engine.utils.qasm3 import coerce_openqasm3_sources
from devqubit_engine.utils.serialization import json_dumps, to_jsonable
from ulid import ULID


logger = logging.getLogger(__name__)

# Maximum artifact size in bytes (20 MB default)
MAX_ARTIFACT_BYTES: int = 20 * 1024 * 1024


def _has_valid_envelope(artifacts: list) -> bool:
    """
    Check if a valid envelope artifact exists in the artifact list.

    Only returns True if there is a valid envelope
    (kind="devqubit.envelope.json"). Invalid envelopes
    (kind="devqubit.envelope.invalid.json") are ignored.

    Parameters
    ----------
    artifacts : list of ArtifactRef
        List of artifact references to check.

    Returns
    -------
    bool
        True if at least one valid envelope artifact exists.
    """
    for artifact in artifacts:
        if artifact.role == "envelope" and artifact.kind == "devqubit.envelope.json":
            return True
    return False


def _synthesize_envelope_for_manual_run(
    record: dict[str, Any],
    artifacts: list,
    store: ObjectStoreProtocol,
) -> Any | None:
    """
    Synthesize an envelope for a manual run.

    This is called during finalization when a manual run doesn't have
    an explicit envelope. The synthesized envelope has limited semantics
    (no program hashes) but provides basic structure.

    Parameters
    ----------
    record : dict
        Raw run record dictionary.
    artifacts : list of ArtifactRef
        Current artifact list.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.

    Returns
    -------
    ExecutionEnvelope or None
        Synthesized envelope, or None if synthesis fails.

    Notes
    -----
    The synthesized envelope will have:

    - metadata.auto_generated=True
    - metadata.manual_run=True (set by synthesize_envelope)
    - No program hashes (engine cannot compute them)
    """
    try:
        from devqubit_engine.tracking.record import RunRecord
        from devqubit_engine.uec.api.synthesize import synthesize_envelope

        temp_record = RunRecord(record=record, artifacts=artifacts)
        envelope = synthesize_envelope(temp_record, store)
        envelope.metadata["auto_generated"] = True
        return envelope

    except Exception as e:
        logger.warning("Failed to synthesize envelope: %s", e)
        return None


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
        # Thread-safety lock for record and artifact mutations
        self._lock = threading.Lock()

        # Generate unique run ID
        ulid_gen = ULID()
        self._run_id = (
            ulid_gen.generate() if hasattr(ulid_gen, "generate") else str(ulid_gen)
        )
        self._project = project
        self._adapter = adapter
        self._run_name = run_name
        self._artifacts: list[ArtifactRef] = []

        # Get config (use provided or global)
        cfg = config or get_config()

        # Use provided backends or create from config
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
            self.record["group_id"] = group_id
        if group_name:
            self.record["group_name"] = group_name
        if parent_run_id:
            self.record["parent_run_id"] = parent_run_id

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

    @property
    def run_id(self) -> str:
        """
        Get the unique run identifier.

        Returns
        -------
        str
            ULID-based run identifier.
        """
        return self._run_id

    @property
    def status(self) -> str:
        """
        Get the current run status.

        Returns
        -------
        str
            One of: "RUNNING", "FINISHED", "FAILED", "KILLED".
        """
        return self.record.get("info", {}).get("status", "RUNNING")

    @property
    def store(self) -> ObjectStoreProtocol:
        """
        Get the object store for artifacts.

        Returns
        -------
        ObjectStoreProtocol
            Object store instance.
        """
        return self._store

    @property
    def registry(self) -> RegistryProtocol:
        """
        Get the run registry.

        Returns
        -------
        RegistryProtocol
            Registry instance.
        """
        return self._registry

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter value.

        Parameters are experimental configuration values that should
        remain constant during the run.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value. Will be converted to JSON-serializable form.
        """
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

        Metrics are numeric values that measure experimental outcomes.

        Parameters
        ----------
        key : str
            Metric name.
        value : float
            Metric value (will be converted to float).
        step : int, optional
            Step number for time series tracking. If provided, the metric
            is stored as a time series point. If None, stores as a scalar
            (overwrites previous value).

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

            # Time series mode
            with self._lock:
                if "metric_series" not in self.record["data"]:
                    self.record["data"]["metric_series"] = {}

                if key not in self.record["data"]["metric_series"]:
                    self.record["data"]["metric_series"][key] = []

                self.record["data"]["metric_series"][key].append(
                    {
                        "value": value_f,
                        "step": step,
                        "timestamp": utc_now_iso(),
                    }
                )
            logger.debug("Logged metric series: %s[%d]=%f", key, step, value_f)
        else:
            # Scalar mode (overwrites previous value)
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
            Common values: "program", "results", "device_snapshot".
        meta : dict, optional
            Additional metadata.
        max_bytes : int, optional
            Maximum allowed size in bytes. Defaults to ``MAX_ARTIFACT_BYTES``.
            Set to 0 or negative to disable size limit.
        truncate : bool, optional
            If True and data exceeds max_bytes, truncate data and add
            a digest of full content to metadata. If False (default),
            raise ValueError for oversized artifacts.

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
                full_digest = sha256_digest(data)
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
        It ensures identical validation behavior across all SDKs.

        For adapter runs, invalid envelope raises EnvelopeValidationError.
        For manual runs, invalid envelope is logged but execution continues.

        Multiple envelopes per run are allowed (e.g., one per circuit batch).

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
            This error MUST NOT be silenced by adapters.
        """
        # Validate envelope
        validation = envelope.validate_schema()

        # Check if this is an adapter run (strict mode)
        is_adapter_run = (
            envelope.producer.adapter
            and envelope.producer.adapter != "manual"
            and envelope.producer.adapter != ""
        )

        # Log based on validation result
        if validation.ok:
            # Log valid envelope
            self.log_json(
                name="execution_envelope",
                obj=envelope.to_dict(),
                role="envelope",
                kind="devqubit.envelope.json",
            )
            logger.debug("Logged valid execution envelope")
        else:
            # For adapter runs: strict enforcement - raise error
            if is_adapter_run:
                error_details = [str(e) for e in validation.errors]
                raise EnvelopeValidationError(
                    adapter=envelope.producer.adapter,
                    errors=error_details,
                )

            # For manual runs: log warning and continue
            logger.warning(
                "Envelope validation failed (manual run, continuing): %d errors",
                validation.error_count,
            )

            # Log validation errors for debugging
            self.log_json(
                name="envelope_validation_error",
                obj={
                    "errors": [str(e) for e in validation.errors],
                    "error_count": validation.error_count,
                },
                role="config",
                kind="devqubit.envelope.validation_error.json",
            )

            # Store summary in tracker record for visibility
            with self._lock:
                self.record["envelope_validation_error"] = {
                    "errors": [str(e) for e in validation.errors],
                    "count": validation.error_count,
                }

            # Log invalid envelope for debugging
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

        Supports both single-circuit and multi-circuit runs. For multiple
        programs, each is stored as separate artifacts with stable indices.

        Parameters
        ----------
        source : str, sequence, or dict
            OpenQASM3 input(s). Accepts:

            - Single string (one circuit)
            - List of strings (multiple circuits)
            - List of dicts with "name" and "source" keys
            - Dict mapping names to sources

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
            Result with keys:

            - ``items``: List of per-circuit results (each with name, index, ref)
            - ``ref``: Top-level convenience key for single-circuit input
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

        # Anchor pointers in the run record
        if anchor:
            with self._lock:
                prog = self.record.setdefault("program", {})
                oq3_list = prog.setdefault("openqasm3", [])

                if not isinstance(oq3_list, list):
                    raise TypeError("record['program']['openqasm3'] must be a list")

                for item in out_items:
                    entry: dict[str, Any] = {
                        "name": item["name"],
                        "index": int(item["index"]),
                        "kind": item["ref"].kind,
                        "digest": item["ref"].digest,
                    }
                    oq3_list.append(entry)

        result: dict[str, Any] = {"items": out_items}

        # Convenience key for single-circuit callers
        if len(out_items) == 1:
            result["ref"] = out_items[0]["ref"]

        logger.debug("Logged %d OpenQASM3 program(s)", len(out_items))
        return result

    def wrap(self, executor: Any, **kwargs: Any) -> Any:
        """
        Wrap an executor (backend/device) for automatic tracking.

        The wrapped executor intercepts execution calls and automatically
        logs circuits, results, and device snapshots.

        Parameters
        ----------
        executor : Any
            SDK executor (e.g., Qiskit backend, PennyLane device,
            Cirq sampler).
        **kwargs : Any
            Adapter-specific options forwarded to wrap_executor().

        Returns
        -------
        Any
            Wrapped executor with the same interface as the original.

        Raises
        ------
        ValueError
            If no adapter supports the given executor type.

        Examples
        --------
        >>> from qiskit_aer import AerSimulator
        >>> with track(project="test") as run:
        ...     backend = run.wrap(AerSimulator())
        ...     job = backend.run(circuit)
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
            Status to set. Default is "FAILED". Use "KILLED" for
            interrupts.
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

    def _has_envelope_artifact(self) -> bool:
        """
        Check if a valid envelope artifact exists.

        Only returns True if there is a valid envelope (kind="devqubit.envelope.json").
        Invalid envelopes (kind="devqubit.envelope.invalid.json") are ignored,
        allowing auto-generation to proceed.

        Returns
        -------
        bool
            True if valid envelope artifact exists in self._artifacts.
        """
        with self._lock:
            return _has_valid_envelope(self._artifacts)

    def _ensure_envelope(self) -> None:
        """
        Ensure an envelope artifact exists before finalization.

        For **manual runs only**: synthesizes envelope from run record.
        For **adapter runs**: missing envelope marks the run as FAILED with
        a structured error. Adapters MUST create envelopes - no exceptions.

        Notes
        -----
        This is called automatically during _finalize(). For manual runs,
        the synthesized envelope will have:

        - metadata.auto_generated=True
        - metadata.manual_run=True
        - No program hashes (engine cannot compute them)

        For adapter runs without envelope, the run is marked FAILED with
        a ``MissingExecutionEnvelope`` error in the errors list. The run
        is still persisted (for debugging) but marked as failed.
        """
        from devqubit_engine.utils.common import is_manual_run_record

        if self._has_envelope_artifact():
            logger.debug("Envelope artifact already exists, skipping auto-generation")
            return

        # Only auto-generate for manual runs
        if not is_manual_run_record(self.record):
            adapter = self.record.get("adapter", "unknown")

            # Mark run as FAILED with structured error
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
                "Run marked as FAILED. This is an adapter integration error.",
                self._run_id,
                adapter,
            )
            return

        # Build envelope for manual run
        with self._lock:
            record_copy = dict(self.record)
            artifacts_copy = list(self._artifacts)

        envelope = _synthesize_envelope_for_manual_run(
            record=record_copy, artifacts=artifacts_copy, store=self._store
        )

        if envelope is not None:
            # Log the envelope (skip validation for auto-generated)
            self.log_json(
                name="execution_envelope",
                obj=envelope.to_dict(),
                role="envelope",
                kind="devqubit.envelope.json",
            )
            logger.debug(
                "Auto-generated envelope for manual run %s",
                self._run_id,
            )
        else:
            logger.warning(
                "Failed to auto-generate envelope for run %s",
                self._run_id,
            )

    def _compute_fingerprints(self, run_record: RunRecord) -> dict[str, str]:
        """
        Compute fingerprints using canonical envelope resolution.

        Uses the canonical resolver (load_all_envelopes) to load envelopes
        and compute fingerprints. Handles errors gracefully by returning
        empty fingerprints and marking the run as FAILED.

        Parameters
        ----------
        run_record : RunRecord
            Run record with artifacts.

        Returns
        -------
        dict
            Fingerprints dictionary. Empty dict if envelope resolution fails.
        """
        from devqubit_engine.uec.api.resolve import load_all_envelopes

        try:
            envelopes = load_all_envelopes(run_record, self._store)

            if not envelopes:
                # No envelopes found - use artifact-based fallback
                logger.debug("No envelopes found, using artifact-based fingerprints")
                return compute_fingerprints(run_record, envelope=None)

            if len(envelopes) == 1:
                return compute_fingerprints(run_record, envelope=envelopes[0])

            # Multi-envelope: aggregate fingerprints
            return compute_fingerprints_from_envelopes(envelopes)

        except (MissingEnvelopeError, EnvelopeValidationError) as e:
            adapter = self.record.get("adapter", "unknown")
            error_type = type(e).__name__

            with self._lock:
                self.record["info"]["status"] = "FAILED"
                self.record.setdefault("errors", []).append(
                    {
                        "type": error_type,
                        "message": str(e),
                        "adapter": adapter,
                    }
                )

            logger.error(
                "Envelope error during fingerprinting for run '%s': %s - %s",
                self._run_id,
                error_type,
                str(e),
            )
            return {}

        except Exception as e:
            logger.warning(
                "Unexpected error computing fingerprints for run '%s': %s",
                self._run_id,
                e,
            )
            # Fall back to artifact-based fingerprints
            return compute_fingerprints(run_record, envelope=None)

    def _finalize(self, success: bool = True) -> None:
        """
        Finalize the run record and persist it.

        Parameters
        ----------
        success : bool, optional
            If True and status is "RUNNING", set to "FINISHED".
        """
        with self._lock:
            if success and self.record["info"]["status"] == "RUNNING":
                self.record["info"]["status"] = "FINISHED"
                self.record["info"]["ended_at"] = utc_now_iso()

        # Ensure envelope exists (auto-generate if needed)
        self._ensure_envelope()

        # Serialize artifacts and compute fingerprints
        with self._lock:
            self.record["artifacts"] = [a.to_dict() for a in self._artifacts]
            run_record = RunRecord(record=self.record, artifacts=list(self._artifacts))

        # Compute fingerprints using canonical envelope resolution
        fingerprints = self._compute_fingerprints(run_record)
        with self._lock:
            self.record["fingerprints"] = fingerprints

        # Validate if enabled
        if self._config.validate:
            validate_run_record(run_record.to_dict())
            logger.debug("Run record validated successfully")

        # Save to registry
        self._registry.save(run_record.to_dict())

        logger.info(
            "Run finalized: run_id=%s, status=%s, artifacts=%d",
            self._run_id,
            self.record["info"]["status"],
            len(self._artifacts),
        )

    def __enter__(self) -> Run:
        """Enter the run context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the run context, handling any exceptions."""
        if exc_type is not None:
            # Determine status based on exception type
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
                # Best-effort: preserve original exception, record finalization error
                with self._lock:
                    self.record.setdefault("errors", []).append(
                        {
                            "type": type(finalize_error).__name__,
                            "message": f"Finalization error: {finalize_error}",
                            "traceback": _tb.format_exc(),
                        }
                    )
                logger.exception("Error during run finalization")

            return False  # Propagate original exception

        # Success path - wrap in try/except for robustness
        try:
            self._finalize(success=True)
        except Exception as finalize_error:
            # Best-effort: record error but don't raise on exit
            with self._lock:
                self.record.setdefault("errors", []).append(
                    {
                        "type": type(finalize_error).__name__,
                        "message": f"Finalization error: {finalize_error}",
                        "traceback": _tb.format_exc(),
                    }
                )
            logger.exception("Error during run finalization (success path)")

            # Attempt to save the run record even with incomplete data
            try:
                self._registry.save(self.record)
            except Exception:
                logger.exception("Failed to save run record after finalization error")

        return False

    def __repr__(self) -> str:
        """Return a string representation of the run."""
        return (
            f"Run(run_id={self._run_id!r}, project={self._project!r}, "
            f"adapter={self._adapter!r}, status={self.status!r})"
        )


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
        Adapter name. Auto-detected when using ``wrap()``.
        Default is "manual".
    run_name : str, optional
        Human-readable run name.
    store : ObjectStoreProtocol, optional
        Object store for artifacts. Created from config if not provided.
    registry : RegistryProtocol, optional
        Run registry. Created from config if not provided.
    config : Config, optional
        Configuration object. Uses global config if not provided.
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

    Grouped runs:

    >>> sweep_id = "sweep_20240101"
    >>> for shots in [100, 1000, 10000]:
    ...     with track(project="bell_state", group_id=sweep_id) as run:
    ...         run.log_param("shots", shots)
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
    The wrapped backend intercepts execution calls and automatically
    logs circuits, results, and device snapshots.

    Parameters
    ----------
    run : Run
        Active experiment run from :func:`track`.
    backend : Any
        Quantum backend or device instance.
    **kwargs : Any
        Adapter-specific options forwarded to the adapter.

    Returns
    -------
    Any
        Wrapped backend with the same interface as the original.

    Raises
    ------
    ValueError
        If no adapter supports the given backend type.

    Examples
    --------
    >>> from devqubit_engine.tracking.run import track, wrap_backend
    >>> from qiskit_aer import AerSimulator
    >>>
    >>> with track(project="bell") as run:
    ...     backend = wrap_backend(run, AerSimulator())
    ...     job = backend.run(qc, shots=1000)
    """
    return run.wrap(backend, **kwargs)
