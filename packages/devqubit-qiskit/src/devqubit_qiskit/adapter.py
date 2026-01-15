# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Qiskit adapter for devqubit tracking system.

Provides integration with Qiskit backends, enabling automatic
tracking of quantum circuit execution, results, and device configurations
following the devqubit Uniform Execution Contract (UEC).

The adapter produces an ExecutionEnvelope containing four canonical snapshots:
- DeviceSnapshot: Backend state and calibration
- ProgramSnapshot: Logical circuit artifacts
- ExecutionSnapshot: Submission and job metadata
- ResultSnapshot: Normalized measurement results

Supported Backends
------------------
This adapter supports Qiskit BackendV2 implementations including:
- qiskit-aer simulators (AerSimulator, etc.)
- qiskit-ibm-runtime backends (when used directly, not via primitives)
- Fake backends for testing

Note: Legacy BackendV1 is not supported. Use BackendV2-based backends.
For Runtime primitives (SamplerV2, EstimatorV2), use the qiskit-runtime adapter.

Example
-------
>>> from qiskit import QuantumCircuit
>>> from qiskit_aer import AerSimulator
>>> from devqubit_engine.core import track
>>>
>>> qc = QuantumCircuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> qc.measure_all()
>>>
>>> with track(project="my_experiment") as run:
...     backend = run.wrap(AerSimulator())
...     job = backend.run(qc, shots=1000)
...     result = job.result()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.core.run import Run
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import ProgramSnapshot
from devqubit_engine.uec.result import ResultSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.circuits import (
    compute_parametric_hash,
    compute_structural_hash,
    materialize_circuits,
    serialize_and_log_circuits,
)
from devqubit_qiskit.envelope import (
    create_execution_snapshot,
    create_failure_result_snapshot,
    create_program_snapshot,
    create_result_snapshot,
    detect_physical_provider,
    finalize_envelope_with_result,
    log_device_snapshot,
)
from devqubit_qiskit.utils import (
    extract_job_id,
    get_adapter_version,
    get_backend_name,
    qiskit_version,
)
from qiskit.providers.backend import BackendV2


logger = logging.getLogger(__name__)


@dataclass
class TrackedJob:
    """
    Wrapper for Qiskit job that tracks result retrieval.

    This class wraps a Qiskit job and logs artifacts when
    results are retrieved, producing a ResultSnapshot and
    finalizing the ExecutionEnvelope.

    Parameters
    ----------
    job : Any
        Original Qiskit job instance.
    tracker : Run
        Tracker instance for logging.
    backend_name : str
        Name of the backend that created this job.
    should_log_results : bool
        Whether to log results for this job.
    envelope : ExecutionEnvelope or None
        Envelope to finalize when result() is called.

    Attributes
    ----------
    job : Any
        The wrapped Qiskit job.
    tracker : Run
        The active run tracker.
    backend_name : str
        Backend name for metadata.
    result_snapshot : ResultSnapshot or None
        Captured result snapshot after result() is called.
    """

    job: Any
    tracker: Run
    backend_name: str
    should_log_results: bool = True
    envelope: ExecutionEnvelope | None = None

    # Set after result() is called
    result_snapshot: ResultSnapshot | None = field(default=None, init=False, repr=False)
    _result_logged: bool = field(default=False, init=False, repr=False)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve job result and log artifacts.

        Always creates an envelope - even when job.result() fails.
        This is a UEC requirement: envelope must exist for
        failure cases to enable debugging and telemetry.

        Idempotent: calling result() multiple times will only log once.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to job.result().
        **kwargs : Any
            Keyword arguments passed to job.result().

        Returns
        -------
        Result
            Qiskit Result object.

        Raises
        ------
        Exception
            Re-raises any exception from job.result() after logging
            the failure envelope.
        """
        # Try to get result - may raise exception
        try:
            result = self.job.result(*args, **kwargs)
        except Exception as exc:
            # ALWAYS create failure envelope before re-raising
            if self.should_log_results and not self._result_logged:
                self._result_logged = True
                self._log_failure(exc)
            raise  # Re-raise original exception with original traceback

        # Happy path - log successful result
        if self.should_log_results and not self._result_logged:
            self._result_logged = True

            try:
                # Create result snapshot
                self.result_snapshot = create_result_snapshot(
                    self.tracker,
                    self.backend_name,
                    result,
                )

                # Finalize envelope with result
                if self.envelope is not None and self.result_snapshot is not None:
                    finalize_envelope_with_result(
                        self.tracker,
                        self.envelope,
                        self.result_snapshot,
                    )

                # Update tracker record
                if self.result_snapshot is not None:
                    self.tracker.record["results"] = {
                        "completed_at": utc_now_iso(),
                        "backend_name": self.backend_name,
                        "success": self.result_snapshot.success,
                        "status": self.result_snapshot.status,
                        "num_items": len(self.result_snapshot.items),
                        **self.result_snapshot.metadata,
                    }

                logger.debug("Logged results on %s", self.backend_name)

            except Exception as e:
                # Log error but don't fail - result retrieval should succeed
                logger.warning(
                    "Failed to log results for %s: %s",
                    self.backend_name,
                    e,
                )
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "result_logging_failed",
                        "message": str(e),
                        "backend_name": self.backend_name,
                    }
                )

        return result

    def _log_failure(self, exc: Exception) -> None:
        """
        Log failure envelope when job.result() raises an exception.

        Parameters
        ----------
        exc : Exception
            The exception that was raised.
        """
        try:
            # Create failure result snapshot
            self.result_snapshot = create_failure_result_snapshot(
                exception=exc,
                backend_name=self.backend_name,
            )

            # Finalize envelope with failure result
            if self.envelope is not None:
                finalize_envelope_with_result(
                    self.tracker,
                    self.envelope,
                    self.result_snapshot,
                )

            # Update tracker record with failure info
            self.tracker.record["results"] = {
                "completed_at": utc_now_iso(),
                "backend_name": self.backend_name,
                "success": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }

            logger.debug(
                "Logged failure envelope for %s: %s",
                self.backend_name,
                type(exc).__name__,
            )

        except Exception as log_error:
            # Last resort - don't let logging failure mask original error
            logger.error(
                "Failed to log failure envelope for %s: %s",
                self.backend_name,
                log_error,
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped job."""
        return getattr(self.job, name)

    def __repr__(self) -> str:
        """Return string representation."""
        job_id = extract_job_id(self.job) or "unknown"
        return f"TrackedJob(backend={self.backend_name!r}, job_id={job_id!r})"


@dataclass
class TrackedBackend:
    """
    Wrapper for Qiskit backend that tracks circuit execution.

    This class wraps a Qiskit backend and logs circuits,
    execution parameters, and device snapshots when circuits
    are submitted, following the UEC with minimal overhead.

    Parameters
    ----------
    backend : Any
        Original Qiskit backend instance (must be BackendV2-compatible).
    tracker : Run
        Tracker instance for logging.
    log_every_n : int
        Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update stats every N executions (default 1000).

    Attributes
    ----------
    backend : Any
        The wrapped Qiskit backend.
    tracker : Run
        The active run tracker.
    device_snapshot : DeviceSnapshot or None
        Cached device snapshot (created once per run).

    Notes
    -----
    Logging Behavior for Parameter Sweeps
    -------------------------------------
    The default settings (log_every_n=0, log_new_circuits=True) log the
    first execution and any new circuit structures. For parameter sweeps
    where the same circuit is executed with different parameter values,
    only the first execution is logged since the circuit structure hash
    ignores parameter values.

    To log all parameter sweep points, use log_every_n=-1 or set
    log_every_n to a positive value for sampling.
    """

    backend: Any
    tracker: Run
    log_every_n: int = 0
    log_new_circuits: bool = True
    stats_update_interval: int = 1000

    # Internal state (not init params)
    _snapshot_logged: bool = field(default=False, init=False, repr=False)
    _execution_count: int = field(default=0, init=False, repr=False)
    _logged_execution_count: int = field(default=0, init=False, repr=False)
    _seen_circuit_hashes: set[str] = field(default_factory=set, init=False, repr=False)
    _logged_circuit_hashes: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _program_snapshot_cache: dict[str, ProgramSnapshot] = field(
        default_factory=dict, init=False, repr=False
    )

    # Cached device snapshot
    device_snapshot: DeviceSnapshot | None = field(default=None, init=False, repr=False)

    def run(self, circuits: Any, *args: Any, **kwargs: Any) -> TrackedJob:
        """
        Execute circuits and log artifacts based on sampling settings.

        Produces ExecutionEnvelope with DeviceSnapshot, ProgramSnapshot,
        and ExecutionSnapshot following the UEC.

        Parameters
        ----------
        circuits : QuantumCircuit or iterable
            Circuit(s) to execute.
        *args : Any
            Additional positional args passed to backend.run().
        **kwargs : Any
            Additional keyword args passed to backend.run() (e.g., shots).

        Returns
        -------
        TrackedJob
            Wrapped job that tracks result retrieval.
        """
        backend_name = get_backend_name(self.backend)
        submitted_at = utc_now_iso()

        # Materialize once to avoid consuming generators during logging
        circuit_list, was_single = materialize_circuits(circuits)

        # Payload for backend.run(): single circuit if user gave single, else list
        run_payload: Any = (
            circuit_list[0] if was_single and circuit_list else circuit_list
        )

        # Increment execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute hashes for structure detection and exact match
        structural_hash = compute_structural_hash(circuit_list)
        parametric_hash = compute_parametric_hash(circuit_list)
        is_new_circuit = (
            structural_hash and structural_hash not in self._seen_circuit_hashes
        )
        if structural_hash:
            self._seen_circuit_hashes.add(structural_hash)

        # Determine what to log based on settings
        should_log_structure = False
        should_log_results = False

        if self.log_every_n == -1:
            # Log all: structure if not logged, results always
            should_log_structure = structural_hash not in self._logged_circuit_hashes
            should_log_results = True
        elif exec_count == 1:
            # First execution: log everything
            should_log_structure = True
            should_log_results = True
        elif self.log_new_circuits and is_new_circuit:
            # New circuit structure: log structure + first result
            should_log_structure = True
            should_log_results = True
        elif self.log_every_n > 0 and exec_count % self.log_every_n == 0:
            # Sampling: log results only
            should_log_results = True

        # Fast path: nothing to log
        if not should_log_structure and not should_log_results:
            job = self.backend.run(run_payload, *args, **kwargs)

            if (
                self.stats_update_interval > 0
                and exec_count % self.stats_update_interval == 0
            ):
                self._update_stats()

            return TrackedJob(
                job=job,
                tracker=self.tracker,
                backend_name=backend_name,
                should_log_results=False,
                envelope=None,
            )

        # Detect physical provider (not SDK)
        detected_provider = detect_physical_provider(self.backend)

        # Set tags
        self.tracker.set_tag("backend_name", backend_name)
        self.tracker.set_tag("sdk", "qiskit")
        self.tracker.set_tag("adapter", "devqubit-qiskit")
        self.tracker.set_tag("provider", detected_provider)

        # Log device snapshot (once per run)
        if not self._snapshot_logged:
            self.device_snapshot = log_device_snapshot(self.backend, self.tracker)
            self._snapshot_logged = True

        # Build program snapshot
        program_snapshot: ProgramSnapshot | None = None
        if should_log_structure and circuit_list:
            # Log execution parameters
            shots = kwargs.get("shots")
            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", int(len(circuit_list)))

            # Check for parameter_binds (parameter sweep indicator)
            if kwargs.get("parameter_binds"):
                self.tracker.log_param(
                    "parameter_binds_count",
                    len(kwargs["parameter_binds"]),
                )

            # Log circuits
            program_artifacts = serialize_and_log_circuits(
                self.tracker,
                circuit_list,
                backend_name,
                structural_hash,
            )

            if structural_hash:
                self._logged_circuit_hashes.add(structural_hash)

            # Create program snapshot (UEC v1.0)
            program_snapshot = create_program_snapshot(
                program_artifacts,
                structural_hash,
                parametric_hash,
                len(circuit_list),
            )

            # Cache program snapshot for reuse in results-only logging
            if structural_hash:
                self._program_snapshot_cache[structural_hash] = program_snapshot

            # Update tracker record (used by fingerprint computation)
            self.tracker.record["backend"] = {
                "name": backend_name,
                "type": self.backend.__class__.__name__,
                "provider": detected_provider,
                "sdk": "qiskit",
            }

            self._logged_execution_count += 1

        # Reuse cached program snapshot when only logging results
        elif should_log_results and structural_hash in self._program_snapshot_cache:
            program_snapshot = self._program_snapshot_cache[structural_hash]

        # Build ExecutionSnapshot
        shots = kwargs.get("shots")
        execution_snapshot = create_execution_snapshot(
            submitted_at=submitted_at,
            shots=int(shots) if shots is not None else None,
            exec_count=exec_count,
            job_ids=[],  # Will be updated after job creation
            options={
                "args": to_jsonable(list(args)),
                "kwargs": to_jsonable(kwargs),
            },
        )

        # Update tracker record (used by fingerprint computation)
        self.tracker.record["execute"] = {
            "sdk": "qiskit",
            "submitted_at": submitted_at,
            "backend_name": backend_name,
            "num_circuits": len(circuit_list),
            "execution_count": exec_count,
            "structural_hash": structural_hash,
            "parametric_hash": parametric_hash,
            "args": to_jsonable(list(args)),
            "kwargs": to_jsonable(kwargs),
        }

        logger.debug(
            "Submitting %d circuits to %s",
            len(circuit_list),
            backend_name,
        )

        # Execute on actual backend
        job = self.backend.run(run_payload, *args, **kwargs)

        # Log job ID if available
        job_id = extract_job_id(job)
        if job_id:
            self.tracker.record["execute"]["job_ids"] = [job_id]
            execution_snapshot.job_ids = [job_id]
            logger.debug("Job ID: %s", job_id)

        # Create envelope (will be finalized when result() is called)
        envelope: ExecutionEnvelope | None = None
        if should_log_results and self.device_snapshot is not None:
            # Use existing or create minimal program snapshot
            if program_snapshot is None:
                program_snapshot = ProgramSnapshot(
                    logical=[],
                    physical=[],
                    structural_hash=structural_hash,
                    parametric_hash=parametric_hash,
                    num_circuits=len(circuit_list),
                )

            # Create ProducerInfo for SDK stack tracking
            producer = ProducerInfo.create(
                adapter="devqubit-qiskit",
                adapter_version=get_adapter_version(),
                sdk="qiskit",
                sdk_version=qiskit_version(),
                frontends=["qiskit"],
            )

            # Create envelope with pending result (will be updated in result())
            pending_result = ResultSnapshot(
                success=False,
                status="failed",  # Will be updated when result() completes
                items=[],
                metadata={"state": "pending"},
            )

            envelope = ExecutionEnvelope(
                envelope_id=uuid.uuid4().hex[:26],
                created_at=utc_now_iso(),
                producer=producer,
                result=pending_result,
                device=self.device_snapshot,
                program=program_snapshot,
                execution=execution_snapshot,
            )

        # Update stats
        self._update_stats()

        return TrackedJob(
            job=job,
            tracker=self.tracker,
            backend_name=backend_name,
            should_log_results=should_log_results,
            envelope=envelope,
        )

    def _update_stats(self) -> None:
        """Update execution statistics in tracker record."""
        self.tracker.record["execution_stats"] = {
            "total_executions": self._execution_count,
            "logged_executions": self._logged_execution_count,
            "unique_circuits": len(self._seen_circuit_hashes),
            "logged_circuits": len(self._logged_circuit_hashes),
            "last_execution_at": utc_now_iso(),
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped backend."""
        return getattr(self.backend, name)

    def __repr__(self) -> str:
        """Return string representation."""
        backend_name = get_backend_name(self.backend)
        run_id = getattr(self.tracker, "run_id", "unknown")
        return f"TrackedBackend(backend={backend_name!r}, run_id={run_id!r})"


class QiskitAdapter:
    """
    Adapter for integrating Qiskit backends with devqubit tracking.

    This adapter wraps Qiskit backends to automatically log circuits,
    execution parameters, device configurations, and results following
    the devqubit Uniform Execution Contract (UEC).

    Attributes
    ----------
    name : str
        Adapter identifier ("qiskit").

    Notes
    -----
    This adapter only supports BackendV2-based backends. Legacy BackendV1
    backends are not supported and will return False from ``supports_executor()``.

    For Runtime primitives (SamplerV2, EstimatorV2), use the ``qiskit-runtime``
    adapter instead.

    Example
    -------
    >>> from qiskit_aer import AerSimulator
    >>> adapter = QiskitAdapter()
    >>> assert adapter.supports_executor(AerSimulator())
    >>> desc = adapter.describe_executor(AerSimulator())
    >>> print(desc["name"])
    'aer_simulator'
    """

    name: str = "qiskit"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a supported Qiskit backend.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a Qiskit BackendV2.
        """
        return isinstance(executor, BackendV2)

    def describe_executor(self, executor: Any) -> dict[str, Any]:
        """
        Create a description of the backend.

        Parameters
        ----------
        executor : Any
            Qiskit backend instance.

        Returns
        -------
        dict
            Backend description with keys: name, type, provider, sdk.
        """
        return {
            "name": get_backend_name(executor),
            "type": executor.__class__.__name__,
            "provider": detect_physical_provider(executor),
            "sdk": "qiskit",
        }

    def wrap_executor(
        self,
        executor: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedBackend:
        """
        Wrap a backend with tracking capabilities.

        Parameters
        ----------
        executor : Any
            Qiskit backend to wrap.
        tracker : Run
            Tracker instance for logging.
        log_every_n : int
            Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
        log_new_circuits : bool
            Auto-log new circuit structures (default True).
        stats_update_interval : int
            Update stats every N executions (default 1000).

        Returns
        -------
        TrackedBackend
            Wrapped backend that logs execution artifacts.

        Notes
        -----
        For parameter sweeps (same circuit with different parameter values),
        the default settings will only log the first execution since circuit
        structure hashing ignores parameter values. Use ``log_every_n=-1``
        to log all executions, or ``log_every_n=N`` for sampling.
        """
        return TrackedBackend(
            backend=executor,
            tracker=tracker,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
