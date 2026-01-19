# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Qiskit adapter for devqubit tracking system.

Provides integration with Qiskit backends, enabling automatic
tracking of quantum circuit execution, results, and device configurations
following the devqubit Uniform Execution Contract (UEC).

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
>>> from devqubit_engine.tracking import track
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

from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.execution import ProducerInfo
from devqubit_engine.uec.models.program import ProgramSnapshot
from devqubit_engine.uec.models.result import ResultSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.circuits import (
    compute_circuit_hashes,
    materialize_circuits,
    serialize_and_log_circuits,
)
from devqubit_qiskit.device import detect_provider
from devqubit_qiskit.envelope import (
    create_execution_snapshot,
    create_failure_result_snapshot,
    create_program_snapshot,
    create_result_snapshot,
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


# =============================================================================
# TrackedJob
# =============================================================================


@dataclass
class TrackedJob:
    """
    Wrapper for Qiskit job that tracks result retrieval.

    This class wraps a Qiskit job and logs artifacts when results are
    retrieved, producing a ResultSnapshot and creating the ExecutionEnvelope.

    Parameters
    ----------
    job : Any
        Original Qiskit job instance.
    tracker : Run
        Tracker instance for logging.
    backend_name : str
        Name of the backend that created this job.
    should_log_results : bool, default=True
        Whether to log results for this job.
    envelope_data : dict or None, default=None
        Components for creating envelope: device, program, execution, producer.
    """

    job: Any
    tracker: Run
    backend_name: str
    should_log_results: bool = True
    envelope_data: dict[str, Any] | None = None

    # Internal state
    result_snapshot: ResultSnapshot | None = field(default=None, init=False, repr=False)
    _result_logged: bool = field(default=False, init=False, repr=False)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve job result and log artifacts.

        Always creates an envelope - even when job.result() fails.
        Idempotent: calling result() multiple times will only log once.

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
        try:
            result = self.job.result(*args, **kwargs)
        except Exception as exc:
            if self.should_log_results and not self._result_logged:
                self._result_logged = True
                self._log_failure(exc)
            raise

        if self.should_log_results and not self._result_logged:
            self._result_logged = True
            self._log_success(result)

        return result

    def _log_success(self, result: Any) -> None:
        """Log successful result and create envelope."""
        try:
            self.result_snapshot = create_result_snapshot(
                self.tracker, self.backend_name, result
            )

            if self.envelope_data is not None and self.result_snapshot is not None:
                self._create_and_log_envelope()

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
            logger.warning("Failed to log results for %s: %s", self.backend_name, e)
            self.tracker.record.setdefault("warnings", []).append(
                {
                    "type": "result_logging_failed",
                    "message": str(e),
                    "backend_name": self.backend_name,
                }
            )

    def _log_failure(self, exc: Exception) -> None:
        """Log failure envelope when job.result() raises an exception."""
        try:
            self.result_snapshot = create_failure_result_snapshot(
                exception=exc,
                backend_name=self.backend_name,
            )

            if self.envelope_data is not None:
                self._create_and_log_envelope()

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
            logger.error(
                "Failed to log failure envelope for %s: %s",
                self.backend_name,
                log_error,
            )

    def _create_and_log_envelope(self) -> None:
        """Create envelope and log it."""
        envelope = ExecutionEnvelope(
            envelope_id=uuid.uuid4().hex[:26],
            created_at=utc_now_iso(),
            producer=self.envelope_data["producer"],
            result=self.result_snapshot,
            device=self.envelope_data["device"],
            program=self.envelope_data["program"],
            execution=self.envelope_data["execution"],
        )
        finalize_envelope_with_result(self.tracker, envelope, self.result_snapshot)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped job."""
        return getattr(self.job, name)

    def __repr__(self) -> str:
        job_id = extract_job_id(self.job) or "unknown"
        return f"TrackedJob(backend={self.backend_name!r}, job_id={job_id!r})"


# =============================================================================
# TrackedBackend
# =============================================================================


@dataclass
class TrackedBackend:
    """
    Wrapper for Qiskit backend that tracks circuit execution.

    Parameters
    ----------
    backend : Any
        Original Qiskit backend instance (must be BackendV2-compatible).
    tracker : Run
        Tracker instance for logging.
    log_every_n : int, default=0
        Logging frequency:
        - 0: Log first execution only (default)
        - N > 0: Log every Nth execution
        - -1: Log all executions
    log_new_circuits : bool, default=True
        Auto-log new circuit structures.
    stats_update_interval : int, default=1000
        Update execution stats every N executions.
    """

    backend: Any
    tracker: Run
    log_every_n: int = 0
    log_new_circuits: bool = True
    stats_update_interval: int = 1000

    # Internal state
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
    device_snapshot: DeviceSnapshot | None = field(default=None, init=False, repr=False)

    def run(self, circuits: Any, *args: Any, **kwargs: Any) -> TrackedJob:
        """
        Execute circuits and log artifacts based on sampling settings.

        Parameters
        ----------
        circuits : QuantumCircuit or iterable
            Circuit(s) to execute.
        *args : Any
            Additional positional args passed to backend.run().
        **kwargs : Any
            Additional keyword args (e.g., shots).

        Returns
        -------
        TrackedJob
            Wrapped job that tracks result retrieval.
        """
        backend_name = get_backend_name(self.backend)
        submitted_at = utc_now_iso()

        # Materialize circuits
        circuit_list, was_single = materialize_circuits(circuits)
        run_payload: Any = (
            circuit_list[0] if was_single and circuit_list else circuit_list
        )

        # Update execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute hashes
        structural_hash, parametric_hash = compute_circuit_hashes(circuit_list)
        is_new_circuit = (
            structural_hash and structural_hash not in self._seen_circuit_hashes
        )
        if structural_hash:
            self._seen_circuit_hashes.add(structural_hash)

        # Determine logging behavior
        should_log_structure, should_log_results = self._determine_logging(
            exec_count, structural_hash, is_new_circuit
        )

        # Fast path: nothing to log
        if not should_log_structure and not should_log_results:
            return self._execute_fast_path(
                run_payload, backend_name, exec_count, *args, **kwargs
            )

        # Full logging path
        return self._execute_with_logging(
            run_payload=run_payload,
            circuit_list=circuit_list,
            backend_name=backend_name,
            submitted_at=submitted_at,
            exec_count=exec_count,
            structural_hash=structural_hash,
            parametric_hash=parametric_hash,
            should_log_structure=should_log_structure,
            should_log_results=should_log_results,
            args=args,
            kwargs=kwargs,
        )

    def _determine_logging(
        self,
        exec_count: int,
        structural_hash: str | None,
        is_new_circuit: bool,
    ) -> tuple[bool, bool]:
        """Determine what to log based on settings."""
        should_log_structure = False
        should_log_results = False

        if self.log_every_n == -1:
            should_log_structure = structural_hash not in self._logged_circuit_hashes
            should_log_results = True
        elif exec_count == 1:
            should_log_structure = True
            should_log_results = True
        elif self.log_new_circuits and is_new_circuit:
            should_log_structure = True
            should_log_results = True
        elif self.log_every_n > 0 and exec_count % self.log_every_n == 0:
            should_log_results = True

        return should_log_structure, should_log_results

    def _execute_fast_path(
        self,
        run_payload: Any,
        backend_name: str,
        exec_count: int,
        *args: Any,
        **kwargs: Any,
    ) -> TrackedJob:
        """Execute without logging (fast path)."""
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
            envelope_data=None,
        )

    def _execute_with_logging(
        self,
        *,
        run_payload: Any,
        circuit_list: list[Any],
        backend_name: str,
        submitted_at: str,
        exec_count: int,
        structural_hash: str | None,
        parametric_hash: str | None,
        should_log_structure: bool,
        should_log_results: bool,
        args: tuple,
        kwargs: dict,
    ) -> TrackedJob:
        """Execute with full logging."""
        detected_provider = detect_provider(self.backend)

        # Set tags
        self.tracker.set_tag("backend_name", backend_name)
        self.tracker.set_tag("sdk", "qiskit")
        self.tracker.set_tag("adapter", "devqubit-qiskit")
        self.tracker.set_tag("provider", detected_provider)

        # Log device snapshot once
        if not self._snapshot_logged:
            self.device_snapshot = log_device_snapshot(self.backend, self.tracker)
            self._snapshot_logged = True

        # Build program snapshot
        program_snapshot: ProgramSnapshot | None = None
        if should_log_structure and circuit_list:
            program_snapshot = self._log_program(
                circuit_list,
                backend_name,
                structural_hash,
                parametric_hash,
                detected_provider,
                kwargs,
            )

        # Reuse cached program snapshot when only logging results
        elif should_log_results and structural_hash in self._program_snapshot_cache:
            program_snapshot = self._program_snapshot_cache[structural_hash]

        # Build execution snapshot
        shots = kwargs.get("shots")
        execution_snapshot = create_execution_snapshot(
            submitted_at=submitted_at,
            shots=int(shots) if shots is not None else None,
            exec_count=exec_count,
            job_ids=[],
            options={"args": to_jsonable(list(args)), "kwargs": to_jsonable(kwargs)},
        )

        # Update tracker record
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

        # Execute
        job = self.backend.run(run_payload, *args, **kwargs)

        # Log job ID
        job_id = extract_job_id(job)
        if job_id:
            self.tracker.record["execute"]["job_ids"] = [job_id]
            execution_snapshot.job_ids = [job_id]

        # Build envelope data
        envelope_data = self._build_envelope_data(
            program_snapshot,
            execution_snapshot,
            circuit_list,
            structural_hash,
            parametric_hash,
            should_log_results,
        )

        self._update_stats()

        return TrackedJob(
            job=job,
            tracker=self.tracker,
            backend_name=backend_name,
            should_log_results=should_log_results,
            envelope_data=envelope_data,
        )

    def _log_program(
        self,
        circuit_list: list[Any],
        backend_name: str,
        structural_hash: str | None,
        parametric_hash: str | None,
        detected_provider: str,
        kwargs: dict,
    ) -> ProgramSnapshot:
        """Log program artifacts and create snapshot."""
        shots = kwargs.get("shots")
        if shots is not None:
            self.tracker.log_param("shots", int(shots))
        self.tracker.log_param("num_circuits", int(len(circuit_list)))

        if kwargs.get("parameter_binds"):
            self.tracker.log_param(
                "parameter_binds_count", len(kwargs["parameter_binds"])
            )

        program_artifacts = serialize_and_log_circuits(
            self.tracker, circuit_list, backend_name, structural_hash
        )

        if structural_hash:
            self._logged_circuit_hashes.add(structural_hash)

        program_snapshot = create_program_snapshot(
            program_artifacts, structural_hash, parametric_hash, len(circuit_list)
        )

        if structural_hash:
            self._program_snapshot_cache[structural_hash] = program_snapshot

        self.tracker.record["backend"] = {
            "name": backend_name,
            "type": self.backend.__class__.__name__,
            "provider": detected_provider,
            "sdk": "qiskit",
        }

        self._logged_execution_count += 1

        return program_snapshot

    def _build_envelope_data(
        self,
        program_snapshot: ProgramSnapshot | None,
        execution_snapshot: Any,
        circuit_list: list[Any],
        structural_hash: str | None,
        parametric_hash: str | None,
        should_log_results: bool,
    ) -> dict[str, Any] | None:
        """Build envelope data for TrackedJob."""
        if not should_log_results or self.device_snapshot is None:
            return None

        if program_snapshot is None:
            program_snapshot = ProgramSnapshot(
                logical=[],
                physical=[],
                structural_hash=structural_hash,
                parametric_hash=parametric_hash,
                num_circuits=len(circuit_list),
            )

        producer = ProducerInfo.create(
            adapter="devqubit-qiskit",
            adapter_version=get_adapter_version(),
            sdk="qiskit",
            sdk_version=qiskit_version(),
            frontends=["qiskit"],
        )

        return {
            "device": self.device_snapshot,
            "program": program_snapshot,
            "execution": execution_snapshot,
            "producer": producer,
        }

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
        backend_name = get_backend_name(self.backend)
        run_id = getattr(self.tracker, "run_id", "unknown")
        return f"TrackedBackend(backend={backend_name!r}, run_id={run_id!r})"


# =============================================================================
# QiskitAdapter
# =============================================================================


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
            "provider": detect_provider(executor),
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
        log_every_n : int, default=0
            Logging frequency:
            - 0: Log first execution only (default)
            - N > 0: Log every Nth execution
            - -1: Log all executions
        log_new_circuits : bool, default=True
            Auto-log new circuit structures.
        stats_update_interval : int, default=1000
            Update stats every N executions.

        Returns
        -------
        TrackedBackend
            Wrapped backend that logs execution artifacts.
        """
        return TrackedBackend(
            backend=executor,
            tracker=tracker,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
