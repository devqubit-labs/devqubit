# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
CUDA-Q adapter for devqubit tracking system.

Provides integration with CUDA-Q execution, enabling automatic tracking
of quantum kernel execution, results, and target configurations
following the devqubit Uniform Execution Contract (UEC).

Example
-------
>>> import cudaq
>>> from devqubit import track
>>>
>>> @cudaq.kernel
>>> def bell():
...     q = cudaq.qvector(2)
...     cudaq.h(q[0])
...     cudaq.cx(q[0], q[1])
...     cudaq.mz(q)
>>>
>>> with track(project="cudaq-experiment") as run:
...     executor = run.wrap(cudaq)
...     result = executor.sample(bell, shots_count=1000)
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from devqubit_cudaq.circuits import compute_circuit_hashes
from devqubit_cudaq.results import (
    build_result_snapshot,
    detect_result_type,
    result_to_raw_artifact,
)
from devqubit_cudaq.serialization import (
    CudaqCircuitSerializer,
    capture_mlir,
    capture_qir,
    kernel_to_text,
)
from devqubit_cudaq.snapshot import create_device_snapshot
from devqubit_cudaq.utils import (
    collect_sdk_versions,
    get_adapter_version,
    get_kernel_name,
    get_target_info,
    is_cudaq_module,
)
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.errors import EnvelopeValidationError
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.execution import ExecutionSnapshot, ProducerInfo
from devqubit_engine.uec.models.program import (
    ProgramArtifact,
    ProgramRole,
    ProgramSnapshot,
    TranspilationInfo,
    TranspilationMode,
)
from devqubit_engine.uec.models.result import ResultSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable


logger = logging.getLogger(__name__)
_serializer = CudaqCircuitSerializer()


# ============================================================================
# Artifact logging helpers
# ============================================================================


def _log_kernel(
    tracker: Run,
    kernel: Any,
    args: tuple[Any, ...],
    structural_hash: str | None = None,
) -> list[ProgramArtifact]:
    """Log kernel artifacts and return program artifact list."""
    artifacts: list[ProgramArtifact] = []

    # Log serialized kernel metadata
    try:
        serialized = _serializer.serialize_circuit(kernel, args=args)
        ref = tracker.log_bytes(
            kind="cudaq.kernel.json",
            data=serialized.as_bytes(),
            media_type="application/json",
            role="program",
            meta={
                "kernel_name": get_kernel_name(kernel),
                "structural_hash": structural_hash,
            },
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="cudaq_json",
                name=get_kernel_name(kernel),
                index=0,
            )
        )
    except Exception as exc:
        logger.debug("Kernel serialization failed: %s", exc)

    # Log human-readable diagram
    try:
        text = kernel_to_text(kernel, args)
        ref = tracker.log_bytes(
            kind="cudaq.kernel.diagram",
            data=text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="diagram",
                name=get_kernel_name(kernel),
                index=0,
            )
        )
    except Exception as exc:
        logger.debug("Kernel diagram generation failed: %s", exc)

    # Log MLIR (Quake) representation
    try:
        mlir = capture_mlir(kernel)
        if mlir is not None:
            ref = tracker.log_bytes(
                kind="cudaq.kernel.mlir",
                data=mlir.encode("utf-8"),
                media_type="text/plain; charset=utf-8",
                role="program",
            )
            artifacts.append(
                ProgramArtifact(
                    ref=ref,
                    role=ProgramRole.LOGICAL,
                    format="mlir",
                    name=get_kernel_name(kernel),
                    index=0,
                )
            )
    except Exception as exc:
        logger.debug("Kernel MLIR capture failed: %s", exc)

    # Log QIR representation
    try:
        qir = capture_qir(kernel)
        if qir is not None:
            ref = tracker.log_bytes(
                kind="cudaq.kernel.qir",
                data=qir.encode("utf-8"),
                media_type="text/plain; charset=utf-8",
                role="program",
            )
            artifacts.append(
                ProgramArtifact(
                    ref=ref,
                    role=ProgramRole.LOGICAL,
                    format="qir",
                    name=get_kernel_name(kernel),
                    index=0,
                )
            )
    except Exception as exc:
        logger.debug("Kernel QIR capture failed: %s", exc)

    return artifacts


def _log_results(
    tracker: Run,
    backend_name: str,
    result: Any,
    shots: int | None,
    result_type: str | None,
    *,
    success: bool = True,
    error_info: dict[str, Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """Log execution results and return ``ResultSnapshot``."""
    raw_result_ref = None
    try:
        raw_result_ref = tracker.log_json(
            name="results",
            obj={
                "results": (
                    result_to_raw_artifact(result, result_type=result_type, shots=shots)
                    if result is not None
                    else None
                ),
                "shots": shots,
                "result_type": result_type,
                "success": success,
                "error": error_info,
                "call_kwargs": to_jsonable(call_kwargs) if call_kwargs else None,
            },
            role="results",
            kind="result.cudaq.output.json",
        )
    except Exception as exc:
        logger.warning("Failed to log raw results: %s", exc)

    tracker.record["results"] = {
        "completed_at": utc_now_iso(),
        "backend_name": backend_name,
        "shots": shots,
        "result_type": result_type,
        "success": success,
    }
    if error_info:
        tracker.record["results"]["error"] = error_info

    return build_result_snapshot(
        result,
        result_type=result_type,
        backend_name=backend_name,
        shots=shots,
        raw_result_ref=raw_result_ref,
        success=success,
        error_info=error_info,
        call_kwargs=call_kwargs,
    )


def _log_device_snapshot(
    tracker: Run,
    runtime_events: list[dict[str, Any]] | None = None,
) -> DeviceSnapshot:
    """Log device (target) snapshot and return ``DeviceSnapshot``."""
    try:
        target_info = get_target_info()
        snapshot = create_device_snapshot(
            target_info,
            tracker=tracker,
            runtime_events=runtime_events,
        )
    except Exception as exc:
        logger.warning(
            "Failed to create target snapshot: %s. Using minimal snapshot.", exc
        )
        snapshot = DeviceSnapshot(
            captured_at=utc_now_iso(),
            backend_name="unknown",
            backend_type="unknown",
            provider="cudaq",
            sdk_versions=collect_sdk_versions(),
        )

    record: dict[str, Any] = {
        "sdk": "cudaq",
        "backend_name": snapshot.backend_name,
        "backend_type": snapshot.backend_type,
        "provider": snapshot.provider,
        "captured_at": snapshot.captured_at,
    }
    if snapshot.frontend:
        record["frontend"] = snapshot.frontend.to_dict()

    tracker.record["device_snapshot"] = record
    return snapshot


def _finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """Finalize envelope with result and log it."""
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    envelope.result = result_snapshot

    try:
        tracker.log_envelope(envelope=envelope)
    except EnvelopeValidationError:
        raise
    except Exception as exc:
        logger.warning("Failed to log envelope: %s", exc)


# ============================================================================
# TrackedCudaqExecutor
# ============================================================================


class TrackedCudaqExecutor:
    """
    Tracked wrapper around ``cudaq`` module-level functions.

    Intercepts ``sample()`` and ``observe()`` calls, logging UEC envelopes
    for each execution. Tracking errors never abort user experiments.

    Parameters
    ----------
    tracker : Run
        devqubit tracker instance.
    executor : module, optional
        The ``cudaq`` module to wrap.  If ``None``, imports ``cudaq``
        directly.
    log_every_n : int
        Logging frequency: 0=first only, N>0=every Nth, -1=all.
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update stats every N executions (default 1000).
    """

    def __init__(
        self,
        tracker: Run,
        *,
        executor: Any = None,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> None:
        self._tracker = tracker
        self._log_every_n = log_every_n
        self._log_new_circuits = log_new_circuits
        self._stats_update_interval = stats_update_interval

        self._execution_count = 0
        self._logged_execution_count = 0
        self._seen_circuit_hashes: set[str] = set()
        self._logged_circuit_hashes: set[str] = set()

        self._device_snapshot: DeviceSnapshot | None = None
        self._runtime_config_events: list[dict[str, Any]] = []

        if executor is not None:
            self.cudaq = executor
        else:
            import cudaq as _cudaq

            self.cudaq = _cudaq

    # ------------------------------------------------------------------
    # Core tracked execution
    # ------------------------------------------------------------------

    def _tracked_execute(
        self,
        method_name: str,
        kernel: Any,
        kernel_args: tuple[Any, ...],
        *,
        shots: int | None = None,
        hamiltonian: Any = None,
        call_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a CUDA-Q function with UEC tracking."""
        tracker = self._tracker
        submitted_at = utc_now_iso()
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute circuit hashes — always produce values (never None)
        structural_hash, parametric_hash = self._safe_compute_hashes(
            kernel, kernel_args
        )

        is_new_circuit = (
            structural_hash and structural_hash not in self._seen_circuit_hashes
        )
        if structural_hash:
            self._seen_circuit_hashes.add(structural_hash)

        # Determine what to log
        should_log_structure = False
        should_log_results = False

        if self._log_every_n == -1:
            should_log_structure = (
                structural_hash not in self._logged_circuit_hashes
                if structural_hash
                else True
            )
            should_log_results = True
        elif exec_count == 1:
            should_log_structure = True
            should_log_results = True
        elif self._log_new_circuits and is_new_circuit:
            should_log_structure = True
            should_log_results = True
        elif self._log_every_n > 0 and exec_count % self._log_every_n == 0:
            should_log_results = True

        # Capture target snapshot once (invalidated on set_target).
        # Must happen before the fast-path return so that the snapshot
        # is available even when logging is skipped.
        if self._device_snapshot is None:
            self._device_snapshot = _log_device_snapshot(
                tracker,
                runtime_events=self._runtime_config_events,
            )

        # Fast path: no logging needed
        if not should_log_structure and not should_log_results:
            result = self._raw_execute(
                method_name,
                kernel,
                kernel_args,
                shots=shots,
                hamiltonian=hamiltonian,
                call_kwargs=call_kwargs,
            )
            if (
                self._stats_update_interval > 0
                and exec_count % self._stats_update_interval == 0
            ):
                tracker.record["execution_stats"] = self._build_stats()
            return result

        backend_name = (
            self._device_snapshot.backend_name if self._device_snapshot else "unknown"
        )

        # Log kernel structure
        program_artifacts: list[ProgramArtifact] = []
        if should_log_structure and structural_hash not in self._logged_circuit_hashes:
            provider = (
                self._device_snapshot.provider if self._device_snapshot else "local"
            )
            tracker.set_tag("backend_name", backend_name)
            tracker.set_tag("provider", provider)
            tracker.set_tag("sdk", "cudaq")
            tracker.set_tag("adapter", "devqubit-cudaq")

            program_artifacts = _log_kernel(
                tracker, kernel, kernel_args, structural_hash
            )

            if structural_hash:
                self._logged_circuit_hashes.add(structural_hash)

            tracker.record["backend"] = {
                "name": backend_name,
                "provider": provider,
                "sdk": "cudaq",
            }
            self._logged_execution_count += 1

        # Determine num_circuits (broadcasting produces list results)
        num_circuits = 1

        # Build ProgramSnapshot
        program_snapshot = ProgramSnapshot(
            logical=program_artifacts,
            physical=[],
            structural_hash=structural_hash,
            parametric_hash=parametric_hash,
            executed_structural_hash=structural_hash,
            executed_parametric_hash=parametric_hash,
            num_circuits=num_circuits,
        )

        # Build ExecutionSnapshot
        transpilation_info = TranspilationInfo(
            mode=TranspilationMode.MANUAL,
            transpiled_by="user",
        )

        execution_options: dict[str, Any] = {
            "method": method_name,
            "kernel_name": get_kernel_name(kernel),
            "shots": shots,
        }
        if hamiltonian is not None:
            execution_options["spin_operator"] = repr(hamiltonian)
            execution_options["spin_operator_type"] = type(hamiltonian).__name__
        if call_kwargs:
            execution_options["kwargs"] = to_jsonable(call_kwargs)

        execution_snapshot = ExecutionSnapshot(
            submitted_at=submitted_at,
            shots=shots,
            job_ids=[],
            execution_count=exec_count,
            transpilation=transpilation_info,
            options=execution_options,
            sdk="cudaq",
        )

        # Execute with error handling
        result = None
        execution_error: dict[str, Any] | None = None
        execution_succeeded = True
        original_exception: BaseException | None = None

        try:
            result = self._raw_execute(
                method_name,
                kernel,
                kernel_args,
                shots=shots,
                hamiltonian=hamiltonian,
                call_kwargs=call_kwargs,
            )
        except Exception as exc:
            execution_succeeded = False
            original_exception = exc
            execution_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            logger.warning(
                "CUDA-Q execution failed: %s: %s",
                execution_error["type"],
                execution_error["message"],
            )
            tracker.log_json(
                name="execution_error",
                obj={
                    "error": execution_error,
                    "execution_count": exec_count,
                    "backend_name": backend_name,
                    "structural_hash": structural_hash,
                    "submitted_at": submitted_at,
                },
                role="results",
                kind="devqubit.execution.error.json",
            )
            tracker.record["execution_error"] = {
                "type": execution_error["type"],
                "message": execution_error["message"],
                "execution_count": exec_count,
            }

        # Update num_circuits for broadcasting
        if execution_succeeded and isinstance(result, list):
            num_circuits = len(result)
            program_snapshot.num_circuits = num_circuits

        # Log results and build envelope
        if should_log_results:
            if not should_log_structure:
                provider = (
                    self._device_snapshot.provider if self._device_snapshot else "local"
                )
                tracker.set_tag("backend_name", backend_name)
                tracker.set_tag("provider", provider)
                tracker.set_tag("sdk", "cudaq")
                tracker.set_tag("adapter", "devqubit-cudaq")

            result_type = detect_result_type(result) if execution_succeeded else None
            result_snapshot = _log_results(
                tracker,
                backend_name,
                result,
                shots,
                result_type,
                success=execution_succeeded,
                error_info=execution_error,
                call_kwargs=call_kwargs,
            )
            if not should_log_structure:
                self._logged_execution_count += 1

            tracker.record["execute"] = {
                "sdk": "cudaq",
                "submitted_at": submitted_at,
                "backend_name": backend_name,
                "execution_count": exec_count,
                "structural_hash": structural_hash,
                "parametric_hash": parametric_hash,
                "success": execution_succeeded,
            }

            # Create and finalize ExecutionEnvelope
            if self._device_snapshot is not None:
                sdk_versions = (
                    self._device_snapshot.sdk_versions or collect_sdk_versions()
                )
                producer = ProducerInfo.create(
                    adapter="devqubit-cudaq",
                    adapter_version=get_adapter_version(),
                    sdk="cudaq",
                    sdk_version=sdk_versions.get("cudaq", "unknown"),
                    frontends=["cudaq"],
                )

                envelope = ExecutionEnvelope(
                    envelope_id=uuid.uuid4().hex[:26],
                    created_at=utc_now_iso(),
                    producer=producer,
                    result=result_snapshot,
                    device=self._device_snapshot,
                    program=program_snapshot,
                    execution=execution_snapshot,
                )
                try:
                    _finalize_envelope_with_result(
                        tracker=tracker,
                        envelope=envelope,
                        result_snapshot=result_snapshot,
                    )
                    logger.debug("Created execution envelope for %s", backend_name)
                except EnvelopeValidationError as val_err:
                    logger.error(
                        "Envelope validation failed for %s: %s "
                        "(this indicates an adapter bug)",
                        backend_name,
                        val_err,
                    )
                    tracker.record.setdefault("errors", []).append(
                        {
                            "type": "envelope_validation_error",
                            "message": str(val_err),
                            "backend_name": backend_name,
                        }
                    )
                except Exception as log_err:
                    logger.warning(
                        "Failed to finalize envelope for %s: %s",
                        backend_name,
                        log_err,
                    )
                    tracker.record.setdefault("warnings", []).append(
                        {
                            "type": "envelope_finalization_failed",
                            "message": str(log_err),
                            "backend_name": backend_name,
                        }
                    )

        # Update stats
        tracker.record["execution_stats"] = self._build_stats()

        # Re-raise execution error after logging
        if not execution_succeeded and original_exception is not None:
            raise original_exception

        return result

    # ------------------------------------------------------------------
    # Hash computation (always returns non-None)
    # ------------------------------------------------------------------

    def _safe_compute_hashes(
        self,
        kernel: Any,
        args: tuple[Any, ...],
    ) -> tuple[str, str]:
        """Compute hashes with guaranteed non-None return."""
        try:
            return compute_circuit_hashes(kernel, args)
        except Exception as exc:
            logger.debug("Hash computation failed, using fallback: %s", exc)
            import hashlib

            # Try str(kernel) for structural content before falling back to name
            try:
                content = str(kernel)
                if content and len(content) > 20:
                    structural = (
                        f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
                    )
                    payload = f"{content}:{repr(args)}"
                    h = f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
                    return structural, h
            except (OSError, TypeError, ValueError) as e:
                logger.debug("Source-based kernel hashing failed: %s", e)
                pass

            payload = f"{get_kernel_name(kernel)}:{repr(args)}"
            h = f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
            structural = (
                f"sha256:{hashlib.sha256(get_kernel_name(kernel).encode()).hexdigest()}"
            )
            return structural, h

    # ------------------------------------------------------------------
    # Raw execution (no tracking)
    # ------------------------------------------------------------------

    def _raw_execute(
        self,
        method_name: str,
        kernel: Any,
        kernel_args: tuple[Any, ...],
        *,
        shots: int | None = None,
        hamiltonian: Any = None,
        call_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute without tracking."""
        kwargs = dict(call_kwargs or {})
        if shots is not None:
            kwargs["shots_count"] = shots

        if method_name == "sample":
            return self.cudaq.sample(kernel, *kernel_args, **kwargs)
        elif method_name == "observe":
            if hamiltonian is None:
                raise ValueError("observe() requires a hamiltonian argument")
            return self.cudaq.observe(kernel, hamiltonian, *kernel_args, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        kernel: Any,
        *args: Any,
        shots_count: int = 1000,
        **kwargs: Any,
    ) -> Any:
        """
        Tracked ``cudaq.sample()``.

        Parameters
        ----------
        kernel : Any
            CUDA-Q kernel.
        *args : Any
            Kernel arguments.
        shots_count : int
            Number of shots (default 1000).
        **kwargs : Any
            Additional arguments to ``cudaq.sample()``.

        Returns
        -------
        SampleResult or list[SampleResult]
            CUDA-Q sample result (list for broadcasting).
        """
        return self._tracked_execute(
            "sample",
            kernel,
            args,
            shots=shots_count,
            call_kwargs=kwargs if kwargs else None,
        )

    def observe(
        self,
        kernel: Any,
        hamiltonian: Any,
        *args: Any,
        shots_count: int = -1,
        **kwargs: Any,
    ) -> Any:
        """
        Tracked ``cudaq.observe()``.

        Parameters
        ----------
        kernel : Any
            CUDA-Q kernel.
        hamiltonian : Any
            Spin operator to observe.
        *args : Any
            Kernel arguments.
        shots_count : int
            Number of shots (-1 for analytic, default).
        **kwargs : Any
            Additional arguments to ``cudaq.observe()``.

        Returns
        -------
        ObserveResult or list[ObserveResult]
            CUDA-Q observe result (list for broadcasting).
        """
        actual_shots = shots_count if shots_count > 0 else None
        return self._tracked_execute(
            "observe",
            kernel,
            args,
            shots=actual_shots,
            hamiltonian=hamiltonian,
            call_kwargs=kwargs if kwargs else None,
        )

    def _build_stats(self) -> dict[str, Any]:
        """Build execution statistics dict."""
        return {
            "total_executions": self._execution_count,
            "logged_executions": self._logged_execution_count,
            "unique_circuits": len(self._seen_circuit_hashes),
            "logged_circuits": len(self._logged_circuit_hashes),
            "last_execution_at": utc_now_iso(),
        }

    def __getattr__(self, name: str) -> Any:
        """
        Passthrough for non-tracked cudaq attributes.

        Intercepts runtime configuration methods to invalidate caches
        and record configuration events for the device snapshot.
        """
        attr = getattr(self.cudaq, name)

        if name == "set_target" and callable(attr):
            return self._wrap_config_method(
                attr, "set_target", invalidate_snapshot=True
            )

        if name == "reset_target" and callable(attr):
            return self._wrap_config_method(
                attr,
                "reset_target",
                invalidate_snapshot=True,
            )

        if name in ("set_noise", "unset_noise", "set_random_seed") and callable(attr):
            return self._wrap_config_method(attr, name)

        return attr

    def _wrap_config_method(
        self,
        method: Any,
        method_name: str,
        *,
        invalidate_snapshot: bool = False,
    ) -> Any:
        """Return a wrapper that records the call and optionally invalidates the snapshot."""

        def _wrapped(*a: Any, **k: Any) -> Any:
            result = method(*a, **k)
            self._runtime_config_events.append(
                {
                    "method": method_name,
                    "args": [str(x) for x in a] if a else [],
                    "kwargs": (
                        {str(key): str(val) for key, val in k.items()} if k else {}
                    ),
                    "timestamp": utc_now_iso(),
                }
            )
            if invalidate_snapshot:
                self._device_snapshot = None
            return result

        return _wrapped


# ============================================================================
# CudaqAdapter (entry point for devqubit adapter registry)
# ============================================================================


class CudaqAdapter:
    """
    Adapter for integrating CUDA-Q with devqubit tracking.

    Wraps the ``cudaq`` module into a ``TrackedCudaqExecutor`` that
    intercepts ``sample()`` and ``observe()`` calls.
    """

    name: str = "cudaq"

    def supports_executor(self, executor: Any) -> bool:
        """Check if executor is the ``cudaq`` module."""
        return is_cudaq_module(executor)

    def describe_executor(self, executor: Any) -> dict[str, Any]:
        """Describe the CUDA-Q execution environment."""
        try:
            target_info = get_target_info()
        except (AttributeError, ImportError, TypeError) as e:
            logger.debug("Failed to get target info: %s", e)
            return {
                "name": "cudaq",
                "type": "module",
                "sdk": "cudaq",
                "provider": "local",
            }

        return {
            "name": target_info.name,
            "type": "cudaq_target",
            "sdk": "cudaq",
            "provider": ("local" if target_info.is_simulator else target_info.name),
            "backend_type": ("simulator" if target_info.is_simulator else "hardware"),
            "platform": target_info.platform,
            "is_remote": target_info.is_remote,
        }

    def wrap_executor(
        self,
        executor: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedCudaqExecutor:
        """
        Wrap the ``cudaq`` module for tracked execution.

        Parameters
        ----------
        executor : Any
            The ``cudaq`` module.
        tracker : Run
            Tracker instance.
        log_every_n : int
            Logging frequency: 0=first only, N>0=every Nth, -1=all.
        log_new_circuits : bool
            Auto-log new circuit structures (default True).
        stats_update_interval : int
            Update stats every N executions (default 1000).

        Returns
        -------
        TrackedCudaqExecutor
            Tracked executor wrapping cudaq functions.
        """
        return TrackedCudaqExecutor(
            tracker=tracker,
            executor=executor,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
