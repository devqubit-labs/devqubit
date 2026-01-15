# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
PennyLane adapter for devqubit tracking system.

Provides integration with PennyLane devices, enabling automatic tracking
of quantum circuit execution, tapes, results, and device configurations
following the devqubit Uniform Execution Contract (UEC).

The adapter produces four canonical snapshots for every execution:
- DeviceSnapshot: Device config with multi-layer stack (PennyLane → Braket/Qiskit/native)
- ProgramSnapshot: Tape artifacts with circuit hash
- ExecutionSnapshot: Submission metadata
- ResultSnapshot: Execution results with type detection

Multi-Layer Stack
-----------------
PennyLane acts as a frontend to multiple execution providers:
- Braket: `qml.device("braket.aws.qubit", ...)`
- Qiskit: `qml.device("qiskit.remote", ...)`
- Native: `qml.device("default.qubit", ...)`

When using external providers, the adapter extracts calibration and
topology information from the underlying backend.

Example
-------
>>> import pennylane as qml
>>> from devqubit import track
>>>
>>> dev = qml.device("default.qubit", wires=2)
>>>
>>> @qml.qnode(dev)
>>> def circuit(x):
...     qml.RX(x, wires=0)
...     qml.CNOT(wires=[0, 1])
...     return qml.expval(qml.PauliZ(0))
>>>
>>> with track(project="pennylane-experiment") as run:
...     dev = run.wrap(dev)
...     result = circuit(0.5)
"""

from __future__ import annotations

import hashlib
import logging
import traceback
import types
import uuid
from typing import Any

from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import (
    ProgramArtifact,
    ProgramSnapshot,
    TranspilationInfo,
)
from devqubit_engine.uec.result import ResultSnapshot
from devqubit_engine.uec.types import (
    ProgramRole,
    TranspilationMode,
)
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_pennylane.results import build_result_snapshot, extract_result_type
from devqubit_pennylane.serialization import PennyLaneCircuitSerializer, tapes_to_text
from devqubit_pennylane.snapshot import (
    create_device_snapshot,
    resolve_pennylane_backend,
)
from devqubit_pennylane.utils import (
    collect_sdk_versions,
    extract_shots_info,
    get_adapter_version,
    get_device_name,
    is_pennylane_device,
)


logger = logging.getLogger(__name__)
_serializer = PennyLaneCircuitSerializer()


def _is_tape_like(obj: Any) -> bool:
    """Check if object has tape-like interface."""
    return hasattr(obj, "operations") and hasattr(obj, "measurements")


def _compute_structural_hash(circuits: Any) -> str | None:
    """
    Compute a structure-only hash for PennyLane tapes.

    Parameters
    ----------
    circuits : Any
        A tape-like object or list of tapes.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if not tape-like.

    Notes
    -----
    This is the STRUCTURAL hash - captures gate names, wires, parameter arity
    but NOT parameter values. Use _compute_parametric_hash for full identity.
    """
    if circuits is None:
        return None

    if _is_tape_like(circuits):
        tape_list = [circuits]
    elif (
        isinstance(circuits, (list, tuple))
        and circuits
        and all(_is_tape_like(t) for t in circuits)
    ):
        tape_list = list(circuits)
    else:
        return None

    def _wires_to_tuple(wires: Any) -> tuple[str, ...]:
        try:
            return tuple(str(w) for w in list(wires))
        except Exception:
            try:
                return (str(wires),)
            except Exception:
                return ("<wires>",)

    def _param_arity(op: Any) -> int:
        params = getattr(op, "parameters", None)
        if isinstance(params, (list, tuple)):
            return len(params)
        data = getattr(op, "data", None)
        if isinstance(data, (list, tuple)):
            return len(data)
        return 0

    def _op_sig(op: Any) -> str:
        name = getattr(op, "name", None)
        op_name = name if isinstance(name, str) and name else type(op).__name__
        wires = _wires_to_tuple(getattr(op, "wires", ()))
        arity = _param_arity(op)
        return f"{op_name}|p{arity}|w{wires}"

    def _meas_sig(m: Any) -> str:
        mtype = type(m).__name__
        wires = _wires_to_tuple(getattr(m, "wires", ()))
        rtype = getattr(m, "return_type", None)
        rtype_s = str(rtype) if rtype is not None else ""
        obs = getattr(m, "obs", None)
        if obs is not None:
            obs_name = getattr(obs, "name", None)
            obs_s = (
                obs_name
                if isinstance(obs_name, str) and obs_name
                else type(obs).__name__
            )
        else:
            obs_s = ""
        return f"{mtype}|rt={rtype_s}|obs={obs_s}|w{wires}"

    tape_signatures: list[str] = []

    for tape in tape_list:
        try:
            ops = getattr(tape, "operations", [])
            meas = getattr(tape, "measurements", [])
            op_sigs = [_op_sig(op) for op in ops]
            meas_sigs = [_meas_sig(m) for m in meas]
            tape_signatures.append(
                "||".join(op_sigs) + "\n--MEAS--\n" + "||".join(meas_sigs)
            )
        except Exception:
            tape_signatures.append(str(tape)[:500])

    payload = "\n\n".join(tape_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _compute_parametric_hash(circuits: Any) -> str | None:
    """
    Compute a parametric hash for PennyLane tapes.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : Any
        A tape-like object or list of tapes.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if not tape-like.

    Notes
    -----
    Includes:
    - All structural information (gates, wires, measurement types)
    - Bound parameter values (rounded to 10 decimal places for stability)
    - Trainable parameter names if present
    """
    if circuits is None:
        return None

    if _is_tape_like(circuits):
        tape_list = [circuits]
    elif (
        isinstance(circuits, (list, tuple))
        and circuits
        and all(_is_tape_like(t) for t in circuits)
    ):
        tape_list = list(circuits)
    else:
        return None

    def _wires_to_tuple(wires: Any) -> tuple[str, ...]:
        try:
            return tuple(str(w) for w in list(wires))
        except Exception:
            try:
                return (str(wires),)
            except Exception:
                return ("<wires>",)

    def _format_param(p: Any) -> str:
        """Format parameter value with stable representation."""
        try:
            # Check if it's a trainable parameter (has name)
            if hasattr(p, "name"):
                return f"<param:{p.name}>"
            # Numeric value - round for stability
            val = float(p)
            return f"{val:.10f}"
        except (TypeError, ValueError):
            return str(p)[:50]

    def _op_sig_with_params(op: Any) -> str:
        name = getattr(op, "name", None)
        op_name = name if isinstance(name, str) and name else type(op).__name__
        wires = _wires_to_tuple(getattr(op, "wires", ()))

        # Get parameter values
        params = getattr(op, "parameters", None) or getattr(op, "data", None) or []
        if isinstance(params, (list, tuple)):
            param_strs = [_format_param(p) for p in params]
        else:
            param_strs = []

        return f"{op_name}|params=[{','.join(param_strs)}]|w{wires}"

    def _meas_sig(m: Any) -> str:
        mtype = type(m).__name__
        wires = _wires_to_tuple(getattr(m, "wires", ()))
        rtype = getattr(m, "return_type", None)
        rtype_s = str(rtype) if rtype is not None else ""
        obs = getattr(m, "obs", None)
        if obs is not None:
            obs_name = getattr(obs, "name", None)
            obs_s = (
                obs_name
                if isinstance(obs_name, str) and obs_name
                else type(obs).__name__
            )
        else:
            obs_s = ""
        return f"{mtype}|rt={rtype_s}|obs={obs_s}|w{wires}"

    tape_signatures: list[str] = []

    for tape in tape_list:
        try:
            ops = getattr(tape, "operations", [])
            meas = getattr(tape, "measurements", [])

            # Include trainable params info if available
            trainable = getattr(tape, "trainable_params", None)
            trainable_str = ""
            if trainable:
                trainable_str = f"\n--TRAINABLE--\n{sorted(trainable)}"

            op_sigs = [_op_sig_with_params(op) for op in ops]
            meas_sigs = [_meas_sig(m) for m in meas]
            tape_signatures.append(
                "||".join(op_sigs)
                + "\n--MEAS--\n"
                + "||".join(meas_sigs)
                + trainable_str
            )
        except Exception:
            tape_signatures.append(str(tape)[:500])

    payload = "\n\n".join(tape_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _get_tapes(circuits: Any) -> list[Any]:
    """
    Extract tape list from circuits.

    Parameters
    ----------
    circuits : Any
        A tape-like object, list of tapes, or None.

    Returns
    -------
    list
        List of tapes (empty if input is invalid).
    """
    if circuits is None:
        return []
    if _is_tape_like(circuits):
        return [circuits]
    if isinstance(circuits, (list, tuple)) and circuits:
        # Only return if all elements are tape-like
        if all(_is_tape_like(t) for t in circuits):
            return list(circuits)
    return []


def _log_tapes(
    tracker: Run,
    tapes: list[Any],
    structural_hash: str | None = None,
) -> list[ProgramArtifact]:
    """Log tapes and return program artifacts."""
    artifacts: list[ProgramArtifact] = []

    try:
        serialized = _serializer.serialize_batch(tapes)
        ref = tracker.log_bytes(
            kind="pennylane.tapes.json",
            data=serialized.as_bytes(),
            media_type="application/json",
            role="program",
            meta={"num_tapes": len(tapes), "structural_hash": structural_hash},
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="tape_json",
                name="tapes",
                index=0,
            )
        )
    except Exception as e:
        logger.debug("Tape serialization failed: %s", e)

    try:
        ref = tracker.log_bytes(
            kind="pennylane.tapes.txt",
            data=tapes_to_text(tapes).encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="diagram",
                name="tapes",
                index=0,
            )
        )
    except Exception:
        pass

    return artifacts


def _log_results(
    tracker: Run,
    backend_name: str,
    results: Any,
    num_circuits: int,
    result_type: str | None,
    *,
    success: bool = True,
    error_info: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Log execution results and return ResultSnapshot.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    backend_name : str
        Backend name.
    results : Any
        Execution results (may be None if execution failed).
    num_circuits : int
        Number of circuits executed.
    result_type : str or None
        Result type string.
    success : bool
        Whether execution succeeded.
    error_info : dict, optional
        Error information if execution failed.

    Returns
    -------
    ResultSnapshot
        Result snapshot (with success=False if execution failed).
    """
    # Log raw results (even if None for failed executions)
    raw_result_ref = None
    try:
        raw_result_ref = tracker.log_json(
            name="results",
            obj={
                "results": to_jsonable(results) if results is not None else None,
                "num_circuits": num_circuits,
                "result_type": result_type,
                "success": success,
                "error": error_info,
            },
            role="results",
            kind="result.pennylane.output.json",
        )
    except Exception as e:
        logger.warning("Failed to log raw results: %s", e)

    tracker.record["results"] = {
        "completed_at": utc_now_iso(),
        "backend_name": backend_name,
        "num_circuits": num_circuits,
        "result_type": result_type,
        "success": success,
    }

    if error_info:
        tracker.record["results"]["error"] = error_info

    return build_result_snapshot(
        results,
        result_type=result_type,
        backend_name=backend_name,
        num_circuits=num_circuits,
        raw_result_ref=raw_result_ref,
        success=success,
        error_info=error_info,
    )


def _log_device_snapshot(
    device: Any,
    tracker: Run,
    *,
    resolve_remote_backend: bool = False,
) -> DeviceSnapshot:
    """
    Log device snapshot and return DeviceSnapshot object.

    Parameters
    ----------
    device : Any
        PennyLane device.
    tracker : Run
        Tracker instance.
    resolve_remote_backend : bool
        Whether to resolve remote backends (may be slow).

    Returns
    -------
    DeviceSnapshot
        Device snapshot with raw_properties_ref set.
    """
    # Create snapshot with tracker for raw_properties logging
    try:
        snapshot = create_device_snapshot(
            device,
            resolve_remote_backend=resolve_remote_backend,
            tracker=tracker,
        )
    except Exception as e:
        logger.warning(
            "Failed to create device snapshot: %s. Using minimal snapshot.", e
        )
        # Create minimal snapshot on failure
        snapshot = DeviceSnapshot(
            captured_at=utc_now_iso(),
            backend_name=get_device_name(device),
            backend_type="unknown",
            provider="pennylane",
            sdk_versions=collect_sdk_versions(),
        )

    # Build record with multi-layer stack info
    record: dict[str, Any] = {
        "sdk": "pennylane",
        "backend_name": snapshot.backend_name,
        "backend_type": snapshot.backend_type,
        "provider": snapshot.provider,
        "captured_at": snapshot.captured_at,
        "num_qubits": snapshot.num_qubits,
    }

    # Add frontend info
    if snapshot.frontend:
        record["frontend"] = snapshot.frontend.to_dict()

    # Add calibration summary if available
    if snapshot.calibration:
        record["calibration_summary"] = snapshot.get_calibration_summary()

    tracker.record["device_snapshot"] = record

    return snapshot


def _finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """
    Finalize envelope with result and log it.

    Validation errors are logged but never crash the experiment.
    Tracking should never abort user experiments.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result_snapshot : ResultSnapshot
        Result snapshot to add.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if result_snapshot is None:
        logger.warning("Finalizing envelope with None result_snapshot")

    # Update execution snapshot with completion time
    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    # Add result to envelope
    envelope.result = result_snapshot

    # Log the envelope using tracker's canonical method
    try:
        tracker.log_envelope(envelope=envelope)
    except Exception as e:
        logger.warning("Failed to log envelope: %s", e)


def patch_device(
    device: Any,
    *,
    log_every_n: int = 0,
    log_new_circuits: bool = True,
    stats_update_interval: int = 1000,
) -> None:
    """
    Patch a PennyLane device for tracking.

    This function patches execute/batch_execute methods in place.
    Tracking errors never abort user experiments.

    Parameters
    ----------
    device : Any
        PennyLane device to patch.
    log_every_n : int
        Logging frequency.
    log_new_circuits : bool
        Auto-log new circuit structures.
    stats_update_interval : int
        Update stats every N executions.
    """
    if getattr(device, "_devqubit_patched", False):
        device._devqubit_log_every_n = log_every_n
        device._devqubit_log_new_circuits = log_new_circuits
        device._devqubit_stats_update_interval = stats_update_interval
        return

    device._devqubit_patched = True
    device._devqubit_tracker = None
    device._devqubit_execution_count = 0
    device._devqubit_logged_execution_count = 0
    device._devqubit_log_every_n = log_every_n
    device._devqubit_log_new_circuits = log_new_circuits
    device._devqubit_stats_update_interval = stats_update_interval
    device._devqubit_seen_circuit_hashes: set[str] = set()
    device._devqubit_logged_circuit_hashes: set[str] = set()

    # UEC snapshots stored on device
    device._devqubit_device_snapshot: DeviceSnapshot | None = None
    device._devqubit_program_snapshot: ProgramSnapshot | None = None
    device._devqubit_execution_snapshot: ExecutionSnapshot | None = None
    device._devqubit_result_snapshot: ResultSnapshot | None = None
    device._devqubit_envelope: ExecutionEnvelope | None = None

    def _make_wrapper(method_name: str):
        original = getattr(device, method_name)
        setattr(device, f"_devqubit_original_{method_name}", original)

        def wrapped(self: Any, circuits: Any, *args: Any, **kwargs: Any) -> Any:
            tracker: Run | None = getattr(self, "_devqubit_tracker", None)
            if tracker is None:
                return getattr(self, f"_devqubit_original_{method_name}")(
                    circuits, *args, **kwargs
                )

            submitted_at = utc_now_iso()
            self._devqubit_execution_count += 1
            exec_count = self._devqubit_execution_count
            log_every_n = self._devqubit_log_every_n
            log_new_circuits = self._devqubit_log_new_circuits
            stats_interval = self._devqubit_stats_update_interval
            seen_hashes: set[str] = self._devqubit_seen_circuit_hashes
            logged_hashes: set[str] = self._devqubit_logged_circuit_hashes

            # Ensure batch_execute gets a list
            exec_payload = (
                [circuits]
                if method_name == "batch_execute" and _is_tape_like(circuits)
                else circuits
            )

            # Fast path: skip all overhead
            if log_every_n == 0 and exec_count > 1 and not log_new_circuits:
                result = getattr(self, f"_devqubit_original_{method_name}")(
                    exec_payload, *args, **kwargs
                )
                if stats_interval > 0 and exec_count % stats_interval == 0:
                    tracker.record["execution_stats"] = {
                        "total_executions": exec_count,
                        "logged_executions": self._devqubit_logged_execution_count,
                        "unique_circuits": len(seen_hashes),
                        "logged_circuits": len(logged_hashes),
                        "last_execution_at": utc_now_iso(),
                    }
                return result

            # Compute structural hash (ignores parameter values - for deduplication)
            structural_hash = _compute_structural_hash(circuits)
            # Compute parametric hash (includes parameter values - for exact match)
            parametric_hash = _compute_parametric_hash(circuits)

            # Use structural hash for deduplication (same circuit template)
            is_new_circuit = structural_hash and structural_hash not in seen_hashes
            if structural_hash:
                seen_hashes.add(structural_hash)

            # Determine logging
            should_log_structure = False
            should_log_results = False

            if log_every_n == -1:
                should_log_structure = structural_hash not in logged_hashes
                should_log_results = True
            elif exec_count == 1:
                should_log_structure = True
                should_log_results = True
            elif log_new_circuits and is_new_circuit:
                should_log_structure = True
                should_log_results = True
            elif log_every_n > 0 and exec_count % log_every_n == 0:
                should_log_results = True

            # Fast path if nothing to log
            if not should_log_structure and not should_log_results:
                result = getattr(self, f"_devqubit_original_{method_name}")(
                    exec_payload, *args, **kwargs
                )
                if stats_interval > 0 and exec_count % stats_interval == 0:
                    tracker.record["execution_stats"] = {
                        "total_executions": exec_count,
                        "logged_executions": self._devqubit_logged_execution_count,
                        "unique_circuits": len(seen_hashes),
                        "logged_circuits": len(logged_hashes),
                        "last_execution_at": utc_now_iso(),
                    }
                return result

            # Get tapes for logging
            tapes = _get_tapes(circuits)
            backend_name = get_device_name(self)

            # Capture device snapshot (once per run)
            if self._devqubit_device_snapshot is None:
                self._devqubit_device_snapshot = _log_device_snapshot(self, tracker)

            # Log structure and build ProgramSnapshot
            program_artifacts: list[ProgramArtifact] = []
            if should_log_structure and tapes and structural_hash not in logged_hashes:
                # Get physical provider from device snapshot
                physical_provider = (
                    self._devqubit_device_snapshot.provider
                    if self._devqubit_device_snapshot
                    else "local"
                )

                tracker.set_tag("backend_name", backend_name)
                tracker.set_tag("provider", physical_provider)  # Physical provider
                tracker.set_tag("sdk", "pennylane")  # SDK frontend
                tracker.set_tag("adapter", "devqubit-pennylane")

                program_artifacts = _log_tapes(tracker, tapes, structural_hash)

                if structural_hash:
                    logged_hashes.add(structural_hash)

                tracker.record["backend"] = {
                    "name": backend_name,
                    "type": self.__class__.__name__,
                    "provider": physical_provider,  # Physical provider
                    "sdk": "pennylane",  # SDK frontend
                }
                self._devqubit_logged_execution_count += 1

            # Build ProgramSnapshot
            # PennyLane doesn't transpile, so executed_*_hash == *_hash
            self._devqubit_program_snapshot = ProgramSnapshot(
                logical=program_artifacts,
                physical=[],  # PennyLane doesn't have separate physical circuits
                structural_hash=structural_hash,
                parametric_hash=parametric_hash,
                # For PennyLane without transpilation, executed hashes equal logical
                executed_structural_hash=structural_hash,
                executed_parametric_hash=parametric_hash,
                num_circuits=len(tapes),
            )

            # Build ExecutionSnapshot with comprehensive shots info
            transpilation_info = TranspilationInfo(
                mode=TranspilationMode.MANUAL,
                transpiled_by="user",
            )

            # Get shots info directly from extract_shots_info
            shots_info = extract_shots_info(self).to_dict()

            self._devqubit_execution_snapshot = ExecutionSnapshot(
                submitted_at=submitted_at,
                shots=shots_info.get("total_shots"),
                job_ids=[],  # PennyLane doesn't have job IDs for local execution
                execution_count=exec_count,
                transpilation=transpilation_info,
                options={
                    "method": method_name,
                    "args": to_jsonable(list(args)) if args else None,
                    "kwargs": to_jsonable(kwargs) if kwargs else None,
                    "shots_info": shots_info,  # Full shots info including shot_vector
                },
                sdk="pennylane",
            )

            # Execute with error handling
            result = None
            execution_error: dict[str, Any] | None = None
            execution_succeeded = True
            original_exception: BaseException | None = None  # Save for re-raise

            try:
                result = getattr(self, f"_devqubit_original_{method_name}")(
                    exec_payload, *args, **kwargs
                )
            except Exception as e:
                execution_succeeded = False
                original_exception = e  # Save original exception
                execution_error = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }

                logger.warning(
                    "Device execution failed: %s: %s",
                    execution_error["type"],
                    execution_error["message"],
                )

                # Log error context
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

            # Log results and build ResultSnapshot + ExecutionEnvelope
            if should_log_results and tapes:
                try:
                    result_type = (
                        extract_result_type(tapes) if execution_succeeded else None
                    )
                except Exception:
                    result_type = None
                self._devqubit_result_snapshot = _log_results(
                    tracker,
                    backend_name,
                    result,
                    len(tapes),
                    result_type,
                    success=execution_succeeded,
                    error_info=execution_error,
                )
                if not should_log_structure:
                    self._devqubit_logged_execution_count += 1

                tracker.record["execute"] = {
                    "sdk": "pennylane",
                    "submitted_at": submitted_at,
                    "backend_name": backend_name,
                    "execution_count": exec_count,
                    "structural_hash": structural_hash,
                    "parametric_hash": parametric_hash,
                    "success": execution_succeeded,
                }

                # Create and finalize ExecutionEnvelope
                if self._devqubit_device_snapshot is not None:
                    # Create ProducerInfo
                    sdk_versions = collect_sdk_versions()
                    producer = ProducerInfo.create(
                        adapter="devqubit-pennylane",
                        adapter_version=get_adapter_version(),
                        sdk="pennylane",
                        sdk_version=sdk_versions.get("pennylane", "unknown"),
                        frontends=["pennylane"],
                    )

                    # Create pending result (will be updated when finalized)
                    pending_result = ResultSnapshot(
                        success=False,
                        status="failed",  # Will be updated by _finalize_envelope_with_result
                        items=[],
                        metadata={"state": "pending"},
                    )

                    self._devqubit_envelope = ExecutionEnvelope(
                        envelope_id=uuid.uuid4().hex[:26],
                        created_at=utc_now_iso(),
                        producer=producer,
                        result=pending_result,
                        device=self._devqubit_device_snapshot,
                        program=self._devqubit_program_snapshot,
                        execution=self._devqubit_execution_snapshot,
                    )
                    try:
                        _finalize_envelope_with_result(
                            tracker=tracker,
                            envelope=self._devqubit_envelope,
                            result_snapshot=self._devqubit_result_snapshot,
                        )
                        logger.debug("Created execution envelope for %s", backend_name)
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
            tracker.record["execution_stats"] = {
                "total_executions": exec_count,
                "logged_executions": self._devqubit_logged_execution_count,
                "unique_circuits": len(seen_hashes),
                "logged_circuits": len(logged_hashes),
                "last_execution_at": utc_now_iso(),
            }

            # Re-raise execution error after logging (preserve original exception type)
            if not execution_succeeded and original_exception is not None:
                raise original_exception

            return result

        return wrapped

    if hasattr(device, "execute"):
        device.execute = types.MethodType(_make_wrapper("execute"), device)
    if hasattr(device, "batch_execute"):
        device.batch_execute = types.MethodType(_make_wrapper("batch_execute"), device)


class PennyLaneAdapter:
    """
    Adapter for integrating PennyLane devices with devqubit tracking.

    Unlike other adapters, this patches the device in place rather than
    wrapping it. This is necessary because PennyLane's QNode mechanism
    captures references to the device object.

    The adapter supports PennyLane's multi-layer stack where PennyLane
    can act as a frontend to Braket, Qiskit, or native simulators.

    Attributes
    ----------
    name : str
        Adapter identifier ("pennylane").
    """

    name: str = "pennylane"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a PennyLane device.

        Uses the centralized detection from utils module.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a PennyLane device.
        """
        return is_pennylane_device(executor)

    def describe_executor(self, device: Any) -> dict[str, Any]:
        """
        Create a description of the device.

        Parameters
        ----------
        device : Any
            PennyLane device instance.

        Returns
        -------
        dict
            Device description with multi-layer stack info.
        """
        # Detect physical execution provider
        backend_info = resolve_pennylane_backend(device)
        physical_provider = backend_info["provider"] if backend_info else "local"

        desc: dict[str, Any] = {
            "name": get_device_name(device),
            "type": device.__class__.__name__,
            "provider": physical_provider,  # Physical provider
            "sdk": "pennylane",  # SDK frontend
        }

        # Add wire info
        if hasattr(device, "wires"):
            try:
                desc["wires"] = list(device.wires)
                desc["num_wires"] = len(device.wires)
            except Exception:
                pass

        # Add shots info
        shots_info = extract_shots_info(device)
        desc["shots_info"] = shots_info.to_dict()

        # Add backend-specific info
        if backend_info:
            desc["backend_type"] = backend_info["backend_type"]
            if backend_info["backend_id"]:
                desc["backend_id"] = backend_info["backend_id"]

        return desc

    def wrap_executor(
        self,
        device: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
        resolve_remote_backend: bool = False,
    ) -> Any:
        """
        Enable tracking on a PennyLane device.

        Parameters
        ----------
        device : Any
            PennyLane device to wrap.
        tracker : Run
            Tracker instance.
        log_every_n : int
            Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
        log_new_circuits : bool
            Auto-log new circuit structures (default True).
        stats_update_interval : int
            Update stats every N executions (default 1000).
        resolve_remote_backend : bool
            If True, attempt to resolve remote backends (may be slow).

        Returns
        -------
        Any
            The same device instance, patched in place.
        """
        patch_device(
            device,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
        device._devqubit_tracker = tracker

        # Capture device snapshot early
        if "device_snapshot" not in tracker.record:
            try:
                device._devqubit_device_snapshot = _log_device_snapshot(
                    device,
                    tracker,
                    resolve_remote_backend=resolve_remote_backend,
                )
            except Exception as e:
                logger.warning("Failed to capture initial device snapshot: %s", e)
                device._devqubit_device_snapshot = None

        return device
