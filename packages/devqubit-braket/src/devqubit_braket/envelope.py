# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Envelope creation for Braket adapter.

Creates UEC compliant ExecutionEnvelopes with proper snapshots
for device, program, execution, and result data.

Notes
-----
Braket uses big-endian bit ordering (qubit 0 = leftmost bit).
In UEC terminology this is ``cbit0_left``. The canonical UEC format
is ``cbit0_right`` (little-endian, like Qiskit).

By default, this adapter preserves Braket's native format and records
``transformed=False`` in CountsFormat.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from devqubit_braket.device import create_device_snapshot
from devqubit_braket.results import extract_counts_payload
from devqubit_braket.serialization import (
    BraketCircuitSerializer,
    circuits_to_text,
    serialize_openqasm,
)
from devqubit_braket.utils import braket_version, get_adapter_version, get_backend_name
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.storage.types import ArtifactRef
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
from devqubit_engine.uec.models.result import (
    CountsFormat,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)

_serializer = BraketCircuitSerializer()

# Pre-computed base CountsFormat dict for Braket (source=braket, cbit0_left).
# Dynamic fields (measured_qubits) are overlaid per-call.
_BRAKET_COUNTS_FORMAT: dict[str, Any] = CountsFormat(
    source_sdk="braket",
    source_key_format="bitstring",
    bit_order="cbit0_left",
    transformed=False,
).to_dict()


# =============================================================================
# Counts Format
# =============================================================================


def _get_braket_counts_format(
    transformed: bool = False,
    measured_qubits: list[int] | None = None,
) -> dict[str, Any]:
    """
    Get CountsFormat metadata for Braket results.

    Braket uses big-endian bit order (qubit 0 = leftmost bit),
    which corresponds to ``cbit0_left`` in UEC canonical terminology.

    Parameters
    ----------
    transformed : bool
        Whether counts have been transformed to canonical ``cbit0_right`` format.
    measured_qubits : list of int, optional
        Ordered list of measured qubit indices (if known). When provided,
        enables accurate cross-SDK comparison even when qubit ordering differs.

    Returns
    -------
    dict
        CountsFormat as dictionary for JSON serialization.
    """
    fmt = CountsFormat(
        source_sdk="braket",
        source_key_format="bitstring",
        bit_order="cbit0_left",
        transformed=transformed,
    )
    result = fmt.to_dict() if transformed else dict(_BRAKET_COUNTS_FORMAT)

    if measured_qubits is not None:
        result["measured_qubits"] = measured_qubits

    return result


def _extract_measured_qubits(result: Any) -> list[int] | None:
    """
    Extract measured qubit indices from a Braket result.

    The measured_qubits list indicates which qubits were measured and in what
    order, enabling accurate bitstring interpretation even when qubit ordering
    differs from the default sequential order.

    Parameters
    ----------
    result : Any
        Braket result object.

    Returns
    -------
    list of int or None
        Ordered list of measured qubit indices, or None if not available.
    """
    if result is None:
        return None

    # Try measured_qubits attribute (GateModelQuantumTaskResult)
    try:
        mq = getattr(result, "measured_qubits", None)
        if mq is not None:
            return [int(q) for q in mq]
    except (TypeError, ValueError) as e:
        logger.debug("Failed to parse int: %s", e)

    # Try measurement_counts_indices (some result types)
    try:
        mci = getattr(result, "measurement_counts_indices", None)
        if mci is not None:
            return [int(q) for q in mci]
    except (TypeError, ValueError) as e:
        logger.debug("Failed to parse int: %s", e)

    return None


# =============================================================================
# Circuit Serialization and Logging
# =============================================================================


def _serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    device_name: str,
) -> list[ProgramArtifact]:
    """
    Serialize circuits and log as artifacts.

    Logs both JAQCD and OpenQASM formats for comprehensive coverage.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    circuits : list
        List of Braket circuits.
    device_name : str
        Backend name for metadata.

    Returns
    -------
    list of ProgramArtifact
        Program artifacts for both JAQCD and OpenQASM formats.
    """
    artifacts: list[ProgramArtifact] = []
    meta = {
        "backend_name": device_name,
        "braket_version": braket_version(),
    }

    for i, circuit in enumerate(circuits):
        circuit_name = getattr(circuit, "name", None) or f"circuit_{i}"

        # JAQCD (native format)
        try:
            jaqcd_data = _serializer.serialize(circuit, CircuitFormat.JAQCD, index=i)
            ref = tracker.log_bytes(
                kind="braket.ir.jaqcd",
                data=jaqcd_data.as_bytes(),
                media_type="application/json",
                role="program",
                meta={**meta, "index": i},
            )
            if ref:
                artifacts.append(
                    ProgramArtifact(
                        ref=ref,
                        role=ProgramRole.LOGICAL,
                        format="jaqcd",
                        name=circuit_name,
                        index=i,
                    )
                )
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to JAQCD: %s", i, e)

        # OpenQASM (canonical format)
        try:
            qasm_data = serialize_openqasm(circuit, index=i)
            ref = tracker.log_bytes(
                kind="braket.ir.openqasm",
                data=qasm_data.as_bytes(),
                media_type="text/x-qasm; charset=utf-8",
                role="program",
                meta={**meta, "index": i, "format": "openqasm3"},
            )
            if ref:
                artifacts.append(
                    ProgramArtifact(
                        ref=ref,
                        role=ProgramRole.LOGICAL,
                        format="openqasm3",
                        name=f"{circuit_name}_qasm",
                        index=i,
                    )
                )
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to OpenQASM: %s", i, e)

    # Circuit diagrams (human-readable)
    try:
        diagram_text = circuits_to_text(circuits)
        ref = tracker.log_bytes(
            kind="braket.circuits.diagram",
            data=diagram_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
            meta={"num_circuits": len(circuits)},
        )
        if ref:
            artifacts.append(
                ProgramArtifact(
                    ref=ref,
                    role=ProgramRole.LOGICAL,
                    format="diagram",
                    name="circuits_diagram",
                    index=0,
                )
            )
    except Exception as e:
        logger.debug("Failed to generate circuit diagrams: %s", e)

    return artifacts


# =============================================================================
# Snapshot Creation
# =============================================================================


def _create_program_snapshot(
    circuits: list[Any],
    artifacts: list[ProgramArtifact],
    structural_hash: str | None,
    parametric_hash: str | None = None,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from circuits and their artifacts.

    Parameters
    ----------
    circuits : list
        List of Braket circuits.
    artifacts : list of ProgramArtifact
        Program artifacts (JAQCD, OpenQASM, diagrams).
    structural_hash : str or None
        Structural hash (ignores parameter values).
    parametric_hash : str or None
        Parametric hash (includes parameter values).

    Returns
    -------
    ProgramSnapshot
        Program snapshot with logical artifacts and hashes.
    """
    # UEC Contract: for circuits without params, parametric == structural
    effective_parametric_hash = parametric_hash or structural_hash

    return ProgramSnapshot(
        logical=artifacts,
        physical=[],  # Braket doesn't expose transpiled circuits
        structural_hash=structural_hash,
        parametric_hash=effective_parametric_hash,
        executed_structural_hash=structural_hash,
        executed_parametric_hash=effective_parametric_hash,
        num_circuits=len(circuits),
    )


def _create_execution_snapshot(
    shots: int | None,
    task_ids: list[str],
    submitted_at: str,
    execution_index: int = 1,
    options: dict[str, Any] | None = None,
) -> ExecutionSnapshot:
    """
    Create an ExecutionSnapshot for a Braket task submission.

    Parameters
    ----------
    shots : int or None
        Number of shots (None means provider default).
    task_ids : list of str
        Task identifiers (Braket-specific).
    submitted_at : str
        ISO 8601 submission timestamp.
    execution_index : int
        Which execution this is (1-indexed sequence number).
    options : dict, optional
        Additional execution options.

    Returns
    -------
    ExecutionSnapshot
        Execution metadata snapshot.
    """
    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=shots,
        task_ids=task_ids,
        execution_count=execution_index,
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANAGED,
            transpiled_by="provider",
        ),
        options=options or {},
        sdk="braket",
    )


def _create_result_snapshot(
    result: Any,
    raw_result_ref: ArtifactRef | None,
    shots: int | None,
    error: Exception | None = None,
    counts_payload: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from Braket result.

    Parameters
    ----------
    result : Any
        Braket result object (may be None on failure).
    raw_result_ref : ArtifactRef or None
        Reference to raw result artifact.
    shots : int or None
        Number of shots used.
    error : Exception or None
        Exception if execution failed.
    counts_payload : dict or None
        Pre-extracted counts payload. When provided, skips redundant
        extraction. Callers should extract once and pass through.

    Returns
    -------
    ResultSnapshot
        Result snapshot with items list and success status.
    """
    items: list[ResultItem] = []
    success = False
    status = "failed"
    result_error: ResultError | None = None

    if error is not None:
        result_error = ResultError(
            type=type(error).__name__,
            message=str(error),
        )
        status = "failed"
    elif result is not None:
        # Extract measured qubits if available (for accurate bit-order semantics)
        measured_qubits = _extract_measured_qubits(result)

        # Use pre-computed counts_payload if provided
        if counts_payload is None:
            if isinstance(result, dict) and "experiments" in result:
                counts_payload = result
            else:
                counts_payload = extract_counts_payload(result)

        if counts_payload and counts_payload.get("experiments"):
            format_dict = _get_braket_counts_format(measured_qubits=measured_qubits)

            for exp in counts_payload["experiments"]:
                counts_data = exp.get("counts", {})
                item_success = bool(counts_data)

                counts_obj = None
                if counts_data:
                    counts_obj = {
                        "counts": counts_data,
                        "shots": shots or sum(counts_data.values()),
                        "format": format_dict,
                    }

                items.append(
                    ResultItem(
                        item_index=exp.get("index", len(items)),
                        success=item_success,
                        counts=counts_obj,
                    )
                )

            success = any(item.success for item in items)
            status = "completed" if success else "partial"

        # Fallback: if we have a result but no experiments extracted
        if not items:
            batch_size = result.get("batch_size", 1) if isinstance(result, dict) else 1
            for i in range(batch_size):
                items.append(ResultItem(item_index=i, success=False, counts=None))
            status = "partial"

        # For shots=0 (analytical), may get statevector/other instead of counts
        if not success and shots == 0:
            if hasattr(result, "values") or hasattr(result, "result_types"):
                success = True
                status = "completed"

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        error=result_error,
        raw_result_ref=raw_result_ref,
        metadata={},
    )


# =============================================================================
# Public API
# =============================================================================


def create_envelope(
    tracker: Run,
    device: Any,
    circuits: list[Any],
    shots: int | None,
    task_ids: list[str],
    submitted_at: str,
    structural_hash: str | None,
    parametric_hash: str | None = None,
    execution_index: int = 1,
    options: dict[str, Any] | None = None,
) -> ExecutionEnvelope:
    """
    Create and log a complete ExecutionEnvelope (pre-result).

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    device : Any
        Braket device.
    circuits : list
        List of circuits.
    shots : int or None
        Number of shots.
    task_ids : list of str
        Task identifiers.
    submitted_at : str
        Submission timestamp.
    structural_hash : str or None
        Structural hash (ignores parameter values).
    parametric_hash : str or None
        Parametric hash (includes parameter values).
    execution_index : int
        Which execution this is (1-indexed sequence number).
    options : dict, optional
        Execution options.

    Returns
    -------
    ExecutionEnvelope
        Envelope with device, program, and execution snapshots.
    """
    device_name = get_backend_name(device=device)

    # Create device snapshot
    try:
        device_snapshot = create_device_snapshot(device=device, tracker=tracker)
    except Exception as e:
        logger.warning(
            "Failed to create device snapshot: %s. Using minimal snapshot.", e
        )
        device_snapshot = DeviceSnapshot(
            captured_at=utc_now_iso(),
            backend_name=device_name,
            backend_type="simulator",
            provider="aws_braket",
            sdk_versions={"braket": braket_version()},
        )

    # Update tracker record
    tracker.record["device_snapshot"] = {
        "sdk": "braket",
        "backend_name": device_name,
        "backend_type": device_snapshot.backend_type,
        "provider": device_snapshot.provider,
        "captured_at": device_snapshot.captured_at,
        "num_qubits": device_snapshot.num_qubits,
        "calibration_summary": device_snapshot.get_calibration_summary(),
    }

    # Log circuits and get artifacts
    artifacts = _serialize_and_log_circuits(
        tracker=tracker,
        circuits=circuits,
        device_name=device_name,
    )

    # Create snapshots
    program_snapshot = _create_program_snapshot(
        circuits=circuits,
        artifacts=artifacts,
        structural_hash=structural_hash,
        parametric_hash=parametric_hash,
    )

    execution_snapshot = _create_execution_snapshot(
        shots=shots,
        task_ids=task_ids,
        submitted_at=submitted_at,
        execution_index=execution_index,
        options=options,
    )

    # Create ProducerInfo
    producer = ProducerInfo.create(
        adapter="devqubit-braket",
        adapter_version=get_adapter_version(),
        sdk="braket",
        sdk_version=braket_version(),
        frontends=["braket-sdk"],
    )

    # Create pending result (will be updated by finalize_envelope)
    pending_result = ResultSnapshot.create_pending(
        metadata={"awaiting_result": True},
    )

    return ExecutionEnvelope(
        envelope_id=uuid.uuid4().hex[:26],
        created_at=utc_now_iso(),
        producer=producer,
        device=device_snapshot,
        program=program_snapshot,
        execution=execution_snapshot,
        result=pending_result,
    )


def finalize_envelope(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result: Any,
    device_name: str,
    shots: int | None,
    error: Exception | None = None,
) -> ExecutionEnvelope:
    """
    Finalize envelope with result and log it.

    This function never raises exceptions - tracking should never crash
    user experiments. Validation errors are logged but execution continues.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result : Any
        Braket result object (may be None on failure).
    device_name : str
        Device name.
    shots : int or None
        Number of shots.
    error : Exception or None
        Exception if execution failed.

    Returns
    -------
    ExecutionEnvelope
        Finalized envelope.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    # Log raw result
    raw_result_ref: ArtifactRef | None = None
    if result is not None:
        try:
            result_payload = to_jsonable(result)
        except (TypeError, ValueError):
            result_payload = {"repr": repr(result)[:2000]}

        try:
            raw_result_ref = tracker.log_json(
                name="braket.result",
                obj=result_payload,
                role="results",
                kind="result.braket.raw.json",
            )
        except Exception as e:
            logger.warning("Failed to log raw result: %s", e)
    elif error:
        try:
            tracker.log_json(
                name="braket.error",
                obj={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "timestamp": utc_now_iso(),
                },
                role="results",
                kind="result.braket.error.json",
            )
        except Exception as e:
            logger.warning("Failed to log error: %s", e)

    # Extract counts once (used for both result snapshot and separate logging)
    counts_payload: dict[str, Any] | None = None
    if result is not None and error is None:
        try:
            if isinstance(result, dict) and "experiments" in result:
                counts_payload = result  # Batch: already structured
            else:
                counts_payload = extract_counts_payload(result)
        except Exception as e:
            logger.debug("Failed to extract counts payload: %s", e)

    # Create result snapshot
    result_snapshot = _create_result_snapshot(
        result, raw_result_ref, shots, error, counts_payload=counts_payload
    )

    # Update execution snapshot with completion time
    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    envelope.result = result_snapshot

    # Validate and log envelope
    # EnvelopeValidationError is raised by log_envelope for adapter runs with
    # invalid envelopes - this MUST propagate to enforce UEC contract

    try:
        tracker.log_envelope(envelope=envelope)
    except EnvelopeValidationError:
        # Re-raise validation errors - adapters must produce valid envelopes
        raise
    except Exception as e:
        logger.warning("Failed to log envelope: %s", e)

    # Log normalized counts
    if counts_payload is not None:
        try:
            tracker.log_json(
                name="counts",
                obj=counts_payload,
                role="results",
                kind="result.counts.json",
            )
        except Exception as e:
            logger.debug("Failed to log counts: %s", e)

    # Update tracker record
    tracker.record["results"] = {
        "completed_at": utc_now_iso(),
        "backend_name": device_name,
        "num_items": len(result_snapshot.items),
        "status": result_snapshot.status,
        "success": result_snapshot.success,
    }
    if error:
        tracker.record["results"]["error"] = str(error)
        tracker.record["results"]["error_type"] = type(error).__name__

    logger.debug("Logged execution envelope for %s", device_name)
    return envelope


def log_submission_failure(
    tracker: Run,
    device_name: str,
    error: Exception,
    circuits: list[Any],
    shots: int | None,
    submitted_at: str,
) -> None:
    """
    Log a task submission failure.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    device_name : str
        Device name.
    error : Exception
        The exception that occurred.
    circuits : list
        Circuits that were being submitted.
    shots : int or None
        Requested shots.
    submitted_at : str
        Submission timestamp.
    """
    error_info = {
        "type": "submission_failure",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "device_name": device_name,
        "num_circuits": len(circuits),
        "shots": shots,
        "submitted_at": submitted_at,
        "failed_at": utc_now_iso(),
    }

    try:
        tracker.log_json(
            name="submission_failure",
            obj=error_info,
            role="error",
            kind="devqubit.submission_failure.json",
        )
    except Exception as e:
        logger.warning("Failed to log submission failure: %s", e)
