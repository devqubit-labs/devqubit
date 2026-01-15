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
``transformed=False`` in CountsFormat. Consumers should check the
``bit_order`` field and transform if needed for cross-SDK comparison.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from devqubit_braket.serialization import (
    BraketCircuitSerializer,
    circuits_to_text,
    serialize_openqasm,
)
from devqubit_braket.snapshot import create_device_snapshot
from devqubit_braket.utils import braket_version, get_adapter_version
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import (
    ProgramArtifact,
    ProgramSnapshot,
    TranspilationInfo,
)
from devqubit_engine.uec.result import (
    CountsFormat,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.uec.types import (
    ArtifactRef,
    ProgramRole,
    TranspilationMode,
)
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run


logger = logging.getLogger(__name__)

# Module-level serializer instance
_serializer = BraketCircuitSerializer()


def _get_braket_counts_format(transformed: bool = False) -> dict[str, Any]:
    """
    Get CountsFormat metadata for Braket results.

    Braket uses big-endian bit order (qubit 0 = leftmost bit),
    which corresponds to ``cbit0_left`` in UEC canonical terminology.

    Parameters
    ----------
    transformed : bool
        Whether counts have been transformed to canonical ``cbit0_right`` format.
        Default False - Braket's native format is preserved.

    Returns
    -------
    dict
        CountsFormat as dictionary for JSON serialization.

    Notes
    -----
    UEC canonical format is ``cbit0_right`` (like Qiskit). When ``transformed=False``,
    consumers must reverse bitstrings themselves for cross-SDK comparison.
    """
    return CountsFormat(
        source_sdk="braket",
        source_key_format="bitstring",
        bit_order="cbit0_left",  # Braket native: big-endian
        transformed=transformed,
    ).to_dict()


def serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    device_name: str,
) -> list[ArtifactRef]:
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
    list of ArtifactRef
        References to logged circuit artifacts.
    """
    artifact_refs: list[ArtifactRef] = []
    meta = {
        "backend_name": device_name,
        "braket_version": braket_version(),
    }

    for i, circuit in enumerate(circuits):
        # Serialize JAQCD (native format)
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
                artifact_refs.append(ref)
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to JAQCD: %s", i, e)

        # Serialize OpenQASM (canonical format, better for diffing)
        try:
            qasm_data = serialize_openqasm(circuit, index=i)
            tracker.log_bytes(
                kind="braket.ir.openqasm",
                data=qasm_data.as_bytes(),
                media_type="text/x-qasm; charset=utf-8",
                role="program",
                meta={**meta, "index": i, "format": "openqasm3"},
            )
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to OpenQASM: %s", i, e)

    # Log circuit diagrams (human-readable)
    try:
        diagram_text = circuits_to_text(circuits)
        tracker.log_bytes(
            kind="braket.circuits.diagram",
            data=diagram_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
            meta={"num_circuits": len(circuits)},
        )
    except Exception as e:
        logger.debug("Failed to generate circuit diagrams: %s", e)

    return artifact_refs


def create_program_snapshot(
    circuits: list[Any],
    artifact_refs: list[ArtifactRef],
    structural_hash: str | None,
    parametric_hash: str | None = None,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from circuits and their artifact refs.

    Parameters
    ----------
    circuits : list
        List of Braket circuits.
    artifact_refs : list of ArtifactRef
        References to logged circuit artifacts.
    structural_hash : str or None
        Structural hash (ignores parameter values).
    parametric_hash : str or None
        Parametric hash (includes parameter values).

    Returns
    -------
    ProgramSnapshot
        Program snapshot with logical artifacts and hashes.
    """
    logical_artifacts: list[ProgramArtifact] = []

    for i, ref in enumerate(artifact_refs):
        circuit_name = None
        if i < len(circuits):
            circuit_name = getattr(circuits[i], "name", None)

        logical_artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="jaqcd",
                name=circuit_name or f"circuit_{i}",
                index=i,
            )
        )

    # If parametric_hash not provided, use structural_hash
    effective_parametric_hash = parametric_hash or structural_hash

    return ProgramSnapshot(
        logical=logical_artifacts,
        physical=[],  # Braket doesn't expose transpiled circuits
        structural_hash=structural_hash,
        parametric_hash=effective_parametric_hash,
        # For Braket without transpilation, executed hashes equal logical
        executed_structural_hash=structural_hash,
        executed_parametric_hash=effective_parametric_hash,
        num_circuits=len(circuits),
    )


def create_execution_snapshot(
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
        Task identifiers.
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
        job_ids=task_ids,
        execution_count=execution_index,
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANAGED,
            transpiled_by="provider",
        ),
        options=options or {},
        sdk="braket",
    )


def create_result_snapshot(
    result: Any,
    raw_result_ref: ArtifactRef | None,
    shots: int | None,
    error: Exception | None = None,
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

    Returns
    -------
    ResultSnapshot
        Result snapshot with items list and success status.
    """
    from devqubit_braket.results import extract_counts_payload

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
        # Check if result is already a combined payload dict (from batch)
        if isinstance(result, dict) and "experiments" in result:
            counts_payload = result
        else:
            counts_payload = extract_counts_payload(result)

        if counts_payload and counts_payload.get("experiments"):
            format_dict = _get_braket_counts_format()

            for exp in counts_payload["experiments"]:
                counts_data = exp.get("counts", {})
                item_success = bool(counts_data)

                # Build counts structure
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

            # Success = at least one item with non-empty counts
            success = any(item.success for item in items)
            status = "completed" if success else "partial"

        # Fallback: if we have a result but no experiments extracted
        if not items:
            batch_size = result.get("batch_size", 1) if isinstance(result, dict) else 1
            for i in range(batch_size):
                items.append(
                    ResultItem(
                        item_index=i,
                        success=False,
                        counts=None,
                    )
                )
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
    from devqubit_braket.utils import get_backend_name

    device_name = get_backend_name(device=device)

    # Create device snapshot with tracker for raw_properties logging
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

    # Log circuits and get artifact refs
    artifact_refs = serialize_and_log_circuits(
        tracker=tracker,
        circuits=circuits,
        device_name=device_name,
    )

    # Create program snapshot
    program_snapshot = create_program_snapshot(
        circuits=circuits,
        artifact_refs=artifact_refs,
        structural_hash=structural_hash,
        parametric_hash=parametric_hash,
    )

    # Create execution snapshot
    execution_snapshot = create_execution_snapshot(
        shots=shots,
        task_ids=task_ids,
        submitted_at=submitted_at,
        execution_index=execution_index,
        options=options,
    )

    # Create ProducerInfo
    sdk_version = braket_version()
    producer = ProducerInfo.create(
        adapter="devqubit-braket",
        adapter_version=get_adapter_version(),
        sdk="braket",
        sdk_version=sdk_version,
        frontends=["braket-sdk"],
    )

    # Create pending result
    pending_result = ResultSnapshot(
        success=False,
        status="failed",  # Will be updated by finalize_envelope
        items=[],
        metadata={"state": "pending"},
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
    from devqubit_braket.results import extract_counts_payload

    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    # Log raw result and get ref
    raw_result_ref: ArtifactRef | None = None
    if result is not None:
        try:
            result_payload = to_jsonable(result)
        except Exception:
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

    # Create result snapshot
    result_snapshot = create_result_snapshot(result, raw_result_ref, shots, error)

    # Update execution snapshot with completion time
    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    # Add result to envelope
    envelope.result = result_snapshot

    # Extract counts for separate logging
    counts_payload = None
    if result is not None:
        try:
            counts_payload = extract_counts_payload(result)
        except Exception as e:
            logger.debug("Failed to extract counts payload: %s", e)

    # Validate and log envelope
    try:
        tracker.log_envelope(envelope=envelope)
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
