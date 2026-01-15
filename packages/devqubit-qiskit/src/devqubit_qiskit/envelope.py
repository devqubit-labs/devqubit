# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Envelope and snapshot utilities for Qiskit adapter.

This module provides functions for creating UEC snapshots and
managing ExecutionEnvelope lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.core.run import Run
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
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
from devqubit_engine.uec.types import ArtifactRef, TranspilationMode
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.results import extract_result_metadata, normalize_result_counts
from devqubit_qiskit.snapshot import create_device_snapshot
from devqubit_qiskit.utils import get_backend_name, qiskit_version


logger = logging.getLogger(__name__)


def detect_physical_provider(backend: Any) -> str:
    """
    Detect physical provider from backend (not SDK).

    UEC requires provider to be the physical backend provider,
    not the SDK name. SDK goes in producer.frontends[].

    Parameters
    ----------
    backend : Any
        Qiskit backend instance.

    Returns
    -------
    str
        Physical provider: "ibm_quantum", "aer", "fake", or "local".
    """
    module_name = type(backend).__module__.lower()
    backend_name = get_backend_name(backend).lower()

    if "ibm" in module_name or "ibm_" in backend_name:
        return "ibm_quantum"
    if "qiskit_aer" in module_name or "aer" in module_name:
        return "aer"
    if "fake" in module_name:
        return "fake"
    return "local"


def create_program_snapshot(
    program_artifacts: list[ProgramArtifact],
    structural_hash: str | None,
    parametric_hash: str | None,
    num_circuits: int,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from logged artifacts (UEC v1.0 compliant).

    Parameters
    ----------
    program_artifacts : list of ProgramArtifact
        References to logged circuit artifacts.
    structural_hash : str or None
        Structural hash of circuits (ignores parameter values).
    parametric_hash : str or None
        Parametric hash of circuits (includes parameter values).
    num_circuits : int
        Number of circuits in the program.

    Returns
    -------
    ProgramSnapshot
        Program snapshot with artifact references and hashes.
    """
    return ProgramSnapshot(
        logical=program_artifacts,
        physical=[],  # Base Qiskit adapter doesn't transpile
        structural_hash=structural_hash,
        parametric_hash=parametric_hash,
        # For base Qiskit without transpilation, executed hashes equal logical
        executed_structural_hash=structural_hash,
        executed_parametric_hash=parametric_hash,
        num_circuits=num_circuits,
    )


def create_execution_snapshot(
    submitted_at: str,
    shots: int | None,
    exec_count: int,
    job_ids: list[str] | None,
    options: dict[str, Any],
) -> ExecutionSnapshot:
    """
    Create an ExecutionSnapshot.

    Parameters
    ----------
    submitted_at : str
        ISO timestamp of submission.
    shots : int or None
        Number of shots requested.
    exec_count : int
        Execution count.
    job_ids : list of str or None
        Job IDs if available.
    options : dict
        Execution options (args, kwargs).

    Returns
    -------
    ExecutionSnapshot
        Execution metadata snapshot.
    """
    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=shots,
        execution_count=exec_count,
        job_ids=job_ids or [],
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANUAL,
            transpiled_by="user",
        ),
        options=options,
        sdk="qiskit",
    )


def create_result_snapshot(
    tracker: Run,
    backend_name: str,
    result: Any,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from a Qiskit Result object.

    Uses UEC 1.0 structure with items[], CountsFormat for
    cross-SDK comparability.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    backend_name : str
        Backend name.
    result : Any
        Qiskit Result object.

    Returns
    -------
    ResultSnapshot
        Structured result snapshot with items[].
    """
    # Handle None result
    if result is None:
        return ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=ResultError(type="NullResult", message="Result is None"),
            metadata={"backend_name": backend_name},
        )

    # Serialize full result as artifact
    raw_result_ref: ArtifactRef | None = None
    try:
        if hasattr(result, "to_dict") and callable(result.to_dict):
            result_dict = result.to_dict()
        else:
            result_dict = result
        payload = to_jsonable(result_dict)
        raw_result_ref = tracker.log_json(
            name="qiskit.result",
            obj=payload,
            role="results",
            kind="result.qiskit.result_json",
        )
    except Exception as e:
        logger.debug("Failed to serialize result to dict: %s", e)

    # Extract measurement counts
    counts_data = normalize_result_counts(result)
    experiments = counts_data.get("experiments", [])

    # Log counts as separate artifact
    if experiments:
        tracker.log_json(
            name="counts",
            obj=counts_data,
            role="results",
            kind="result.counts.json",
        )

    # Qiskit counts format metadata
    # Qiskit uses little-endian (cbit[0] on right) = UEC canonical
    counts_format = CountsFormat(
        source_sdk="qiskit",
        source_key_format="qiskit_little_endian",
        bit_order="cbit0_right",  # Qiskit native = UEC canonical
        transformed=False,  # No transformation needed
    )

    # Build ResultItem list
    items: list[ResultItem] = []
    for exp in experiments:
        counts = exp.get("counts", {})
        shots = exp.get("shots")
        item_index = exp.get("index", 0)

        # Ensure counts keys are strings and values are ints
        normalized_counts = {str(k): int(v) for k, v in counts.items()}

        items.append(
            ResultItem(
                item_index=item_index,
                success=True,
                counts={
                    "counts": normalized_counts,
                    "shots": shots,
                    "format": counts_format.to_dict(),
                },
            )
        )

    # Extract metadata for status
    meta = extract_result_metadata(result)
    success = meta.get("success", True)
    status = "completed" if success else "failed"

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        raw_result_ref=raw_result_ref,
        metadata={
            "backend_name": backend_name,
            "num_experiments": len(experiments),
            **meta,
        },
    )


def create_failure_result_snapshot(
    exception: BaseException,
    backend_name: str,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot for a failed execution.

    Used when job.result() raises an exception. Ensures envelope
    is always created even on failures (UEC requirement).

    Parameters
    ----------
    exception : BaseException
        The exception that caused the failure.
    backend_name : str
        Backend name for metadata.

    Returns
    -------
    ResultSnapshot
        Failed result snapshot with error details.
    """
    return ResultSnapshot.create_failed(
        exception=exception,
        metadata={"backend_name": backend_name},
    )


def finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """
    Finalize envelope with result and log as artifact.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result_snapshot : ResultSnapshot
        Result to add to envelope.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if result_snapshot is None:
        logger.warning("Finalizing envelope with None result_snapshot")

    # Add result to envelope
    envelope.result = result_snapshot

    # Set completion time
    if envelope.execution is not None:
        envelope.execution.completed_at = utc_now_iso()

    # Validate and log envelope using tracker's canonical method
    tracker.log_envelope(envelope=envelope)


def create_minimal_device_snapshot(
    backend: Any,
    captured_at: str,
    error_msg: str | None = None,
) -> DeviceSnapshot:
    """
    Create a minimal DeviceSnapshot when full snapshot creation fails.

    Ensures envelope can always be completed even if backend introspection
    fails due to network issues or unsupported backend types.

    Parameters
    ----------
    backend : Any
        Qiskit backend (may be partially functional).
    captured_at : str
        ISO timestamp.
    error_msg : str, optional
        Error message explaining why full snapshot failed.

    Returns
    -------
    DeviceSnapshot
        Minimal snapshot with available information.
    """
    backend_name = get_backend_name(backend)
    name_lower = backend_name.lower()
    type_lower = type(backend).__name__.lower()

    # Determine backend type - default to simulator (safer fallback)
    # Schema allows: "hardware", "simulator", "emulator"
    backend_type = "simulator"
    if any(s in name_lower for s in ("ibm_", "ionq", "rigetti", "oqc")):
        backend_type = "hardware"
    elif any(s in name_lower or s in type_lower for s in ("sim", "emulator", "fake")):
        backend_type = "simulator"

    # Detect physical provider (not SDK)
    provider = detect_physical_provider(backend)

    # Try to get num_qubits
    num_qubits = None
    try:
        num_qubits = backend.num_qubits
    except Exception:
        pass

    snapshot = DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        num_qubits=num_qubits,
        sdk_versions={"qiskit": qiskit_version()},
    )

    if error_msg:
        logger.warning(
            "Created minimal device snapshot for %s: %s",
            backend_name,
            error_msg,
        )

    return snapshot


def log_device_snapshot(backend: Any, tracker: Run) -> DeviceSnapshot:
    """
    Log device snapshot with fallback to minimal snapshot on failure.

    Logs both the snapshot summary and raw properties as separate artifacts
    for complete backend state capture.

    Parameters
    ----------
    backend : Any
        Qiskit backend.
    tracker : Run
        Tracker instance.

    Returns
    -------
    DeviceSnapshot
        Created device snapshot (full or minimal).
    """
    backend_name = get_backend_name(backend)
    captured_at = utc_now_iso()

    try:
        # Create snapshot with tracker for raw_properties logging
        snapshot = create_device_snapshot(
            backend,
            refresh_properties=True,
            tracker=tracker,
        )
    except Exception as e:
        # Generate minimal snapshot on failure instead of propagating
        logger.warning(
            "Full device snapshot failed for %s: %s. Using minimal snapshot.",
            backend_name,
            e,
        )
        snapshot = create_minimal_device_snapshot(
            backend, captured_at, error_msg=str(e)
        )

    # Update tracker record with summary (for querying and fingerprinting)
    tracker.record["device_snapshot"] = {
        "sdk": "qiskit",
        "backend_name": backend_name,
        "backend_type": snapshot.backend_type,
        "provider": snapshot.provider,
        "captured_at": snapshot.captured_at,
        "num_qubits": snapshot.num_qubits,
        "calibration_summary": snapshot.get_calibration_summary(),
    }

    logger.debug("Logged device snapshot for %s", backend_name)

    return snapshot
