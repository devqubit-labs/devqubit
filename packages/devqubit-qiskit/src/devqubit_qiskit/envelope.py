# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Envelope and snapshot utilities for Qiskit adapter.

Provides functions for creating UEC snapshots and managing
ExecutionEnvelope lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.execution import ExecutionSnapshot
from devqubit_engine.uec.models.program import (
    ProgramArtifact,
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
from devqubit_qiskit.device import create_device_snapshot, detect_provider
from devqubit_qiskit.results import (
    detect_result_type,
    extract_quasi_distributions,
    extract_result_metadata,
    normalize_result_counts,
)
from devqubit_qiskit.utils import get_backend_name, qiskit_version


logger = logging.getLogger(__name__)


# =============================================================================
# Program Snapshot
# =============================================================================


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

    Notes
    -----
    For the base Qiskit adapter (no transpilation), executed hashes
    equal logical hashes since circuits are executed as-is.
    """
    return ProgramSnapshot(
        logical=program_artifacts,
        physical=[],
        structural_hash=structural_hash,
        parametric_hash=parametric_hash,
        executed_structural_hash=structural_hash,
        executed_parametric_hash=parametric_hash,
        num_circuits=num_circuits,
    )


# =============================================================================
# Execution Snapshot
# =============================================================================


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

    Notes
    -----
    The base Qiskit adapter uses MANUAL transpilation mode since
    users are expected to transpile circuits before submission.
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


# =============================================================================
# Result Snapshot
# =============================================================================


def create_result_snapshot(
    tracker: Run,
    backend_name: str,
    result: Any,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from a Qiskit Result object.

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

    Notes
    -----
    Qiskit uses little-endian bit order (cbit[0] on right), which
    is the UEC canonical format. No transformation is needed.

    Supported result types:
    - Measurement counts (standard)
    - Quasi-distributions (Runtime Sampler)
    - Statevector (simulators) - logged as raw artifact only
    """
    if result is None:
        return ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=ResultError(type="NullResult", message="Result is None"),
            metadata={"backend_name": backend_name},
        )

    # Log raw result
    raw_result_ref = _log_raw_result(tracker, result)

    # Detect result type
    result_type = detect_result_type(result)

    # Normalize and log counts
    counts_data = normalize_result_counts(result)
    experiments = counts_data.get("experiments", [])

    if experiments:
        tracker.log_json(
            name="counts",
            obj=counts_data,
            role="results",
            kind="result.counts.json",
        )

    # Build result items
    counts_format = CountsFormat(
        source_sdk="qiskit",
        source_key_format="qiskit_little_endian",
        bit_order="cbit0_right",
        transformed=False,
    )

    items: list[ResultItem] = []
    for exp in experiments:
        counts = exp.get("counts", {})
        shots = exp.get("shots")
        item_index = exp.get("index", 0)

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

    # Handle quasi-distributions (Runtime Sampler)
    quasi_dists = extract_quasi_distributions(result)
    if quasi_dists is not None and not items:
        for i, qd in enumerate(quasi_dists):
            from devqubit_engine.uec.models.result import QuasiProbability

            quasi_prob = QuasiProbability(
                distribution=qd,
                sum_probs=sum(qd.values()) if qd else None,
                min_prob=min(qd.values()) if qd else None,
                max_prob=max(qd.values()) if qd else None,
            )
            items.append(
                ResultItem(
                    item_index=i,
                    success=True,
                    quasi_probability=quasi_prob,
                )
            )
        tracker.log_json(
            name="quasi_distributions",
            obj={"distributions": quasi_dists},
            role="results",
            kind="result.quasi_dist.json",
        )

    # Handle statevector (simulators) - log as artifact only
    statevector_ref = _log_statevector_if_present(tracker, result)

    meta = extract_result_metadata(result)
    success = meta.get("success", True)
    status = "completed" if success else "failed"

    result_metadata: dict[str, Any] = {
        "backend_name": backend_name,
        "num_experiments": len(experiments),
        "result_type": result_type.value,
        **meta,
    }
    if statevector_ref is not None:
        result_metadata["statevector_ref"] = str(statevector_ref)

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        raw_result_ref=raw_result_ref,
        metadata=result_metadata,
    )


def _log_statevector_if_present(tracker: Run, result: Any) -> ArtifactRef | None:
    """
    Log statevector as artifact if present (simulator results).

    Statevectors grow exponentially with qubit count. To prevent
    memory exhaustion, amplitudes are only expanded for circuits
    with <= 20 qubits (~1M amplitudes). Larger statevectors are
    logged as metadata-only references.
    """
    _MAX_SV_QUBITS = 20  # 2^20 = 1M amplitudes

    try:
        if not hasattr(result, "get_statevector"):
            return None

        sv = result.get_statevector()
        if sv is None:
            return None

        # Convert to serializable format
        sv_data: dict[str, Any] = {}

        if hasattr(sv, "data"):
            # Qiskit Statevector object
            num_qubits = int(getattr(sv, "num_qubits", 0))
            sv_data["num_qubits"] = num_qubits

            if num_qubits <= _MAX_SV_QUBITS:
                data = sv.data
                sv_data["amplitudes_real"] = [float(x.real) for x in data]
                sv_data["amplitudes_imag"] = [float(x.imag) for x in data]
            else:
                sv_data["truncated"] = True
                sv_data["reason"] = (
                    f"Statevector too large to expand ({num_qubits} qubits, "
                    f"{2**num_qubits} amplitudes)"
                )
                logger.debug(
                    "Skipping statevector amplitude expansion: %d qubits",
                    num_qubits,
                )
        else:
            # Raw array
            sv_data["raw"] = str(sv)[:10000]

        return tracker.log_json(
            name="statevector",
            obj=sv_data,
            role="results",
            kind="result.statevector.json",
        )

    except Exception as e:
        logger.debug("Failed to log statevector: %s", e)
        return None


def create_failure_result_snapshot(
    exception: BaseException,
    backend_name: str,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot for a failed execution.

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


def _log_raw_result(tracker: Run, result: Any) -> ArtifactRef | None:
    """Log raw result as JSON artifact."""
    try:
        if hasattr(result, "to_dict") and callable(result.to_dict):
            result_dict = result.to_dict()
        else:
            result_dict = result
        payload = to_jsonable(result_dict)
        return tracker.log_json(
            name="qiskit.result",
            obj=payload,
            role="results",
            kind="result.qiskit.result_json",
        )
    except Exception as e:
        logger.debug("Failed to serialize result to dict: %s", e)
        return None


# =============================================================================
# Envelope Lifecycle
# =============================================================================


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

    envelope.result = result_snapshot

    if envelope.execution is not None:
        envelope.execution.completed_at = utc_now_iso()

    tracker.log_envelope(envelope=envelope)


# =============================================================================
# Device Snapshot Logging
# =============================================================================


def log_device_snapshot(backend: Any, tracker: Run) -> DeviceSnapshot:
    """
    Log device snapshot with fallback to minimal snapshot on failure.

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
        snapshot = create_device_snapshot(
            backend,
            refresh_properties=True,
            tracker=tracker,
        )
    except Exception as e:
        logger.warning(
            "Full device snapshot failed for %s: %s. Using minimal snapshot.",
            backend_name,
            e,
        )
        snapshot = _create_minimal_device_snapshot(backend, captured_at, str(e))

    tracker.record["device_snapshot"] = {
        "sdk": "qiskit",
        "backend_name": backend_name,
        "backend_type": snapshot.backend_type,
        "provider": snapshot.provider,
        "captured_at": snapshot.captured_at,
        "num_qubits": snapshot.num_qubits,
        "calibration_summary": snapshot.get_calibration_summary(),
    }

    return snapshot


def _create_minimal_device_snapshot(
    backend: Any,
    captured_at: str,
    error_msg: str | None = None,
) -> DeviceSnapshot:
    """Create a minimal DeviceSnapshot when full snapshot creation fails."""
    backend_name = get_backend_name(backend)
    name_lower = backend_name.lower()
    type_lower = type(backend).__name__.lower()

    backend_type = "simulator"
    if any(s in name_lower for s in ("ibm_", "ionq", "rigetti", "oqc")):
        backend_type = "hardware"
    elif any(s in name_lower or s in type_lower for s in ("sim", "emulator", "fake")):
        backend_type = "simulator"

    provider = detect_provider(backend)

    num_qubits = None
    try:
        num_qubits = backend.num_qubits
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to get backend num_qubits: %s", e)

    if error_msg:
        logger.warning(
            "Created minimal device snapshot for %s: %s",
            backend_name,
            error_msg,
        )

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        num_qubits=num_qubits,
        sdk_versions={"qiskit": qiskit_version()},
    )
