# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC envelope synthesis from RunRecord.

This module provides functions for synthesizing ExecutionEnvelope from
RunRecord data when no envelope artifact exists. This is intended for
**manual runs only** - adapter runs should always have envelope created
by the adapter.

Synthesized envelopes have limitations:

- ``metadata.synthesized_from_run=True`` marks as synthesized
- ``metadata.manual_run=True`` marks as manual (if no adapter)
- ``program.structural_hash`` is None (engine cannot compute)
- ``program.parametric_hash`` is None (engine cannot compute)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.models.calibration import DeviceCalibration
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.execution import ExecutionSnapshot, ProducerInfo
from devqubit_engine.uec.models.program import (
    ProgramArtifact,
    ProgramRole,
    ProgramSnapshot,
)
from devqubit_engine.uec.models.result import (
    CountsFormat,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.utils.common import is_manual_run_record


if TYPE_CHECKING:
    from devqubit_engine.core.record import RunRecord
    from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol


logger = logging.getLogger(__name__)


# Keys in execute section that are volatile (change between runs)
# and should be stripped when computing fingerprints or comparing
VOLATILE_EXECUTE_KEYS: frozenset[str] = frozenset(
    {
        "submitted_at",
        "job_id",
        "job_ids",
        "completed_at",
        "session_id",
        "task_id",
        "task_ids",
    }
)


def _build_producer(record: "RunRecord") -> ProducerInfo:
    """
    Build ProducerInfo from RunRecord.

    Parameters
    ----------
    record : RunRecord
        Run record to extract producer info from.

    Returns
    -------
    ProducerInfo
        Constructed producer info (best effort).
    """
    adapter = record.record.get("adapter", "manual")
    if not adapter or adapter == "":
        adapter = "manual"

    # Try to get engine version from environment
    env = record.record.get("environment") or {}
    packages = env.get("packages") or {}
    engine_version = packages.get("devqubit-engine", "unknown")

    # Build frontends list
    frontends = [adapter] if adapter != "manual" else ["manual"]

    return ProducerInfo(
        name="devqubit",
        engine_version=engine_version,
        adapter=adapter,
        adapter_version="unknown",
        sdk=(
            adapter.replace("devqubit-", "")
            if adapter.startswith("devqubit-")
            else adapter
        ),
        sdk_version="unknown",
        frontends=frontends,
    )


def _build_device(record: "RunRecord") -> DeviceSnapshot | None:
    """
    Build DeviceSnapshot from RunRecord.

    Parameters
    ----------
    record : RunRecord
        Run record to extract device info from.

    Returns
    -------
    DeviceSnapshot or None
        Constructed device snapshot, or None if insufficient data.
    """
    backend = record.record.get("backend") or {}
    if not isinstance(backend, dict):
        backend = {}

    # Need at least backend name to create snapshot
    backend_name = backend.get("name", "")
    if not backend_name:
        return None

    # Get device_snapshot summary from record
    snapshot_summary = record.record.get("device_snapshot") or {}
    if not isinstance(snapshot_summary, dict):
        snapshot_summary = {}

    # Build calibration if available
    calibration = None
    cal_data = snapshot_summary.get("calibration")
    if isinstance(cal_data, dict):
        try:
            calibration = DeviceCalibration.from_dict(cal_data)
        except Exception as e:
            logger.debug("Failed to parse calibration data: %s", e)

    # Determine captured_at
    captured_at = snapshot_summary.get("captured_at") or record.created_at

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend.get("type", "unknown"),
        provider=backend.get("provider", "unknown"),
        backend_id=backend.get("backend_id"),
        num_qubits=snapshot_summary.get("num_qubits"),
        connectivity=snapshot_summary.get("connectivity"),
        native_gates=snapshot_summary.get("native_gates"),
        calibration=calibration,
    )


def _build_execution(record: "RunRecord") -> ExecutionSnapshot | None:
    """
    Build ExecutionSnapshot from RunRecord.

    Parameters
    ----------
    record : RunRecord
        Run record to extract execution info from.

    Returns
    -------
    ExecutionSnapshot or None
        Constructed execution snapshot, or None if insufficient data.
    """
    execute = record.record.get("execute") or {}
    if not isinstance(execute, dict):
        execute = {}

    # Get submission time
    submitted_at = execute.get("submitted_at") or record.created_at

    # Get shots from execute or params
    shots = execute.get("shots")
    if shots is None:
        data = record.record.get("data") or {}
        params = data.get("params") or {}
        shots = params.get("shots")

    # Build options (strip volatile keys)
    options = {k: v for k, v in execute.items() if k not in VOLATILE_EXECUTE_KEYS}

    # Get job IDs
    job_ids = []
    if execute.get("job_id"):
        job_ids.append(str(execute["job_id"]))
    if execute.get("job_ids"):
        job_ids.extend([str(j) for j in execute["job_ids"]])

    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=shots,
        job_ids=job_ids,
        options=options if options else {},
        completed_at=record.record.get("info", {}).get("ended_at"),
    )


def _build_program(record: "RunRecord") -> ProgramSnapshot | None:
    """
    Build ProgramSnapshot from RunRecord and artifacts.

    Parameters
    ----------
    record : RunRecord
        Run record to extract program info from.

    Returns
    -------
    ProgramSnapshot or None
        Constructed program snapshot, or None if no program artifacts.

    Notes
    -----
    For manual runs, structural_hash and parametric_hash are not available.
    Compare will report "hash unavailable" for these runs.
    """
    logical: list[ProgramArtifact] = []
    physical: list[ProgramArtifact] = []
    structural_hash = None

    # Check for circuit_hash in execute metadata (pre-UEC field)
    execute = record.record.get("execute") or {}
    if isinstance(execute, dict) and execute.get("circuit_hash"):
        structural_hash = str(execute["circuit_hash"])

    # Build from program artifacts
    for artifact in record.artifacts:
        if artifact.role != "program":
            continue

        # Determine format from kind
        fmt = "unknown"
        if "openqasm3" in artifact.kind.lower():
            fmt = "openqasm3"
        elif "qpy" in artifact.kind.lower():
            fmt = "qpy"
        elif "json" in artifact.kind.lower():
            fmt = "json"

        # Get metadata
        meta = artifact.meta or {}
        name = meta.get("program_name") or meta.get("name")
        index = meta.get("program_index")

        # All program artifacts are logical (user-provided)
        prog_artifact = ProgramArtifact(
            ref=artifact,
            role=ProgramRole.LOGICAL,
            format=fmt,
            name=name,
            index=index,
        )
        logical.append(prog_artifact)

        # Check for circuit_hash in artifact metadata
        if meta.get("circuit_hash") and structural_hash is None:
            structural_hash = str(meta["circuit_hash"])

    # If no program artifacts, check record["program"] anchors
    program_section = record.record.get("program") or {}
    if isinstance(program_section, dict):
        oq3_anchors = program_section.get("openqasm3", [])
        if isinstance(oq3_anchors, list):
            for anchor in oq3_anchors:
                if not isinstance(anchor, dict):
                    continue

                # Build artifact refs from anchor info
                if "raw" in anchor and isinstance(anchor["raw"], dict):
                    try:
                        ref = ArtifactRef(
                            kind=anchor["raw"].get("kind", "source.openqasm3"),
                            digest=anchor["raw"]["digest"],
                            media_type="application/openqasm",
                            role="program",
                        )
                        logical.append(
                            ProgramArtifact(
                                ref=ref,
                                role=ProgramRole.LOGICAL,
                                format="openqasm3",
                                name=anchor.get("name"),
                                index=anchor.get("index"),
                            )
                        )
                    except (KeyError, ValueError):
                        pass

    if not logical and not physical:
        return None

    return ProgramSnapshot(
        logical=logical,
        physical=physical,
        structural_hash=structural_hash,
        num_circuits=len(logical) if logical else None,
    )


def _build_result(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
) -> ResultSnapshot:
    """
    Build ResultSnapshot from RunRecord and artifacts.

    Parameters
    ----------
    record : RunRecord
        Run record to extract results from.
    store : ObjectStoreProtocol
        Object store for loading counts artifacts.

    Returns
    -------
    ResultSnapshot
        Constructed result snapshot (always returns a valid snapshot).

    Notes
    -----
    If the counts payload includes ``counts_format`` metadata, it is
    preserved in the result. Otherwise, canonical format (cbit0_right)
    is assumed and marked appropriately in metadata.
    """
    # Lazy import to avoid circular dependencies
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.storage.artifacts.lookup import find_artifact

    status = record.record.get("info", {}).get("status", "RUNNING")

    # Determine success and normalized status
    if status == "FINISHED":
        success = True
        normalized_status = "completed"
    elif status == "FAILED":
        success = False
        normalized_status = "failed"
    elif status == "KILLED":
        success = False
        normalized_status = "cancelled"
    else:
        success = False
        normalized_status = "failed"

    items: list[ResultItem] = []
    error: ResultError | None = None

    # Build error from record errors
    errors_list = record.record.get("errors") or []
    if errors_list and isinstance(errors_list, list) and errors_list:
        first_error = errors_list[0]
        if isinstance(first_error, dict):
            error = ResultError(
                type=str(first_error.get("type", "UnknownError")),
                message=str(first_error.get("message", ""))[:500],
            )

    # Try to load counts from artifact
    counts_artifact = find_artifact(
        record,
        role="results",
        kind_contains="counts",
    )

    # Track whether format was assumed (for metadata)
    format_was_assumed = False

    if counts_artifact:
        try:
            payload = load_artifact_json(counts_artifact, store)
            if isinstance(payload, dict):
                # Check if payload includes counts_format metadata
                payload_format = payload.get("counts_format")

                # Handle batch format
                experiments = payload.get("experiments")
                if isinstance(experiments, list) and experiments:
                    for idx, exp in enumerate(experiments):
                        if isinstance(exp, dict) and exp.get("counts"):
                            counts_data = exp["counts"]
                            shots = sum(counts_data.values()) if counts_data else 0

                            # Use format from payload if available
                            if payload_format and isinstance(payload_format, dict):
                                format_dict = {
                                    "source_sdk": payload_format.get(
                                        "source_sdk",
                                        record.record.get("adapter", "manual"),
                                    ),
                                    "source_key_format": payload_format.get(
                                        "source_key_format", "run"
                                    ),
                                    "bit_order": payload_format.get(
                                        "bit_order", "cbit0_right"
                                    ),
                                    "transformed": payload_format.get(
                                        "transformed", False
                                    ),
                                }
                            else:
                                # Assume canonical format for manual runs
                                format_was_assumed = True
                                format_dict = CountsFormat(
                                    source_sdk=record.record.get("adapter", "manual"),
                                    source_key_format="run",
                                    bit_order="cbit0_right",
                                    transformed=False,
                                ).to_dict()

                            items.append(
                                ResultItem(
                                    item_index=idx,
                                    success=True,
                                    counts={
                                        "counts": counts_data,
                                        "shots": shots,
                                        "format": format_dict,
                                    },
                                )
                            )
                else:
                    # Simple format
                    counts_data = payload.get("counts", {})
                    if counts_data:
                        shots = sum(counts_data.values())

                        # Use format from payload if available
                        if payload_format and isinstance(payload_format, dict):
                            format_dict = {
                                "source_sdk": payload_format.get(
                                    "source_sdk",
                                    record.record.get("adapter", "manual"),
                                ),
                                "source_key_format": payload_format.get(
                                    "source_key_format", "run"
                                ),
                                "bit_order": payload_format.get(
                                    "bit_order", "cbit0_right"
                                ),
                                "transformed": payload_format.get("transformed", False),
                            }
                        else:
                            format_was_assumed = True
                            format_dict = CountsFormat(
                                source_sdk=record.record.get("adapter", "manual"),
                                source_key_format="run",
                                bit_order="cbit0_right",
                                transformed=False,
                            ).to_dict()

                        items.append(
                            ResultItem(
                                item_index=0,
                                success=True,
                                counts={
                                    "counts": counts_data,
                                    "shots": shots,
                                    "format": format_dict,
                                },
                            )
                        )
        except Exception as e:
            logger.debug("Failed to load counts from artifact: %s", e)

    # Mark as completed if we have results even without explicit status
    if items and not success:
        success = True
        normalized_status = "completed"

    result_metadata: dict[str, Any] = {"synthesized_from_run": True}
    if format_was_assumed:
        result_metadata["counts_format_assumed"] = True

    return ResultSnapshot(
        success=success,
        status=normalized_status,
        items=items,
        error=error,
        metadata=result_metadata,
    )


def synthesize_envelope(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
) -> ExecutionEnvelope:
    """
    Synthesize ExecutionEnvelope from RunRecord and artifacts.

    This function synthesizes an envelope from data when no envelope
    artifact exists. Intended for **manual runs only** - adapter runs
    should always have envelope created by the adapter.

    Parameters
    ----------
    record : RunRecord
        Run record to build envelope from.
    store : ObjectStoreProtocol
        Object store for loading artifacts.

    Returns
    -------
    ExecutionEnvelope
        Synthesized envelope (best effort, valid structure).

    Notes
    -----
    Synthesized envelopes have limitations:

    - ``metadata.synthesized_from_run=True`` marks as synthesized
    - ``metadata.manual_run=True`` marks as manual (if no adapter)
    - ``program.structural_hash`` is None (engine cannot compute)
    - ``program.parametric_hash`` is None (engine cannot compute)

    Compare operations will report "hash unavailable" for these runs.

    Examples
    --------
    >>> envelope = synthesize_envelope(record, store)
    >>> envelope.metadata.get("synthesized_from_run")
    True
    >>> envelope.program.structural_hash  # None for manual runs
    """
    producer = _build_producer(record)
    device = _build_device(record)
    execution = _build_execution(record)
    program = _build_program(record)
    result = _build_result(record, store)

    is_manual = is_manual_run_record(record.record)

    metadata: dict[str, Any] = {
        "synthesized_from_run": True,
        "source_run_id": record.run_id,
    }

    if is_manual:
        metadata["manual_run"] = True

    # Add warning if counts format was assumed
    if result.items:
        for item in result.items:
            if item.counts:
                fmt = item.counts.get("format", {})
                if fmt.get("source_key_format") == "run":
                    metadata["counts_format_assumed"] = True
                    break

    envelope = ExecutionEnvelope.create(
        producer=producer,
        result=result,
        device=device,
        program=program,
        execution=execution,
        metadata=metadata,
    )

    logger.debug(
        "Synthesized envelope from run: run=%s, envelope_id=%s, manual=%s",
        record.run_id,
        envelope.envelope_id,
        is_manual,
    )

    return envelope
