# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC envelope synthesis and enrichment utilities.

This module provides:

1. **Envelope synthesis** - Creating ExecutionEnvelope from RunRecord when
   no envelope artifact exists (manual runs only).

2. **Envelope enrichment** - Adding tracker namespace metadata (params,
   metrics, project, fingerprints) to envelopes.

Synthesized envelopes have limitations:

- ``metadata.synthesized_from_run=True`` marks as synthesized
- ``metadata.manual_run=True`` marks as manual (if no adapter)
- ``program.structural_hash`` computed from artifact digests (not circuit structure)
- ``program.parametric_hash`` is None (engine cannot compute)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.storage.types import ArtifactRef
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
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.tracking.record import RunRecord


logger = logging.getLogger(__name__)


# Keys in execute section that are volatile (change between runs)
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


# ---------------------------------------------------------------------------
# Envelope enrichment helpers (used by both run.py and synthesize_envelope)
# ---------------------------------------------------------------------------


def build_tracker_namespace(
    record: dict[str, Any],
    fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build tracker metadata namespace from run record.

    Extracts params, metrics, project from the run record and optionally
    includes fingerprints. This namespace enables the compare module to
    extract data uniformly from envelopes.

    Parameters
    ----------
    record : dict
        Raw run record dictionary.
    fingerprints : dict, optional
        Pre-computed fingerprints to include.

    Returns
    -------
    dict
        Tracker namespace with available fields. Empty dict if no data.
    """
    tracker_ns: dict[str, Any] = {}

    # Extract project (handle both string and dict formats)
    project = record.get("project")
    if project:
        if isinstance(project, dict):
            project_name = project.get("name")
            if project_name:
                tracker_ns["project"] = project_name
        elif isinstance(project, str):
            tracker_ns["project"] = project

    # Extract params and metrics from data section
    data = record.get("data") or {}
    if isinstance(data, dict):
        params = data.get("params")
        if params and isinstance(params, dict):
            tracker_ns["params"] = dict(params)

        metrics = data.get("metrics")
        if metrics and isinstance(metrics, dict):
            tracker_ns["metrics"] = dict(metrics)

    # Include fingerprints if provided
    if fingerprints:
        tracker_ns["fingerprints"] = dict(fingerprints)

    return tracker_ns


def enrich_envelope_with_tracker(
    envelope: ExecutionEnvelope,
    record: dict[str, Any],
    fingerprints: dict[str, str] | None = None,
) -> None:
    """
    Add tracker namespace to envelope metadata (in-place).

    This enriches the envelope with project, params, metrics, and
    fingerprints so that compare module can extract this data uniformly.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to enrich (modified in-place).
    record : dict
        Run record dictionary containing project, params, metrics.
    fingerprints : dict, optional
        Pre-computed fingerprints to include.
    """
    tracker_ns = build_tracker_namespace(record, fingerprints)
    if tracker_ns:
        if "tracker" not in envelope.metadata:
            envelope.metadata["tracker"] = {}
        envelope.metadata["tracker"].update(tracker_ns)


def add_fingerprints_to_envelope(
    envelope: ExecutionEnvelope,
    fingerprints: dict[str, str],
) -> None:
    """
    Add fingerprints to envelope.metadata.tracker (in-place).

    If tracker namespace doesn't exist, creates it with only fingerprints.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to update (modified in-place).
    fingerprints : dict
        Fingerprints to add (program, device, intent, run).
    """
    if not fingerprints:
        return

    if "tracker" not in envelope.metadata:
        envelope.metadata["tracker"] = {}

    envelope.metadata["tracker"]["fingerprints"] = dict(fingerprints)


# ---------------------------------------------------------------------------
# Internal builders for envelope synthesis
# ---------------------------------------------------------------------------


def _build_producer(record: RunRecord) -> ProducerInfo:
    """Build ProducerInfo from RunRecord."""
    adapter = record.record.get("adapter", "manual")
    if not adapter:
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


def build_device_from_record(record: RunRecord) -> DeviceSnapshot | None:
    """
    Build DeviceSnapshot from RunRecord.

    This is used for synthesizing envelopes and as a fallback in
    resolve_device_snapshot for manual runs.

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
        return None

    backend_name = backend.get("name", "")
    if not backend_name:
        return None

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


def _build_execution(record: RunRecord) -> ExecutionSnapshot | None:
    """Build ExecutionSnapshot from RunRecord."""
    execute = record.record.get("execute") or {}
    if not isinstance(execute, dict):
        execute = {}

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


def _build_program(record: RunRecord) -> ProgramSnapshot | None:
    """
    Build ProgramSnapshot from RunRecord and artifacts.

    For synthesized envelopes, structural_hash is computed from sorted
    artifact digests when not explicitly available.
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
        fmt = _detect_program_format(artifact.kind)
        meta = artifact.meta or {}

        prog_artifact = ProgramArtifact(
            ref=artifact,
            role=ProgramRole.LOGICAL,
            format=fmt,
            name=meta.get("program_name") or meta.get("name"),
            index=meta.get("program_index"),
        )
        logical.append(prog_artifact)

        # Check for circuit_hash in artifact metadata
        if meta.get("circuit_hash") and structural_hash is None:
            structural_hash = str(meta["circuit_hash"])

    # If no program artifacts, check record["program"] anchors
    program_section = record.record.get("program") or {}
    if isinstance(program_section, dict):
        _extract_program_anchors(program_section, logical)

    if not logical and not physical:
        return None

    # Compute structural_hash from artifact digests if not already set
    if structural_hash is None and logical:
        digests = sorted(
            art.ref.digest for art in logical if art.ref and art.ref.digest
        )
        if digests:
            combined = "|".join(digests)
            structural_hash = f"sha256:{hashlib.sha256(combined.encode()).hexdigest()}"

    return ProgramSnapshot(
        logical=logical,
        physical=physical,
        structural_hash=structural_hash,
        num_circuits=len(logical) if logical else None,
    )


def _detect_program_format(kind: str) -> str:
    """Detect program format from artifact kind."""
    kind_lower = kind.lower()
    if "openqasm3" in kind_lower:
        return "openqasm3"
    elif "qpy" in kind_lower:
        return "qpy"
    elif "json" in kind_lower:
        return "json"
    return "unknown"


def _extract_program_anchors(
    program_section: dict[str, Any],
    logical: list[ProgramArtifact],
) -> None:
    """Extract program artifacts from record anchors."""
    oq3_anchors = program_section.get("openqasm3", [])
    if not isinstance(oq3_anchors, list):
        return

    for anchor in oq3_anchors:
        if not isinstance(anchor, dict):
            continue

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


def _build_result(record: RunRecord, store: ObjectStoreProtocol) -> ResultSnapshot:
    """Build ResultSnapshot from RunRecord and artifacts."""
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.storage.artifacts.lookup import find_artifact

    status = record.record.get("info", {}).get("status", "RUNNING")
    success, normalized_status = _normalize_status(status)

    items: list[ResultItem] = []
    error = _build_result_error(record)

    # Try to load counts from artifact
    counts_artifact = find_artifact(record, role="results", kind_contains="counts")
    format_was_assumed = False

    if counts_artifact:
        try:
            payload = load_artifact_json(counts_artifact, store)
            if isinstance(payload, dict):
                items, format_was_assumed = _parse_counts_payload(payload, record)
        except Exception as e:
            logger.debug("Failed to load counts from artifact: %s", e)

    # Mark as completed if we have results
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


def _normalize_status(status: str) -> tuple[bool, str]:
    """Convert run status to (success, normalized_status)."""
    status_map = {
        "FINISHED": (True, "completed"),
        "FAILED": (False, "failed"),
        "KILLED": (False, "cancelled"),
        "RUNNING": (False, "running"),
        "QUEUED": (False, "running"),
    }
    return status_map.get(status, (False, "failed"))


def _build_result_error(record: RunRecord) -> ResultError | None:
    """Build ResultError from record errors."""
    errors_list = record.record.get("errors") or []
    if not errors_list or not isinstance(errors_list, list):
        return None

    first_error = errors_list[0]
    if isinstance(first_error, dict):
        return ResultError(
            type=str(first_error.get("type", "UnknownError")),
            message=str(first_error.get("message", ""))[:500],
        )
    return None


def _parse_counts_payload(
    payload: dict[str, Any],
    record: RunRecord,
) -> tuple[list[ResultItem], bool]:
    """Parse counts payload into ResultItems."""
    items: list[ResultItem] = []
    format_was_assumed = False
    payload_format = payload.get("counts_format")
    adapter = record.record.get("adapter", "manual")

    # Handle batch format
    experiments = payload.get("experiments")
    if isinstance(experiments, list) and experiments:
        for idx, exp in enumerate(experiments):
            if isinstance(exp, dict) and exp.get("counts"):
                counts_data = exp["counts"]
                shots = sum(counts_data.values()) if counts_data else 0
                format_dict, assumed = _build_counts_format(payload_format, adapter)
                if assumed:
                    format_was_assumed = True

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
            format_dict, assumed = _build_counts_format(payload_format, adapter)
            if assumed:
                format_was_assumed = True

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

    return items, format_was_assumed


def _build_counts_format(
    payload_format: dict[str, Any] | None,
    adapter: str,
) -> tuple[dict[str, Any], bool]:
    """
    Build counts format dict.

    Returns (format_dict, was_assumed).
    """
    if payload_format and isinstance(payload_format, dict):
        return {
            "source_sdk": payload_format.get("source_sdk", adapter),
            "source_key_format": payload_format.get("source_key_format", "run"),
            "bit_order": payload_format.get("bit_order", "cbit0_right"),
            "transformed": payload_format.get("transformed", False),
        }, False

    # Assume canonical format
    return (
        CountsFormat(
            source_sdk=adapter,
            source_key_format="run",
            bit_order="cbit0_right",
            transformed=False,
        ).to_dict(),
        True,
    )


# ---------------------------------------------------------------------------
# Main synthesis function
# ---------------------------------------------------------------------------


def synthesize_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
    fingerprints: dict[str, str] | None = None,
) -> ExecutionEnvelope:
    """
    Synthesize ExecutionEnvelope from RunRecord and artifacts.

    Creates an envelope from run data when no envelope artifact exists.
    Intended for **manual runs only** - adapter runs should always have
    envelope created by the adapter.

    Parameters
    ----------
    record : RunRecord
        Run record to build envelope from.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    fingerprints : dict, optional
        Pre-computed fingerprints to include in tracker namespace.

    Returns
    -------
    ExecutionEnvelope
        Synthesized envelope with valid structure.

    Notes
    -----
    Synthesized envelopes have limitations:

    - ``metadata.synthesized_from_run=True`` marks as synthesized
    - ``metadata.manual_run=True`` marks as manual (if no adapter)
    - ``program.structural_hash`` computed from artifact digests
    - ``program.parametric_hash`` is None
    """
    producer = _build_producer(record)
    device = build_device_from_record(record)
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
    if result.metadata and result.metadata.get("counts_format_assumed"):
        metadata["counts_format_assumed"] = True

    # Add tracker namespace with params, metrics, project, fingerprints
    tracker_ns = build_tracker_namespace(record.record, fingerprints)
    if tracker_ns:
        metadata["tracker"] = tracker_ns

    envelope = ExecutionEnvelope.create(
        producer=producer,
        result=result,
        device=device,
        program=program,
        execution=execution,
        metadata=metadata,
    )

    logger.debug(
        "Synthesized envelope: run=%s, envelope_id=%s, manual=%s",
        record.run_id,
        envelope.envelope_id,
        is_manual,
    )

    return envelope
