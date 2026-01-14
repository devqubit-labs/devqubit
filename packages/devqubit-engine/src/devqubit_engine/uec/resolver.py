# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC Resolver - Single entry point for envelope resolution.

This module provides the canonical interface for obtaining ExecutionEnvelope
from any run record. It implements the "UEC-first" strategy with strict
contract enforcement:

1. **Adapter runs**: Envelope MUST exist (created by adapter). Missing envelope
   is an integration error and raises MissingEnvelopeError.
2. **Manual runs**: Envelope is synthesized from RunRecord if not present.
   This is best-effort with limited semantics (no program hashes).

This ensures that adapters are responsible for producing complete UEC data,
while manual runs still work (with documented limitations).
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.artifacts import find_artifact, load_json_artifact
from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.protocols import ObjectStoreProtocol
from devqubit_engine.uec.calibration import DeviceCalibration
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import ProgramArtifact, ProgramSnapshot
from devqubit_engine.uec.result import (
    CountsFormat,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.uec.types import ArtifactRef, ProgramRole


logger = logging.getLogger(__name__)


class MissingEnvelopeError(Exception):
    """
    Raised when adapter run is missing required envelope.

    Adapter runs MUST have an envelope artifact. If missing, this indicates
    an adapter integration error that should be fixed in the adapter, not
    papered over by the engine.
    """

    def __init__(self, run_id: str, adapter: str):
        self.run_id = run_id
        self.adapter = adapter
        super().__init__(
            f"Adapter run '{run_id}' (adapter={adapter}) is missing envelope. "
            f"This is an adapter integration error - adapters must create envelope."
        )


VOLATILE_EXECUTE_KEYS = frozenset(
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


def load_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    include_invalid: bool = False,
) -> ExecutionEnvelope | None:
    """
    Load ExecutionEnvelope from stored artifact.

    This function attempts to load an existing envelope artifact from
    the run record. It prefers valid envelopes but can optionally
    return invalid ones for debugging purposes.

    Parameters
    ----------
    record : RunRecord
        Run record to load envelope from.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, also return invalid envelopes (kind contains "invalid").
        If False, only return valid envelopes.

    Returns
    -------
    ExecutionEnvelope or None
        Loaded envelope if found, None otherwise.

    Notes
    -----
    Selection priority:
    1. role="envelope", kind="devqubit.envelope.json" (valid)
    2. role="envelope", kind="devqubit.envelope.invalid.json" (if include_invalid)
    """
    # Search for envelope artifact with exact kind match
    valid_artifact = None
    invalid_artifact = None

    for artifact in record.artifacts:
        if artifact.role != "envelope":
            continue

        # Exact match for valid envelope
        if artifact.kind == "devqubit.envelope.json":
            valid_artifact = artifact
            break  # Found valid, stop searching

        # Track invalid envelope as fallback
        if artifact.kind == "devqubit.envelope.invalid.json":
            invalid_artifact = artifact

    # Use valid envelope if found
    if valid_artifact is not None:
        target_artifact = valid_artifact
    elif include_invalid and invalid_artifact is not None:
        target_artifact = invalid_artifact
    else:
        logger.debug("No envelope artifact found for run %s", record.run_id)
        return None

    # Load and parse envelope
    try:
        envelope_data = load_json_artifact(target_artifact, store)
        if not isinstance(envelope_data, dict):
            logger.warning(
                "Envelope artifact is not a dict for run %s",
                record.run_id,
            )
            return None

        envelope = ExecutionEnvelope.from_dict(envelope_data)
        logger.debug(
            "Loaded envelope from artifact: run=%s, envelope_id=%s",
            record.run_id,
            envelope.envelope_id,
        )
        return envelope

    except Exception as e:
        logger.warning(
            "Failed to parse envelope for run %s: %s",
            record.run_id,
            e,
        )
        return None


def _build_producer_from_run(record: RunRecord) -> ProducerInfo:
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


def _build_device_from_run(record: RunRecord) -> DeviceSnapshot | None:
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


def _build_execution_from_run(record: RunRecord) -> ExecutionSnapshot | None:
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


def _build_program_from_run(record: RunRecord) -> ProgramSnapshot | None:
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

    # Check for circuit_hash in execute metadata (legacy field)
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


def _build_result_from_run(
    record: RunRecord,
    store: ObjectStoreProtocol,
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
    """
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

    if counts_artifact:
        try:
            payload = load_json_artifact(counts_artifact, store)
            if isinstance(payload, dict):
                # Handle batch format
                experiments = payload.get("experiments")
                if isinstance(experiments, list) and experiments:
                    for idx, exp in enumerate(experiments):
                        if isinstance(exp, dict) and exp.get("counts"):
                            counts_data = exp["counts"]
                            shots = sum(counts_data.values()) if counts_data else 0
                            # NOTE: For manual runs, we assume cbit0_right format.
                            # See detailed comment below in simple format case.
                            items.append(
                                ResultItem(
                                    item_index=idx,
                                    success=True,
                                    counts={
                                        "counts": counts_data,
                                        "shots": shots,
                                        "format": CountsFormat(
                                            source_sdk=record.record.get(
                                                "adapter", "manual"
                                            ),
                                            source_key_format="run",
                                            bit_order="cbit0_right",
                                            transformed=False,
                                        ).to_dict(),
                                    },
                                )
                            )
                else:
                    # Simple format
                    counts_data = payload.get("counts", {})
                    if counts_data:
                        shots = sum(counts_data.values())
                        # NOTE: For manual runs without adapter, we assume
                        # cbit0_right (canonical) format. This is a best-effort
                        # assumption since manual runs don't have SDK metadata.
                        # Compare operations will use this assumption for
                        # canonicalization - if the actual format differs,
                        # TVD results may be incorrect.
                        items.append(
                            ResultItem(
                                item_index=0,
                                success=True,
                                counts={
                                    "counts": counts_data,
                                    "shots": shots,
                                    "format": CountsFormat(
                                        source_sdk=record.record.get(
                                            "adapter", "manual"
                                        ),
                                        source_key_format="run",
                                        bit_order="cbit0_right",
                                        transformed=False,
                                    ).to_dict(),
                                },
                            )
                        )
        except Exception as e:
            logger.debug("Failed to load counts from artifact: %s", e)

    # Mark as completed if we have results even without explicit status
    if items and not success:
        success = True
        normalized_status = "completed"

    return ResultSnapshot(
        success=success,
        status=normalized_status,
        items=items,
        error=error,
        metadata={"synthesized_from_run": True},
    )


def build_envelope_from_run(
    record: RunRecord,
    store: ObjectStoreProtocol,
) -> ExecutionEnvelope:
    """
    Build ExecutionEnvelope from RunRecord and artifacts.

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

    - ``metadata.synthesized_from_run=True`` - marks as synthesized
    - ``metadata.manual_run=True`` - marks as manual (if no adapter)
    - ``program.structural_hash`` - None (engine cannot compute)
    - ``program.parametric_hash`` - None (engine cannot compute)

    Compare operations will report "hash unavailable" for these runs.

    Examples
    --------
    >>> # Direct usage for manual runs
    >>> envelope = build_envelope_from_run(record, store)
    >>> envelope.metadata.get("synthesized_from_run")
    True
    >>> envelope.program.structural_hash  # None for manual runs
    """
    producer = _build_producer_from_run(record)
    device = _build_device_from_run(record)
    execution = _build_execution_from_run(record)
    program = _build_program_from_run(record)
    result = _build_result_from_run(record, store)

    is_manual = _is_manual_run(record)

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
        "Built envelope from run: run=%s, envelope_id=%s, manual=%s",
        record.run_id,
        envelope.envelope_id,
        is_manual,
    )

    return envelope


def _is_manual_run(record: RunRecord) -> bool:
    """
    Check if run is a manual run (no adapter).

    Parameters
    ----------
    record : RunRecord
        Run record to check.

    Returns
    -------
    bool
        True if manual run, False if adapter run.
    """
    adapter = record.record.get("adapter")
    if not adapter or adapter == "" or adapter == "manual":
        return True
    return False


def resolve_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    include_invalid: bool = False,
    strict: bool = True,
) -> ExecutionEnvelope:
    """
    Resolve ExecutionEnvelope for a run (UEC-first with strict contract).

    This is the **primary entry point** for obtaining envelope data.
    All compare/diff/verify operations should use this function rather
    than directly accessing RunRecord fields or artifacts.

    Strategy:
    1. Try to load existing envelope artifact (UEC-first)
    2. If not found:
       - **Adapter run**: Raise MissingEnvelopeError (strict mode) or
         synthesize with warning (non-strict mode)
       - **Manual run**: Synthesize from RunRecord and artifacts

    Parameters
    ----------
    record : RunRecord
        Run record to resolve envelope for.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, include invalid envelopes in search.
    strict : bool, default=True
        If True, raise error for adapter runs without envelope.
        If False, synthesize envelope with warning (for backward compat).

    Returns
    -------
    ExecutionEnvelope
        Resolved envelope.

    Raises
    ------
    MissingEnvelopeError
        If strict=True and adapter run is missing envelope.

    Notes
    -----
    For manual runs, synthesized envelope will have:
    - ``metadata.synthesized_from_run=True``
    - ``metadata.manual_run=True``
    - No program hashes (compare will report "hash unavailable")

    Examples
    --------
    Basic usage in compare/diff:

    >>> from devqubit_engine.uec.resolver import resolve_envelope
    >>>
    >>> env_a = resolve_envelope(run_a, store_a)
    >>> env_b = resolve_envelope(run_b, store_b)
    >>>
    >>> # Now compare using UEC structure
    >>> if env_a.device and env_b.device:
    ...     drift = compute_drift(env_a.device, env_b.device)

    Checking envelope source:

    >>> envelope = resolve_envelope(record, store)
    >>> if envelope.metadata.get("synthesized_from_run"):
    ...     print("Envelope was synthesized (manual run)")
    >>> if envelope.metadata.get("manual_run"):
    ...     print("This is a manual run - program hashes unavailable")
    """
    # Try to load existing envelope
    envelope = load_envelope(
        record,
        store,
        include_invalid=include_invalid,
    )

    if envelope is not None:
        return envelope

    # No envelope found - check if this is allowed
    is_manual = _is_manual_run(record)
    adapter = record.record.get("adapter", "manual")

    if not is_manual and strict:
        # Adapter run without envelope is an error
        raise MissingEnvelopeError(record.run_id, str(adapter))

    if not is_manual:
        # Non-strict mode: log warning and proceed
        logger.warning(
            "Adapter run '%s' (adapter=%s) missing envelope. "
            "Synthesizing from RunRecord. This should be fixed in adapter.",
            record.run_id,
            adapter,
        )

    # Build envelope for manual run (or non-strict adapter run)
    return build_envelope_from_run(record, store)


def get_counts_from_envelope(
    envelope: ExecutionEnvelope,
    *,
    item_index: int = 0,
) -> dict[str, int] | None:
    """
    Extract measurement counts from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to extract counts from.
    item_index : int, default=0
        Index of result item (for batch executions).

    Returns
    -------
    dict or None
        Counts as {bitstring: count}, or None if not available.

    Examples
    --------
    >>> envelope = resolve_envelope(record, store)
    >>> counts = get_counts_from_envelope(envelope)
    >>> if counts:
    ...     print(f"Got {sum(counts.values())} total shots")
    """
    if not envelope.result.items:
        return None

    if item_index >= len(envelope.result.items):
        return None

    item = envelope.result.items[item_index]
    if not item.counts:
        return None

    counts_data = item.counts.get("counts")
    if isinstance(counts_data, dict):
        return {str(k): int(v) for k, v in counts_data.items()}

    return None


def get_shots_from_envelope(envelope: ExecutionEnvelope) -> int | None:
    """
    Extract shot count from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to extract shots from.

    Returns
    -------
    int or None
        Number of shots, or None if not available.
    """
    # Try execution snapshot first
    if envelope.execution and envelope.execution.shots:
        return envelope.execution.shots

    # Fall back to counts
    counts = get_counts_from_envelope(envelope)
    if counts:
        return sum(counts.values())

    return None


def get_program_hash_from_envelope(envelope: ExecutionEnvelope) -> str | None:
    """
    Extract structural hash from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to extract hash from.

    Returns
    -------
    str or None
        Structural hash, or None if not available.
    """
    if envelope.program:
        return (
            envelope.program.structural_hash
            or envelope.program.executed_structural_hash
        )
    return None
