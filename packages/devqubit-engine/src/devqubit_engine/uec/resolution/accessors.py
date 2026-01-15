# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC accessor functions.

This module provides read-only functions for extracting data from
ExecutionEnvelope. These are the canonical accessors that should be
used by compare/diff/verify operations instead of directly accessing
envelope internals or implementing custom extraction logic.

The functions implement UEC-first strategy with appropriate fallbacks
for synthesized (manual) envelopes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from devqubit_engine.uec.models.result import canonicalize_bitstrings


if TYPE_CHECKING:
    from devqubit_engine.circuit.summary import CircuitSummary
    from devqubit_engine.core.record import RunRecord
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.uec.models.device import DeviceSnapshot
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


def get_counts_from_envelope(
    envelope: "ExecutionEnvelope",
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


def get_shots_from_envelope(envelope: "ExecutionEnvelope") -> int | None:
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


def get_program_hash_from_envelope(envelope: "ExecutionEnvelope") -> str | None:
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


def resolve_counts(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
    envelope: "ExecutionEnvelope | None" = None,
    *,
    canonicalize: bool = True,
) -> dict[str, int] | None:
    """
    Extract counts from envelope or Run artifacts with UEC-first strategy.

    This is the canonical function for getting counts in compare operations.
    It extracts counts from ExecutionEnvelope, falling back to Run counts
    artifact only for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings to cbit0_right format.

    Returns
    -------
    dict or None
        Counts as {bitstring: count} in canonical format, or None if not available.

    Notes
    -----
    Fallback to Run counts artifact is only allowed for synthesized (manual)
    envelopes. For adapter envelopes, missing counts returns None.
    """
    from devqubit_engine.artifacts import get_counts
    from devqubit_engine.uec.resolution.resolve import (
        resolve_envelope as _resolve_envelope,
    )

    # Use provided envelope or resolve
    if envelope is None:
        envelope = _resolve_envelope(record, store)

    # Try to get counts from envelope (UEC-first)
    if envelope.result.items:
        item = envelope.result.items[0]
        if item.counts:
            raw_counts = item.counts.get("counts")
            if isinstance(raw_counts, dict):
                if canonicalize:
                    format_info = item.counts.get("format", {})
                    bit_order = format_info.get("bit_order", "cbit0_right")
                    transformed = format_info.get("transformed", False)
                    canonical = canonicalize_bitstrings(
                        raw_counts,
                        bit_order=bit_order,
                        transformed=transformed,
                    )
                    return {k: int(v) for k, v in canonical.items()}
                else:
                    return {str(k): int(v) for k, v in raw_counts.items()}

    # Fallback: Run counts artifact - ONLY for synthesized (manual) envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        # Adapter envelope without counts - this is an integration issue
        logger.debug(
            "Adapter envelope for run %s has no counts in result items",
            record.run_id,
        )
        return None

    # Manual/synthesized envelope - fallback to Run artifact is allowed
    counts_info = get_counts(record, store)
    if counts_info is None:
        return None

    # Canonicalize fallback counts (assume cbit0_right if not specified)
    if canonicalize:
        return canonicalize_bitstrings(
            counts_info.counts,
            bit_order="cbit0_right",
            transformed=False,
        )
    return counts_info.counts


def resolve_device_snapshot(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
    envelope: "ExecutionEnvelope | None" = None,
) -> "DeviceSnapshot | None":
    """
    Load device snapshot from envelope or run record with UEC-first strategy.

    This is the canonical function for getting device snapshot in compare
    operations. It extracts device from ExecutionEnvelope, falling back to
    Run record metadata only for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record to extract device snapshot from.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.

    Returns
    -------
    DeviceSnapshot or None
        Device snapshot if available, None otherwise.

    Notes
    -----
    Fallback to Run record metadata is only allowed for synthesized
    (manual) envelopes. For adapter envelopes, if device is not present
    in the envelope, None is returned.
    """
    from devqubit_engine.uec.models.calibration import DeviceCalibration
    from devqubit_engine.uec.models.device import DeviceSnapshot
    from devqubit_engine.uec.resolution.resolve import (
        resolve_envelope as _resolve_envelope,
    )

    # Use provided envelope or resolve
    if envelope is None:
        envelope = _resolve_envelope(record, store)

    # Get device from envelope (UEC-first)
    if envelope.device is not None:
        return envelope.device

    # Fallback: construct from record metadata - ONLY for synthesized envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        # Adapter envelope without device - this may be intentional (no device info)
        logger.debug(
            "Adapter envelope for run %s has no device snapshot",
            record.run_id,
        )
        return None

    # Manual/synthesized envelope - fallback to Run record is allowed
    backend = record.record.get("backend") or {}
    if not isinstance(backend, dict):
        return None

    snapshot_summary = record.record.get("device_snapshot") or {}
    if not isinstance(snapshot_summary, dict):
        snapshot_summary = {}

    calibration = None
    cal_data = snapshot_summary.get("calibration")
    if isinstance(cal_data, dict):
        try:
            calibration = DeviceCalibration.from_dict(cal_data)
        except Exception as e:
            logger.debug("Failed to parse calibration data: %s", e)

    try:
        return DeviceSnapshot(
            captured_at=snapshot_summary.get("captured_at", record.created_at),
            backend_name=backend.get("name", ""),
            backend_type=backend.get("type", ""),
            provider=backend.get("provider", ""),
            num_qubits=snapshot_summary.get("num_qubits"),
            connectivity=snapshot_summary.get("connectivity"),
            native_gates=snapshot_summary.get("native_gates"),
            calibration=calibration,
        )
    except Exception:
        return None


def resolve_circuit_summary(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
    envelope: "ExecutionEnvelope | None" = None,
    *,
    which: str = "logical",
) -> "CircuitSummary | None":
    """
    Extract circuit summary from envelope or run record with UEC-first strategy.

    This is the canonical function for getting circuit summary in compare
    operations. It extracts circuit from envelope refs, falling back to
    run record artifacts only for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record for fallback extraction.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope.
    which : str, default="logical"
        Which circuit to extract: "logical" or "physical".

    Returns
    -------
    CircuitSummary or None
        Extracted circuit summary, or None if not found.
    """
    from devqubit_engine.circuit.extractors import (
        extract_circuit,
        extract_circuit_from_envelope,
    )
    from devqubit_engine.circuit.summary import summarize_circuit_data
    from devqubit_engine.uec.resolution.resolve import (
        resolve_envelope as _resolve_envelope,
    )

    # Resolve envelope if not provided
    if envelope is None:
        envelope = _resolve_envelope(record, store)

    circuit_data = None

    # Try envelope first (UEC-first)
    circuit_data = extract_circuit_from_envelope(envelope, store, which=which)

    # Fallback to Run artifacts - ONLY for synthesized (manual) envelopes
    if circuit_data is None:
        is_synthesized = envelope.metadata.get("synthesized_from_run", False)

        if is_synthesized:
            # Manual envelope - fallback is allowed
            circuit_data = extract_circuit(record, store)
        else:
            # Adapter envelope without circuit refs - log and return None
            logger.debug(
                "Adapter envelope for run %s has no circuit refs for '%s'",
                record.run_id,
                which,
            )

    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)

    return None
