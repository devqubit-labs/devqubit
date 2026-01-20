# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC accessor functions.

This module provides read-only functions for extracting data from
ExecutionEnvelope. These are the canonical accessors that should be
used by compare/diff/verify operations.

The functions implement UEC-first strategy with appropriate fallbacks
for synthesized (manual) envelopes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from devqubit_engine.uec.models.result import canonicalize_bitstrings


if TYPE_CHECKING:
    from devqubit_engine.circuit.summary import CircuitSummary
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.tracking.record import RunRecord
    from devqubit_engine.uec.models.device import DeviceSnapshot
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


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
    if envelope.execution and envelope.execution.shots is not None:
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


def resolve_counts(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
    *,
    item_index: int = 0,
    canonicalize: bool = True,
) -> dict[str, int] | None:
    """
    Extract counts with UEC-first strategy.

    This is the canonical function for getting counts in compare operations.
    Falls back to Run counts artifact only for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.
    item_index : int, default=0
        Index of result item for batch executions.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings to cbit0_right format.

    Returns
    -------
    dict or None
        Counts as {bitstring: count}, or None if not available.
    """
    from devqubit_engine.storage.artifacts.counts import get_counts
    from devqubit_engine.uec.api.resolve import resolve_envelope as _resolve_envelope

    if envelope is None:
        envelope = _resolve_envelope(record, store)

    # Try to get counts from envelope (UEC-first)
    if envelope.result.items and item_index < len(envelope.result.items):
        item = envelope.result.items[item_index]
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

    # Fallback: Run counts artifact - ONLY for synthesized envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        logger.debug(
            "Adapter envelope for run %s has no counts in result items[%d]",
            record.run_id,
            item_index,
        )
        return None

    # Manual/synthesized envelope - fallback allowed
    counts_info = get_counts(record, store)
    if counts_info is None:
        return None

    if canonicalize:
        return canonicalize_bitstrings(
            counts_info.counts,
            bit_order="cbit0_right",
            transformed=False,
        )
    return counts_info.counts


def resolve_device_snapshot(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
) -> DeviceSnapshot | None:
    """
    Load device snapshot with UEC-first strategy.

    This is the canonical function for getting device snapshot in compare
    operations. Falls back to Run record metadata only for synthesized
    (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.

    Returns
    -------
    DeviceSnapshot or None
        Device snapshot if available.
    """
    from devqubit_engine.uec.api.resolve import resolve_envelope as _resolve_envelope
    from devqubit_engine.uec.api.synthesize import build_device_from_record

    if envelope is None:
        envelope = _resolve_envelope(record, store)

    # Get device from envelope (UEC-first)
    if envelope.device is not None:
        return envelope.device

    # Fallback: construct from record - ONLY for synthesized envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        logger.debug(
            "Adapter envelope for run %s has no device snapshot",
            record.run_id,
        )
        return None

    # Manual/synthesized envelope - use shared builder from synthesize module
    return build_device_from_record(record)


def resolve_circuit_summary(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
    *,
    which: str = "logical",
) -> CircuitSummary | None:
    """
    Extract circuit summary with UEC-first strategy.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope.
    which : str, default="logical"
        Which circuit to extract: "logical" or "physical".

    Returns
    -------
    CircuitSummary or None
        Circuit summary, or None if not found.
    """
    from devqubit_engine.circuit.extractors import extract_circuit
    from devqubit_engine.circuit.summary import summarize_circuit_data
    from devqubit_engine.uec.api.resolve import resolve_envelope as _resolve_envelope

    if envelope is None:
        envelope = _resolve_envelope(record, store)

    circuit_data = extract_circuit(
        record,
        store,
        envelope=envelope,
        which=which,
        uec_first=True,
    )

    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)

    return None
