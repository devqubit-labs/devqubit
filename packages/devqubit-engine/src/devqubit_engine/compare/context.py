# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Execution context and envelope extraction utilities.

This module provides the RunContext abstraction and helper functions for
extracting data from ExecutionEnvelopes. These are the building blocks
for comparison, verification, and verdict operations.

The extraction functions are part of the public API and are used by
both the diff and verdict modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.uec.models.device import DeviceSnapshot
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Execution context for comparison operations.

    Encapsulates all data needed for comparing a single run: the resolved
    envelope and the object store for artifact access. This is the primary
    input type for core comparison functions.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    envelope : ExecutionEnvelope
        Resolved execution envelope containing all run data.
    store : ObjectStoreProtocol
        Object store for loading artifacts referenced in the envelope.

    Notes
    -----
    RunContext is created by IO layer functions (diff, verify_against_baseline)
    and passed to core comparison functions (diff_contexts, verify_contexts).
    This separation ensures the core comparison logic is pure and testable.
    """

    run_id: str
    envelope: ExecutionEnvelope
    store: ObjectStoreProtocol


# =============================================================================
# Metadata extraction
# =============================================================================


def extract_tracker_metadata(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """
    Extract tracker namespace from envelope metadata.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    dict
        The tracker metadata namespace, or empty dict if not present.
    """
    tracker_meta = envelope.metadata.get("tracker")
    if tracker_meta is None:
        return {}
    if not isinstance(tracker_meta, dict):
        logger.warning(
            "envelope.metadata.tracker is not a dict (got %s), treating as empty",
            type(tracker_meta).__name__,
        )
        return {}
    return tracker_meta


def extract_params(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """
    Extract parameters from envelope metadata.tracker namespace.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    dict
        Parameters dictionary, or empty dict if not present.
    """
    tracker_meta = extract_tracker_metadata(envelope)
    return tracker_meta.get("params", {}) or {}


def extract_metrics(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """
    Extract metrics from envelope metadata.tracker namespace.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    dict
        Metrics dictionary, or empty dict if not present.
    """
    tracker_meta = extract_tracker_metadata(envelope)
    return tracker_meta.get("metrics", {}) or {}


def extract_project(envelope: ExecutionEnvelope) -> str | None:
    """
    Extract project name from envelope metadata.tracker namespace.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    str or None
        Project name if present.
    """
    tracker_meta = extract_tracker_metadata(envelope)
    return tracker_meta.get("project")


def extract_fingerprint(envelope: ExecutionEnvelope) -> str | None:
    """
    Extract run fingerprint from envelope metadata.tracker namespace.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    str or None
        Run fingerprint if present.
    """
    tracker_meta = extract_tracker_metadata(envelope)
    fingerprints = tracker_meta.get("fingerprints", {}) or {}
    return fingerprints.get("run")


def extract_backend_name(envelope: ExecutionEnvelope) -> str | None:
    """
    Extract backend name from envelope device snapshot.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    str or None
        Backend name if device snapshot present.
    """
    if envelope.device:
        return envelope.device.backend_name
    return None


# =============================================================================
# Result extraction
# =============================================================================


def get_counts_from_envelope(
    envelope: ExecutionEnvelope,
    item_index: int = 0,
    *,
    canonicalize: bool = True,
    skip_failed: bool = True,
) -> dict[str, int] | None:
    """
    Extract counts from envelope result item.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    item_index : int, default=0
        Index of the result item to extract counts from.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings to cbit0_right format.
    skip_failed : bool, default=True
        If True, skip items with success=False and return None for them.

    Returns
    -------
    dict or None
        Counts as {bitstring: count} in canonical format, or None if not
        available.
    """
    from devqubit_engine.uec.models.result import canonicalize_bitstrings

    if not envelope.result.items:
        return None

    if item_index >= len(envelope.result.items):
        return None

    item = envelope.result.items[item_index]

    if skip_failed and item.success is False:
        logger.debug("Skipping failed result item at index %d", item_index)
        return None

    if not item.counts:
        return None

    raw_counts = item.counts.get("counts")
    if not isinstance(raw_counts, dict):
        return None

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


def get_all_counts_from_envelope(
    envelope: ExecutionEnvelope,
    *,
    canonicalize: bool = True,
    skip_failed: bool = True,
) -> list[tuple[int, dict[str, int]]]:
    """
    Extract counts from all result items in the envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings.
    skip_failed : bool, default=True
        If True, skip items with success=False.

    Returns
    -------
    list of tuple
        List of (item_index, counts) tuples for each successful result item.
        Empty list if no counts available.
    """
    results: list[tuple[int, dict[str, int]]] = []
    skipped_count = 0

    for idx, item in enumerate(envelope.result.items):
        if skip_failed and item.success is False:
            skipped_count += 1
            continue

        counts = get_counts_from_envelope(
            envelope, idx, canonicalize=canonicalize, skip_failed=False
        )
        if counts is not None:
            results.append((idx, counts))

    if skipped_count > 0:
        logger.debug(
            "Skipped %d failed result items when extracting counts", skipped_count
        )

    return results


# =============================================================================
# Device and circuit extraction
# =============================================================================


def get_device_snapshot(envelope: ExecutionEnvelope) -> DeviceSnapshot | None:
    """
    Get device snapshot from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    DeviceSnapshot or None
        Device snapshot if present.
    """
    return envelope.device


def extract_circuit_summary(
    envelope: ExecutionEnvelope,
    store: ObjectStoreProtocol,
    *,
    which: str = "logical",
):
    """
    Extract circuit summary from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    which : str, default="logical"
        Which circuit to extract: "logical" or "physical".

    Returns
    -------
    CircuitSummary or None
        Extracted circuit summary, or None if not found.
    """
    from devqubit_engine.circuit.extractors import extract_circuit_from_envelope
    from devqubit_engine.circuit.summary import summarize_circuit_data

    circuit_data = extract_circuit_from_envelope(envelope, store, which=which)

    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)

    return None
