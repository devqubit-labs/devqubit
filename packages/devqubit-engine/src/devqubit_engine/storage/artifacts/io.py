# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact I/O and formatting utilities.

This module provides:
- Low-level functions for loading artifact content from object stores
- Functions for formatting artifact data as human-readable ASCII tables
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from devqubit_engine.storage.artifacts.counts import CountsInfo
    from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol


logger = logging.getLogger(__name__)


def load_artifact_bytes(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
) -> bytes | None:
    """
    Load artifact bytes from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference containing digest.
    store : ObjectStoreProtocol
        Object store to retrieve data from.

    Returns
    -------
    bytes or None
        Raw bytes, or None on failure.
    """
    try:
        return store.get_bytes(artifact.digest)
    except Exception as e:
        logger.debug("Failed to load artifact %s: %s", artifact.digest[:16], e)
        return None


def load_artifact_text(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
    *,
    encoding: str = "utf-8",
) -> str | None:
    """
    Load artifact as text from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference.
    store : ObjectStoreProtocol
        Object store.
    encoding : str, default="utf-8"
        Text encoding.

    Returns
    -------
    str or None
        Decoded text, or None on failure.
    """
    data = load_artifact_bytes(artifact, store)
    if data is None:
        return None
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        logger.debug("Failed to decode artifact as %s: %s", encoding, e)
        return None


def load_artifact_json(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
) -> Any | None:
    """
    Load and parse JSON artifact from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference containing digest.
    store : ObjectStoreProtocol
        Object store to retrieve data from.

    Returns
    -------
    Any or None
        Parsed JSON payload, or None on failure.
    """
    text = load_artifact_text(artifact, store)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse artifact as JSON: %s", e)
        return None


# Legacy alias
load_json_artifact = load_artifact_json


def format_counts_table(counts: "CountsInfo", top_k: int = 10) -> str:
    """
    Format counts as ASCII table.

    Parameters
    ----------
    counts : CountsInfo
        Counts to format.
    top_k : int, default=10
        Number of outcomes to show.

    Returns
    -------
    str
        Formatted table.

    Examples
    --------
    >>> print(format_counts_table(counts))
    Total shots: 1,000
    Unique outcomes: 4

    Outcome              Count       Prob
    ------------------------------------------
    00                       500     0.5000
    11                       300     0.3000
    """
    lines = [
        f"Total shots: {counts.total_shots:,}",
        f"Unique outcomes: {counts.num_outcomes}",
        "",
        f"{'Outcome':<20} {'Count':>10} {'Prob':>10}",
        "-" * 42,
    ]

    for bitstring, count, prob in counts.top_k(top_k):
        lines.append(f"{bitstring:<20} {count:>10,} {prob:>10.4f}")

    if counts.num_outcomes > top_k:
        lines.append(f"... and {counts.num_outcomes - top_k} more outcomes")

    return "\n".join(lines)
