# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Measurement counts utilities.

This module provides functions for extracting and comparing measurement
counts from run artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from devqubit_engine.storage.artifacts.browse import find_artifact
from devqubit_engine.storage.artifacts.io import load_artifact_json


if TYPE_CHECKING:
    from devqubit_engine.core.record import RunRecord
    from devqubit_engine.storage.types import ObjectStoreProtocol


logger = logging.getLogger(__name__)


@dataclass
class CountsInfo:
    """
    Measurement counts information.

    Attributes
    ----------
    counts : dict
        Raw counts as {bitstring: count}.
    total_shots : int
        Total number of shots.
    num_outcomes : int
        Number of unique outcomes.
    """

    counts: dict[str, int]
    total_shots: int
    num_outcomes: int

    @property
    def probabilities(self) -> dict[str, float]:
        """Get normalized probabilities."""
        if self.total_shots == 0:
            return {}
        return {k: v / self.total_shots for k, v in self.counts.items()}

    def top_k(self, k: int = 10) -> list[tuple[str, int, float]]:
        """
        Get top-k outcomes by count.

        Parameters
        ----------
        k : int, default=10
            Number of outcomes to return.

        Returns
        -------
        list of tuple
            List of (bitstring, count, probability) tuples.
        """
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        result: list[tuple[str, int, float]] = []
        for bitstring, count in sorted_counts[:k]:
            prob = count / self.total_shots if self.total_shots > 0 else 0.0
            result.append((bitstring, count, prob))
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CountsInfo(shots={self.total_shots}, outcomes={self.num_outcomes})"


def _extract_counts_from_payload(
    payload: dict[str, Any],
    experiment_index: int | None,
) -> dict[str, int]:
    """Extract raw counts from payload, handling batch format."""
    experiments = payload.get("experiments")

    if not isinstance(experiments, list) or not experiments:
        # Simple format
        return payload.get("counts", {})

    if experiment_index is not None:
        # Specific experiment
        if experiment_index >= len(experiments):
            logger.debug(
                "Experiment index %d out of range (max: %d)",
                experiment_index,
                len(experiments) - 1,
            )
            return {}
        return experiments[experiment_index].get("counts", {})

    # Aggregate all experiments
    aggregated: dict[str, int] = {}
    for exp in experiments:
        if isinstance(exp, dict):
            for k, v in exp.get("counts", {}).items():
                aggregated[str(k)] = aggregated.get(str(k), 0) + int(v)
    return aggregated


def get_counts(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    experiment_index: int | None = None,
) -> CountsInfo | None:
    """
    Get measurement counts from a run.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    experiment_index : int, optional
        If provided, get counts for specific experiment in batch.
        If None, aggregates counts from all experiments.

    Returns
    -------
    CountsInfo or None
        Counts information or None if not found.

    Examples
    --------
    >>> counts = get_counts(record, store)
    >>> if counts:
    ...     print(f"Total shots: {counts.total_shots}")
    ...     for bitstring, count, prob in counts.top_k(5):
    ...         print(f"  {bitstring}: {count} ({prob:.2%})")
    """
    artifact = find_artifact(record, role="results", kind_contains="counts")
    if not artifact:
        logger.debug("No counts artifact found in run %s", record.run_id)
        return None

    payload = load_artifact_json(artifact, store)
    if not isinstance(payload, dict):
        return None

    raw_counts = _extract_counts_from_payload(payload, experiment_index)
    if not raw_counts:
        return None

    counts = {str(k): int(v) for k, v in raw_counts.items()}
    total = sum(counts.values())

    logger.debug("Loaded counts: %d shots, %d outcomes", total, len(counts))
    return CountsInfo(counts=counts, total_shots=total, num_outcomes=len(counts))
