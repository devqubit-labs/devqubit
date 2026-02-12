# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for Braket adapter.

Provides functions for extracting measurement counts from Braket task results,
including Program Set results, with optional canonicalization to UEC format.

Notes
-----
Braket uses big-endian bit ordering (qubit 0 = leftmost bit, cbit0_left).
UEC canonical format uses little-endian (qubit 0 = rightmost bit, cbit0_right).

When extracting counts, use ``canonicalize=True`` to transform bitstrings
to the canonical format for cross-SDK comparison.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_braket.utils import canonicalize_counts


logger = logging.getLogger(__name__)


def _to_counts_dict(
    x: Any,
    *,
    canonicalize: bool = False,
) -> dict[str, int] | None:
    """
    Convert a Counter/dict-like object into a {bitstring: count} dict.

    Parameters
    ----------
    x : Any
        Counter-like or dict-like object.
    canonicalize : bool
        If True, reverse bitstrings to canonical cbit0_right format.

    Returns
    -------
    dict or None
        Normalized counts dict or None if conversion fails.
    """
    if x is None:
        return None
    try:
        d = dict(x)
        result = {str(k): int(v) for k, v in d.items()}
        if canonicalize:
            result = canonicalize_counts(result)
        return result
    except (TypeError, ValueError):
        return None


def extract_measurement_counts(
    result: Any,
    *,
    canonicalize: bool = False,
) -> dict[str, int] | None:
    """
    Extract measurement counts from a single Braket result-like object.

    Parameters
    ----------
    result : Any
        Braket result object (e.g., GateModelQuantumTaskResult).
    canonicalize : bool
        If True, reverse bitstrings to canonical cbit0_right format.
        Default False (preserves Braket's native big-endian format).

    Returns
    -------
    dict or None
        Counts dictionary {bitstring: count} or None if extraction fails.

    Notes
    -----
    This is best-effort for *single* results. For Program Set results that
    contain multiple executables, prefer `extract_counts_payload()`.
    """
    if result is None:
        return None

    for key in ("measurement_counts", "counts", "measurementCounts"):
        try:
            if hasattr(result, key):
                v = getattr(result, key)
                v = v() if callable(v) else v
                out = _to_counts_dict(v, canonicalize=canonicalize)
                if out is not None:
                    return out
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug("Failed in return expression: %s", e)

    return None


def extract_counts_payload(
    result: Any,
    *,
    canonicalize: bool = False,
) -> dict[str, Any] | None:
    """
    Extract a devqubit-style counts payload from a Braket result.

    Parameters
    ----------
    result : Any
        Braket result object. Supports:
        - GateModelQuantumTaskResult-like objects (single executable)
        - ProgramSetQuantumTaskResult-like objects (multiple executables)
    canonicalize : bool
        If True, reverse bitstrings to canonical cbit0_right format.
        Default False (preserves Braket's native big-endian format).

    Returns
    -------
    dict or None
        Counts payload with structure::

            {
                "experiments": [
                    {"index": i, "counts": {...}, ...},
                    ...
                ]
            }

        Returns None if no counts could be extracted.

    Examples
    --------
    >>> task = device.run(circuit, shots=100)
    >>> result = task.result()
    >>> payload = extract_counts_payload(result)
    >>> payload["experiments"][0]["counts"]
    {'00': 48, '11': 52}
    """
    if result is None:
        return None

    # Try Program Set result structure (has .entries)
    experiments = _extract_program_set_experiments(result, canonicalize)
    if experiments:
        return {"experiments": experiments}

    # Single-result fallback
    try:
        counts = extract_measurement_counts(result, canonicalize=canonicalize)
    except (TypeError, ValueError, AttributeError):
        counts = None

    if counts is None:
        return None

    return {"experiments": [{"index": 0, "counts": counts}]}


def _extract_program_set_experiments(
    result: Any,
    canonicalize: bool,
) -> list[dict[str, Any]] | None:
    """
    Extract experiments from a Program Set result structure.

    Parameters
    ----------
    result : Any
        Result object with potential .entries attribute.
    canonicalize : bool
        Whether to canonicalize bitstrings.

    Returns
    -------
    list or None
        List of experiment dicts if Program Set, None otherwise.
    """
    try:
        top_entries = getattr(result, "entries", None)
        if not isinstance(top_entries, list) or not top_entries:
            return None

        experiments: list[dict[str, Any]] = []
        idx = 0

        for program_index, composite in enumerate(top_entries):
            inner_entries = getattr(composite, "entries", None)
            if not isinstance(inner_entries, list):
                continue

            for executable_index, measured in enumerate(inner_entries):
                counts_obj = getattr(measured, "counts", None)
                counts = _to_counts_dict(counts_obj, canonicalize=canonicalize)

                if counts is None:
                    counts = extract_measurement_counts(
                        measured, canonicalize=canonicalize
                    )

                if counts is None:
                    continue

                experiments.append(
                    {
                        "index": idx,
                        "program_index": int(program_index),
                        "executable_index": int(executable_index),
                        "counts": counts,
                    }
                )
                idx += 1

        return experiments if experiments else None

    except (TypeError, ValueError, AttributeError):
        return None
