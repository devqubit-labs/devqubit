# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for Braket adapter.

Provides functions for extracting measurement counts from
Braket task results, including Program Set results.
"""

from __future__ import annotations

from typing import Any


def _to_counts_dict(x: Any) -> dict[str, int] | None:
    """
    Convert a Counter/dict-like object into a {bitstring: count} dict.

    Parameters
    ----------
    x : Any
        Counter-like or dict-like object.

    Returns
    -------
    dict or None
        Normalized counts dict or None if conversion fails.
    """
    if x is None:
        return None
    try:
        d = dict(x)
        return {str(k): int(v) for k, v in d.items()}
    except Exception:
        return None


def extract_measurement_counts(result: Any) -> dict[str, int] | None:
    """
    Extract measurement counts from a single Braket result-like object.

    Parameters
    ----------
    result : Any
        Braket result object (e.g., GateModelQuantumTaskResult).

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

    # Try common attribute names for measurement counts
    for key in ("measurement_counts", "counts", "measurementCounts"):
        try:
            if hasattr(result, key):
                v = getattr(result, key)
                v = v() if callable(v) else v
                out = _to_counts_dict(v)
                if out is not None:
                    return out
        except Exception:
            pass

    return None


def extract_counts_payload(result: Any) -> dict[str, Any] | None:
    """
    Extract a devqubit-style counts payload from a Braket result.

    Parameters
    ----------
    result : Any
        Braket result object. Supports:
        - GateModelQuantumTaskResult-like objects (single executable)
        - ProgramSetQuantumTaskResult-like objects (multiple executables)

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
    try:
        top_entries = getattr(result, "entries", None)
        if isinstance(top_entries, list) and top_entries:
            experiments: list[dict[str, Any]] = []
            idx = 0
            for program_index, composite in enumerate(top_entries):
                inner_entries = getattr(composite, "entries", None)
                if not isinstance(inner_entries, list):
                    continue
                for executable_index, measured in enumerate(inner_entries):
                    counts_obj = getattr(measured, "counts", None)
                    counts = _to_counts_dict(counts_obj)
                    if counts is None:
                        # Fallback: see if measured itself has measurement_counts
                        counts = extract_measurement_counts(measured)

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

            if experiments:
                return {"experiments": experiments}
    except Exception:
        pass

    # Single-result fallback
    try:
        counts = extract_measurement_counts(result)
    except Exception:
        counts = None

    if counts is None:
        return None

    return {"experiments": [{"index": 0, "counts": counts}]}
