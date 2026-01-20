# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for Cirq adapter.

Provides functions for extracting measurements, building counts,
and normalizing results to devqubit's canonical format.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Use CountsFormat from UEC for format metadata
from devqubit_engine.uec.models.result import CountsFormat


def _get_cirq_counts_format() -> dict[str, Any]:
    """
    Get CountsFormat metadata for Cirq results.

    Cirq measurement arrays are converted to bitstring counts by:
    1. Sorting measurement keys alphabetically
    2. Concatenating bits from each key in sorted order
    3. Converting each row to a bitstring (left-to-right = first-to-last bit)

    Cirq uses big-endian bit order (first qubit = leftmost bit),
    which corresponds to cbit0_left in UEC canonical terminology.

    Returns
    -------
    dict
        CountsFormat as dictionary for JSON serialization.
    """
    return CountsFormat(
        source_sdk="cirq",
        source_key_format="measurement_key_concatenated",
        bit_order="cbit0_left",  # Cirq big-endian = cbit0_left in UEC terminology
        transformed=False,  # Not transformed to canonical cbit0_right
    ).to_dict()


def get_result_measurements(result: Any) -> dict[str, Any]:
    """
    Extract measurements dictionary from a Cirq result.

    Parameters
    ----------
    result : Any
        Cirq result object with `measurements` attribute.

    Returns
    -------
    dict
        Measurements dictionary mapping keys to numpy arrays.
        Returns empty dict if measurements not found or result is None.
    """
    if result is None:
        return {}
    try:
        meas = getattr(result, "measurements", None)
        return meas if isinstance(meas, dict) else {}
    except Exception:
        return {}


def counts_from_measurements(
    measurements: dict[str, Any] | None,
) -> tuple[dict[str, int], list[str], int, list[dict[str, Any]]]:
    """
    Build bitstring counts from Cirq measurement arrays.

    Concatenates measurement bits in sorted key order to form
    complete bitstrings, then counts occurrences.

    Parameters
    ----------
    measurements : dict or None
        Dictionary of measurement key to numpy array.
        Each array has shape (repetitions, num_bits).

    Returns
    -------
    counts : dict
        Mapping of bitstring to count.
    keys : list
        Sorted list of measurement keys used.
    total_bits : int
        Total number of bits in each bitstring.
    key_bit_ranges : list of dict
        Mapping describing which bits correspond to each measurement key.
        Each dict contains: key, start_bit, end_bit, num_bits.
        Bit positions are 0-indexed from the left of the bitstring.

    Raises
    ------
    ValueError
        If measurement arrays have inconsistent repetition counts.

    Notes
    -----
    When multiple measurement keys are present, bits are concatenated in
    sorted key order. The returned ``key_bit_ranges`` documents this layout
    to enable unambiguous interpretation of the combined bitstring.

    Examples
    --------
    >>> measurements = {"m": np.array([[0, 1], [1, 0], [0, 1]])}
    >>> counts, keys, nbits, ranges = counts_from_measurements(measurements)
    >>> counts
    {'01': 2, '10': 1}
    >>> ranges
    [{'key': 'm', 'start_bit': 0, 'end_bit': 2, 'num_bits': 2}]

    >>> # Multi-key example
    >>> measurements = {"a": np.array([[0], [1]]), "b": np.array([[1], [0]])}
    >>> counts, keys, nbits, ranges = counts_from_measurements(measurements)
    >>> keys
    ['a', 'b']
    >>> ranges
    [{'key': 'a', 'start_bit': 0, 'end_bit': 1, 'num_bits': 1},
     {'key': 'b', 'start_bit': 1, 'end_bit': 2, 'num_bits': 1}]
    """
    if not measurements:
        return {}, [], 0, []

    keys = sorted(measurements.keys())
    arrays = []
    for k in keys:
        arr = np.asarray(measurements[k], dtype=int)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)

    if not arrays:
        return {}, [], 0, []

    # Validate consistent repetition counts
    reps = arrays[0].shape[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape[0] != reps:
            raise ValueError(
                f"Inconsistent repetition counts: measurement key '{keys[0]}' "
                f"has {reps} reps, but '{keys[i]}' has {arr.shape[0]} reps. "
                f"All measurement arrays must have the same number of repetitions."
            )

    # Build bit range mapping for each key
    key_bit_ranges: list[dict[str, Any]] = []
    bit_offset = 0
    for key, arr in zip(keys, arrays):
        num_bits = arr.shape[1]
        key_bit_ranges.append(
            {
                "key": key,
                "start_bit": bit_offset,
                "end_bit": bit_offset + num_bits,
                "num_bits": num_bits,
            }
        )
        bit_offset += num_bits

    total_bits = bit_offset

    # Build bitstrings and count
    counts: dict[str, int] = {}
    for i in range(reps):
        bits = "".join("".join("1" if b else "0" for b in arr[i]) for arr in arrays)
        counts[bits] = counts.get(bits, 0) + 1

    return counts, keys, total_bits, key_bit_ranges


def _is_result_like(obj: Any) -> bool:
    """
    Check if object looks like a Cirq Result.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True if object has a dict-like `measurements` attribute.
    """
    try:
        return isinstance(getattr(obj, "measurements", None), dict)
    except Exception:
        return False


def _extract_params(result: Any) -> dict[str, Any] | None:
    """
    Extract parameter resolver from a Cirq result.

    Parameters
    ----------
    result : Any
        Cirq result object with optional `params` attribute.

    Returns
    -------
    dict or None
        JSON-serializable parameter dictionary, or None if not available.
    """
    params = getattr(result, "params", None)
    if params is None:
        return None

    def _to_serializable(v: Any) -> Any:
        return float(v) if hasattr(v, "__float__") else str(v)

    try:
        # ParamResolver has param_dict attribute
        param_dict = getattr(params, "param_dict", None)
        if param_dict is not None:
            return {str(k): _to_serializable(v) for k, v in param_dict.items()}

        # Fallback: try to convert directly
        if hasattr(params, "items"):
            return {str(k): _to_serializable(v) for k, v in params.items()}
    except Exception:
        pass

    return None


def _process_result(
    result: Any, index: int, batch_idx: int | None = None, sweep_idx: int | None = None
) -> dict[str, Any]:
    """
    Process a single result into experiment dict.

    Parameters
    ----------
    result : Any
        Cirq result object.
    index : int
        Global experiment index.
    batch_idx : int, optional
        Batch index for nested results.
    sweep_idx : int, optional
        Sweep index within batch.

    Returns
    -------
    dict
        Experiment dictionary with counts and metadata.
    """
    key_bit_ranges: list[dict[str, Any]] = []
    try:
        meas = get_result_measurements(result)
        counts, keys, nbits, key_bit_ranges = counts_from_measurements(meas)
    except Exception:
        counts, keys, nbits = {}, [], 0

    try:
        params = _extract_params(result)
    except Exception:
        params = None

    exp: dict[str, Any] = {
        "index": index,
        "counts": counts,
        "measurement_keys": keys,
        "num_bits": nbits,
    }
    # Include key_bit_ranges for multi-key measurements to document bit layout
    if key_bit_ranges:
        exp["key_bit_ranges"] = key_bit_ranges
    if batch_idx is not None:
        exp["batch_index"] = batch_idx
        exp["sweep_index"] = sweep_idx
    if params is not None:
        exp["params"] = params

    return exp


def normalize_counts_payload(results: Any) -> dict[str, Any]:
    """
    Convert Cirq results to devqubit's canonical counts format.

    Handles single results, flat lists of results (from run_sweep),
    and nested lists (from run_batch). Preserves parameter information
    for sweep/batch executions.

    Parameters
    ----------
    results : Any
        Single result, list of results, or nested list of results.

    Returns
    -------
    dict
        Normalized counts payload with structure::

            {
                "experiments": [
                    {
                        "index": int,
                        "counts": dict[str, int],
                        "measurement_keys": list[str],
                        "num_bits": int,
                        "key_bit_ranges": list[dict],  # Bit layout for multi-key
                        "params": dict[str, Any] | None,  # Parameter values
                        # Optional for nested results:
                        "batch_index": int,
                        "sweep_index": int,
                    },
                    ...
                ]
            }

        The ``key_bit_ranges`` field documents which bits in the concatenated
        bitstring correspond to each measurement key, enabling unambiguous
        interpretation when multiple measurement keys are present.

    Examples
    --------
    >>> result = simulator.run(circuit, repetitions=100)
    >>> payload = normalize_counts_payload(result)
    >>> payload["experiments"][0]["counts"]
    {'00': 48, '11': 52}
    >>> payload["format"]["source_sdk"]
    'cirq'
    """
    if results is None:
        return {"experiments": [], "format": _get_cirq_counts_format()}

    experiments: list[dict[str, Any]] = []

    # Single Result
    if _is_result_like(results):
        try:
            experiments.append(_process_result(results, 0))
        except Exception:
            pass
        return {
            "experiments": experiments,
            "format": _get_cirq_counts_format(),
        }

    # List of results
    if isinstance(results, (list, tuple)) and results:
        first = results[0]

        # Nested structure (run_batch): list of lists
        if isinstance(first, (list, tuple)) and not _is_result_like(first):
            idx = 0
            for batch_idx, batch in enumerate(results):
                for sweep_idx, r in enumerate(batch):
                    try:
                        experiments.append(
                            _process_result(r, idx, batch_idx, sweep_idx)
                        )
                    except Exception:
                        pass
                    idx += 1
            return {
                "experiments": experiments,
                "format": _get_cirq_counts_format(),
            }

        # Flat list of results (run_sweep)
        for i, r in enumerate(results):
            try:
                experiments.append(_process_result(r, i))
            except Exception:
                pass

    return {
        "experiments": experiments,
        "format": _get_cirq_counts_format(),
    }
