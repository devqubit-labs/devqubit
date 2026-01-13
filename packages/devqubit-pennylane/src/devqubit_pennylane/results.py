# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for PennyLane adapter.

Extracts and normalizes execution results from PennyLane devices
following the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from devqubit_engine.uec.result import (
    NormalizedCounts,
    NormalizedExpectation,
    ResultSnapshot,
)
from devqubit_engine.uec.types import ResultType


logger = logging.getLogger(__name__)


def _result_type_for_tape(tape: Any) -> str:
    """
    Determine a single-tape result type based on its first measurement.

    Parameters
    ----------
    tape : Any
        A PennyLane tape.

    Returns
    -------
    str
        A short type description.
    """
    measurements = getattr(tape, "measurements", []) or []
    if not measurements:
        return "unknown"

    m = measurements[0]

    # Try return_type attribute first (standard approach)
    rtype = getattr(m, "return_type", None)
    if rtype is not None:
        return str(rtype.name if hasattr(rtype, "name") else rtype)

    # Fallback: infer from measurement class name
    class_name = type(m).__name__.lower()
    if "expval" in class_name or "expectation" in class_name:
        return "Expectation"
    elif "sample" in class_name:
        return "Sample"
    elif "counts" in class_name:
        return "Counts"
    elif "probs" in class_name or "probability" in class_name:
        return "Probability"
    elif "state" in class_name:
        return "State"
    elif "var" in class_name or "variance" in class_name:
        return "Variance"

    return type(m).__name__


def extract_result_type(tapes: list[Any]) -> str:
    """
    Determine the result type based on measurements in tapes.

    For batches, returns:
    - the common type if all tapes match
    - "mixed" if tapes have different measurement return types

    Parameters
    ----------
    tapes : list
        List of executed tapes.

    Returns
    -------
    str
        Result type description (e.g., "Expectation", "Probability", "Sample", "mixed").

    Examples
    --------
    >>> import pennylane as qml
    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.Hadamard(wires=0)
    ...     qml.expval(qml.PauliZ(0))
    >>> extract_result_type([tape])
    'Expectation'
    """
    if not tapes:
        return "unknown"

    types = {_result_type_for_tape(t) for t in tapes}
    if len(types) == 1:
        return next(iter(types))
    return "mixed"


def _map_result_type_to_enum(result_type: str | None) -> ResultType:
    """
    Map PennyLane result type string to UEC ResultType enum.

    Parameters
    ----------
    result_type : str or None
        PennyLane result type string.

    Returns
    -------
    ResultType
        UEC result type enum.
    """
    if result_type is None:
        return ResultType.OTHER

    rt_lower = result_type.lower()

    if "expectation" in rt_lower or "expval" in rt_lower:
        return ResultType.EXPECTATION
    elif "sample" in rt_lower:
        return ResultType.SAMPLES
    elif "counts" in rt_lower:
        return ResultType.COUNTS
    elif "probability" in rt_lower or "probs" in rt_lower:
        return ResultType.QUASI_DIST
    elif "state" in rt_lower:
        return ResultType.STATEVECTOR
    elif "variance" in rt_lower or "var" in rt_lower:
        return ResultType.EXPECTATION  # Variance is expectation-like
    else:
        return ResultType.OTHER


def _to_numpy(arr: Any) -> np.ndarray | None:
    """
    Safely convert array-like to numpy array.

    Parameters
    ----------
    arr : Any
        Array-like object.

    Returns
    -------
    np.ndarray or None
        Numpy array, or None if conversion fails.
    """
    try:
        if isinstance(arr, np.ndarray):
            return arr
        return np.asarray(arr)
    except Exception:
        return None


def _sample_to_bitstring(sample: Any) -> str:
    """
    Convert a measurement sample to a bitstring.

    Handles various sample formats: arrays, lists, scalars.

    Parameters
    ----------
    sample : Any
        Single measurement sample (array of 0/1 values).

    Returns
    -------
    str
        Bitstring representation (e.g., "0101").
    """
    try:
        # Handle numpy array
        if isinstance(sample, np.ndarray):
            return "".join(str(int(b)) for b in sample.flatten())

        # Handle list/tuple
        if isinstance(sample, (list, tuple)):
            return "".join(str(int(b)) for b in sample)

        # Handle scalar (single qubit)
        if isinstance(sample, (int, np.integer)):
            return str(int(sample))

        # Fallback: try iteration
        return "".join(str(int(b)) for b in sample)

    except Exception as e:
        logger.debug("Failed to convert sample to bitstring: %s", e)
        # Last resort: hash for uniqueness
        return f"sample_{hash(str(sample)) % 10000:04d}"


def _extract_expectation_values(
    results: Any,
    num_circuits: int = 1,
) -> list[NormalizedExpectation]:
    """
    Extract expectation values from PennyLane results.

    Parameters
    ----------
    results : Any
        PennyLane execution results.
    num_circuits : int
        Number of circuits executed.

    Returns
    -------
    list of NormalizedExpectation
        Normalized expectation values.
    """
    if results is None:
        return []

    expectations: list[NormalizedExpectation] = []

    try:
        arr = _to_numpy(results)

        # Single circuit case
        if num_circuits == 1:
            if arr is not None and arr.ndim == 0:
                # Scalar result
                expectations.append(
                    NormalizedExpectation(
                        circuit_index=0,
                        observable_index=0,
                        value=float(arr),
                        std_error=None,
                    )
                )
            elif arr is not None and arr.ndim == 1:
                # Multiple observables for single circuit
                for j, val in enumerate(arr):
                    expectations.append(
                        NormalizedExpectation(
                            circuit_index=0,
                            observable_index=j,
                            value=float(val),
                            std_error=None,
                        )
                    )
            elif not isinstance(results, (str, dict)):
                # Single value
                expectations.append(
                    NormalizedExpectation(
                        circuit_index=0,
                        observable_index=0,
                        value=float(results),
                        std_error=None,
                    )
                )
            return expectations

        # Batch results
        if hasattr(results, "__iter__") and not isinstance(results, (str, dict)):
            for i, res in enumerate(results):
                if hasattr(res, "__iter__") and not isinstance(res, (str, dict)):
                    # Multiple measurements per circuit
                    for j, val in enumerate(res):
                        expectations.append(
                            NormalizedExpectation(
                                circuit_index=i,
                                observable_index=j,
                                value=float(val),
                                std_error=None,
                            )
                        )
                else:
                    # Single measurement per circuit
                    expectations.append(
                        NormalizedExpectation(
                            circuit_index=i,
                            observable_index=0,
                            value=float(res),
                            std_error=None,
                        )
                    )
        else:
            # Single result
            expectations.append(
                NormalizedExpectation(
                    circuit_index=0,
                    observable_index=0,
                    value=float(results),
                    std_error=None,
                )
            )
    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract expectation values: %s", e)

    return expectations


def _extract_sample_counts(
    results: Any,
    num_circuits: int = 1,
) -> list[NormalizedCounts]:
    """
    Extract sample counts from PennyLane results.

    Properly converts samples to bitstrings (e.g., "0101") instead of
    string representations of numpy arrays.

    Parameters
    ----------
    results : Any
        PennyLane execution results (samples or counts).
    num_circuits : int
        Number of circuits executed.

    Returns
    -------
    list of NormalizedCounts
        Normalized counts.
    """
    if results is None:
        return []

    from collections import Counter

    counts_list: list[NormalizedCounts] = []

    try:
        # Case 1: Already counts-like (dict)
        if isinstance(results, dict):
            counts_dict = {str(k): int(v) for k, v in results.items()}
            counts_list.append(
                NormalizedCounts(
                    circuit_index=0,
                    counts=counts_dict,
                    shots=sum(counts_dict.values()),
                )
            )
            return counts_list

        # Case 2: Single circuit samples (2D array: shots x wires)
        arr = _to_numpy(results)
        if num_circuits == 1 and arr is not None:
            if arr.ndim == 2:
                # (shots, num_wires) -> each row is a sample
                bitstrings = [_sample_to_bitstring(row) for row in arr]
                counter = Counter(bitstrings)
                counts_dict = dict(counter)
                counts_list.append(
                    NormalizedCounts(
                        circuit_index=0,
                        counts=counts_dict,
                        shots=len(bitstrings),
                    )
                )
                return counts_list
            elif arr.ndim == 1:
                # Single wire samples or pre-aggregated
                bitstrings = [_sample_to_bitstring(s) for s in arr]
                counter = Counter(bitstrings)
                counts_dict = dict(counter)
                counts_list.append(
                    NormalizedCounts(
                        circuit_index=0,
                        counts=counts_dict,
                        shots=len(bitstrings),
                    )
                )
                return counts_list

        # Case 3: Batch results (iterable)
        if hasattr(results, "__iter__") and not isinstance(results, (str, dict)):
            for i, res in enumerate(results):
                if isinstance(res, dict):
                    # Already counts
                    counts_dict = {str(k): int(v) for k, v in res.items()}
                    counts_list.append(
                        NormalizedCounts(
                            circuit_index=i,
                            counts=counts_dict,
                            shots=sum(counts_dict.values()),
                        )
                    )
                else:
                    # Samples array
                    res_arr = _to_numpy(res)
                    if res_arr is not None and res_arr.size > 0:
                        if res_arr.ndim == 2:
                            # (shots, num_wires)
                            bitstrings = [_sample_to_bitstring(row) for row in res_arr]
                        elif res_arr.ndim == 1:
                            # Single wire or flat samples
                            bitstrings = [_sample_to_bitstring(s) for s in res_arr]
                        else:
                            bitstrings = [_sample_to_bitstring(res_arr)]

                        counter = Counter(bitstrings)
                        counts_dict = dict(counter)
                        counts_list.append(
                            NormalizedCounts(
                                circuit_index=i,
                                counts=counts_dict,
                                shots=len(bitstrings),
                            )
                        )

    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract sample counts: %s", e)

    return counts_list


def _extract_probabilities(
    results: Any,
    num_circuits: int = 1,
) -> list[NormalizedCounts]:
    """
    Extract probabilities from PennyLane results as pseudo-counts.

    Handles both single circuit (1D array) and batch (list of arrays) cases.

    Parameters
    ----------
    results : Any
        PennyLane probability results.
    num_circuits : int
        Number of circuits executed.

    Returns
    -------
    list of NormalizedCounts
        Normalized probabilities as counts (values sum to ~1).
    """
    if results is None:
        return []

    counts_list: list[NormalizedCounts] = []

    try:
        arr = _to_numpy(results)

        # Single circuit case: results is 1D probability array
        if num_circuits == 1 and arr is not None and arr.ndim == 1:
            probs = arr.tolist()
            num_bits = max(1, (len(probs) - 1).bit_length()) if probs else 0
            counts_dict = {
                format(j, f"0{num_bits}b"): float(p)
                for j, p in enumerate(probs)
                if p > 1e-10  # Filter near-zero probabilities
            }
            if counts_dict:
                counts_list.append(
                    NormalizedCounts(
                        circuit_index=0,
                        counts=counts_dict,
                        shots=None,  # Probabilities don't have shots
                    )
                )
            return counts_list

        # Batch case: iterable of probability arrays
        if hasattr(results, "__iter__") and not isinstance(results, (str, dict)):
            for i, res in enumerate(results):
                res_arr = _to_numpy(res)
                if res_arr is not None and res_arr.ndim >= 1:
                    probs = res_arr.flatten().tolist()
                    num_bits = max(1, (len(probs) - 1).bit_length()) if probs else 0
                    counts_dict = {
                        format(j, f"0{num_bits}b"): float(p)
                        for j, p in enumerate(probs)
                        if p > 1e-10
                    }
                    if counts_dict:
                        counts_list.append(
                            NormalizedCounts(
                                circuit_index=i,
                                counts=counts_dict,
                                shots=None,
                            )
                        )

    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract probabilities: %s", e)

    return counts_list


def build_result_snapshot(
    results: Any,
    *,
    result_type: str | None = None,
    backend_name: str | None = None,
    num_circuits: int = 1,
    raw_result_ref: Any = None,
    success: bool = True,
    error_info: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Build a ResultSnapshot from PennyLane execution results.

    Parameters
    ----------
    results : Any
        PennyLane execution results.
    result_type : str, optional
        Result type string from extract_result_type.
    backend_name : str, optional
        Backend name for metadata.
    num_circuits : int
        Number of circuits executed.
    raw_result_ref : Any, optional
        Reference to stored raw result artifact.
    success : bool
        Whether execution succeeded.
    error_info : dict, optional
        Error information if execution failed.

    Returns
    -------
    ResultSnapshot
        Structured result snapshot.

    Examples
    --------
    >>> snapshot = build_result_snapshot(
    ...     [0.5, -0.3],
    ...     result_type="Expectation",
    ...     num_circuits=2,
    ... )
    >>> snapshot.result_type
    ResultType.EXPECTATION
    """
    uec_result_type = _map_result_type_to_enum(result_type)

    # Extract normalized results based on type
    counts: list[NormalizedCounts] = []
    expectations: list[NormalizedExpectation] = []

    if success and results is not None:
        try:
            if uec_result_type == ResultType.EXPECTATION:
                expectations = _extract_expectation_values(results, num_circuits)
            elif uec_result_type in (ResultType.COUNTS, ResultType.SAMPLES):
                counts = _extract_sample_counts(results, num_circuits)
            elif uec_result_type == ResultType.QUASI_DIST:
                counts = _extract_probabilities(results, num_circuits)
            # For STATEVECTOR and OTHER, we just store raw results
        except Exception as e:
            logger.debug("Failed to extract normalized results: %s", e)

    # Build metadata
    metadata: dict[str, Any] = {
        "backend_name": backend_name,
        "pennylane_result_type": result_type,
    }
    if error_info:
        metadata["error"] = error_info

    return ResultSnapshot(
        result_type=uec_result_type,
        raw_result_ref=raw_result_ref,
        counts=counts if counts else None,
        expectations=expectations if expectations else None,
        num_experiments=num_circuits,
        success=success,
        metadata=metadata,
    )
