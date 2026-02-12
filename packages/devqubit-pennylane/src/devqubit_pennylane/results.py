# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for PennyLane adapter.

Extracts and normalizes execution results from PennyLane devices
following the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from devqubit_engine.uec.models.result import (
    CountsFormat,
    NormalizedExpectation,
    QuasiProbability,
    ResultError,
    ResultItem,
    ResultSnapshot,
)


logger = logging.getLogger(__name__)

# Pre-built counts format for PennyLane (static, never changes)
# PennyLane uses wire[0] as leftmost bit = cbit0_left (big-endian) in UEC
_PENNYLANE_COUNTS_FORMAT: dict[str, Any] = CountsFormat(
    source_sdk="pennylane",
    source_key_format="pennylane_bitstring",
    bit_order="cbit0_left",
    transformed=False,
).to_dict()


# Internal dataclass for intermediate counts extraction
@dataclass
class _CountsData:
    """Internal representation for extracted counts."""

    circuit_index: int
    counts: dict[str, int]
    shots: int | None


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
    except (TypeError, ValueError) as e:
        logger.debug("Failed to convert to numpy array: %s", e)
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
        digest = hashlib.sha256(str(sample).encode("utf-8")).hexdigest()[:8]
        return f"sample_{digest}"


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
) -> list[_CountsData]:
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
    list of _CountsData
        Normalized counts.
    """
    if results is None:
        return []

    from collections import Counter

    counts_list: list[_CountsData] = []

    try:
        # Case 1: Already counts-like (dict)
        if isinstance(results, dict):
            counts_dict = {str(k): int(v) for k, v in results.items()}
            counts_list.append(
                _CountsData(
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
                    _CountsData(
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
                    _CountsData(
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
                        _CountsData(
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
                            _CountsData(
                                circuit_index=i,
                                counts=counts_dict,
                                shots=len(bitstrings),
                            )
                        )

    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract sample counts: %s", e)

    return counts_list


@dataclass
class _ProbabilityData:
    """Internal representation for extracted probability distribution."""

    circuit_index: int
    distribution: dict[str, float]


def _extract_probabilities(
    results: Any,
    num_circuits: int = 1,
) -> list[_ProbabilityData]:
    """
    Extract probabilities from PennyLane results as quasi-probability distributions.

    Handles both single circuit (1D array) and batch (list of arrays) cases.
    Returns data suitable for QuasiProbability.

    Parameters
    ----------
    results : Any
        PennyLane probability results.
    num_circuits : int
        Number of circuits executed.

    Returns
    -------
    list of _ProbabilityData
        Probability distributions with float values.

    Notes
    -----
    PennyLane qml.probs() returns computational basis state probabilities.
    Bitstring format is wire[0] as leftmost bit (cbit0_left in UEC terms).
    """
    if results is None:
        return []

    prob_list: list[_ProbabilityData] = []

    try:
        arr = _to_numpy(results)

        # Single circuit case: results is 1D probability array
        if num_circuits == 1 and arr is not None and arr.ndim == 1:
            probs = arr.tolist()
            num_bits = max(1, (len(probs) - 1).bit_length()) if probs else 0
            distribution = {
                format(j, f"0{num_bits}b"): float(p)
                for j, p in enumerate(probs)
                if p > 1e-10  # Filter near-zero probabilities
            }
            if distribution:
                prob_list.append(
                    _ProbabilityData(
                        circuit_index=0,
                        distribution=distribution,
                    )
                )
            return prob_list

        # Batch case: iterable of probability arrays
        if hasattr(results, "__iter__") and not isinstance(results, (str, dict)):
            for i, res in enumerate(results):
                res_arr = _to_numpy(res)
                if res_arr is not None and res_arr.ndim >= 1:
                    probs = res_arr.flatten().tolist()
                    num_bits = max(1, (len(probs) - 1).bit_length()) if probs else 0
                    distribution = {
                        format(j, f"0{num_bits}b"): float(p)
                        for j, p in enumerate(probs)
                        if p > 1e-10
                    }
                    if distribution:
                        prob_list.append(
                            _ProbabilityData(
                                circuit_index=i,
                                distribution=distribution,
                            )
                        )

    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract probabilities: %s", e)

    return prob_list


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

    Uses UEC structure with items[] for per-circuit results.

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
    """
    # Determine status
    status = "completed" if success else "failed"

    # Build error object if execution failed
    error: ResultError | None = None
    if not success and error_info:
        error = ResultError(
            type=error_info.get("type", "UnknownError"),
            message=error_info.get("message", "Unknown error"),
        )

    # Build items list
    items: list[ResultItem] = []

    if success and results is not None:
        try:
            rt_lower = (result_type or "").lower()

            # Handle expectation values
            if "expectation" in rt_lower or "expval" in rt_lower or "var" in rt_lower:
                expectations = _extract_expectation_values(results, num_circuits)
                for exp in expectations:
                    items.append(
                        ResultItem(
                            item_index=exp.circuit_index,
                            success=True,
                            expectation=exp,
                        )
                    )

            # Handle counts/samples
            elif "counts" in rt_lower or "sample" in rt_lower:
                counts_list = _extract_sample_counts(results, num_circuits)
                for cd in counts_list:
                    items.append(
                        ResultItem(
                            item_index=cd.circuit_index,
                            success=True,
                            counts={
                                "counts": cd.counts,
                                "shots": cd.shots,
                                "format": dict(_PENNYLANE_COUNTS_FORMAT),
                            },
                        )
                    )

            # Handle probabilities - use quasi_probability
            elif "probability" in rt_lower or "probs" in rt_lower:
                probs_list = _extract_probabilities(results, num_circuits)
                for pd in probs_list:
                    # Create QuasiProbability with computed stats
                    probs_values = list(pd.distribution.values())
                    quasi = QuasiProbability(
                        distribution=pd.distribution,
                        sum_probs=sum(probs_values) if probs_values else None,
                        min_prob=min(probs_values) if probs_values else None,
                        max_prob=max(probs_values) if probs_values else None,
                    )
                    items.append(
                        ResultItem(
                            item_index=pd.circuit_index,
                            success=True,
                            quasi_probability=quasi,
                        )
                    )

        except Exception as e:
            logger.debug("Failed to extract normalized results: %s", e)

    # Build metadata
    metadata: dict[str, Any] = {
        "backend_name": backend_name,
        "pennylane_result_type": result_type,
        "num_circuits": num_circuits,
    }

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        error=error,
        raw_result_ref=raw_result_ref,
        metadata=metadata,
    )
