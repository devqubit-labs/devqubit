# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for Qiskit adapter.

Extracts and normalizes measurement results from Qiskit jobs,
producing structures compatible with the devqubit Uniform Execution
Contract (UEC).
"""

from __future__ import annotations

from typing import Any

from devqubit_engine.uec.models.result import ResultType
from devqubit_engine.utils.serialization import to_jsonable


# =============================================================================
# Result Type Detection
# =============================================================================


def detect_result_type(result: Any) -> ResultType:
    """
    Detect the result type from a Qiskit result object.

    Handles counts, quasi-distributions, expectation values,
    statevectors, and density matrices.

    Parameters
    ----------
    result : Any
        Qiskit Result object.

    Returns
    -------
    ResultType
        Detected result type.
    """
    if result is None:
        return ResultType.OTHER

    # Check for statevector (simulator)
    try:
        if hasattr(result, "get_statevector") and callable(result.get_statevector):
            sv = result.get_statevector()
            if sv is not None:
                return ResultType.STATEVECTOR
    except Exception:
        pass

    # Check for quasi-distributions (Runtime Sampler)
    try:
        if extract_quasi_distributions(result) is not None:
            return ResultType.QUASI_DIST
    except Exception:
        pass

    # Check for expectation values (Runtime Estimator)
    try:
        if extract_expectation_values(result) is not None:
            return ResultType.EXPECTATION
    except Exception:
        pass

    # Check for measurement counts (standard)
    try:
        if hasattr(result, "get_counts") and callable(result.get_counts):
            result.get_counts()
            return ResultType.COUNTS
    except Exception:
        pass

    return ResultType.OTHER


# =============================================================================
# Counts Extraction
# =============================================================================


def normalize_result_counts(result: Any) -> dict[str, Any]:
    """
    Extract and normalize measurement counts from a Qiskit result.

    Parameters
    ----------
    result : Any
        Qiskit Result object from job.result().

    Returns
    -------
    dict
        Normalized counts with structure::

            {
                "experiments": [
                    {"index": 0, "counts": {"00": 500, "11": 500}, "shots": 1000},
                    ...
                ]
            }

    Notes
    -----
    Qiskit's Result.get_counts() supports selecting an experiment by index.
    This function iterates through all experiments to extract counts.
    """
    output: dict[str, Any] = {"experiments": []}

    if result is None:
        return output

    # Determine number of experiments
    num_experiments = _get_num_experiments(result)

    # Single experiment fallback
    if num_experiments is None:
        exp = _extract_single_experiment(result)
        if exp:
            output["experiments"].append(exp)
        return output

    # Multiple experiments
    for idx in range(int(num_experiments)):
        exp = _extract_experiment_by_index(result, idx)
        if exp:
            output["experiments"].append(exp)

    return output


def _get_num_experiments(result: Any) -> int | None:
    """Get number of experiments from result."""
    try:
        if hasattr(result, "results") and result.results is not None:
            return len(result.results)
    except Exception:
        pass
    return None


def _extract_single_experiment(result: Any) -> dict[str, Any] | None:
    """Extract counts from a single-experiment result."""
    try:
        if hasattr(result, "get_counts") and callable(result.get_counts):
            counts = result.get_counts()
            counts_dict = to_jsonable(counts)
            shots = sum(counts_dict.values()) if isinstance(counts_dict, dict) else None
            return {"index": 0, "counts": counts_dict, "shots": shots}
    except Exception:
        pass
    return None


def _extract_experiment_by_index(result: Any, idx: int) -> dict[str, Any] | None:
    """Extract counts for a specific experiment index."""
    try:
        counts = result.get_counts(idx)
        counts_dict = to_jsonable(counts)
        shots = sum(counts_dict.values()) if isinstance(counts_dict, dict) else None

        exp_data: dict[str, Any] = {"index": idx, "counts": counts_dict, "shots": shots}

        # Try to get experiment name
        try:
            if hasattr(result.results[idx], "header"):
                name = getattr(result.results[idx].header, "name", None)
                if name:
                    exp_data["name"] = str(name)
        except Exception:
            pass

        return exp_data
    except Exception:
        return None


# =============================================================================
# Result Metadata
# =============================================================================


def extract_result_metadata(result: Any) -> dict[str, Any]:
    """
    Extract metadata from a Qiskit result object.

    Parameters
    ----------
    result : Any
        Qiskit Result object.

    Returns
    -------
    dict
        Result metadata including backend name, job ID, success status,
        and execution timestamps.
    """
    metadata: dict[str, Any] = {}

    if result is None:
        return metadata

    attrs = [
        ("backend_name", str),
        ("job_id", str),
        ("success", bool),
        ("status", str),
        ("date", str),
        ("time_taken", float),
    ]

    for attr, converter in attrs:
        val = _safe_getattr(result, attr, converter)
        if val is not None:
            metadata[attr] = val

    return metadata


def _safe_getattr(obj: Any, attr: str, converter: type | None = None) -> Any | None:
    """Safely get and optionally convert an attribute value."""
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return None
        if converter is not None:
            return converter(val)
        return val
    except Exception:
        return None


# =============================================================================
# Quasi-Distribution Extraction
# =============================================================================


def extract_quasi_distributions(result: Any) -> list[dict[str, float]] | None:
    """
    Extract quasi-probability distributions from a Qiskit result.

    This is primarily used for results from Qiskit Runtime Sampler
    which returns quasi-distributions rather than raw counts.

    Parameters
    ----------
    result : Any
        Qiskit result object (typically SamplerResult).

    Returns
    -------
    list of dict or None
        List of quasi-distributions (one per circuit), or None if
        not available.

    Notes
    -----
    Quasi-distributions can have negative values for error-mitigated
    results. The values should sum to approximately 1.0.
    """
    if result is None:
        return None

    quasi_dists: list[dict[str, float]] = []

    try:
        if hasattr(result, "quasi_dists") and result.quasi_dists is not None:
            for qd in result.quasi_dists:
                if hasattr(qd, "binary_probabilities") and callable(
                    qd.binary_probabilities
                ):
                    quasi_dists.append(dict(qd.binary_probabilities()))
                elif isinstance(qd, dict):
                    if qd:
                        num_bits = max(len(bin(k)) - 2 for k in qd.keys())
                    else:
                        num_bits = 1
                    quasi_dists.append(
                        {format(k, f"0{num_bits}b"): v for k, v in qd.items()}
                    )
            if quasi_dists:
                return quasi_dists
    except Exception:
        pass

    return None


def extract_expectation_values(
    result: Any,
) -> list[tuple[float, float | None]] | None:
    """
    Extract expectation values from a Qiskit Estimator result.

    Parameters
    ----------
    result : Any
        Qiskit result object (typically EstimatorResult).

    Returns
    -------
    list of tuple or None
        List of (value, std_error) tuples for each observable,
        or None if not available.
    """
    if result is None:
        return None

    try:
        if hasattr(result, "values") and result.values is not None:
            values = list(result.values)
            std_errors: list[float | None]
            if (
                hasattr(result, "metadata")
                and result.metadata
                and len(result.metadata) > 0
            ):
                std_errors = list(result.metadata[0].get("std_error", []))
            else:
                std_errors = [None] * len(values)
            return list(zip(values, std_errors))
    except Exception:
        pass

    return None


# =============================================================================
# Primitive Result Normalization
# =============================================================================


def normalize_primitive_result(result: Any) -> dict[str, Any]:
    """
    Normalize results from Qiskit Runtime primitives.

    Handles SamplerV2 and EstimatorV2 result formats which differ
    from standard Result objects.

    Parameters
    ----------
    result : Any
        Qiskit Runtime primitive result.

    Returns
    -------
    dict
        Normalized result data with type indicator.
    """
    output: dict[str, Any] = {"result_type": "unknown"}

    if result is None:
        return output

    # Handle SamplerV2 results
    quasi_dists = extract_quasi_distributions(result)
    if quasi_dists is not None:
        output["result_type"] = "quasi_dist"
        output["quasi_distributions"] = quasi_dists
        return output

    # Handle EstimatorV2 results
    expectations = extract_expectation_values(result)
    if expectations is not None:
        output["result_type"] = "expectation"
        output["expectation_values"] = [
            {"value": v, "std_error": e} for v, e in expectations
        ]
        return output

    # Fallback to counts
    counts_data = normalize_result_counts(result)
    if counts_data.get("experiments"):
        output["result_type"] = "counts"
        output["experiments"] = counts_data["experiments"]
        return output

    return output
