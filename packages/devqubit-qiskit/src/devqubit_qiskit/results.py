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

    # Check for unitary/density matrix (simulator)
    try:
        if hasattr(result, "get_unitary") and callable(result.get_unitary):
            unitary = result.get_unitary()
            if unitary is not None:
                return ResultType.DENSITY_MATRIX
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

    # Best-effort number of experiments
    num_experiments: int | None = None
    try:
        if hasattr(result, "results") and result.results is not None:
            num_experiments = len(result.results)
    except Exception:
        pass

    # If we cannot determine experiments, try a single get_counts()
    if not num_experiments:
        try:
            if hasattr(result, "get_counts") and callable(result.get_counts):
                counts = result.get_counts()
                counts_dict = to_jsonable(counts)
                shots = (
                    sum(counts_dict.values()) if isinstance(counts_dict, dict) else None
                )
                output["experiments"].append(
                    {
                        "index": 0,
                        "counts": counts_dict,
                        "shots": shots,
                    }
                )
        except Exception:
            pass
        return output

    # Multiple experiments
    for idx in range(int(num_experiments)):
        try:
            counts = result.get_counts(idx)
            counts_dict = to_jsonable(counts)
            shots = sum(counts_dict.values()) if isinstance(counts_dict, dict) else None

            # Try to get experiment name
            name = None
            try:
                if hasattr(result.results[idx], "header"):
                    name = getattr(result.results[idx].header, "name", None)
            except Exception:
                pass

            exp_data: dict[str, Any] = {
                "index": idx,
                "counts": counts_dict,
                "shots": shots,
            }
            if name:
                exp_data["name"] = str(name)

            output["experiments"].append(exp_data)
        except Exception:
            continue

    return output


def _safe_getattr(obj: Any, attr: str, converter: type | None = None) -> Any | None:
    """
    Safely get and optionally convert an attribute value.

    Parameters
    ----------
    obj : Any
        Object to get attribute from.
    attr : str
        Attribute name.
    converter : type, optional
        Type to convert value to (str, bool, float).

    Returns
    -------
    Any or None
        Converted value or None if attribute missing or conversion fails.
    """
    try:
        if not hasattr(obj, attr):
            return None
        val = getattr(obj, attr)
        if converter is not None:
            return converter(val)
        return val
    except Exception:
        return None


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

    # Define attributes to extract: (attr_name, output_key, converter)
    attrs = [
        ("backend_name", "backend_name", str),
        ("job_id", "job_id", str),
        ("success", "success", bool),
        ("status", "status", str),
        ("date", "date", str),
        ("time_taken", "time_taken", float),
    ]

    for attr, key, converter in attrs:
        val = _safe_getattr(result, attr, converter)
        if val is not None:
            metadata[key] = val

    return metadata


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

    # Try direct quasi_dists attribute (Runtime Sampler)
    try:
        if hasattr(result, "quasi_dists") and result.quasi_dists is not None:
            for qd in result.quasi_dists:
                if hasattr(qd, "binary_probabilities") and callable(
                    qd.binary_probabilities
                ):
                    quasi_dists.append(dict(qd.binary_probabilities()))
                elif isinstance(qd, dict):
                    # Convert integer keys to binary strings
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


def extract_statevector(result: Any) -> dict[str, Any] | None:
    """
    Extract statevector from a Qiskit simulator result.

    Parameters
    ----------
    result : Any
        Qiskit result object (typically from Aer simulator).

    Returns
    -------
    dict or None
        Statevector data with amplitude information, or None if
        not available.

    Notes
    -----
    Statevectors are only available from simulator backends and
    can be very large for many qubits. The returned dict contains:

    - "num_qubits": Number of qubits
    - "dim": Statevector dimension (2^n)
    - "data_type": "complex128" or similar
    - "amplitudes": List of [real, imag] pairs (truncated for large states)
    """
    try:
        if hasattr(result, "get_statevector"):
            sv = result.get_statevector()
            if sv is not None:
                import numpy as np

                data = np.asarray(sv.data)
                num_qubits = int(np.log2(len(data)))

                # For large statevectors, only store metadata
                if num_qubits > 10:
                    return {
                        "num_qubits": num_qubits,
                        "dim": len(data),
                        "data_type": str(data.dtype),
                        "truncated": True,
                        "note": "Statevector too large for inline storage",
                    }

                # Store full statevector for small states
                amplitudes = [[float(x.real), float(x.imag)] for x in data]
                return {
                    "num_qubits": num_qubits,
                    "dim": len(data),
                    "data_type": str(data.dtype),
                    "amplitudes": amplitudes,
                }
    except Exception:
        pass

    return None


def extract_unitary(result: Any) -> dict[str, Any] | None:
    """
    Extract unitary matrix from a Qiskit simulator result.

    Parameters
    ----------
    result : Any
        Qiskit result object (typically from Aer UnitarySimulator).

    Returns
    -------
    dict or None
        Unitary matrix metadata, or None if not available.

    Notes
    -----
    Full unitary matrices are typically too large to store inline.
    This function returns metadata about the unitary.
    """
    try:
        if hasattr(result, "get_unitary"):
            unitary = result.get_unitary()
            if unitary is not None:
                import numpy as np

                data = np.asarray(unitary.data)
                dim = data.shape[0]
                num_qubits = int(np.log2(dim))

                return {
                    "num_qubits": num_qubits,
                    "dim": dim,
                    "data_type": str(data.dtype),
                    "shape": list(data.shape),
                    "note": "Full unitary matrix stored separately as artifact",
                }
    except Exception:
        pass

    return None


def extract_density_matrix(result: Any) -> dict[str, Any] | None:
    """
    Extract density matrix from a Qiskit simulator result.

    Parameters
    ----------
    result : Any
        Qiskit result object (typically from Aer DensityMatrixSimulator).

    Returns
    -------
    dict or None
        Density matrix metadata, or None if not available.
    """
    try:
        if hasattr(result, "data"):
            data_dict = result.data()
            if isinstance(data_dict, dict) and "density_matrix" in data_dict:
                import numpy as np

                dm = data_dict["density_matrix"]
                dm_data = np.asarray(dm)
                dim = dm_data.shape[0]
                num_qubits = int(np.log2(dim))

                return {
                    "num_qubits": num_qubits,
                    "dim": dim,
                    "data_type": str(dm_data.dtype),
                    "shape": list(dm_data.shape),
                    "purity": float(np.real(np.trace(dm_data @ dm_data))),
                }
    except Exception:
        pass

    return None


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
