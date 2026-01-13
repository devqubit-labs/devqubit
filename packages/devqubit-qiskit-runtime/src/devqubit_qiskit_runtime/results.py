# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for Qiskit Runtime adapter.

Provides functions for extracting measurement counts from Sampler
results and expectation values from Estimator results, following
the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.result import (
    ExpectationValue,
    NormalizedCounts,
    ResultSnapshot,
)
from devqubit_engine.uec.types import ResultType
from devqubit_engine.utils.serialization import to_jsonable


if TYPE_CHECKING:
    pass


def is_bitarray_like(x: Any) -> bool:
    """
    Check if object has BitArray-like interface.

    Parameters
    ----------
    x : Any
        Object to check.

    Returns
    -------
    bool
        True if object has get_counts() or get_bitstrings() methods.
    """
    return callable(getattr(x, "get_counts", None)) or callable(
        getattr(x, "get_bitstrings", None)
    )


def extract_bitarrays_from_databin(databin: Any) -> list[tuple[str, Any]]:
    """
    Extract BitArray-like objects from a DataBin container.

    IBM Runtime commonly uses result[0].data.meas for sampled bits.

    Parameters
    ----------
    databin : Any
        DataBin-like container.

    Returns
    -------
    list of tuple
        List of (name, bitarray) tuples.
    """
    if databin is None:
        return []

    # Check canonical 'meas' attribute first
    meas = getattr(databin, "meas", None)
    if meas is not None and is_bitarray_like(meas):
        return [("meas", meas)]

    # Scan for other BitArray-like attributes
    out: list[tuple[str, Any]] = []
    for name in dir(databin):
        if name.startswith("_"):
            continue
        try:
            val = getattr(databin, name)
        except Exception:
            continue
        if is_bitarray_like(val):
            out.append((name, val))

    return out


def counts_from_bitarrays(bitarrays: list[tuple[str, Any]]) -> dict[str, int] | None:
    """
    Convert BitArray objects to measurement counts.

    Parameters
    ----------
    bitarrays : list of tuple
        List of (name, bitarray) tuples.

    Returns
    -------
    dict or None
        Counts dictionary {bitstring: count} or None if extraction fails.
    """
    if not bitarrays:
        return None

    # Single register case
    if len(bitarrays) == 1:
        _, ba = bitarrays[0]
        if callable(getattr(ba, "get_counts", None)):
            try:
                c = ba.get_counts()
                return {str(k): int(v) for k, v in dict(c).items()}
            except Exception:
                return None
        return None

    # Multiple registers: join via get_bitstrings()
    bitstrings_per_reg: list[list[str]] = []
    for _, ba in bitarrays:
        getter = getattr(ba, "get_bitstrings", None)
        if not callable(getter):
            bitstrings_per_reg = []
            break
        try:
            bitstrings_per_reg.append([str(s) for s in list(getter())])
        except Exception:
            bitstrings_per_reg = []
            break

    if bitstrings_per_reg:
        nshots = min(len(x) for x in bitstrings_per_reg)
        ctr: Counter[str] = Counter()
        for i in range(nshots):
            parts = [
                bitstrings_per_reg[r][i].replace(" ", "")
                for r in range(len(bitstrings_per_reg))
            ]
            ctr["|".join(parts)] += 1
        return {k: int(v) for k, v in ctr.items()}

    # Fallback: first register get_counts
    _, ba0 = bitarrays[0]
    if callable(getattr(ba0, "get_counts", None)):
        try:
            c = ba0.get_counts()
            return {str(k): int(v) for k, v in dict(c).items()}
        except Exception:
            return None

    return None


def extract_sampler_results(result: Any) -> dict[str, Any] | None:
    """
    Extract counts artifact from a Sampler primitive result.

    Converts to devqubit's canonical format:
    {"experiments": [{"index": i, "counts": {...}}, ...]}

    Parameters
    ----------
    result : Any
        Primitive result object (PrimitiveResult).

    Returns
    -------
    dict or None
        Counts payload or None if extraction fails.
    """
    if result is None:
        return None

    exps: list[dict[str, Any]] = []

    try:
        pubs = list(result)
    except Exception:
        return None

    for i, pubres in enumerate(pubs):
        # Try join_data() first (combines multiple registers)
        join_data = getattr(pubres, "join_data", None)
        if callable(join_data):
            try:
                joined = join_data()
                if callable(getattr(joined, "get_counts", None)):
                    counts = joined.get_counts()
                    counts_dict = {str(k): int(v) for k, v in dict(counts).items()}
                    exps.append(
                        {
                            "index": i,
                            "counts": counts_dict,
                            "shots": sum(counts_dict.values()),
                        }
                    )
                    continue
            except Exception:
                pass

        # Fallback to extracting from databin
        databin = getattr(pubres, "data", None)
        bitarrays = extract_bitarrays_from_databin(databin)
        counts = counts_from_bitarrays(bitarrays)
        if counts is not None:
            exps.append(
                {
                    "index": i,
                    "counts": counts,
                    "shots": sum(counts.values()),
                }
            )

    return {"experiments": exps} if exps else None


def build_sampler_result_snapshot(
    result: Any,
    *,
    backend_name: str | None = None,
    raw_result_ref: Any = None,
) -> ResultSnapshot:
    """
    Build a ResultSnapshot from a Sampler result.

    Parameters
    ----------
    result : Any
        Primitive result object (PrimitiveResult).
    backend_name : str, optional
        Backend name for metadata.
    raw_result_ref : Any, optional
        Reference to stored raw result artifact.

    Returns
    -------
    ResultSnapshot
        Structured result snapshot with normalized counts.
    """
    if result is None:
        return ResultSnapshot(
            result_type=ResultType.COUNTS,
            raw_result_ref=raw_result_ref,
            counts=[],
            num_experiments=0,
            success=False,
            error_message="Result is None",
            metadata={
                "backend_name": backend_name,
                "primitive_type": "sampler",
            },
        )

    counts_data = extract_sampler_results(result)
    normalized_counts: list[NormalizedCounts] = []

    if counts_data:
        for exp in counts_data.get("experiments", []):
            normalized_counts.append(
                NormalizedCounts(
                    circuit_index=exp.get("index", 0),
                    counts=exp.get("counts", {}),
                    shots=exp.get("shots"),
                )
            )

    return ResultSnapshot(
        result_type=ResultType.COUNTS,
        raw_result_ref=raw_result_ref,
        counts=normalized_counts,
        num_experiments=len(normalized_counts),
        success=True,
        metadata={
            "backend_name": backend_name,
            "primitive_type": "sampler",
        },
    )


def extract_estimator_results(result: Any) -> dict[str, Any] | None:
    """
    Extract expectation values from an Estimator primitive result.

    Parameters
    ----------
    result : Any
        Primitive result object (PrimitiveResult).

    Returns
    -------
    dict or None
        Estimator values payload or None if extraction fails.
    """
    if result is None:
        return None

    exps: list[dict[str, Any]] = []

    try:
        pubs = list(result)
    except Exception:
        return None

    for i, pubres in enumerate(pubs):
        exp_entry: dict[str, Any] = {"index": i}

        databin = getattr(pubres, "data", None)
        if databin is not None:
            # Expectation values
            evs = getattr(databin, "evs", None)
            if evs is not None:
                try:
                    exp_entry["expectation_values"] = to_jsonable(evs)
                except Exception:
                    pass

            # Standard deviations
            stds = getattr(databin, "stds", None)
            if stds is not None:
                try:
                    exp_entry["standard_deviations"] = to_jsonable(stds)
                except Exception:
                    pass

        # Top-level metadata
        metadata = getattr(pubres, "metadata", None)
        if metadata is not None:
            try:
                exp_entry["metadata"] = to_jsonable(metadata)
            except Exception:
                pass

        if len(exp_entry) > 1:  # Has more than just index
            exps.append(exp_entry)

    return {"experiments": exps} if exps else None


def build_estimator_result_snapshot(
    result: Any,
    *,
    backend_name: str | None = None,
    raw_result_ref: Any = None,
) -> ResultSnapshot:
    """
    Build a ResultSnapshot from an Estimator result.

    Parameters
    ----------
    result : Any
        Primitive result object (PrimitiveResult).
    backend_name : str, optional
        Backend name for metadata.
    raw_result_ref : Any, optional
        Reference to stored raw result artifact.

    Returns
    -------
    ResultSnapshot
        Structured result snapshot with normalized expectations.
    """
    if result is None:
        return ResultSnapshot(
            result_type=ResultType.EXPECTATION,
            raw_result_ref=raw_result_ref,
            expectations=[],
            num_experiments=0,
            success=False,
            error_message="Result is None",
            metadata={
                "backend_name": backend_name,
                "primitive_type": "estimator",
            },
        )

    est_data = extract_estimator_results(result)
    normalized_expectations: list[ExpectationValue] = []

    if est_data:
        for exp in est_data.get("experiments", []):
            evs = exp.get("expectation_values", [])
            stds = exp.get("standard_deviations", [])

            # Handle single value or array
            if not isinstance(evs, list):
                evs = [evs]
            if not isinstance(stds, list):
                stds = [stds] if stds else []

            for obs_idx, ev in enumerate(evs):
                std = stds[obs_idx] if obs_idx < len(stds) else None
                normalized_expectations.append(
                    ExpectationValue(
                        circuit_index=exp.get("index", 0),
                        observable_index=obs_idx,
                        value=float(ev) if ev is not None else 0.0,
                        std_error=float(std) if std is not None else None,
                    )
                )

    return ResultSnapshot(
        result_type=ResultType.EXPECTATION,
        raw_result_ref=raw_result_ref,
        expectations=normalized_expectations,
        num_experiments=len(est_data.get("experiments", [])) if est_data else 0,
        success=True,
        metadata={
            "backend_name": backend_name,
            "primitive_type": "estimator",
        },
    )


def extract_quasi_distributions(result: Any) -> list[dict[str, float]] | None:
    """
    Extract quasi-probability distributions from a result.

    This is primarily used for results that return quasi-distributions
    rather than raw counts (e.g., with error mitigation).

    Parameters
    ----------
    result : Any
        Result object.

    Returns
    -------
    list of dict or None
        List of quasi-distributions (one per circuit), or None if not available.

    Notes
    -----
    Quasi-distributions can have negative values for error-mitigated results.
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
