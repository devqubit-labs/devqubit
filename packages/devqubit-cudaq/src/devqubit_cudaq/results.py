# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for CUDA-Q adapter.

Extracts and normalizes execution results from CUDA-Q ``SampleResult``
and ``ObserveResult`` objects following the devqubit Uniform Execution
Contract (UEC).

Bitstring Convention
--------------------
CUDA-Q uses MSB-first (qubit 0 = leftmost bit), which maps to
``cbit0_left`` in UEC terms. We document this without transforming.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.uec.models.result import (
    CountsFormat,
    NormalizedExpectation,
    QuasiProbability,
    ResultError,
    ResultItem,
    ResultSnapshot,
)


logger = logging.getLogger(__name__)

# Pre-built counts format for CUDA-Q (static, never changes)
# CUDA-Q uses qubit 0 as leftmost bit = cbit0_left (big-endian) in UEC
_CUDAQ_COUNTS_FORMAT: dict[str, Any] = CountsFormat(
    source_sdk="cudaq",
    source_key_format="cudaq_bitstring",
    bit_order="cbit0_left",
    transformed=False,
).to_dict()


# ============================================================================
# Result type detection
# ============================================================================


def detect_result_type(result: Any) -> str:
    """
    Determine the result type from a CUDA-Q result object.

    Parameters
    ----------
    result : Any
        CUDA-Q execution result (``SampleResult`` or ``ObserveResult``).

    Returns
    -------
    str
        Result type: ``"sample"``, ``"observe"``, or ``"unknown"``.
    """
    if result is None:
        return "unknown"

    type_name = type(result).__name__.lower()

    if "observeresult" in type_name:
        return "observe"
    if "sampleresult" in type_name:
        return "sample"

    # Check for ObserveResult methods
    if hasattr(result, "expectation") and callable(getattr(result, "expectation")):
        return "observe"

    # Check for SampleResult dict-like interface
    if hasattr(result, "items") and hasattr(result, "most_probable"):
        return "sample"

    return "unknown"


# ============================================================================
# Extraction helpers
# ============================================================================


def _extract_counts_from_sample(result: Any) -> dict[str, int] | None:
    """
    Extract bitstring counts from a ``SampleResult``.

    Parameters
    ----------
    result : Any
        CUDA-Q ``SampleResult``.

    Returns
    -------
    dict or None
        Mapping of bitstring → count, or None on failure.
    """
    try:
        # SampleResult supports dict-like .items()
        if hasattr(result, "items"):
            return {str(k): int(v) for k, v in result.items()}

        # Fallback: try __iter__ for register-based results
        if hasattr(result, "register_names"):
            combined: dict[str, int] = {}
            for reg_name in result.register_names:
                sub = result.get_register_counts(reg_name)
                for k, v in sub.items():
                    combined[str(k)] = combined.get(str(k), 0) + int(v)
            if combined:
                return combined

    except Exception as exc:
        logger.debug("Failed to extract counts from SampleResult: %s", exc)

    return None


def _extract_expectation_from_observe(result: Any) -> float | None:
    """
    Extract expectation value from an ``ObserveResult``.

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.

    Returns
    -------
    float or None
        Expectation value, or None on failure.
    """
    try:
        return float(result.expectation())
    except Exception as exc:
        logger.debug("Failed to extract expectation from ObserveResult: %s", exc)
        return None


def _extract_counts_from_observe(result: Any) -> dict[str, int] | None:
    """
    Extract counts from an ``ObserveResult`` (shot-based observe).

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.

    Returns
    -------
    dict or None
        Bitstring counts if available, or None.
    """
    try:
        sample_result = result.counts()
        if sample_result and hasattr(sample_result, "items"):
            return {str(k): int(v) for k, v in sample_result.items()}
    except Exception as exc:
        logger.debug("Failed to extract counts from ObserveResult: %s", exc)
    return None


# ============================================================================
# Public API
# ============================================================================


def build_result_snapshot(
    result: Any,
    *,
    result_type: str | None = None,
    backend_name: str | None = None,
    shots: int | None = None,
    raw_result_ref: Any = None,
    success: bool = True,
    error_info: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Build a ``ResultSnapshot`` from CUDA-Q execution results.

    Follows UEC structure with ``items[]`` for per-circuit results.

    Parameters
    ----------
    result : Any
        CUDA-Q execution result (``SampleResult`` or ``ObserveResult``).
    result_type : str, optional
        Override result type. Auto-detected if not provided.
    backend_name : str, optional
        Target/backend name for metadata.
    shots : int, optional
        Number of shots used.
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
    status = "completed" if success else "failed"

    error: ResultError | None = None
    if not success and error_info:
        error = ResultError(
            type=error_info.get("type", "UnknownError"),
            message=error_info.get("message", "Unknown error"),
        )

    items: list[ResultItem] = []

    if success and result is not None:
        if result_type is None:
            result_type = detect_result_type(result)

        try:
            if result_type == "observe":
                # Extract expectation value
                exp_val = _extract_expectation_from_observe(result)
                if exp_val is not None:
                    items.append(
                        ResultItem(
                            item_index=0,
                            success=True,
                            expectation=NormalizedExpectation(
                                circuit_index=0,
                                observable_index=0,
                                value=exp_val,
                                std_error=None,
                            ),
                        )
                    )

                # Also extract counts if shots were used
                obs_counts = _extract_counts_from_observe(result)
                if obs_counts:
                    total_shots = sum(obs_counts.values())
                    items.append(
                        ResultItem(
                            item_index=1,
                            success=True,
                            counts={
                                "counts": obs_counts,
                                "shots": total_shots,
                                "format": dict(_CUDAQ_COUNTS_FORMAT),
                            },
                        )
                    )

            elif result_type == "sample":
                counts = _extract_counts_from_sample(result)
                if counts:
                    total_shots = sum(counts.values())
                    items.append(
                        ResultItem(
                            item_index=0,
                            success=True,
                            counts={
                                "counts": counts,
                                "shots": total_shots,
                                "format": dict(_CUDAQ_COUNTS_FORMAT),
                            },
                        )
                    )

                    # Derive quasi-probability distribution
                    if total_shots > 0:
                        distribution = {
                            k: float(v) / total_shots for k, v in counts.items()
                        }
                        probs_values = list(distribution.values())
                        items.append(
                            ResultItem(
                                item_index=1,
                                success=True,
                                quasi_probability=QuasiProbability(
                                    distribution=distribution,
                                    sum_probs=sum(probs_values),
                                    min_prob=min(probs_values),
                                    max_prob=max(probs_values),
                                ),
                            )
                        )

        except Exception as exc:
            logger.debug("Failed to extract normalized results: %s", exc)

    metadata: dict[str, Any] = {
        "backend_name": backend_name,
        "cudaq_result_type": result_type,
        "shots": shots,
    }

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        error=error,
        raw_result_ref=raw_result_ref,
        metadata=metadata,
    )
