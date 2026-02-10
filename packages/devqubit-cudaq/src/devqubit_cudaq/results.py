# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for CUDA-Q adapter.

Extracts and normalizes execution results from CUDA-Q ``SampleResult``
and ``ObserveResult`` objects following the devqubit Uniform Execution
Contract (UEC).

Bitstring Convention
--------------------
CUDA-Q returns bitstrings in allocation order (qubit 0 = leftmost bit),
which corresponds to ``cbit0_left`` in UEC terms. This module transforms
all bitstrings to the UEC canonical ``cbit0_right`` order and marks the
result format accordingly.

When ``explicit_measurements=True`` is passed to ``sample()``, the global
register contains bits in **measurement execution order** rather than
allocation order. In that mode bitstrings are kept as-is and the format
is marked ``bit_order='measurement_order'`` so downstream consumers know
the semantics differ.
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


# ============================================================================
# Bitstring normalization
# ============================================================================


def _normalize_bitstrings_to_cbit0_right(
    counts: dict[str, int],
) -> dict[str, int]:
    """
    Reverse every bitstring key so that qubit-0 is the rightmost bit.

    CUDA-Q allocation order places qubit-0 on the left.  UEC canonical
    form is ``cbit0_right`` (qubit-0 on the right), so we reverse each key.

    Parameters
    ----------
    counts : dict[str, int]
        Raw counts from CUDA-Q (allocation order, cbit0_left).

    Returns
    -------
    dict[str, int]
        Counts with reversed bitstring keys (cbit0_right).
    """
    return {k[::-1]: v for k, v in counts.items()}


# Pre-built counts format for standard (non-explicit) CUDA-Q results
_CUDAQ_COUNTS_FORMAT: dict[str, Any] = CountsFormat(
    source_sdk="cudaq",
    source_key_format="cudaq.__global__",
    bit_order="cbit0_right",
    transformed=True,
).to_dict()

# Counts format for explicit_measurements mode (measurement order, no reversal)
_CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER: dict[str, Any] = CountsFormat(
    source_sdk="cudaq",
    source_key_format="cudaq.__global__.explicit_measurements",
    bit_order="measurement_order",
    transformed=False,
).to_dict()


# ============================================================================
# Result type detection
# ============================================================================


def detect_result_type(result: Any) -> str:
    """
    Determine the result type from a CUDA-Q result object.

    Handles both single results and broadcasting (list results).

    Parameters
    ----------
    result : Any
        CUDA-Q execution result (``SampleResult``, ``ObserveResult``,
        or a list of either).

    Returns
    -------
    str
        Result type: ``"sample"``, ``"observe"``, ``"batch_sample"``,
        ``"batch_observe"``, or ``"unknown"``.
    """
    if result is None:
        return "unknown"

    # Handle broadcasting (list results)
    if isinstance(result, list):
        if not result:
            return "unknown"
        inner = detect_result_type(result[0])
        if inner in ("sample", "observe"):
            return f"batch_{inner}"
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


def _extract_counts_and_meta(
    sample_obj: Any,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """
    Extract bitstring counts and metadata from a ``SampleResult``.

    Uses the proper ``SampleResult`` API: ``.items()`` for iteration
    and ``.get_total_shots()`` for shot count.

    Parameters
    ----------
    sample_obj : Any
        CUDA-Q ``SampleResult``.

    Returns
    -------
    tuple[dict[str, int] | None, dict]
        (counts dict, metadata dict with shots/register_names).
    """
    meta: dict[str, Any] = {}

    # Extract total shots via API
    if hasattr(sample_obj, "get_total_shots"):
        try:
            meta["shots"] = int(sample_obj.get_total_shots())
        except Exception:
            pass

    # Extract register names
    if hasattr(sample_obj, "register_names"):
        try:
            names = list(sample_obj.register_names)
            if names:
                meta["register_names"] = names
        except Exception:
            pass

    # Extract counts via .items() (primary API)
    try:
        if hasattr(sample_obj, "items"):
            counts = {str(k): int(v) for k, v in sample_obj.items()}
            return counts, meta
    except Exception as exc:
        logger.debug("SampleResult.items() failed: %s", exc)

    # Fallback: try dict-like iteration
    try:
        counts = {str(k): int(v) for k, v in dict(sample_obj).items()}
        return counts, meta
    except Exception as exc:
        logger.debug("Failed to extract counts from SampleResult: %s", exc)

    return None, meta


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


def _extract_counts_from_observe(
    result: Any,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """
    Extract counts from an ``ObserveResult`` (shot-based observe).

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.

    Returns
    -------
    tuple[dict[str, int] | None, dict]
        (counts dict, metadata) or (None, {}).
    """
    try:
        sample_result = result.counts()
        if sample_result is not None:
            return _extract_counts_and_meta(sample_result)
    except Exception as exc:
        logger.debug("Failed to extract counts from ObserveResult: %s", exc)
    return None, {}


# ============================================================================
# Single-result item builders
# ============================================================================


def _build_sample_items(
    result: Any,
    *,
    item_offset: int = 0,
    call_kwargs: dict[str, Any] | None = None,
) -> list[ResultItem]:
    """
    Build ResultItems for a single SampleResult.

    Parameters
    ----------
    result : Any
        CUDA-Q ``SampleResult``.
    item_offset : int
        Starting item index.
    call_kwargs : dict, optional
        Keyword arguments that were passed to ``cudaq.sample()``.
        Used to detect ``explicit_measurements`` mode.
    """
    items: list[ResultItem] = []

    counts, meta = _extract_counts_and_meta(result)
    if not counts:
        return items

    # Determine normalization strategy based on explicit_measurements flag
    explicit = bool((call_kwargs or {}).get("explicit_measurements"))

    if explicit:
        # Measurement-order mode: keep bitstrings as-is
        normalized_counts = dict(counts)
        counts_format = dict(_CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER)
    else:
        # Standard mode: reverse to UEC canonical cbit0_right
        normalized_counts = _normalize_bitstrings_to_cbit0_right(counts)
        counts_format = dict(_CUDAQ_COUNTS_FORMAT)

    total_shots = meta.get("shots") or sum(normalized_counts.values())

    items.append(
        ResultItem(
            item_index=item_offset,
            success=True,
            counts={
                "counts": normalized_counts,
                "shots": total_shots,
                "format": counts_format,
            },
        )
    )

    # Derive quasi-probability distribution
    if total_shots > 0:
        distribution = {k: float(v) / total_shots for k, v in normalized_counts.items()}
        probs_values = list(distribution.values())
        items.append(
            ResultItem(
                item_index=item_offset + 1,
                success=True,
                quasi_probability=QuasiProbability(
                    distribution=distribution,
                    sum_probs=sum(probs_values),
                    min_prob=min(probs_values),
                    max_prob=max(probs_values),
                ),
            )
        )

    return items


def _build_observe_items(
    result: Any,
    *,
    item_offset: int = 0,
    call_kwargs: dict[str, Any] | None = None,
) -> list[ResultItem]:
    """
    Build ResultItems for a single ObserveResult.

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.
    item_offset : int
        Starting item index.
    call_kwargs : dict, optional
        Keyword arguments that were passed to ``cudaq.observe()``.
    """
    items: list[ResultItem] = []

    exp_val = _extract_expectation_from_observe(result)
    if exp_val is not None:
        items.append(
            ResultItem(
                item_index=item_offset,
                success=True,
                expectation=NormalizedExpectation(
                    circuit_index=0,
                    observable_index=0,
                    value=exp_val,
                    std_error=None,
                ),
            )
        )

    obs_counts, _ = _extract_counts_from_observe(result)
    if obs_counts:
        obs_counts = _normalize_bitstrings_to_cbit0_right(obs_counts)
        total_shots = sum(obs_counts.values())
        items.append(
            ResultItem(
                item_index=item_offset + 1,
                success=True,
                counts={
                    "counts": obs_counts,
                    "shots": total_shots,
                    "format": dict(_CUDAQ_COUNTS_FORMAT),
                },
            )
        )

    return items


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
    call_kwargs: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Build a ``ResultSnapshot`` from CUDA-Q execution results.

    Handles both single results and broadcasting (list results).
    Bitstrings are normalized to UEC canonical ``cbit0_right`` order
    unless ``explicit_measurements=True`` was passed to the execution call.

    Parameters
    ----------
    result : Any
        CUDA-Q execution result (``SampleResult``, ``ObserveResult``,
        or a list of either for broadcasting).
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
    call_kwargs : dict, optional
        Keyword arguments passed to the CUDA-Q execution call.

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
                items = _build_observe_items(result, call_kwargs=call_kwargs)

            elif result_type == "sample":
                items = _build_sample_items(result, call_kwargs=call_kwargs)

            elif result_type == "batch_sample" and isinstance(result, list):
                offset = 0
                for i, single in enumerate(result):
                    sub_items = _build_sample_items(
                        single,
                        item_offset=offset,
                        call_kwargs=call_kwargs,
                    )
                    # tag each sub-item with the batch index
                    for si in sub_items:
                        si.item_index = offset
                        offset += 1
                    items.extend(sub_items)

            elif result_type == "batch_observe" and isinstance(result, list):
                offset = 0
                for i, single in enumerate(result):
                    sub_items = _build_observe_items(
                        single,
                        item_offset=offset,
                        call_kwargs=call_kwargs,
                    )
                    for si in sub_items:
                        si.item_index = offset
                        offset += 1
                    items.extend(sub_items)

        except Exception as exc:
            logger.debug("Failed to extract normalized results: %s", exc)

    metadata: dict[str, Any] = {
        "backend_name": backend_name,
        "cudaq_result_type": result_type,
        "shots": shots,
    }
    if call_kwargs and call_kwargs.get("explicit_measurements"):
        metadata["cudaq.explicit_measurements"] = True

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        error=error,
        raw_result_ref=raw_result_ref,
        metadata=metadata,
    )
