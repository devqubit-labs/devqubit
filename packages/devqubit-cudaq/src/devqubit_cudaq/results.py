# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result processing for CUDA-Q adapter.

This module owns **all knowledge about CUDA-Q result objects**
(``SampleResult`` and ``ObserveResult``).  It exposes two consumer APIs:

UEC normalized results
    ``build_result_snapshot()`` — bitstring normalization, quasi-probability
    derivation, expectation extraction.

Raw artifact (``result.cudaq.output.json``)
    ``result_to_raw_artifact()`` — SDK-native extraction through CUDA-Q
    public API (``expectation()``, ``counts()``, ``items()``,
    ``register_names``, ``get_register_counts()``).

Bitstring Convention
--------------------
CUDA-Q returns bitstrings in allocation order (qubit 0 = leftmost bit),
which corresponds to ``cbit0_left`` in UEC terms.  This module transforms
all bitstrings to the UEC canonical ``cbit0_right`` order and marks the
result format accordingly.

When ``explicit_measurements=True`` is passed to ``sample()``, the global
register contains bits in **measurement execution order** rather than
allocation order.  In that mode bitstrings are kept as-is and the format
is marked ``bit_order='measurement_order'``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from devqubit_engine.uec.models.result import (
    CountsFormat,
    NormalizedExpectation,
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
        first = result[0]
        # Nested list: broadcast over arguments x spin operators
        if isinstance(first, list):
            if first:
                inner = detect_result_type(first[0])
                if inner == "observe":
                    return "batch_observe"
            return "unknown"
        inner = detect_result_type(first)
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
# Raw artifact extraction (for result.cudaq.output.json)
# ============================================================================


def _scrub_memory_addresses(text: str) -> str:
    """Replace process-specific memory addresses with a stable placeholder."""
    return re.sub(r"0x[0-9a-fA-F]+", "0x…", text)


def _extract_sample_raw(sample_obj: Any) -> dict[str, Any]:
    """
    Extract raw data from a ``SampleResult`` into JSON-safe form.

    Uses the public CUDA-Q API: ``items()``, ``get_total_shots()``,
    ``most_probable()``, ``register_names``, ``get_register_counts()``.

    Parameters
    ----------
    sample_obj : Any
        CUDA-Q sample result.

    Returns
    -------
    dict
        JSON-safe raw representation.
    """
    out: dict[str, Any] = {"type": "sample"}

    shots = None
    if hasattr(sample_obj, "get_total_shots"):
        try:
            shots = int(sample_obj.get_total_shots())
        except Exception:
            shots = None

    counts: dict[str, int] | None = None
    if hasattr(sample_obj, "items"):
        try:
            counts = {str(k): int(v) for k, v in sample_obj.items()}
        except Exception:
            counts = None
    if counts is None:
        try:
            counts = {str(k): int(v) for k, v in dict(sample_obj).items()}
        except Exception:
            counts = None

    if counts is not None:
        out["counts"] = counts
        if shots is None:
            try:
                shots = int(sum(counts.values()))
            except Exception:
                shots = None

    if shots is not None:
        out["shots"] = shots

    if hasattr(sample_obj, "register_names"):
        try:
            names = list(sample_obj.register_names)
            if names:
                out["register_names"] = [str(n) for n in names]
        except Exception:
            pass

    if hasattr(sample_obj, "most_probable"):
        try:
            out["most_probable"] = str(sample_obj.most_probable())
        except Exception:
            pass

    if hasattr(sample_obj, "get_register_counts") and out.get("register_names"):
        reg_counts: dict[str, Any] = {}
        for reg in out["register_names"]:
            try:
                r = sample_obj.get_register_counts(reg)
                if r is not None and hasattr(r, "items"):
                    reg_counts[str(reg)] = {str(k): int(v) for k, v in r.items()}
            except Exception:
                continue
        if reg_counts:
            out["register_counts"] = reg_counts

    return out


def _extract_observe_counts_raw(sample_obj: Any) -> dict[str, Any]:
    """
    Extract counts from an observe-derived ``SampleResult`` safely.

    The ``SampleResult`` returned by ``ObserveResult.counts()`` has
    partially-initialized C++ internals.  Methods like
    ``most_probable()``, ``register_names``, and
    ``get_register_counts()`` trigger segfaults at the pybind11 layer
    that Python ``try/except`` cannot catch.  This helper restricts
    itself to the safe subset: ``items()`` and ``get_total_shots()``.

    Parameters
    ----------
    sample_obj : Any
        ``SampleResult`` obtained from ``ObserveResult.counts()``.

    Returns
    -------
    dict
        JSON-safe counts representation with ``type``, ``counts``,
        and ``shots`` keys.
    """
    out: dict[str, Any] = {"type": "sample"}

    shots = None
    if hasattr(sample_obj, "get_total_shots"):
        try:
            shots = int(sample_obj.get_total_shots())
        except Exception:
            shots = None

    counts: dict[str, int] | None = None
    if hasattr(sample_obj, "items"):
        try:
            counts = {str(k): int(v) for k, v in sample_obj.items()}
        except Exception:
            counts = None
    if counts is None:
        try:
            counts = {str(k): int(v) for k, v in dict(sample_obj).items()}
        except Exception:
            counts = None

    if counts is not None:
        out["counts"] = counts
        if shots is None:
            try:
                shots = int(sum(counts.values()))
            except Exception:
                shots = None

    if shots is not None:
        out["shots"] = shots

    return out


def _extract_observe_raw(
    observe_obj: Any,
    *,
    shots: int | None = None,
) -> dict[str, Any]:
    """
    Extract raw data from an ``ObserveResult`` into JSON-safe form.

    Uses the public CUDA-Q API: ``expectation()`` and ``counts()``.

    Parameters
    ----------
    observe_obj : Any
        CUDA-Q observe result.
    shots : int or None
        Shot count from the execution call.  When ``None`` (analytic
        mode) ``.counts()`` is **not** called — CUDA-Q returns an
        invalid C++ ``SampleResult`` that segfaults in pybind11 before
        Python can catch the exception.  Even in shot-based mode the
        returned ``SampleResult`` has partially-initialized internals,
        so only safe methods (``items()``, ``get_total_shots()``) are
        called via ``_extract_observe_counts_raw``.

    Returns
    -------
    dict
        JSON-safe raw representation.
    """
    out: dict[str, Any] = {"type": "observe"}

    if hasattr(observe_obj, "expectation"):
        try:
            out["expectation"] = float(observe_obj.expectation())
        except Exception:
            pass

    # Guard: .counts() on an analytic ObserveResult segfaults (pybind11
    # returns an invalid SampleResult whose methods crash the process).
    # Even in shot-based mode, only safe methods may be called — see
    # _extract_observe_counts_raw docstring.
    if shots is not None and hasattr(observe_obj, "counts"):
        try:
            sr = observe_obj.counts()
        except Exception:
            sr = None
        if sr is not None:
            try:
                out["counts_obj"] = _extract_observe_counts_raw(sr)
            except Exception:
                pass

    return out


def result_to_raw_artifact(
    result: Any,
    *,
    result_type: str | None = None,
    shots: int | None = None,
) -> Any:
    """
    Convert a CUDA-Q result to a JSON-safe raw artifact payload.

    This is the data logged as ``result.cudaq.output.json``.  It extracts
    real values through the CUDA-Q public API rather than falling back to
    ``repr()`` (which embeds non-deterministic memory addresses).

    Parameters
    ----------
    result : Any
        CUDA-Q result object (``SampleResult``, ``ObserveResult``,
        list of either, or unknown).
    result_type : str, optional
        Explicit result type if already known.
    shots : int or None
        Shot count from the execution call.  Passed through to
        ``_extract_observe_raw`` to guard against the analytic-mode
        segfault in ``ObserveResult.counts()``.

    Returns
    -------
    Any
        JSON-safe representation.
    """
    if result is None:
        return None

    if isinstance(result, list):
        return [result_to_raw_artifact(r, shots=shots) for r in result]

    rt = result_type or detect_result_type(result)

    if rt == "sample":
        try:
            return _extract_sample_raw(result)
        except Exception:
            pass

    if rt == "observe":
        try:
            return _extract_observe_raw(result, shots=shots)
        except Exception:
            pass

    # Unknown result type — scrubbed str/repr fallback
    try:
        s = _scrub_memory_addresses(str(result))
    except Exception:
        s = "<unavailable>"

    try:
        r = _scrub_memory_addresses(repr(result))
    except Exception:
        r = "<unavailable>"

    return {"type": type(result).__name__, "str": s, "repr": r}


# ============================================================================
# UEC extraction helpers
# ============================================================================


def _extract_counts_and_meta(
    sample_obj: Any,
    *,
    skip_registers: bool = False,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """
    Extract bitstring counts and metadata from a ``SampleResult``.

    Parameters
    ----------
    sample_obj : Any
        CUDA-Q ``SampleResult``.
    skip_registers : bool
        When ``True``, skip ``register_names`` access.  Must be set
        for ``SampleResult`` objects obtained from
        ``ObserveResult.counts()`` whose ``register_names`` property
        can segfault at the C++ pybind11 layer.

    Returns
    -------
    tuple[dict[str, int] | None, dict]
        (counts dict, metadata dict with shots/register_names).
    """
    meta: dict[str, Any] = {}

    if hasattr(sample_obj, "get_total_shots"):
        try:
            meta["shots"] = int(sample_obj.get_total_shots())
        except Exception:
            pass

    if not skip_registers and hasattr(sample_obj, "register_names"):
        try:
            names = list(sample_obj.register_names)
            if names:
                meta["register_names"] = names
        except Exception:
            pass

    try:
        if hasattr(sample_obj, "items"):
            counts = {str(k): int(v) for k, v in sample_obj.items()}
            return counts, meta
    except Exception as exc:
        logger.debug("SampleResult.items() failed: %s", exc)

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
    """
    try:
        return float(result.expectation())
    except Exception as exc:
        logger.debug("Failed to extract expectation from ObserveResult: %s", exc)
        return None


def _extract_counts_from_observe(
    result: Any,
    *,
    shots: int | None = None,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """
    Extract counts from an ``ObserveResult`` (shot-based observe).

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.
    shots : int or None
        Shot count from the execution call.  When ``None`` (analytic
        mode) ``.counts()`` is skipped — it segfaults on CUDA-Q's
        pybind11 layer.

    Returns
    -------
    tuple[dict[str, int] | None, dict]
    """
    if shots is None:
        return None, {}
    try:
        sample_result = result.counts()
        if sample_result is not None:
            return _extract_counts_and_meta(
                sample_result,
                skip_registers=True,
            )
    except Exception as exc:
        logger.debug("Failed to extract counts from ObserveResult: %s", exc)
    return None, {}


# ============================================================================
# UEC single-result item builders
# ============================================================================


def _build_sample_item(
    result: Any,
    *,
    item_index: int = 0,
    call_kwargs: dict[str, Any] | None = None,
) -> ResultItem | None:
    """
    Build a single ``ResultItem`` for one ``SampleResult``.

    Each ``SampleResult`` maps to exactly one ``ResultItem`` carrying
    measurement counts.  This keeps ``item_index`` aligned with the
    broadcast position (UEC contract: 1 item per circuit/parameter set).

    Parameters
    ----------
    result : Any
        CUDA-Q ``SampleResult``.
    item_index : int
        Position in the batch (0-based).
    call_kwargs : dict, optional
        Keyword arguments passed to ``cudaq.sample()``.

    Returns
    -------
    ResultItem or None
        ``None`` when counts extraction fails.
    """
    counts, meta = _extract_counts_and_meta(result)
    if not counts:
        return None

    explicit = bool((call_kwargs or {}).get("explicit_measurements"))

    if explicit:
        normalized_counts = dict(counts)
        counts_format = dict(_CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER)
    else:
        normalized_counts = _normalize_bitstrings_to_cbit0_right(counts)
        counts_format = dict(_CUDAQ_COUNTS_FORMAT)

    total_shots = meta.get("shots") or sum(normalized_counts.values())

    return ResultItem(
        item_index=item_index,
        success=True,
        counts={
            "counts": normalized_counts,
            "shots": total_shots,
            "format": counts_format,
        },
    )


def _build_observe_item(
    result: Any,
    *,
    item_index: int = 0,
    circuit_index: int = 0,
    observable_index: int = 0,
) -> ResultItem | None:
    """
    Build a single ``ResultItem`` for one ``ObserveResult``.

    The canonical value is the expectation; shot-based counts (when
    available) are preserved in the raw artifact only.

    Parameters
    ----------
    result : Any
        CUDA-Q ``ObserveResult``.
    item_index : int
        Position in the flattened batch (0-based).
    circuit_index : int
        Index of the argument set in a broadcast.
    observable_index : int
        Index of the spin operator in a broadcast.

    Returns
    -------
    ResultItem or None
        ``None`` when expectation extraction fails.
    """
    exp_val = _extract_expectation_from_observe(result)
    if exp_val is None:
        return None

    return ResultItem(
        item_index=item_index,
        success=True,
        expectation=NormalizedExpectation(
            circuit_index=circuit_index,
            observable_index=observable_index,
            value=exp_val,
        ),
    )


# ============================================================================
# Public API — UEC ResultSnapshot
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

    Parameters
    ----------
    result : Any
        CUDA-Q execution result.
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
            if result_type == "sample":
                item = _build_sample_item(result, call_kwargs=call_kwargs)
                if item is not None:
                    items.append(item)

            elif result_type == "observe":
                item = _build_observe_item(result)
                if item is not None:
                    items.append(item)

            elif result_type == "batch_sample" and isinstance(result, list):
                for i, single in enumerate(result):
                    item = _build_sample_item(
                        single,
                        item_index=i,
                        call_kwargs=call_kwargs,
                    )
                    if item is not None:
                        items.append(item)

            elif result_type == "batch_observe" and isinstance(result, list):
                if result and isinstance(result[0], list):
                    # Nested: broadcast arguments × broadcast spin operators
                    idx = 0
                    for circuit_i, sub in enumerate(result):
                        for obs_i, single in enumerate(sub):
                            item = _build_observe_item(
                                single,
                                item_index=idx,
                                circuit_index=circuit_i,
                                observable_index=obs_i,
                            )
                            if item is not None:
                                items.append(item)
                            idx += 1
                else:
                    for i, single in enumerate(result):
                        item = _build_observe_item(
                            single,
                            item_index=i,
                            circuit_index=i,
                        )
                        if item is not None:
                            items.append(item)

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
