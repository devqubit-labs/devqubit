# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run comparison with drift detection.

This module provides comprehensive comparison of quantum experiment runs,
including parameter comparison, metrics comparison, program artifact comparison,
device calibration drift analysis, result distribution comparison (TVD),
sampling noise context, and circuit semantic comparison.

The comparison logic operates entirely on ExecutionEnvelope (UEC) data.
All run metadata (params, metrics, project, fingerprints) is extracted from
envelope.metadata.devqubit namespace, ensuring consistent behavior across
adapter and manual runs.

Architecture
------------
- Core functions (diff_contexts) operate entirely on RunContext/envelope
- Adapter functions (diff, diff_runs) handle RunRecord/registry/bundle IO
- Extraction helpers are public for use by verdict module
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterator, Literal

from devqubit_engine.bundle.reader import Bundle, is_bundle_path
from devqubit_engine.circuit.summary import (
    CircuitSummary,
    diff_summaries,
    summarize_circuit_data,
)
from devqubit_engine.compare.drift import (
    DEFAULT_THRESHOLDS,
    DriftThresholds,
    compute_drift,
)
from devqubit_engine.compare.results import (
    ComparisonResult,
    ProgramComparison,
    ProgramMatchMode,
)
from devqubit_engine.config import Config, get_config
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.types import (
    ArtifactRef,
    ObjectStoreProtocol,
    RegistryProtocol,
)
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.uec.api.resolve import resolve_envelope
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.result import canonicalize_bitstrings
from devqubit_engine.utils.distributions import (
    NoiseContext,
    compute_noise_context,
    normalize_counts,
    total_variation_distance,
)


logger = logging.getLogger(__name__)


# Tolerance for TVD comparison (floating point precision)
_TVD_TOLERANCE = 1e-12


# =============================================================================
# RunContext: Core abstraction for comparison
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Execution context for comparison operations.

    Encapsulates all data needed for comparing a single run: the resolved
    envelope and the object store for artifact access. This is the primary
    input type for core comparison functions.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    envelope : ExecutionEnvelope
        Resolved execution envelope containing all run data.
    store : ObjectStoreProtocol
        Object store for loading artifacts referenced in the envelope.

    Notes
    -----
    RunContext is created by IO layer functions (diff, verify_against_baseline)
    and passed to core comparison functions (diff_contexts, verify_contexts).
    This separation ensures the core comparison logic is pure and testable.
    """

    run_id: str
    envelope: ExecutionEnvelope
    store: ObjectStoreProtocol


# =============================================================================
# Envelope extraction helpers (public API for verdict module)
# =============================================================================


def extract_devqubit_metadata(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """
    Extract devqubit namespace from envelope metadata.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.

    Returns
    -------
    dict
        The devqubit metadata namespace, or empty dict if not present.
    """
    devqubit = envelope.metadata.get("devqubit")
    if devqubit is None:
        return {}
    if not isinstance(devqubit, dict):
        logger.warning(
            "envelope.metadata.devqubit is not a dict (got %s), treating as empty",
            type(devqubit).__name__,
        )
        return {}
    return devqubit


def extract_params(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """Extract parameters from envelope metadata.devqubit namespace."""
    devqubit = extract_devqubit_metadata(envelope)
    return devqubit.get("params", {}) or {}


def extract_metrics(envelope: ExecutionEnvelope) -> dict[str, Any]:
    """Extract metrics from envelope metadata.devqubit namespace."""
    devqubit = extract_devqubit_metadata(envelope)
    return devqubit.get("metrics", {}) or {}


def extract_project(envelope: ExecutionEnvelope) -> str | None:
    """Extract project name from envelope metadata.devqubit namespace."""
    devqubit = extract_devqubit_metadata(envelope)
    return devqubit.get("project")


def extract_fingerprint(envelope: ExecutionEnvelope) -> str | None:
    """Extract run fingerprint from envelope metadata.devqubit namespace."""
    devqubit = extract_devqubit_metadata(envelope)
    fingerprints = devqubit.get("fingerprints", {}) or {}
    return fingerprints.get("run")


def extract_backend_name(envelope: ExecutionEnvelope) -> str | None:
    """Extract backend name from envelope device snapshot."""
    if envelope.device:
        return envelope.device.backend_name
    return None


def get_counts_from_envelope(
    envelope: ExecutionEnvelope,
    item_index: int = 0,
    *,
    canonicalize: bool = True,
    skip_failed: bool = True,
) -> dict[str, int] | None:
    """
    Extract counts from envelope result item.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    item_index : int, default=0
        Index of the result item to extract counts from.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings to cbit0_right format.
    skip_failed : bool, default=True
        If True, skip items with success=False and return None for them.

    Returns
    -------
    dict or None
        Counts as {bitstring: count} in canonical format, or None if not available.
    """
    if not envelope.result.items:
        return None

    if item_index >= len(envelope.result.items):
        return None

    item = envelope.result.items[item_index]

    # Skip failed items if requested
    if skip_failed and item.success is False:
        logger.debug("Skipping failed result item at index %d", item_index)
        return None

    if not item.counts:
        return None

    raw_counts = item.counts.get("counts")
    if not isinstance(raw_counts, dict):
        return None

    if canonicalize:
        format_info = item.counts.get("format", {})
        bit_order = format_info.get("bit_order", "cbit0_right")
        transformed = format_info.get("transformed", False)
        canonical = canonicalize_bitstrings(
            raw_counts,
            bit_order=bit_order,
            transformed=transformed,
        )
        return {k: int(v) for k, v in canonical.items()}
    else:
        return {str(k): int(v) for k, v in raw_counts.items()}


def get_all_counts_from_envelope(
    envelope: ExecutionEnvelope,
    *,
    canonicalize: bool = True,
    skip_failed: bool = True,
) -> list[tuple[int, dict[str, int]]]:
    """
    Extract counts from all result items in the envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings.
    skip_failed : bool, default=True
        If True, skip items with success=False.

    Returns
    -------
    list of tuple
        List of (item_index, counts) tuples for each successful result item.
        Empty list if no counts available.
    """
    results: list[tuple[int, dict[str, int]]] = []
    skipped_count = 0

    for idx in range(len(envelope.result.items)):
        item = envelope.result.items[idx]

        # Track skipped failed items for warning
        if skip_failed and item.success is False:
            skipped_count += 1
            continue

        counts = get_counts_from_envelope(
            envelope, idx, canonicalize=canonicalize, skip_failed=False
        )
        if counts is not None:
            results.append((idx, counts))

    if skipped_count > 0:
        logger.debug(
            "Skipped %d failed result items when extracting counts", skipped_count
        )

    return results


def get_device_snapshot(envelope: ExecutionEnvelope) -> DeviceSnapshot | None:
    """Get device snapshot from envelope."""
    return envelope.device


def extract_circuit_summary(
    envelope: ExecutionEnvelope,
    store: ObjectStoreProtocol,
    *,
    which: str = "logical",
) -> CircuitSummary | None:
    """
    Extract circuit summary from envelope.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Execution envelope.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    which : str, default="logical"
        Which circuit to extract: "logical" or "physical".

    Returns
    -------
    CircuitSummary or None
        Extracted circuit summary, or None if not found.
    """
    from devqubit_engine.circuit.extractors import extract_circuit_from_envelope

    circuit_data = extract_circuit_from_envelope(envelope, store, which=which)

    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)

    return None


# =============================================================================
# Internal comparison helpers
# =============================================================================


def _num_equal(a: Any, b: Any, tolerance: float) -> bool:
    """
    Compare two values with numeric tolerance.

    Handles bool, Real (including numpy types, Decimal), NaN, and Inf correctly.
    """
    # Booleans should not be compared as numbers
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b

    # Handle numeric types (including numpy.float64, Decimal, etc.)
    if isinstance(a, Real) and isinstance(b, Real):
        af, bf = float(a), float(b)

        # NaN equals NaN for comparison purposes
        if math.isnan(af) and math.isnan(bf):
            return True

        # Both NaN check failed, but one is NaN
        if math.isnan(af) or math.isnan(bf):
            return False

        # Inf values must match exactly
        if math.isinf(af) or math.isinf(bf):
            return af == bf

        return abs(af - bf) <= tolerance

    return a == b


def _diff_dict(
    dict_a: dict[str, Any],
    dict_b: dict[str, Any],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Compute difference between two dictionaries.

    Returns
    -------
    dict
        Comparison result with keys: match, added, removed, changed.
    """
    keys_a: set[str] = set(dict_a.keys())
    keys_b: set[str] = set(dict_b.keys())

    added = {k: dict_b[k] for k in keys_b - keys_a}
    removed = {k: dict_a[k] for k in keys_a - keys_b}

    changed: dict[str, dict[str, Any]] = {}
    for k in keys_a & keys_b:
        val_a = dict_a[k]
        val_b = dict_b[k]
        if not _num_equal(val_a, val_b, tolerance):
            changed[k] = {"a": val_a, "b": val_b}

    return {
        "match": not added and not removed and not changed,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _compare_programs(
    envelope_a: ExecutionEnvelope,
    envelope_b: ExecutionEnvelope,
) -> ProgramComparison:
    """
    Compare program artifacts between two envelopes.

    Uses envelope program refs for exact matching and structural/parametric
    hashes for semantic matching.
    """
    # Exact match: compare artifact digests from envelope refs
    digests_a: list[str] = []
    digests_b: list[str] = []

    if envelope_a.program:
        logical_digests_a = [a.ref.digest for a in envelope_a.program.logical if a.ref]
        physical_digests_a = [
            a.ref.digest for a in envelope_a.program.physical if a.ref
        ]
        digests_a = sorted(set(logical_digests_a + physical_digests_a))

    if envelope_b.program:
        logical_digests_b = [a.ref.digest for a in envelope_b.program.logical if a.ref]
        physical_digests_b = [
            a.ref.digest for a in envelope_b.program.physical if a.ref
        ]
        digests_b = sorted(set(logical_digests_b + physical_digests_b))

    # Exact match: identical digests (including both empty = both have no programs)
    exact_match = digests_a == digests_b

    # Extract hashes from envelope program snapshot
    hash_a: str | None = None
    hash_b: str | None = None
    param_hash_a: str | None = None
    param_hash_b: str | None = None
    exec_struct_hash_a: str | None = None
    exec_struct_hash_b: str | None = None
    exec_param_hash_a: str | None = None
    exec_param_hash_b: str | None = None
    hash_available = True

    if envelope_a.program:
        hash_a = envelope_a.program.structural_hash
        param_hash_a = envelope_a.program.parametric_hash
        exec_struct_hash_a = envelope_a.program.executed_structural_hash
        exec_param_hash_a = envelope_a.program.executed_parametric_hash

    if envelope_b.program:
        hash_b = envelope_b.program.structural_hash
        param_hash_b = envelope_b.program.parametric_hash
        exec_struct_hash_b = envelope_b.program.executed_structural_hash
        exec_param_hash_b = envelope_b.program.executed_parametric_hash

    # Check if hashes are available (manual runs won't have them)
    is_manual_a = envelope_a.metadata.get("manual_run", False)
    is_manual_b = envelope_b.metadata.get("manual_run", False)

    if is_manual_a or is_manual_b:
        if not hash_a or not hash_b:
            hash_available = False

    # Structural match: same circuit structure (ignores parameter values)
    # Both empty = match (no programs to compare)
    structural_match = exact_match  # Start with exact_match (handles empty case)
    if hash_a and hash_b:
        structural_match = hash_a == hash_b

    # Parametric match: same structure AND same parameter values
    parametric_match = False
    if param_hash_a and param_hash_b:
        parametric_match = param_hash_a == param_hash_b
    elif exact_match:
        parametric_match = True

    # Executed hashes match (for physical circuits)
    executed_structural_match = False
    executed_parametric_match = False

    if exec_struct_hash_a and exec_struct_hash_b:
        executed_structural_match = exec_struct_hash_a == exec_struct_hash_b

    if exec_param_hash_a and exec_param_hash_b:
        executed_parametric_match = exec_param_hash_a == exec_param_hash_b

    if structural_match and not parametric_match:
        logger.debug(
            "Programs match in structure but differ in params "
            "(structural_hash=%s, parametric_hash_a=%s, parametric_hash_b=%s)",
            hash_a,
            param_hash_a,
            param_hash_b,
        )

    return ProgramComparison(
        exact_match=exact_match,
        structural_match=structural_match,
        parametric_match=parametric_match,
        digests_a=digests_a,
        digests_b=digests_b,
        circuit_hash_a=hash_a,
        circuit_hash_b=hash_b,
        parametric_hash_a=param_hash_a,
        parametric_hash_b=param_hash_b,
        executed_structural_hash_a=exec_struct_hash_a,
        executed_structural_hash_b=exec_struct_hash_b,
        executed_parametric_hash_a=exec_param_hash_a,
        executed_parametric_hash_b=exec_param_hash_b,
        executed_structural_match=executed_structural_match,
        executed_parametric_match=executed_parametric_match,
        hash_available=hash_available,
    )


def _compute_tvd_for_item_pair(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    *,
    include_noise_context: bool,
    noise_alpha: float,
    noise_n_boot: int,
    noise_seed: int,
) -> tuple[float, NoiseContext | None]:
    """
    Compute TVD and optional noise context for a pair of counts.

    Returns
    -------
    tuple
        (tvd, noise_context) where noise_context may be None if not requested.
    """
    probs_a = normalize_counts(counts_a)
    probs_b = normalize_counts(counts_b)
    tvd = total_variation_distance(probs_a, probs_b)

    noise_ctx = None
    if include_noise_context:
        noise_ctx = compute_noise_context(
            counts_a,
            counts_b,
            tvd,
            n_boot=noise_n_boot,
            alpha=noise_alpha,
            seed=noise_seed,
        )

    return tvd, noise_ctx


# =============================================================================
# Core: Envelope-only comparison
# =============================================================================


def diff_contexts(
    ctx_a: RunContext,
    ctx_b: RunContext,
    *,
    thresholds: DriftThresholds | None = None,
    include_circuit_diff: bool = True,
    include_noise_context: bool = True,
    item_index: int | Literal["all"] = 0,
    noise_alpha: float = 0.95,
    noise_n_boot: int = 1000,
    noise_seed: int = 12345,
) -> ComparisonResult:
    """
    Compare two run contexts comprehensively.

    This is the core comparison function operating entirely on envelope data.
    All metadata (params, metrics, project, fingerprint) is extracted from
    the envelope's metadata.devqubit namespace.

    Parameters
    ----------
    ctx_a : RunContext
        Baseline run context.
    ctx_b : RunContext
        Candidate run context.
    thresholds : DriftThresholds, optional
        Drift detection thresholds. Uses defaults if not provided.
    include_circuit_diff : bool, default=True
        Include semantic circuit comparison.
    include_noise_context : bool, default=True
        Include sampling noise context estimation.
    item_index : int or "all", default=0
        Which result item(s) to use for TVD computation:
        - int: Use specific item index
        - "all": Aggregate across all items (worst case: max TVD)
    noise_alpha : float, default=0.95
        Quantile level for noise_p95 threshold (0.99 for stricter CI).
    noise_n_boot : int, default=1000
        Number of bootstrap iterations for noise estimation.
    noise_seed : int, default=12345
        Random seed for reproducible noise estimation.

    Returns
    -------
    ComparisonResult
        Complete comparison result with all analysis dimensions.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    envelope_a = ctx_a.envelope
    envelope_b = ctx_b.envelope

    logger.info("Comparing runs: %s vs %s", ctx_a.run_id, ctx_b.run_id)

    # Extract metadata from envelopes
    project_a = extract_project(envelope_a)
    project_b = extract_project(envelope_b)
    backend_a = extract_backend_name(envelope_a)
    backend_b = extract_backend_name(envelope_b)
    fingerprint_a = extract_fingerprint(envelope_a)
    fingerprint_b = extract_fingerprint(envelope_b)

    result = ComparisonResult(
        run_id_a=ctx_a.run_id,
        run_id_b=ctx_b.run_id,
        fingerprint_a=fingerprint_a,
        fingerprint_b=fingerprint_b,
    )

    # Metadata comparison with proper warnings for missing data
    project_match: bool | None = None
    backend_match: bool | None = None

    if project_a and project_b:
        project_match = project_a == project_b
    elif project_a or project_b:
        # One has project, other doesn't - warn and treat as mismatch
        project_match = False
        result.warnings.append(
            f"Project metadata incomplete: baseline={project_a!r}, candidate={project_b!r}"
        )
    # else: both None - no project info, match=None (unknown)

    if backend_a and backend_b:
        backend_match = backend_a == backend_b
    elif backend_a or backend_b:
        backend_match = False
        result.warnings.append(
            f"Backend metadata incomplete: baseline={backend_a!r}, candidate={backend_b!r}"
        )

    result.metadata = {
        "project_match": project_match if project_match is not None else True,
        "backend_match": backend_match if backend_match is not None else True,
        "project_a": project_a,
        "project_b": project_b,
        "backend_a": backend_a,
        "backend_b": backend_b,
        "envelope_a_synthesized": envelope_a.metadata.get(
            "synthesized_from_run", False
        ),
        "envelope_b_synthesized": envelope_b.metadata.get(
            "synthesized_from_run", False
        ),
    }

    # Batch result warning
    num_items_a = len(envelope_a.result.items) if envelope_a.result else 0
    num_items_b = len(envelope_b.result.items) if envelope_b.result else 0

    if item_index != "all" and (num_items_a > 1 or num_items_b > 1):
        result.warnings.append(
            f"Batch results detected (items: a={num_items_a}, b={num_items_b}). "
            f"Using item_index={item_index}. Consider using item_index='all' "
            f"for comprehensive comparison."
        )

    # Parameter comparison
    params_a = extract_params(envelope_a)
    params_b = extract_params(envelope_b)
    result.params = _diff_dict(params_a, params_b)

    # Metrics comparison
    metrics_a = extract_metrics(envelope_a)
    metrics_b = extract_metrics(envelope_b)
    result.metrics = _diff_dict(metrics_a, metrics_b)

    # Program comparison
    result.program = _compare_programs(envelope_a, envelope_b)

    if result.program.structural_only_match:
        result.warnings.append(
            "Program artifacts differ in content but match in structure\n"
            "(same circuit template with different parameter values)."
        )

    logger.debug(
        "Comparison: params_match=%s, metrics_match=%s, program_exact=%s, program_structural=%s",
        result.params.get("match"),
        result.metrics.get("match"),
        result.program.exact_match,
        result.program.structural_match,
    )

    # Device drift analysis
    snapshot_a = get_device_snapshot(envelope_a)
    snapshot_b = get_device_snapshot(envelope_b)

    if snapshot_a is None:
        result.warnings.append("Baseline envelope missing device snapshot")
    if snapshot_b is None:
        result.warnings.append("Candidate envelope missing device snapshot")

    if snapshot_a and snapshot_b:
        result.device_drift = compute_drift(snapshot_a, snapshot_b, thresholds)
        if result.device_drift.significant_drift:
            result.warnings.append(
                "Significant calibration drift detected. "
                "Results may not be directly comparable."
            )

    # Results comparison (TVD)
    if item_index == "all":
        _compute_tvd_all_items(
            result,
            envelope_a,
            envelope_b,
            include_noise_context=include_noise_context,
            noise_alpha=noise_alpha,
            noise_n_boot=noise_n_boot,
            noise_seed=noise_seed,
        )
    else:
        _compute_tvd_single_item(
            result,
            envelope_a,
            envelope_b,
            item_index=item_index,
            include_noise_context=include_noise_context,
            noise_alpha=noise_alpha,
            noise_n_boot=noise_n_boot,
            noise_seed=noise_seed,
        )

    # Circuit diff
    if include_circuit_diff:
        summary_a = extract_circuit_summary(envelope_a, ctx_a.store)
        summary_b = extract_circuit_summary(envelope_b, ctx_b.store)
        if summary_a and summary_b:
            result.circuit_diff = diff_summaries(summary_a, summary_b)
        elif not result.program.matches(ProgramMatchMode.EITHER):
            result.warnings.append(
                "Programs differ but circuit data not available for comparison."
            )

    # Determine overall identity (with tolerance for TVD)
    tvd_match = result.tvd is None or result.tvd <= _TVD_TOLERANCE
    drift_ok = not (result.device_drift and result.device_drift.significant_drift)

    result.identical = (
        result.metadata.get("project_match", False)
        and result.metadata.get("backend_match", False)
        and result.params.get("match", False)
        and result.metrics.get("match", True)
        and result.program.matches(ProgramMatchMode.EITHER)
        and drift_ok
        and tvd_match
    )

    logger.info(
        "Comparison complete: %s",
        "identical" if result.identical else "differ",
    )

    return result


def _compute_tvd_single_item(
    result: ComparisonResult,
    envelope_a: ExecutionEnvelope,
    envelope_b: ExecutionEnvelope,
    *,
    item_index: int,
    include_noise_context: bool,
    noise_alpha: float,
    noise_n_boot: int,
    noise_seed: int,
) -> None:
    """Compute TVD for a single item index."""
    result.counts_a = get_counts_from_envelope(envelope_a, item_index)
    result.counts_b = get_counts_from_envelope(envelope_b, item_index)

    if result.counts_a is not None and result.counts_b is not None:
        result.tvd, result.noise_context = _compute_tvd_for_item_pair(
            result.counts_a,
            result.counts_b,
            include_noise_context=include_noise_context,
            noise_alpha=noise_alpha,
            noise_n_boot=noise_n_boot,
            noise_seed=noise_seed,
        )
        logger.debug("TVD: %.6f", result.tvd)


def _compute_tvd_all_items(
    result: ComparisonResult,
    envelope_a: ExecutionEnvelope,
    envelope_b: ExecutionEnvelope,
    *,
    include_noise_context: bool,
    noise_alpha: float,
    noise_n_boot: int,
    noise_seed: int,
) -> None:
    """
    Compute TVD across all items using worst-case (max TVD) approach.

    IMPORTANT: Keeps counts_a/b consistent with the item that produced max TVD.
    This fixes the bug where counts were from the first item but TVD was from
    the worst item.
    """
    all_counts_a = get_all_counts_from_envelope(envelope_a)
    all_counts_b = get_all_counts_from_envelope(envelope_b)

    if not all_counts_a or not all_counts_b:
        result.tvd = None
        return

    # Check for batch size mismatch
    len_a, len_b = len(all_counts_a), len(all_counts_b)
    if len_a != len_b:
        result.warnings.append(
            f"Batch size mismatch: baseline has {len_a} items, "
            f"candidate has {len_b} items. TVD comparison skipped."
        )
        result.tvd = None
        return

    # Find worst case (max TVD) - track EVERYTHING together
    # Use >= to ensure we always have a valid worst case (even if TVD=0)
    worst_tvd: float | None = None
    worst_counts_a: dict[str, int] | None = None
    worst_counts_b: dict[str, int] | None = None
    worst_noise_ctx: NoiseContext | None = None
    worst_item_idx: int | None = None

    for i in range(len_a):
        idx_a, ca = all_counts_a[i]
        idx_b, cb = all_counts_b[i]

        tvd, noise_ctx = _compute_tvd_for_item_pair(
            ca,
            cb,
            include_noise_context=include_noise_context,
            noise_alpha=noise_alpha,
            noise_n_boot=noise_n_boot,
            noise_seed=noise_seed + i,
        )

        # Use >= to ensure we always have a valid worst case (fixes TVD=0 bug)
        if worst_tvd is None or tvd >= worst_tvd:
            worst_tvd = tvd
            worst_counts_a = ca
            worst_counts_b = cb
            worst_noise_ctx = noise_ctx
            worst_item_idx = i

    # Set result with consistent data from worst item
    result.tvd = worst_tvd
    result.counts_a = worst_counts_a
    result.counts_b = worst_counts_b
    result.noise_context = worst_noise_ctx

    if worst_item_idx is not None and len_a > 1:
        logger.debug(
            "Worst TVD %.6f found at item pair %d (of %d pairs)",
            worst_tvd or 0.0,
            worst_item_idx,
            len_a,
        )


# =============================================================================
# Adapters: RunRecord to RunContext conversion
# =============================================================================


def diff_runs(
    run_a: RunRecord,
    run_b: RunRecord,
    *,
    store_a: ObjectStoreProtocol,
    store_b: ObjectStoreProtocol,
    thresholds: DriftThresholds | None = None,
    include_circuit_diff: bool = True,
    include_noise_context: bool = True,
    item_index: int | Literal["all"] = 0,
    noise_alpha: float = 0.95,
    noise_n_boot: int = 1000,
    noise_seed: int = 12345,
) -> ComparisonResult:
    """
    Compare two run records comprehensively.

    Resolves ExecutionEnvelope for each run and delegates to diff_contexts
    for envelope-only comparison.

    Parameters
    ----------
    run_a : RunRecord
        Baseline run record.
    run_b : RunRecord
        Candidate run record.
    store_a : ObjectStoreProtocol
        Object store for baseline artifacts.
    store_b : ObjectStoreProtocol
        Object store for candidate artifacts.
    thresholds : DriftThresholds, optional
        Drift detection thresholds. Uses defaults if not provided.
    include_circuit_diff : bool, default=True
        Include semantic circuit comparison.
    include_noise_context : bool, default=True
        Include sampling noise context estimation.
    item_index : int or "all", default=0
        Which result item(s) to use for TVD computation.
    noise_alpha : float, default=0.95
        Quantile level for noise_p95 threshold.
    noise_n_boot : int, default=1000
        Number of bootstrap iterations.
    noise_seed : int, default=12345
        Random seed for reproducibility.

    Returns
    -------
    ComparisonResult
        Complete comparison result with all analysis dimensions.
    """
    # Resolve envelopes
    envelope_a = resolve_envelope(run_a, store_a)
    envelope_b = resolve_envelope(run_b, store_b)

    # Build contexts
    ctx_a = RunContext(run_id=run_a.run_id, envelope=envelope_a, store=store_a)
    ctx_b = RunContext(run_id=run_b.run_id, envelope=envelope_b, store=store_b)

    return diff_contexts(
        ctx_a,
        ctx_b,
        thresholds=thresholds,
        include_circuit_diff=include_circuit_diff,
        include_noise_context=include_noise_context,
        item_index=item_index,
        noise_alpha=noise_alpha,
        noise_n_boot=noise_n_boot,
        noise_seed=noise_seed,
    )


# =============================================================================
# Bundle loading helpers
# =============================================================================


class _BundleContext:
    """Context manager for loading run records from bundles."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._bundle: Bundle | None = None
        self._record: RunRecord | None = None

    def __enter__(self) -> tuple[RunRecord, ObjectStoreProtocol]:
        """Open bundle and return record and store."""
        self._bundle = Bundle(self.path)
        self._bundle.__enter__()

        record_dict = self._bundle.run_record
        artifacts = [
            ArtifactRef.from_dict(a)
            for a in record_dict.get("artifacts", [])
            if isinstance(a, dict)
        ]
        self._record = RunRecord(record=record_dict, artifacts=artifacts)

        return self._record, self._bundle.store

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close bundle."""
        if self._bundle is not None:
            self._bundle.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def load_from_bundle(path: Path) -> Iterator[tuple[RunRecord, ObjectStoreProtocol]]:
    """
    Load run record and store from a bundle file.

    Parameters
    ----------
    path : Path
        Path to the bundle zip file.

    Yields
    ------
    tuple
        (RunRecord, ObjectStoreProtocol) from the bundle.
    """
    ctx = _BundleContext(path)
    try:
        yield ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


# =============================================================================
# High-level API: diff by reference (run_id, path, or bundle)
# =============================================================================


def diff(
    ref_a: str | Path,
    ref_b: str | Path,
    *,
    registry: RegistryProtocol | None = None,
    store: ObjectStoreProtocol | None = None,
    thresholds: DriftThresholds | None = None,
    include_circuit_diff: bool = True,
    include_noise_context: bool = True,
    item_index: int | Literal["all"] = 0,
    noise_alpha: float = 0.95,
    noise_n_boot: int = 1000,
    noise_seed: int = 12345,
) -> ComparisonResult:
    """
    Compare two runs or bundles by reference.

    Accepts run IDs or bundle file paths and loads the appropriate
    records and stores automatically.

    Parameters
    ----------
    ref_a : str or Path
        Baseline run ID or bundle path.
    ref_b : str or Path
        Candidate run ID or bundle path.
    registry : RegistryProtocol, optional
        Run registry. Uses global config if not provided.
    store : ObjectStoreProtocol, optional
        Object store. Uses global config if not provided.
    thresholds : DriftThresholds, optional
        Drift detection thresholds.
    include_circuit_diff : bool, default=True
        Include semantic circuit comparison.
    include_noise_context : bool, default=True
        Include sampling noise context.
    item_index : int or "all", default=0
        Which result item(s) to use for TVD computation.
    noise_alpha : float, default=0.95
        Quantile level for noise_p95 threshold.
    noise_n_boot : int, default=1000
        Number of bootstrap iterations.
    noise_seed : int, default=12345
        Random seed for reproducibility.

    Returns
    -------
    ComparisonResult
        Complete comparison result.
    """
    bundle_contexts: list[_BundleContext] = []

    _registry: RegistryProtocol | None = registry
    _store: ObjectStoreProtocol | None = store
    _cfg: Config | None = None

    def get_cfg() -> Config:
        nonlocal _cfg
        if _cfg is None:
            _cfg = get_config()
        return _cfg

    def get_registry_() -> RegistryProtocol:
        nonlocal _registry
        if _registry is None:
            _registry = create_registry(config=get_cfg())
        return _registry

    def get_store_() -> ObjectStoreProtocol:
        nonlocal _store
        if _store is None:
            _store = create_store(config=get_cfg())
        return _store

    try:
        # Load run A
        if is_bundle_path(ref_a):
            logger.debug("Loading baseline from bundle: %s", ref_a)
            ctx_a = _BundleContext(Path(ref_a))
            bundle_contexts.append(ctx_a)
            run_a, store_a = ctx_a.__enter__()
        else:
            logger.debug("Loading baseline from registry: %s", ref_a)
            run_a = get_registry_().load(str(ref_a))
            store_a = get_store_()

        # Load run B
        if is_bundle_path(ref_b):
            logger.debug("Loading candidate from bundle: %s", ref_b)
            ctx_b = _BundleContext(Path(ref_b))
            bundle_contexts.append(ctx_b)
            run_b, store_b = ctx_b.__enter__()
        else:
            logger.debug("Loading candidate from registry: %s", ref_b)
            run_b = get_registry_().load(str(ref_b))
            store_b = get_store_()

        return diff_runs(
            run_a,
            run_b,
            store_a=store_a,
            store_b=store_b,
            thresholds=thresholds,
            include_circuit_diff=include_circuit_diff,
            include_noise_context=include_noise_context,
            item_index=item_index,
            noise_alpha=noise_alpha,
            noise_n_boot=noise_n_boot,
            noise_seed=noise_seed,
        )
    finally:
        for ctx in bundle_contexts:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
