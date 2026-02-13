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
from devqubit_engine.circuit.summary import diff_summaries
from devqubit_engine.compare.context import (
    RunContext,
    extract_backend_name,
    extract_circuit_summary,
    extract_fingerprint,
    extract_metrics,
    extract_params,
    extract_project,
    get_all_counts_from_envelope,
    get_counts_from_envelope,
    get_device_snapshot,
)
from devqubit_engine.compare.drift import (
    DEFAULT_THRESHOLDS,
    DriftThresholds,
    compute_drift,
)
from devqubit_engine.compare.results import ComparisonResult, ProgramComparison
from devqubit_engine.compare.types import ProgramMatchMode
from devqubit_engine.config import Config, get_config
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.types import (
    ArtifactRef,
    ObjectStoreProtocol,
    RegistryProtocol,
)
from devqubit_engine.tracking.record import RunRecord, resolve_run_id
from devqubit_engine.uec.api.resolve import resolve_envelope
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.utils.distributions import (
    NoiseContext,
    compute_noise_context,
    compute_noise_context_max,
    normalize_counts,
    total_variation_distance,
)


logger = logging.getLogger(__name__)

# Tolerance for TVD comparison (floating point precision)
_TVD_TOLERANCE = 1e-12


# =============================================================================
# Internal comparison helpers
# =============================================================================


def _num_equal(a: Any, b: Any, tolerance: float) -> bool:
    """
    Compare two values with numeric tolerance.

    Parameters
    ----------
    a : Any
        First value.
    b : Any
        Second value.
    tolerance : float
        Numeric tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if values are considered equal.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b

    if isinstance(a, Real) and isinstance(b, Real):
        af, bf = float(a), float(b)

        if math.isnan(af) and math.isnan(bf):
            return True

        if math.isnan(af) or math.isnan(bf):
            return False

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

    Parameters
    ----------
    dict_a : dict
        First dictionary.
    dict_b : dict
        Second dictionary.
    tolerance : float, default=1e-9
        Numeric tolerance for value comparison.

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


# =============================================================================
# Program comparison
# =============================================================================


@dataclass(frozen=True, slots=True)
class _ProgramView:
    """
    Read-only view of comparison-relevant fields from an envelope's program.

    Parameters
    ----------
    digests : list of str
        Sorted unique artifact digests (logical + physical).
    structural_hash : str or None
        Structural hash from program snapshot.
    parametric_hash : str or None
        Parametric hash from program snapshot.
    exec_structural_hash : str or None
        Executed structural hash (physical circuit).
    exec_parametric_hash : str or None
        Executed parametric hash (physical circuit).
    is_manual : bool
        Whether this envelope is a manual run.
    """

    digests: list[str]
    structural_hash: str | None
    parametric_hash: str | None
    exec_structural_hash: str | None
    exec_parametric_hash: str | None
    is_manual: bool

    @classmethod
    def from_envelope(cls, envelope: ExecutionEnvelope) -> _ProgramView:
        """
        Extract a program view from an envelope.

        Parameters
        ----------
        envelope : ExecutionEnvelope
            Source envelope.

        Returns
        -------
        _ProgramView
            Extracted program view.
        """
        digests: list[str] = []
        structural_hash: str | None = None
        parametric_hash: str | None = None
        exec_structural_hash: str | None = None
        exec_parametric_hash: str | None = None

        if envelope.program:
            logical_digests = [a.ref.digest for a in envelope.program.logical if a.ref]
            physical_digests = [
                a.ref.digest for a in envelope.program.physical if a.ref
            ]
            digests = sorted(set(logical_digests + physical_digests))
            structural_hash = envelope.program.structural_hash
            parametric_hash = envelope.program.parametric_hash
            exec_structural_hash = envelope.program.executed_structural_hash
            exec_parametric_hash = envelope.program.executed_parametric_hash

        return cls(
            digests=digests,
            structural_hash=structural_hash,
            parametric_hash=parametric_hash,
            exec_structural_hash=exec_structural_hash,
            exec_parametric_hash=exec_parametric_hash,
            is_manual=bool(envelope.metadata.get("manual_run", False)),
        )


def _both_present_and_equal(a: str | None, b: str | None) -> bool:
    """Return True if both values are non-None and equal.

    Parameters
    ----------
    a : str or None
        First value.
    b : str or None
        Second value.

    Returns
    -------
    bool
        True only if both are truthy and equal.
    """
    return bool(a) and bool(b) and a == b


def _compare_programs(
    envelope_a: ExecutionEnvelope,
    envelope_b: ExecutionEnvelope,
) -> ProgramComparison:
    """
    Compare program artifacts between two envelopes.

    Uses envelope program refs for exact matching and structural/parametric
    hashes for semantic matching.

    Parameters
    ----------
    envelope_a : ExecutionEnvelope
        Baseline envelope.
    envelope_b : ExecutionEnvelope
        Candidate envelope.

    Returns
    -------
    ProgramComparison
        Detailed comparison of program artifacts and hashes.
    """
    va = _ProgramView.from_envelope(envelope_a)
    vb = _ProgramView.from_envelope(envelope_b)

    has_programs = bool(va.digests) or bool(vb.digests)
    exact_match = va.digests == vb.digests

    # Hash availability: programs exist, and hashes present for non-manual runs
    hash_available = has_programs
    if (va.is_manual or vb.is_manual) and (
        not va.structural_hash or not vb.structural_hash
    ):
        hash_available = False

    # Structural match: same circuit structure (ignores parameter values)
    structural_match = exact_match
    if va.structural_hash and vb.structural_hash:
        structural_match = va.structural_hash == vb.structural_hash

    # Parametric match: same structure AND same parameter values
    parametric_match = (
        _both_present_and_equal(va.parametric_hash, vb.parametric_hash) or exact_match
    )

    # Executed hashes match (for physical circuits)
    executed_structural_match = _both_present_and_equal(
        va.exec_structural_hash, vb.exec_structural_hash
    )
    executed_parametric_match = _both_present_and_equal(
        va.exec_parametric_hash, vb.exec_parametric_hash
    )

    if structural_match and not parametric_match:
        logger.debug(
            "Programs match in structure but differ in params "
            "(structural_hash=%s, parametric_hash_a=%s, parametric_hash_b=%s)",
            va.structural_hash,
            va.parametric_hash,
            vb.parametric_hash,
        )

    return ProgramComparison(
        has_programs=has_programs,
        exact_match=exact_match,
        structural_match=structural_match,
        parametric_match=parametric_match,
        digests_a=va.digests,
        digests_b=vb.digests,
        circuit_hash_a=va.structural_hash,
        circuit_hash_b=vb.structural_hash,
        parametric_hash_a=va.parametric_hash,
        parametric_hash_b=vb.parametric_hash,
        executed_structural_hash_a=va.exec_structural_hash,
        executed_structural_hash_b=vb.exec_structural_hash,
        executed_parametric_hash_a=va.exec_parametric_hash,
        executed_parametric_hash_b=vb.exec_parametric_hash,
        executed_structural_match=executed_structural_match,
        executed_parametric_match=executed_parametric_match,
        hash_available=hash_available,
    )


# =============================================================================
# TVD computation helpers
# =============================================================================


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

    Parameters
    ----------
    counts_a : dict
        Baseline measurement counts.
    counts_b : dict
        Candidate measurement counts.
    include_noise_context : bool
        Whether to compute noise context.
    noise_alpha : float
        Quantile level for noise threshold.
    noise_n_boot : int
        Number of bootstrap iterations.
    noise_seed : int
        Random seed for reproducibility.

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
    """
    Compute TVD for a single item index.

    Parameters
    ----------
    result : ComparisonResult
        Result object to populate (modified in-place).
    envelope_a : ExecutionEnvelope
        Baseline envelope.
    envelope_b : ExecutionEnvelope
        Candidate envelope.
    item_index : int
        Index of result item to compare.
    include_noise_context : bool
        Whether to compute noise context.
    noise_alpha : float
        Quantile level for noise threshold.
    noise_n_boot : int
        Number of bootstrap iterations.
    noise_seed : int
        Random seed for reproducibility.
    """
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

    Items are matched by their ``item_index`` (not list position) to avoid
    silent mispairing when items are skipped or reordered. Requires that
    both envelopes have the same set of item indices.

    Keeps counts_a/b consistent with the item that produced max TVD.

    Parameters
    ----------
    result : ComparisonResult
        Result object to populate (modified in-place).
    envelope_a : ExecutionEnvelope
        Baseline envelope.
    envelope_b : ExecutionEnvelope
        Candidate envelope.
    include_noise_context : bool
        Whether to compute noise context.
    noise_alpha : float
        Quantile level for noise threshold.
    noise_n_boot : int
        Number of bootstrap iterations.
    noise_seed : int
        Random seed for reproducibility.
    """
    all_counts_a = get_all_counts_from_envelope(envelope_a)
    all_counts_b = get_all_counts_from_envelope(envelope_b)

    if not all_counts_a or not all_counts_b:
        result.tvd = None
        return

    # Build maps keyed by item_index for correct pairing
    map_a = {idx: c for idx, c in all_counts_a}
    map_b = {idx: c for idx, c in all_counts_b}

    idx_a = set(map_a.keys())
    idx_b = set(map_b.keys())

    if idx_a != idx_b:
        missing_in_b = sorted(idx_a - idx_b)
        missing_in_a = sorted(idx_b - idx_a)
        result.warnings.append(
            "Batch items mismatch (by item_index). "
            f"missing_in_baseline={missing_in_a}, "
            f"missing_in_candidate={missing_in_b}. "
            "TVD comparison skipped."
        )
        result.tvd = None
        return

    indices = sorted(idx_a)
    batch_size = len(indices)

    worst_tvd: float | None = None
    worst_counts_a: dict[str, int] | None = None
    worst_counts_b: dict[str, int] | None = None
    worst_item_idx: int | None = None

    for idx in indices:
        ca = map_a[idx]
        cb = map_b[idx]

        probs_a = normalize_counts(ca)
        probs_b = normalize_counts(cb)
        tvd = total_variation_distance(probs_a, probs_b)

        if worst_tvd is None or tvd >= worst_tvd:
            worst_tvd = tvd
            worst_counts_a = ca
            worst_counts_b = cb
            worst_item_idx = idx

    result.tvd = worst_tvd
    result.counts_a = worst_counts_a
    result.counts_b = worst_counts_b
    result.tvd_item_index = worst_item_idx
    result.tvd_batch_size = batch_size
    result.tvd_aggregation = "max"

    # Noise context for max-TVD statistic (FWER-correct bootstrap)
    if include_noise_context and worst_tvd is not None:
        pairs = [(map_a[idx], map_b[idx]) for idx in indices]
        result.noise_context = compute_noise_context_max(
            pairs,
            worst_tvd,
            n_boot=noise_n_boot,
            alpha=noise_alpha,
            seed=noise_seed,
        )

    if worst_item_idx is not None and batch_size > 1:
        logger.debug(
            "Worst TVD %.6f found at item_index %d (of %d items)",
            worst_tvd or 0.0,
            worst_item_idx,
            batch_size,
        )


# =============================================================================
# Core: Envelope-only comparison
# =============================================================================


def _match_metadata(
    val_a: str | None,
    val_b: str | None,
    label: str,
    warnings: list[str],
) -> bool | None:
    """
    Compare optional metadata values, emitting a warning on partial data.

    Parameters
    ----------
    val_a : str or None
        Baseline value.
    val_b : str or None
        Candidate value.
    label : str
        Human-readable label for warning messages.
    warnings : list of str
        Warning accumulator (appended in-place).

    Returns
    -------
    bool or None
        True if both present and equal, False if mismatch or partial,
        None if both absent.
    """
    if val_a and val_b:
        return val_a == val_b
    if val_a or val_b:
        warnings.append(
            f"{label} metadata incomplete: baseline={val_a!r}, " f"candidate={val_b!r}"
        )
        return False
    return None


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
    """Compare two run contexts comprehensively.

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

    # Metadata comparison
    project_match = _match_metadata(project_a, project_b, "Project", result.warnings)
    backend_match = _match_metadata(backend_a, backend_b, "Backend", result.warnings)

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
        "Comparison: params_match=%s, metrics_match=%s, "
        "program_exact=%s, program_structural=%s",
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

    # Determine overall identity
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
    """Compare two run records comprehensively.

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
    envelope_a = resolve_envelope(run_a, store_a)
    envelope_b = resolve_envelope(run_b, store_b)

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
    """Context manager for loading run records from bundles.

    Parameters
    ----------
    path : Path
        Path to the bundle zip file.
    """

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
        self._record.mark_finalized()

        return self._record, self._bundle.store

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close bundle."""
        if self._bundle is not None:
            self._bundle.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def load_from_bundle(path: Path) -> Iterator[tuple[RunRecord, ObjectStoreProtocol]]:
    """Load run record and store from a bundle file.

    Parameters
    ----------
    path : Path
        Path to the bundle zip file.

    Yields
    ------
    tuple
        (RunRecord, ObjectStoreProtocol) from the bundle.
    """
    with _BundleContext(path) as result:
        yield result


# =============================================================================
# High-level API: diff by reference (run_id, path, or bundle)
# =============================================================================


def diff(
    ref_a: str | Path,
    ref_b: str | Path,
    *,
    project: str | None = None,
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
    """Compare two runs or bundles by reference.

    Accepts run IDs, run names (with project), or bundle file paths
    and loads the appropriate records and stores automatically.

    Parameters
    ----------
    ref_a : str or Path
        Baseline run ID, run name, or bundle path.
    ref_b : str or Path
        Candidate run ID, run name, or bundle path.
    project : str, optional
        Project name. Required when using run names instead of IDs.
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

    def ensure_cfg() -> Config:
        nonlocal _cfg
        if _cfg is None:
            _cfg = get_config()
        return _cfg

    def ensure_registry() -> RegistryProtocol:
        nonlocal _registry
        if _registry is None:
            _registry = create_registry(config=ensure_cfg())
        return _registry

    def ensure_store() -> ObjectStoreProtocol:
        nonlocal _store
        if _store is None:
            _store = create_store(config=ensure_cfg())
        return _store

    def _load_ref(
        ref: str | Path,
        label: str,
    ) -> tuple[RunRecord, ObjectStoreProtocol]:
        """Load run record from bundle path or registry.

        Parameters
        ----------
        ref : str or Path
            Run ID, run name, or bundle path.
        label : str
            Human-readable label for log messages.

        Returns
        -------
        tuple
            (RunRecord, ObjectStoreProtocol) for the run.
        """
        if is_bundle_path(ref):
            logger.debug("Loading %s from bundle: %s", label, ref)
            ctx = _BundleContext(Path(ref))
            bundle_contexts.append(ctx)
            return ctx.__enter__()

        run_id = resolve_run_id(str(ref), project, ensure_registry())
        logger.debug("Loading %s from registry: %s", label, run_id)
        return ensure_registry().load(run_id), ensure_store()

    try:
        run_a, store_a = _load_ref(ref_a, "baseline")
        run_b, store_b = _load_ref(ref_b, "candidate")

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
            except Exception as exc:
                logger.debug("Failed to close bundle context: %s", exc)
