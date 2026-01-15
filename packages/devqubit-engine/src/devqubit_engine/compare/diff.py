# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run comparison with drift detection.

This module provides comprehensive comparison of quantum experiment runs,
including parameter comparison, metrics comparison, program artifact comparison,
device calibration drift analysis, result distribution comparison (TVD),
sampling noise context, and circuit semantic comparison.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Iterator

from devqubit_engine.bundle.reader import Bundle, is_bundle_path
from devqubit_engine.circuit.extractors import extract_circuit
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
from devqubit_engine.core.config import Config, get_config
from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.artifacts.counts import get_counts
from devqubit_engine.storage.artifacts.lookup import get_artifact_digests
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.types import (
    ArtifactRef,
    ObjectStoreProtocol,
    RegistryProtocol,
)
from devqubit_engine.uec.models.calibration import DeviceCalibration
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.result import canonicalize_bitstrings
from devqubit_engine.uec.resolution.resolve import resolve_envelope
from devqubit_engine.utils.distributions import (
    compute_noise_context,
    normalize_counts,
    total_variation_distance,
)


logger = logging.getLogger(__name__)


def _num_equal(a: Any, b: Any, tolerance: float) -> bool:
    """
    Compare two values with numeric tolerance.

    Handles bool, Real (including numpy types, Decimal), NaN, and Inf correctly.

    Parameters
    ----------
    a : Any
        First value.
    b : Any
        Second value.
    tolerance : float
        Tolerance for float comparison.

    Returns
    -------
    bool
        True if values are equal within tolerance.
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

    Parameters
    ----------
    dict_a : dict
        Baseline dictionary.
    dict_b : dict
        Candidate dictionary.
    tolerance : float
        Tolerance for float comparison.

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


def _extract_params(record: RunRecord) -> dict[str, Any]:
    """Extract parameters from run record."""
    data = record.record.get("data") or {}
    if isinstance(data, dict):
        return data.get("params", {}) or {}
    return {}


def _extract_metrics(record: RunRecord) -> dict[str, Any]:
    """Extract metrics from run record."""
    # Try direct metrics attribute
    if hasattr(record, "metrics") and record.metrics:
        return dict(record.metrics)

    # Try record dict
    record_data = record.record if hasattr(record, "record") else {}
    if isinstance(record_data, dict):
        if "metrics" in record_data and isinstance(record_data["metrics"], dict):
            return dict(record_data["metrics"])
        # Try data.metrics path
        data = record_data.get("data", {})
        if isinstance(data, dict) and "metrics" in data:
            return dict(data["metrics"])

    return {}


def _extract_counts(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
    *,
    canonicalize: bool = True,
) -> dict[str, int] | None:
    """
    Extract counts from envelope or Run artifacts.

    Uses UEC-first strategy: extracts counts from ExecutionEnvelope,
    falling back to Run counts artifact only for synthesized (manual)
    envelopes. For adapter envelopes, missing counts is reported as None.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.
    canonicalize : bool, default=True
        Whether to canonicalize bitstrings to cbit0_right format.

    Returns
    -------
    dict or None
        Counts as {bitstring: count} in canonical format, or None if not available.
    """
    # Use provided envelope or resolve
    if envelope is None:
        envelope = resolve_envelope(record, store)

    # Try to get counts from envelope (UEC-first)
    if envelope.result.items:
        item = envelope.result.items[0]
        if item.counts:
            raw_counts = item.counts.get("counts")
            if isinstance(raw_counts, dict):
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

    # Fallback: Run counts artifact - ONLY for synthesized (manual) envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        # Adapter envelope without counts - this is an integration issue
        logger.debug(
            "Adapter envelope for run %s has no counts in result items",
            record.run_id,
        )
        return None

    # Manual/synthesized envelope - fallback to Run artifact is allowed
    counts_info = get_counts(record, store)
    if counts_info is None:
        return None

    # Canonicalize fallback counts (assume cbit0_right if not specified)
    if canonicalize:
        return canonicalize_bitstrings(
            counts_info.counts,
            bit_order="cbit0_right",
            transformed=False,
        )
    return counts_info.counts


def _load_device_snapshot(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
) -> DeviceSnapshot | None:
    """
    Load device snapshot from envelope or run record.

    Uses UEC-first strategy: extracts device from ExecutionEnvelope,
    falling back to Run record metadata only for synthesized (manual)
    envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record to extract device snapshot from.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None, will be resolved internally.

    Returns
    -------
    DeviceSnapshot or None
        Device snapshot if available, None otherwise.

    Notes
    -----
    Fallback to Run record metadata is only allowed for synthesized
    (manual) envelopes. For adapter envelopes, if device is not present
    in the envelope, None is returned.
    """
    # Use provided envelope or resolve
    if envelope is None:
        envelope = resolve_envelope(record, store)

    # Get device from envelope (UEC-first)
    if envelope.device is not None:
        return envelope.device

    # Fallback: construct from record metadata - ONLY for synthesized envelopes
    is_synthesized = envelope.metadata.get("synthesized_from_run", False)

    if not is_synthesized:
        # Adapter envelope without device - this may be intentional (no device info)
        logger.debug(
            "Adapter envelope for run %s has no device snapshot",
            record.run_id,
        )
        return None

    # Manual/synthesized envelope - fallback to Run record is allowed
    backend = record.record.get("backend") or {}
    if not isinstance(backend, dict):
        return None

    snapshot_summary = record.record.get("device_snapshot") or {}
    if not isinstance(snapshot_summary, dict):
        snapshot_summary = {}

    calibration = None
    cal_data = snapshot_summary.get("calibration")
    if isinstance(cal_data, dict):
        try:
            calibration = DeviceCalibration.from_dict(cal_data)
        except Exception as e:
            logger.debug("Failed to parse calibration data: %s", e)

    try:
        return DeviceSnapshot(
            captured_at=snapshot_summary.get("captured_at", record.created_at),
            backend_name=backend.get("name", ""),
            backend_type=backend.get("type", ""),
            provider=backend.get("provider", ""),
            num_qubits=snapshot_summary.get("num_qubits"),
            connectivity=snapshot_summary.get("connectivity"),
            native_gates=snapshot_summary.get("native_gates"),
            calibration=calibration,
        )
    except Exception:
        return None


def _extract_circuit_summary(
    record: RunRecord,
    store: ObjectStoreProtocol,
    envelope: ExecutionEnvelope | None = None,
    *,
    which: str = "logical",
) -> CircuitSummary | None:
    """
    Extract circuit summary from envelope or run record.

    Uses UEC-first strategy: extracts circuit from envelope refs, falling
    back to run record artifacts only for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record for fallback extraction.
    store : ObjectStoreProtocol
        Object store for loading artifacts.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope.
    which : str, default="logical"
        Which circuit to extract: "logical" or "physical".

    Returns
    -------
    CircuitSummary or None
        Extracted circuit summary, or None if not found.
    """
    from devqubit_engine.circuit.extractors import extract_circuit_from_envelope

    circuit_data = None

    # Try envelope first (UEC-first)
    if envelope is not None:
        circuit_data = extract_circuit_from_envelope(envelope, store, which=which)

    # Fallback to Run artifacts - ONLY for synthesized (manual) envelopes
    if circuit_data is None:
        is_synthesized = envelope is not None and envelope.metadata.get(
            "synthesized_from_run", False
        )

        if is_synthesized:
            # Manual envelope - fallback is allowed
            circuit_data = extract_circuit(record, store)
        elif envelope is not None:
            # Adapter envelope without circuit refs - log and return None
            logger.debug(
                "Adapter envelope for run %s has no circuit refs for '%s'",
                record.run_id,
                which,
            )

    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)

    return None


def _compare_programs(
    run_a: RunRecord,
    run_b: RunRecord,
    envelope_a: ExecutionEnvelope | None = None,
    envelope_b: ExecutionEnvelope | None = None,
) -> ProgramComparison:
    """
    Compare program artifacts between two runs.

    Uses UEC-first strategy for structural and parametric hash comparison.
    Computes exact (digest), structural (structural_hash), and parametric
    (parametric_hash) matching.

    Parameters
    ----------
    run_a : RunRecord
        Baseline run.
    run_b : RunRecord
        Candidate run.
    envelope_a : ExecutionEnvelope, optional
        Pre-resolved envelope for run_a.
    envelope_b : ExecutionEnvelope, optional
        Pre-resolved envelope for run_b.

    Returns
    -------
    ProgramComparison
        Detailed comparison with status, exact_match, structural_match,
        parametric_match, and hash availability.
    """
    # Exact match: prefer envelope refs if available (ignores auxiliary artifacts
    # like diagrams), else fall back to all role=program artifacts
    if envelope_a and envelope_a.program:
        logical_digests_a = [a.ref.digest for a in envelope_a.program.logical if a.ref]
        physical_digests_a = [
            a.ref.digest for a in envelope_a.program.physical if a.ref
        ]
        digests_a = sorted(set(logical_digests_a + physical_digests_a))
    else:
        digests_a = get_artifact_digests(run_a, role="program")

    if envelope_b and envelope_b.program:
        logical_digests_b = [a.ref.digest for a in envelope_b.program.logical if a.ref]
        physical_digests_b = [
            a.ref.digest for a in envelope_b.program.physical if a.ref
        ]
        digests_b = sorted(set(logical_digests_b + physical_digests_b))
    else:
        digests_b = get_artifact_digests(run_b, role="program")

    exact_match = digests_a == digests_b

    # Extract hashes from UEC (structural_hash and parametric_hash)
    hash_a = None
    hash_b = None
    param_hash_a = None
    param_hash_b = None
    hash_available = True

    if envelope_a and envelope_a.program:
        hash_a = envelope_a.program.structural_hash
        param_hash_a = envelope_a.program.parametric_hash
    if envelope_b and envelope_b.program:
        hash_b = envelope_b.program.structural_hash
        param_hash_b = envelope_b.program.parametric_hash

    # Check if hashes are available (manual runs won't have them)
    if (envelope_a and envelope_a.metadata.get("manual_run")) or (
        envelope_b and envelope_b.metadata.get("manual_run")
    ):
        # At least one is manual run
        if not hash_a or not hash_b:
            hash_available = False

    # Structural match: same circuit structure (ignores parameter values)
    structural_match = False
    if hash_a and hash_b:
        structural_match = hash_a == hash_b
    elif exact_match:
        # If exact match, structural also matches
        structural_match = True

    # Parametric match: same structure AND same parameter values
    parametric_match = False
    if param_hash_a and param_hash_b:
        parametric_match = param_hash_a == param_hash_b
    elif exact_match:
        # If exact match, parametric also matches
        parametric_match = True

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
        hash_available=hash_available,
    )


def diff_runs(
    run_a: RunRecord,
    run_b: RunRecord,
    *,
    store_a: ObjectStoreProtocol,
    store_b: ObjectStoreProtocol,
    thresholds: DriftThresholds | None = None,
    include_circuit_diff: bool = True,
    include_noise_context: bool = True,
) -> ComparisonResult:
    """
    Compare two run records comprehensively.

    Uses UEC-first strategy: resolves ExecutionEnvelope for each run
    and performs comparisons through the unified envelope structure.
    This ensures consistent behavior whether runs were created with
    adapters or manually.

    Performs multi-dimensional comparison including metadata, parameters,
    metrics, program artifacts, device calibration drift, and result
    distributions.

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

    Returns
    -------
    ComparisonResult
        Complete comparison result with all analysis dimensions.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    logger.info("Comparing runs: %s vs %s", run_a.run_id, run_b.run_id)

    # Resolve envelopes (UEC-first strategy, strict for adapters)
    envelope_a = resolve_envelope(run_a, store_a)
    envelope_b = resolve_envelope(run_b, store_b)

    result = ComparisonResult(
        run_id_a=run_a.run_id,
        run_id_b=run_b.run_id,
        fingerprint_a=run_a.run_fingerprint,
        fingerprint_b=run_b.run_fingerprint,
    )

    # Metadata comparison
    result.metadata = {
        "project_match": run_a.project == run_b.project,
        "backend_match": run_a.backend_name == run_b.backend_name,
        "project_a": run_a.project,
        "project_b": run_b.project,
        "backend_a": run_a.backend_name,
        "backend_b": run_b.backend_name,
        "envelope_a_synthesized": envelope_a.metadata.get(
            "synthesized_from_run", False
        ),
        "envelope_b_synthesized": envelope_b.metadata.get(
            "synthesized_from_run", False
        ),
    }

    # Parameter comparison
    params_a = _extract_params(run_a)
    params_b = _extract_params(run_b)
    result.params = _diff_dict(params_a, params_b)

    # Metrics comparison
    metrics_a = _extract_metrics(run_a)
    metrics_b = _extract_metrics(run_b)
    result.metrics = _diff_dict(metrics_a, metrics_b)

    # Program comparison (both exact and structural) - uses envelope
    result.program = _compare_programs(run_a, run_b, envelope_a, envelope_b)

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

    # Device drift analysis - uses envelope
    snapshot_a = _load_device_snapshot(run_a, store_a, envelope_a)
    snapshot_b = _load_device_snapshot(run_b, store_b, envelope_b)

    if snapshot_a and snapshot_b:
        result.device_drift = compute_drift(snapshot_a, snapshot_b, thresholds)
        if result.device_drift.significant_drift:
            result.warnings.append(
                "Significant calibration drift detected. "
                "Results may not be directly comparable."
            )

    # Results comparison (TVD) - uses envelope
    result.counts_a = _extract_counts(run_a, store_a, envelope_a)
    result.counts_b = _extract_counts(run_b, store_b, envelope_b)

    if result.counts_a is not None and result.counts_b is not None:
        probs_a = normalize_counts(result.counts_a)
        probs_b = normalize_counts(result.counts_b)
        result.tvd = total_variation_distance(probs_a, probs_b)

        if include_noise_context:
            result.noise_context = compute_noise_context(
                result.counts_a, result.counts_b, result.tvd
            )

        logger.debug("TVD: %.6f", result.tvd)

    # Circuit diff
    if include_circuit_diff:
        summary_a = _extract_circuit_summary(run_a, store_a, envelope_a)
        summary_b = _extract_circuit_summary(run_b, store_b, envelope_b)
        if summary_a and summary_b:
            result.circuit_diff = diff_summaries(summary_a, summary_b)
        elif not result.program.matches(ProgramMatchMode.EITHER):
            result.warnings.append(
                "Programs differ but circuit data not captured for comparison."
            )

    # Determine overall identity
    tvd_match = result.tvd == 0.0 if result.tvd is not None else True
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


def diff(
    ref_a: str | Path,
    ref_b: str | Path,
    *,
    registry: RegistryProtocol | None = None,
    store: ObjectStoreProtocol | None = None,
    thresholds: DriftThresholds | None = None,
    include_circuit_diff: bool = True,
    include_noise_context: bool = True,
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
        )
    finally:
        for ctx in bundle_contexts:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
