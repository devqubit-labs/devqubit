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

from devqubit_engine.artifacts import (
    find_artifact,
    get_artifact_digests,
    get_counts,
    load_json_artifact,
)
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
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.protocols import ObjectStoreProtocol, RegistryProtocol
from devqubit_engine.uec.calibration import DeviceCalibration
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.types import ArtifactRef
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
) -> dict[str, int] | None:
    """Extract counts from run record using artifacts module."""
    counts_info = get_counts(record, store)
    if counts_info is None:
        return None
    return counts_info.counts


def _load_device_snapshot(
    record: RunRecord,
    store: ObjectStoreProtocol,
) -> DeviceSnapshot | None:
    """
    Load device snapshot from run record.

    Loads DeviceSnapshot from ExecutionEnvelope artifact, falling back
    to record metadata if envelope is not available.

    Parameters
    ----------
    record : RunRecord
        Run record to extract device snapshot from.
    store : ObjectStoreProtocol
        Object store for loading artifacts.

    Returns
    -------
    DeviceSnapshot or None
        Device snapshot if available, None otherwise.
    """
    # Load from ExecutionEnvelope
    envelope_artifact = find_artifact(record, kind_contains="envelope")
    if envelope_artifact:
        try:
            envelope_data = load_json_artifact(envelope_artifact, store)
            if isinstance(envelope_data, dict) and "device" in envelope_data:
                device_data = envelope_data["device"]
                if isinstance(device_data, dict):
                    return DeviceSnapshot.from_dict(device_data)
        except Exception as e:
            logger.debug("Failed to load device from envelope: %s", e)

    # Fallback: construct from record metadata
    backend = record.record.get("backend") or {}
    if not isinstance(backend, dict):
        return None

    snapshot_summary = record.record.get("device_snapshot") or {}
    if not isinstance(snapshot_summary, dict):
        return None

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
) -> CircuitSummary | None:
    """Extract circuit summary from run record."""
    circuit_data = extract_circuit(record, store)
    if circuit_data is not None:
        try:
            return summarize_circuit_data(circuit_data)
        except Exception as e:
            logger.debug("Failed to summarize circuit: %s", e)
    return None


def _extract_circuit_hash(record: RunRecord) -> str | None:
    """
    Extract circuit_hash from run record or artifact metadata.

    The circuit_hash is a structural hash that ignores parameter values,
    making it suitable for comparing parameterized circuits that were
    executed with different parameter values.

    Parameters
    ----------
    record : RunRecord
        Run record to extract from.

    Returns
    -------
    str or None
        Circuit hash if available.
    """
    # Try execute metadata
    execute = record.record.get("execute", {})
    if isinstance(execute, dict) and execute.get("circuit_hash"):
        return str(execute["circuit_hash"])

    # Try artifacts metadata
    for artifact in record.artifacts:
        meta = artifact.meta or {}
        if meta.get("circuit_hash"):
            return str(meta["circuit_hash"])

    return None


def _compare_programs(
    run_a: RunRecord,
    run_b: RunRecord,
) -> ProgramComparison:
    """
    Compare program artifacts between two runs.

    Computes both exact (digest) and structural (structural/circuit_hash)
    matching to support different verification policies.

    Parameters
    ----------
    run_a : RunRecord
        Baseline run.
    run_b : RunRecord
        Candidate run.

    Returns
    -------
    ProgramComparison
        Detailed comparison with exact_match, structural_match, and metadata.
    """
    digests_a = get_artifact_digests(run_a, role="program")
    digests_b = get_artifact_digests(run_b, role="program")

    # Exact match on content digests
    exact_match = digests_a == digests_b

    # Extract circuit hashes for structural comparison
    hash_a = _extract_circuit_hash(run_a)
    hash_b = _extract_circuit_hash(run_b)

    # Template match: same circuit structure (ignores parameter values)
    structural_match = False
    if hash_a and hash_b:
        structural_match = hash_a == hash_b
    elif exact_match:
        # If exact match, structural also matches
        structural_match = True

    if structural_match and not exact_match:
        logger.debug(
            "Programs differ in content but match in structure (circuit_hash=%s)",
            hash_a,
        )

    return ProgramComparison(
        exact_match=exact_match,
        structural_match=structural_match,
        digests_a=digests_a,
        digests_b=digests_b,
        circuit_hash_a=hash_a,
        circuit_hash_b=hash_b,
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
    }

    # Parameter comparison
    params_a = _extract_params(run_a)
    params_b = _extract_params(run_b)
    result.params = _diff_dict(params_a, params_b)

    # Metrics comparison
    metrics_a = _extract_metrics(run_a)
    metrics_b = _extract_metrics(run_b)
    result.metrics = _diff_dict(metrics_a, metrics_b)

    # Program comparison (both exact and structural)
    result.program = _compare_programs(run_a, run_b)

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
    snapshot_a = _load_device_snapshot(run_a, store_a)
    snapshot_b = _load_device_snapshot(run_b, store_b)

    if snapshot_a and snapshot_b:
        result.device_drift = compute_drift(snapshot_a, snapshot_b, thresholds)
        if result.device_drift.significant_drift:
            result.warnings.append(
                "Significant calibration drift detected. "
                "Results may not be directly comparable."
            )

    # Results comparison (TVD)
    result.counts_a = _extract_counts(run_a, store_a)
    result.counts_b = _extract_counts(run_b, store_b)

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
        summary_a = _extract_circuit_summary(run_a, store_a)
        summary_b = _extract_circuit_summary(run_b, store_b)
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
