# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for run comparison module."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.compare.diff import diff_runs
from devqubit_engine.compare.drift import DriftThresholds, compute_drift
from devqubit_engine.compare.results import ProgramComparison, ProgramMatchMode
from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.uec.models.calibration import DeviceCalibration
from devqubit_engine.utils.distributions import (
    compute_noise_context,
    normalize_counts,
    total_variation_distance,
    tvd_from_counts,
)


class TestDiffRuns:
    """Integration tests for diff_runs - the main comparison entry point."""

    def test_identical_runs_are_identical(self, store, run_factory):
        """Same run compared to itself should be identical."""
        run = run_factory(
            run_id="RUN_SAME",
            params={"shots": 1000},
            metrics={"fidelity": 0.95},
        )

        result = diff_runs(run, run, store_a=store, store_b=store)

        assert result.identical
        assert result.params["match"]
        assert result.metrics["match"]

    def test_detects_param_changes(self, store, run_factory):
        """Parameter differences are detected and reported."""
        run_a = run_factory(run_id="A", params={"shots": 1000, "opt_level": 2})
        run_b = run_factory(run_id="B", params={"shots": 2000, "opt_level": 2})

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]
        assert result.params["changed"]["shots"] == {"a": 1000, "b": 2000}

    def test_detects_metric_changes(self, store, run_factory):
        """Metric differences are detected."""
        run_a = run_factory(run_id="A", metrics={"fidelity": 0.95})
        run_b = run_factory(run_id="B", metrics={"fidelity": 0.85})

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        assert not result.metrics["match"]
        assert "fidelity" in result.metrics["changed"]

    def test_computes_tvd_from_counts(
        self, store, run_factory, counts_artifact_factory
    ):
        """TVD is computed when count artifacts are present."""
        art_a = counts_artifact_factory({"00": 500, "11": 500})
        art_b = counts_artifact_factory({"00": 400, "11": 600})

        run_a = run_factory(run_id="A", artifacts=[art_a])
        run_b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        assert result.tvd == pytest.approx(0.1)
        assert result.counts_a == {"00": 500, "11": 500}
        assert result.counts_b == {"00": 400, "11": 600}

    def test_includes_noise_context(self, store, run_factory, counts_artifact_factory):
        """Noise context is computed for statistical interpretation."""
        artifact = counts_artifact_factory({"00": 500, "11": 500})
        run_a = run_factory(run_id="A", artifacts=[artifact])
        run_b = run_factory(run_id="B", artifacts=[artifact])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        assert result.noise_context is not None
        assert result.noise_context.noise_p95 >= 0

    def test_result_serializable(self, store, run_factory, counts_artifact_factory):
        """ComparisonResult can be JSON serialized for API responses."""
        artifact = counts_artifact_factory({"00": 500, "11": 500})
        run_a = run_factory(run_id="A", artifacts=[artifact])
        run_b = run_factory(run_id="B", artifacts=[artifact])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        # Should not raise
        json_str = json.dumps(result.to_dict(), default=str)
        parsed = json.loads(json_str)

        assert parsed["run_a"] == "A"
        assert parsed["run_b"] == "B"
        assert "noise_context" in parsed

    def test_runs_without_programs_are_identical(self, store, run_factory):
        """Two runs with no program artifacts should be identical."""
        run = run_factory(
            run_id="NO_PROGRAM",
            params={"shots": 1000},
            # No program artifacts
        )

        result = diff_runs(run, run, store_a=store, store_b=store)

        assert result.identical
        assert result.program.exact_match  # [] == [] is True
        assert result.program.structural_match

    def test_item_index_all_keeps_counts_consistent_with_tvd(self, store, run_factory):
        """item_index='all' should keep counts_a/b from worst TVD item."""
        # Create batch counts artifact with multiple experiments
        batch_a = {
            "experiments": [
                {"counts": {"00": 500, "11": 500}},  # TVD=0 with itself
                {"counts": {"00": 100, "11": 900}},  # Different
            ]
        }
        batch_b = {
            "experiments": [
                {"counts": {"00": 500, "11": 500}},  # TVD=0 (same as A)
                {"counts": {"00": 900, "11": 100}},  # TVD=0.8 with A (worst)
            ]
        }

        digest_a = store.put_bytes(json.dumps(batch_a).encode())
        digest_b = store.put_bytes(json.dumps(batch_b).encode())

        art_a = ArtifactRef(
            kind="result.counts.json",
            digest=digest_a,
            media_type="application/json",
            role="results",
        )
        art_b = ArtifactRef(
            kind="result.counts.json",
            digest=digest_b,
            media_type="application/json",
            role="results",
        )

        run_a = run_factory(run_id="BATCH_A", artifacts=[art_a])
        run_b = run_factory(run_id="BATCH_B", artifacts=[art_b])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store, item_index="all")

        # Worst TVD should be from item 1 (0.8), not item 0 (0.0)
        assert result.tvd == pytest.approx(0.8)
        # Counts should be from item 1, not item 0
        assert result.counts_a == {"00": 100, "11": 900}
        assert result.counts_b == {"00": 900, "11": 100}


class TestTVD:
    """Total Variation Distance calculation tests."""

    def test_identical_distributions_zero_tvd(self):
        """Identical distributions have TVD of 0."""
        p = {"00": 0.5, "11": 0.5}
        assert total_variation_distance(p, p) == 0.0

    def test_disjoint_distributions_max_tvd(self):
        """Completely disjoint distributions have TVD of 1."""
        assert total_variation_distance({"00": 1.0}, {"11": 1.0}) == 1.0

    def test_tvd_from_counts(self):
        """TVD computed correctly from raw counts."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 400, "11": 600}

        tvd = tvd_from_counts(counts_a, counts_b)

        assert tvd == pytest.approx(0.1)

    def test_normalize_counts(self):
        """Counts are normalized to probabilities."""
        probs = normalize_counts({"00": 300, "11": 700})

        assert probs["00"] == pytest.approx(0.3)
        assert probs["11"] == pytest.approx(0.7)

    def test_normalize_empty_returns_empty(self):
        """Empty or zero counts return empty dict."""
        assert normalize_counts({}) == {}
        assert normalize_counts({"00": 0}) == {}


class TestNoiseContext:
    """Bootstrap-calibrated noise estimation for CI decisions."""

    def test_small_difference_consistent_with_noise(self):
        """Small TVD within expected noise range."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 495, "11": 505}  # 1% difference

        ctx = compute_noise_context(counts_a, counts_b, n_boot=200)

        assert not ctx.exceeds_noise
        assert "consistent" in ctx.interpretation().lower()

    def test_large_difference_exceeds_noise(self):
        """Large TVD exceeds noise threshold."""
        counts_a = {"00": 900, "11": 100}
        counts_b = {"00": 100, "11": 900}  # 80% difference

        ctx = compute_noise_context(counts_a, counts_b, n_boot=200)

        assert ctx.exceeds_noise
        assert "exceeds" in ctx.interpretation().lower()

    def test_reproducible_with_seed(self):
        """Same seed gives deterministic results."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}

        ctx1 = compute_noise_context(counts_a, counts_b, n_boot=100, seed=42)
        ctx2 = compute_noise_context(counts_a, counts_b, n_boot=100, seed=42)

        assert ctx1.noise_p95 == ctx2.noise_p95
        assert ctx1.p_value == ctx2.p_value

    def test_serialization_has_required_fields(self):
        """Serialized context includes all CI-relevant fields."""
        ctx = compute_noise_context(
            {"00": 500, "11": 500},
            {"00": 480, "11": 520},
            n_boot=100,
        )
        d = ctx.to_dict()

        assert "tvd" in d
        assert "noise_p95" in d
        assert "p_value" in d
        assert "exceeds_noise" in d
        assert d["method"] == "bootstrap"


class TestDriftDetection:
    """Device calibration drift detection."""

    def test_no_drift_for_identical_calibration(self, snapshot_factory):
        """Identical calibrations show no drift."""
        cal = DeviceCalibration(median_t1_us=100.0, median_t2_us=50.0)
        snap = snapshot_factory(calibration=cal)

        result = compute_drift(snap, snap)

        assert not result.significant_drift

    def test_detects_significant_t1_drift(self, snapshot_factory):
        """T1 change exceeding threshold is flagged."""
        cal_a = DeviceCalibration(median_t1_us=100.0)
        cal_b = DeviceCalibration(median_t1_us=80.0)  # 20% drop

        result = compute_drift(
            snapshot_factory(calibration=cal_a),
            snapshot_factory(calibration=cal_b),
        )

        assert result.significant_drift
        t1_drift = next(m for m in result.metrics if "t1" in m.metric)
        assert t1_drift.significant
        assert t1_drift.percent_change == pytest.approx(20.0)

    def test_custom_thresholds(self, snapshot_factory):
        """Custom thresholds override defaults."""
        cal_a = DeviceCalibration(median_t1_us=100.0)
        cal_b = DeviceCalibration(median_t1_us=95.0)  # 5% drop

        # Default threshold (10%) - not significant
        result_default = compute_drift(
            snapshot_factory(calibration=cal_a),
            snapshot_factory(calibration=cal_b),
        )
        assert not result_default.significant_drift

        # Stricter threshold (3%) - significant
        strict = DriftThresholds(t1_us=0.03)
        result_strict = compute_drift(
            snapshot_factory(calibration=cal_a),
            snapshot_factory(calibration=cal_b),
            thresholds=strict,
        )
        assert result_strict.significant_drift

    def test_no_calibration_data_no_drift(self, snapshot_factory):
        """Missing calibration data doesn't flag drift."""
        snap_a = snapshot_factory(calibration=None)
        snap_b = snapshot_factory(calibration=None)

        result = compute_drift(snap_a, snap_b)

        assert not result.significant_drift
        assert not result.has_calibration_data


class TestProgramMatching:
    """Program comparison with exact vs structural matching."""

    def test_exact_match(self):
        """EXACT mode requires identical artifacts."""
        comp = ProgramComparison(
            exact_match=True,
            structural_match=True,
            has_programs=True,
        )

        assert comp.matches(ProgramMatchMode.EXACT)
        assert comp.matches(ProgramMatchMode.STRUCTURAL)
        assert comp.matches(ProgramMatchMode.EITHER)

    def test_structural_only_match(self):
        """Structural match without exact match (VQE scenario)."""
        comp = ProgramComparison(
            exact_match=False,
            structural_match=True,
            has_programs=True,
        )

        assert not comp.matches(ProgramMatchMode.EXACT)
        assert comp.matches(ProgramMatchMode.STRUCTURAL)
        assert comp.matches(ProgramMatchMode.EITHER)
        assert comp.structural_only_match  # Flag for parameter variation

    def test_no_match(self):
        """No match fails all modes."""
        comp = ProgramComparison(
            exact_match=False,
            structural_match=False,
            has_programs=True,
        )

        assert not comp.matches(ProgramMatchMode.EXACT)
        assert not comp.matches(ProgramMatchMode.STRUCTURAL)
        assert not comp.matches(ProgramMatchMode.EITHER)

    def test_empty_programs_match(self):
        """Two runs with no programs should match."""
        comp = ProgramComparison(
            exact_match=True,  # [] == []
            structural_match=True,
            digests_a=[],
            digests_b=[],
            has_programs=False,
        )

        assert comp.matches(ProgramMatchMode.EXACT)
        assert comp.matches(ProgramMatchMode.EITHER)
