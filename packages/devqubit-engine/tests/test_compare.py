# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for run comparison, distributions, and drift detection."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.compare.diff import diff_runs
from devqubit_engine.compare.drift import compute_drift
from devqubit_engine.compare.results import (
    ProgramComparison,
    ProgramMatchMode,
)
from devqubit_engine.core.snapshot import DeviceCalibration
from devqubit_engine.utils.distributions import (
    compute_noise_context,
    counts_to_arrays,
    normalize_counts,
    total_variation_distance,
    tvd_from_counts,
)


class TestTVDCalculation:
    """Core TVD calculation tests."""

    def test_identical_distributions_zero_tvd(self):
        """Identical distributions have TVD of 0."""
        p = {"00": 0.5, "11": 0.5}
        assert total_variation_distance(p, p) == 0.0

    def test_disjoint_distributions_max_tvd(self):
        """Completely disjoint distributions have TVD of 1."""
        assert total_variation_distance({"00": 1.0}, {"11": 1.0}) == 1.0

    def test_tvd_from_counts_basic(self):
        """TVD computed correctly from raw counts."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 400, "11": 600}
        assert tvd_from_counts(counts_a, counts_b) == pytest.approx(0.1)

    def test_counts_to_arrays_fills_missing(self):
        """Missing outcomes filled with zero."""
        p_a, p_b, keys = counts_to_arrays({"00": 1000}, {"11": 1000})

        assert keys == ["00", "11"]
        assert p_a[0] == pytest.approx(1.0)  # 00 in A
        assert p_b[1] == pytest.approx(1.0)  # 11 in B

    def test_normalize_counts_basic(self):
        """Counts normalized to probabilities."""
        probs = normalize_counts({"00": 300, "11": 700})
        assert probs["00"] == pytest.approx(0.3)
        assert probs["11"] == pytest.approx(0.7)

    def test_normalize_empty_returns_empty(self):
        """Empty/zero counts return empty dict."""
        assert normalize_counts({}) == {}
        assert normalize_counts({"00": 0}) == {}


class TestBootstrapNoiseContext:
    """Bootstrap-calibrated noise estimation for CI decisions."""

    def test_small_difference_consistent_with_noise(self):
        """Small TVD is classified as noise."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 495, "11": 505}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=500)

        assert not ctx.exceeds_noise
        assert "consistent" in ctx.interpretation().lower()

    def test_large_difference_exceeds_noise(self):
        """Large TVD is classified as real difference."""
        counts_a = {"00": 900, "11": 100}
        counts_b = {"00": 100, "11": 900}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=500)

        assert ctx.exceeds_noise
        assert "exceeds" in ctx.interpretation().lower()

    def test_pvalue_never_zero(self):
        """P-value with +1 correction is never exactly 0."""
        # Extreme case: completely disjoint
        counts_a = {"00": 1000}
        counts_b = {"11": 1000}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=100)

        assert ctx.p_value is not None
        assert ctx.p_value > 0  # +1 correction ensures this

    def test_noise_p95_in_valid_range(self):
        """noise_p95 is always in [0, 1]."""
        test_cases = [
            ({"0": 10}, {"1": 10}),  # Few shots
            ({"00": 1000, "11": 1000}, {"00": 999, "11": 1001}),  # Many shots
        ]

        for counts_a, counts_b in test_cases:
            ctx = compute_noise_context(counts_a, counts_b, n_boot=100)
            assert 0 <= ctx.noise_p95 <= 1.0

    def test_reproducibility_with_seed(self):
        """Same seed gives same results."""
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}

        ctx1 = compute_noise_context(
            counts_a,
            counts_b,
            n_boot=100,
            seed=42,
        )
        ctx2 = compute_noise_context(
            counts_a,
            counts_b,
            n_boot=100,
            seed=42,
        )

        assert ctx1.noise_p95 == ctx2.noise_p95
        assert ctx1.p_value == ctx2.p_value

    def test_to_dict_has_required_fields(self):
        """Serialization includes all decision-relevant fields."""
        ctx = compute_noise_context(
            {"00": 500, "11": 500},
            {"00": 480, "11": 520},
            n_boot=100,
        )
        d = ctx.to_dict()

        # Fields needed for CI decisions
        assert "tvd" in d
        assert "noise_p95" in d
        assert "p_value" in d
        assert "exceeds_noise" in d
        assert "method" in d
        assert d["method"] == "bootstrap"


class TestProgramMatching:
    """Program comparison with exact vs structural matching."""

    def test_exact_match_requires_identical_digests(self):
        """EXACT mode requires byte-identical artifacts."""
        comp = ProgramComparison(exact_match=True, structural_match=True)

        assert comp.matches(ProgramMatchMode.EXACT)
        assert comp.matches(ProgramMatchMode.STRUCTURAL)
        assert comp.matches(ProgramMatchMode.EITHER)

    def test_structural_only_match(self):
        """STRUCTURAL mode allows different params, same structure."""
        comp = ProgramComparison(exact_match=False, structural_match=True)

        assert not comp.matches(ProgramMatchMode.EXACT)
        assert comp.matches(ProgramMatchMode.STRUCTURAL)
        assert comp.matches(ProgramMatchMode.EITHER)
        assert comp.structural_only_match  # Flag for VQE detection

    def test_no_match_fails_all_modes(self):
        """No match fails all modes."""
        comp = ProgramComparison(exact_match=False, structural_match=False)

        assert not comp.matches(ProgramMatchMode.EXACT)
        assert not comp.matches(ProgramMatchMode.STRUCTURAL)
        assert not comp.matches(ProgramMatchMode.EITHER)


class TestDriftDetection:
    """Device calibration drift detection."""

    def test_no_drift_for_identical_calibration(self, snapshot_factory):
        """Identical calibrations show no drift."""
        cal = DeviceCalibration(median_t1_us=100.0, median_t2_us=50.0)
        snap = snapshot_factory(calibration=cal)

        result = compute_drift(snap, snap)

        assert not result.significant_drift

    def test_significant_drift_detected(self, snapshot_factory):
        """Large calibration changes are flagged."""
        cal_a = DeviceCalibration(median_t1_us=100.0)
        cal_b = DeviceCalibration(median_t1_us=80.0)  # 20% change

        result = compute_drift(
            snapshot_factory(calibration=cal_a),
            snapshot_factory(calibration=cal_b),
        )

        assert result.significant_drift
        t1_drift = next(m for m in result.metrics if "t1" in m.metric)
        assert t1_drift.significant

    def test_top_drifts_sorted(self, snapshot_factory):
        """top_drifts returns sorted by magnitude."""
        cal_a = DeviceCalibration(median_t1_us=100.0, median_t2_us=50.0)
        cal_b = DeviceCalibration(median_t1_us=85.0, median_t2_us=30.0)  # 15% vs 40%

        result = compute_drift(
            snapshot_factory(calibration=cal_a),
            snapshot_factory(calibration=cal_b),
        )

        top = result.top_drifts
        if len(top) >= 2:
            assert abs(top[0].percent_change) >= abs(top[1].percent_change)


class TestDiffRuns:
    """Integration tests for diff_runs."""

    def test_identical_run_is_identical(self, store, run_factory):
        """Same run compared to itself is identical."""
        run = run_factory(run_id="RUN_SAME")

        result = diff_runs(run, run, store_a=store, store_b=store)

        assert result.identical
        assert result.params["match"]
        assert result.program.matches(ProgramMatchMode.EITHER)

    def test_param_differences_detected(self, store, run_factory):
        """Parameter changes are detected."""
        a = run_factory(run_id="A", params={"shots": 1000})
        b = run_factory(run_id="B", params={"shots": 2000})

        result = diff_runs(a, b, store_a=store, store_b=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]

    def test_tvd_computed_from_counts(
        self,
        store,
        run_factory,
        counts_artifact_factory,
    ):
        """TVD computed when count artifacts present."""
        art_a = counts_artifact_factory({"00": 500, "11": 500})
        art_b = counts_artifact_factory({"00": 400, "11": 600})

        a = run_factory(run_id="A", artifacts=[art_a])
        b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(a, b, store_a=store, store_b=store)

        assert result.tvd == pytest.approx(0.1)
        assert result.noise_context is not None


class TestSerialization:
    """Result serialization for API responses and CI."""

    def test_comparison_result_serializable(
        self,
        store,
        run_factory,
        counts_artifact_factory,
    ):
        """ComparisonResult can be JSON serialized."""
        artifact = counts_artifact_factory({"00": 500, "11": 500})
        a = run_factory(run_id="A", artifacts=[artifact])
        b = run_factory(run_id="B", artifacts=[artifact])

        result = diff_runs(a, b, store_a=store, store_b=store)

        # Should not raise
        json_str = json.dumps(result.to_dict(), default=str)
        parsed = json.loads(json_str)

        assert parsed["run_a"] == "A"
        assert "noise_context" in parsed

    def test_noise_context_in_serialization(
        self, store, run_factory, counts_artifact_factory
    ):
        """noise_context included in serialized result."""
        artifact = counts_artifact_factory({"00": 500, "11": 500})
        a = run_factory(run_id="A", artifacts=[artifact])
        b = run_factory(run_id="B", artifacts=[artifact])

        result = diff_runs(a, b, store_a=store, store_b=store)
        d = result.to_dict()

        assert "noise_context" in d
        assert "noise_p95" in d["noise_context"]
        assert "p_value" in d["noise_context"]
