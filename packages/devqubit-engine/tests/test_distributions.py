# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for bootstrap noise estimation in distributions module."""

from devqubit_engine.utils.distributions import compute_noise_context


class TestBootstrapNoiseContext:
    """Test compute_noise_context with bootstrap."""

    def test_pvalue_never_zero(self):
        """P-value with +1 correction should never be exactly 0."""

        # Create counts where TVD is very high (should give low p-value)
        counts_a = {"00": 1000, "11": 0}
        counts_b = {"00": 0, "11": 1000}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=100)

        # P-value should be positive (not zero) due to +1 correction
        assert ctx.p_value is not None
        assert ctx.p_value > 0, "P-value should never be exactly 0 with +1 correction"
        # Should be approximately 1/(n_boot+1) = 1/101 ≈ 0.0099
        assert ctx.p_value >= 1 / (100 + 1) - 0.001

    def test_pvalue_upper_bound(self):
        """P-value should be at most 1."""

        # Create identical counts (TVD = 0)
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 500, "11": 500}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=100)

        assert ctx.p_value is not None
        assert ctx.p_value <= 1.0

    def test_noise_p95_clamped_to_unit(self):
        """noise_p95 should always be in [0, 1]."""

        # Various count configurations
        test_cases = [
            ({"0": 10}, {"1": 10}),  # Very few shots
            ({"00": 1000, "11": 1000}, {"00": 1000, "11": 1000}),  # Many shots
            ({"0" * 10: 100}, {"1" * 10: 100}),  # Many outcomes
        ]

        for counts_a, counts_b in test_cases:
            ctx = compute_noise_context(counts_a, counts_b, n_boot=100)
            assert 0 <= ctx.noise_p95 <= 1.0, f"noise_p95={ctx.noise_p95} out of range"

    def test_method_is_bootstrap(self):
        """Method should be 'bootstrap' when successful."""

        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 490, "11": 510}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=100)

        assert ctx.method == "bootstrap"
        assert ctx.n_boot == 100
        assert ctx.p_value is not None

    def test_exceeds_noise_uses_p95(self):
        """exceeds_noise should be based on noise_p95 threshold."""

        # Create counts with moderate difference
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 400, "11": 600}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=500)

        # Check consistency
        expected_exceeds = ctx.tvd > ctx.noise_p95
        assert ctx.exceeds_noise == expected_exceeds

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""

        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}

        ctx1 = compute_noise_context(counts_a, counts_b, n_boot=100, seed=42)
        ctx2 = compute_noise_context(counts_a, counts_b, n_boot=100, seed=42)

        assert ctx1.noise_p95 == ctx2.noise_p95
        assert ctx1.p_value == ctx2.p_value

    def test_alpha_affects_threshold(self):
        """Different alpha should give different thresholds."""

        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}

        ctx_95 = compute_noise_context(counts_a, counts_b, n_boot=500, alpha=0.95)
        ctx_99 = compute_noise_context(counts_a, counts_b, n_boot=500, alpha=0.99)

        # 99th percentile should be >= 95th percentile
        assert ctx_99.noise_p95 >= ctx_95.noise_p95
        assert ctx_95.alpha == 0.95
        assert ctx_99.alpha == 0.99


class TestNoiseContextInterpretation:
    """Test NoiseContext interpretation."""

    def test_interpretation_high_pvalue(self):
        """High p-value should indicate noise."""

        # Nearly identical counts
        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 498, "11": 502}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=500)

        interp = ctx.interpretation()
        assert "consistent with" in interp.lower() or "noise" in interp.lower()

    def test_interpretation_low_pvalue(self):
        """Low p-value should indicate real difference."""

        # Very different counts
        counts_a = {"00": 900, "11": 100}
        counts_b = {"00": 100, "11": 900}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=500)

        interp = ctx.interpretation()
        assert "exceeds" in interp.lower() or "significantly" in interp.lower()


class TestHeuristicFallback:
    """Test heuristic fallback when bootstrap fails."""

    def test_empty_counts_uses_heuristic(self):
        """Empty counts should fall back to heuristic."""

        counts_a: dict[str, int] = {}
        counts_b: dict[str, int] = {}

        ctx = compute_noise_context(counts_a, counts_b)

        assert ctx.method == "heuristic"
        assert ctx.p_value is None
        assert ctx.n_boot == 0

    def test_zero_shots_uses_heuristic(self):
        """Zero total shots should use heuristic."""
        counts_a = {"00": 0}
        counts_b = {"11": 0}

        ctx = compute_noise_context(counts_a, counts_b)

        # Should not crash, should use heuristic
        assert ctx.method == "heuristic"


class TestToDict:
    """Test serialization."""

    def test_to_dict_includes_bootstrap_fields(self):
        """to_dict should include all bootstrap fields."""

        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}

        ctx = compute_noise_context(counts_a, counts_b, n_boot=100)
        d = ctx.to_dict()

        assert "tvd" in d
        assert "expected_noise" in d
        assert "noise_ratio" in d
        assert "noise_p95" in d
        assert "p_value" in d
        assert "method" in d
        assert "alpha" in d
        assert d["method"] == "bootstrap"
        assert d["n_boot"] == 100
