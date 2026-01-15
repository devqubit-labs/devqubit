# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for baseline verification."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.compare.results import ProgramMatchMode, VerifyResult
from devqubit_engine.compare.verify import (
    VerifyPolicy,
    verify,
    verify_against_baseline,
)
from devqubit_engine.tracking.run import track


class TestVerifyPolicy:
    """VerifyPolicy configuration tests."""

    def test_default_policy_values(self):
        """Default policy has sensible CI defaults."""
        policy = VerifyPolicy()

        assert policy.params_must_match is True
        assert policy.program_must_match is True
        assert policy.program_match_mode == ProgramMatchMode.EITHER
        assert policy.tvd_max is None
        assert policy.noise_factor is None

    def test_from_dict_parses_correctly(self):
        """Policy can be created from config dict."""
        policy = VerifyPolicy.from_dict(
            {
                "params_must_match": False,
                "program_match_mode": "structural",
                "tvd_max": 0.1,
                "noise_factor": 1.5,
            }
        )

        assert policy.params_must_match is False
        assert policy.program_match_mode == ProgramMatchMode.STRUCTURAL
        assert policy.tvd_max == 0.1
        assert policy.noise_factor == 1.5

    def test_to_dict_roundtrip(self):
        """Policy survives dict serialization roundtrip."""
        original = VerifyPolicy(
            tvd_max=0.05,
            noise_factor=2.0,
            program_match_mode=ProgramMatchMode.STRUCTURAL,
        )

        d = original.to_dict()
        restored = VerifyPolicy.from_dict(d)

        assert restored.tvd_max == original.tvd_max
        assert restored.noise_factor == original.noise_factor
        assert restored.program_match_mode == original.program_match_mode


class TestVerify:
    """Core verification tests."""

    def test_identical_runs_pass(self, store, registry, config):
        """Identical runs pass verification."""
        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as base:
            base.log_param("shots", 1000)
            base.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3;",
                media_type="text/plain",
                role="program",
            )
            base_id = base.run_id

        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as cand:
            cand.log_param("shots", 1000)
            cand.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3;",
                media_type="text/plain",
                role="program",
            )
            cand_id = cand.run_id

        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
        )

        assert result.ok
        assert len(result.failures) == 0

    def test_param_change_fails(self, store, registry, config):
        """Parameter changes cause failure when required."""
        with track(project="params", config=config) as base:
            base.log_param("shots", 1000)
            base_id = base.run_id

        with track(project="params", config=config) as cand:
            cand.log_param("shots", 2000)
            cand_id = cand.run_id

        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(params_must_match=True),
        )

        assert not result.ok
        assert any("param" in f.lower() for f in result.failures)

    def test_program_change_fails(self, store, registry, config):
        """Program changes cause failure when required."""
        with track(project="prog", config=config) as base:
            base.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; h q;",
                media_type="text/plain",
                role="program",
            )
            base_id = base.run_id

        with track(project="prog", config=config) as cand:
            cand.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; x q;",
                media_type="text/plain",
                role="program",
            )
            cand_id = cand.run_id

        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(program_must_match=True),
        )

        assert not result.ok
        assert any("program" in f.lower() for f in result.failures)


class TestTVDThresholds:
    """TVD-based verification thresholds."""

    def test_tvd_max_enforced(self, store, registry, config):
        """tvd_max threshold is enforced."""
        with track(project="tvd", config=config) as base:
            base.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 500, "11": 500}}).encode(),
                media_type="application/json",
                role="results",
            )
            base_id = base.run_id

        with track(project="tvd", config=config) as cand:
            cand.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 300, "11": 700}}).encode(),
                media_type="application/json",
                role="results",
            )
            cand_id = cand.run_id

        # TVD = 0.2, threshold = 0.1 -> should fail
        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(
                params_must_match=False,
                program_must_match=False,
                tvd_max=0.1,
            ),
        )

        assert not result.ok
        assert any("tvd" in f.lower() for f in result.failures)

    def test_noise_factor_uses_bootstrap_p95(self, store, registry, config):
        """noise_factor multiplies bootstrap-calibrated noise_p95."""
        with track(project="noise", config=config) as base:
            base.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 500, "11": 500}}).encode(),
                media_type="application/json",
                role="results",
            )
            base_id = base.run_id

        with track(project="noise", config=config) as cand:
            # Large difference that exceeds any reasonable noise threshold
            cand.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 100, "11": 900}}).encode(),
                media_type="application/json",
                role="results",
            )
            cand_id = cand.run_id

        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(
                params_must_match=False,
                program_must_match=False,
                noise_factor=1.0,  # Use raw p95 threshold
            ),
        )

        # TVD = 0.4 should exceed noise threshold
        assert not result.ok
        assert any("noise" in f.lower() or "p95" in f.lower() for f in result.failures)

    def test_analytic_mode_skips_tvd_check(self, store, registry, config):
        """Runs without counts skip TVD check gracefully."""
        with track(project="analytic", config=config) as base:
            base.log_json(name="exp", obj={"value": 0.5}, role="expectation")
            base_id = base.run_id

        with track(project="analytic", config=config) as cand:
            cand.log_json(name="exp", obj={"value": 0.6}, role="expectation")
            cand_id = cand.run_id

        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(
                params_must_match=False,
                program_must_match=False,
                noise_factor=2.0,
            ),
        )

        # Should pass (TVD check skipped) with warning
        assert result.ok
        assert any("tvd check skipped" in w.lower() for w in result.comparison.warnings)


class TestBaselineManagement:
    """Baseline workflow tests."""

    def test_missing_baseline_raises(self, store, registry, config):
        """Missing baseline raises error by default."""
        with track(project="no_base", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        with pytest.raises(ValueError, match="No baseline"):
            verify_against_baseline(
                registry.load(run_id),
                project="no_base",
                store=store,
                registry=registry,
            )

    def test_allow_missing_baseline_passes(self, store, registry, config):
        """allow_missing_baseline=True allows first run."""
        with track(project="first", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        result = verify_against_baseline(
            registry.load(run_id),
            project="first",
            store=store,
            registry=registry,
            policy=VerifyPolicy(allow_missing_baseline=True),
        )

        assert result.ok
        assert result.baseline_run_id is None

    def test_promote_on_pass(self, store, registry, config):
        """promote_on_pass updates baseline on success."""
        with track(project="promote", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        verify_against_baseline(
            registry.load(run_id),
            project="promote",
            store=store,
            registry=registry,
            policy=VerifyPolicy(allow_missing_baseline=True),
            promote_on_pass=True,
        )

        baseline = registry.get_baseline("promote")
        assert baseline is not None
        assert baseline["run_id"] == run_id

    def test_no_promote_on_fail(self, store, registry, config):
        """Failed verification does not update baseline."""
        with track(project="nopromote", config=config) as base:
            base.log_param("x", 1)
            base_id = base.run_id

        registry.set_baseline("nopromote", base_id)

        with track(project="nopromote", config=config) as cand:
            cand.log_param("x", 999)  # Different
            cand_id = cand.run_id

        result = verify_against_baseline(
            registry.load(cand_id),
            project="nopromote",
            store=store,
            registry=registry,
            policy=VerifyPolicy(params_must_match=True),
            promote_on_pass=True,
        )

        assert not result.ok
        # Baseline unchanged
        assert registry.get_baseline("nopromote")["run_id"] == base_id


class TestVerifyResultSerialization:
    """VerifyResult serialization for CI systems."""

    def test_to_dict_structure(self):
        """to_dict has required CI fields."""
        result = VerifyResult(
            ok=False,
            failures=["param change", "TVD exceeded"],
            baseline_run_id="BASE",
            candidate_run_id="CAND",
            duration_ms=150,
        )

        d = result.to_dict()

        assert d["ok"] is False
        assert len(d["failures"]) == 2
        assert d["baseline_run_id"] == "BASE"
        assert d["candidate_run_id"] == "CAND"
        assert d["duration_ms"] == 150

    def test_json_serializable(self):
        """Result can be JSON serialized."""
        result = VerifyResult(
            ok=True,
            failures=[],
            baseline_run_id="BASE",
            candidate_run_id="CAND",
            duration_ms=100,
        )

        # Should not raise
        json_str = json.dumps(result.to_dict(), default=str)
        parsed = json.loads(json_str)

        assert parsed["ok"] is True
