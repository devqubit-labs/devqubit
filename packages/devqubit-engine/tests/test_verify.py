# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for baseline verification workflow."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.compare.ci import (
    result_to_github_annotations,
    result_to_junit,
    write_junit,
)
from devqubit_engine.compare.results import ProgramMatchMode, VerifyResult
from devqubit_engine.compare.verify import (
    VerifyPolicy,
    verify,
    verify_against_baseline,
)
from devqubit_engine.tracking.run import track


class TestVerifyWorkflow:
    """End-to-end verification tests simulating CI scenarios."""

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
                data=b"OPENQASM 3; qubit q; h q;",
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
                data=b"OPENQASM 3; qubit q; h q;",
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

    def test_param_change_fails_when_required(self, store, registry, config):
        """Parameter changes fail verification when params_must_match=True."""
        with track(project="params", config=config) as base:
            base.log_param("shots", 1000)
            base_id = base.run_id

        with track(project="params", config=config) as cand:
            cand.log_param("shots", 2000)  # Changed
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

    def test_program_change_fails_when_required(self, store, registry, config):
        """Program changes fail verification when program_must_match=True."""
        with track(project="prog", config=config) as base:
            base.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; qubit q; h q;",
                media_type="text/plain",
                role="program",
            )
            base_id = base.run_id

        with track(project="prog", config=config) as cand:
            cand.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; qubit q; x q;",  # Changed
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

    def test_tvd_threshold_enforced(self, store, registry, config):
        """TVD exceeding threshold fails verification."""
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

        # TVD = 0.2, threshold = 0.1 -> fail
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

    def test_noise_factor_uses_bootstrap(self, store, registry, config):
        """noise_factor multiplies bootstrap noise_p95 threshold."""
        with track(project="noise", config=config) as base:
            base.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 500, "11": 500}}).encode(),
                media_type="application/json",
                role="results",
            )
            base_id = base.run_id

        with track(project="noise", config=config) as cand:
            # Large difference that should exceed any noise
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
                noise_factor=1.0,
            ),
        )

        assert not result.ok

    def test_tvd_max_is_hard_limit_over_noise_threshold(self, store, registry, config):
        """tvd_max should be a hard limit even when noise_threshold is higher."""
        # Create runs with TVD = 0.2
        with track(project="threshold", config=config) as base:
            base.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 500, "11": 500}}).encode(),
                media_type="application/json",
                role="results",
            )
            base_id = base.run_id

        with track(project="threshold", config=config) as cand:
            cand.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 300, "11": 700}}).encode(),
                media_type="application/json",
                role="results",
            )
            cand_id = cand.run_id

        # tvd_max=0.1 should fail even if noise_factor would allow higher
        # (noise_p95 for 1000 shots is ~0.03, so noise_factor=10 = 0.3)
        # Old bug: max(0.1, 0.3) = 0.3, so TVD=0.2 would pass
        # Fixed: min(0.1, 0.3) = 0.1, so TVD=0.2 fails
        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(
                params_must_match=False,
                program_must_match=False,
                tvd_max=0.1,
                noise_factor=10.0,  # Would give ~0.3 threshold
            ),
        )

        assert not result.ok
        assert any("tvd" in f.lower() for f in result.failures)

    def test_noise_pvalue_cutoff_triggers_noise_context(self, store, registry, config):
        """noise_pvalue_cutoff alone should trigger noise context computation."""
        with track(project="pvalue", config=config) as base:
            base.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 500, "11": 500}}).encode(),
                media_type="application/json",
                role="results",
            )
            base_id = base.run_id

        with track(project="pvalue", config=config) as cand:
            cand.log_bytes(
                kind="result.counts.json",
                data=json.dumps({"counts": {"00": 495, "11": 505}}).encode(),
                media_type="application/json",
                role="results",
            )
            cand_id = cand.run_id

        # Only noise_pvalue_cutoff, no noise_factor
        # Old bug: noise_context not computed because noise_factor=None
        result = verify(
            registry.load(base_id),
            registry.load(cand_id),
            store_baseline=store,
            store_candidate=store,
            policy=VerifyPolicy(
                params_must_match=False,
                program_must_match=False,
                tvd_max=0.001,  # Very strict
                noise_pvalue_cutoff=0.05,  # Should trigger noise context
            ),
        )

        # Should have noise_context computed for p-value check
        assert result.comparison is not None
        assert result.comparison.noise_context is not None


class TestBaselineWorkflow:
    """Baseline management tests."""

    def test_missing_baseline_raises(self, store, registry, config):
        """Missing baseline raises error by default."""
        with track(project="no_baseline", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        with pytest.raises(ValueError, match="No baseline"):
            verify_against_baseline(
                registry.load(run_id),
                project="no_baseline",
                store=store,
                registry=registry,
            )

    def test_allow_missing_baseline_passes(self, store, registry, config):
        """First run passes when allow_missing_baseline=True."""
        with track(project="first_run", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        result = verify_against_baseline(
            registry.load(run_id),
            project="first_run",
            store=store,
            registry=registry,
            policy=VerifyPolicy(allow_missing_baseline=True),
        )

        assert result.ok
        assert result.baseline_run_id is None

    def test_promote_on_pass_updates_baseline(self, store, registry, config):
        """Passing verification with promote_on_pass updates baseline."""
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
        with track(project="no_promote", config=config) as base:
            base.log_param("x", 1)
            base_id = base.run_id

        registry.set_baseline("no_promote", base_id)

        with track(project="no_promote", config=config) as cand:
            cand.log_param("x", 999)  # Different
            cand_id = cand.run_id

        result = verify_against_baseline(
            registry.load(cand_id),
            project="no_promote",
            store=store,
            registry=registry,
            policy=VerifyPolicy(params_must_match=True),
            promote_on_pass=True,
        )

        assert not result.ok
        # Baseline unchanged
        assert registry.get_baseline("no_promote")["run_id"] == base_id


class TestVerifyPolicy:
    """VerifyPolicy configuration and serialization."""

    def test_default_values(self):
        """Default policy has sensible CI defaults."""
        policy = VerifyPolicy()

        assert policy.params_must_match is True
        assert policy.program_must_match is True
        assert policy.program_match_mode == ProgramMatchMode.EITHER
        assert policy.tvd_max is None
        assert policy.noise_factor is None

    def test_from_dict(self):
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

    def test_roundtrip(self):
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


class TestJUnitOutput:
    """JUnit XML output for CI systems."""

    def test_passing_verification(self, tmp_path):
        """Passing verification produces valid JUnit XML."""
        result = VerifyResult(
            ok=True,
            failures=[],
            baseline_run_id="BASE123",
            candidate_run_id="CAND456",
            duration_ms=150,
        )

        junit_path = tmp_path / "results.xml"
        write_junit(result, junit_path)

        content = junit_path.read_text()
        assert 'failures="0"' in content
        assert "CAND456" in content
        assert "<testsuite" in content

    def test_failing_verification(self, tmp_path):
        """Failing verification includes failure details."""
        result = VerifyResult(
            ok=False,
            failures=["params mismatch", "TVD exceeded threshold"],
            baseline_run_id="BASE123",
            candidate_run_id="CAND456",
            duration_ms=200,
        )

        junit_path = tmp_path / "results.xml"
        write_junit(result, junit_path)

        content = junit_path.read_text()
        assert 'failures="1"' in content
        assert "<failure" in content

    def test_result_to_junit_string(self):
        """result_to_junit returns XML string."""
        result = VerifyResult(
            ok=True,
            failures=[],
            baseline_run_id="BASE",
            candidate_run_id="CAND",
            duration_ms=100,
        )

        xml = result_to_junit(result)

        assert xml.startswith("<testsuite")
        assert "CAND" in xml


class TestGitHubAnnotations:
    """GitHub Actions annotation output."""

    def test_pass_uses_notice(self):
        """Passing verification uses ::notice."""
        result = VerifyResult(
            ok=True,
            failures=[],
            baseline_run_id="BASE",
            candidate_run_id="CAND",
            duration_ms=100,
        )

        output = result_to_github_annotations(result)

        assert "::notice" in output
        assert "::error" not in output

    def test_fail_uses_error(self):
        """Failing verification uses ::error for each failure."""
        result = VerifyResult(
            ok=False,
            failures=["params mismatch", "TVD too high"],
            baseline_run_id="BASE",
            candidate_run_id="CAND",
            duration_ms=100,
        )

        output = result_to_github_annotations(result)

        assert "::error" in output
        assert "params mismatch" in output


class TestVerifyResultSerialization:
    """VerifyResult serialization for API responses."""

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

        json_str = json.dumps(result.to_dict(), default=str)
        parsed = json.loads(json_str)

        assert parsed["ok"] is True
