# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for baseline verification.

These tests CI output formats and the verification.
"""

from __future__ import annotations

from devqubit_engine.compare.ci import (
    result_to_github_annotations,
    result_to_junit,
    write_junit,
)
from devqubit_engine.compare.results import VerifyResult


class TestJUnitOutput:
    """Tests for JUnit XML output."""

    def test_write_junit_pass(self, tmp_path):
        """Write JUnit XML for passing verification."""
        result = VerifyResult(
            ok=True,
            failures=[],
            baseline_run_id="BASE123",
            candidate_run_id="CAND456",
            duration_ms=150,
        )

        junit_path = tmp_path / "results.xml"
        write_junit(result, str(junit_path))

        content = junit_path.read_text()
        assert 'failures="0"' in content
        assert "CAND456" in content

    def test_write_junit_fail(self, tmp_path):
        """Write JUnit XML for failing verification."""
        result = VerifyResult(
            ok=False,
            failures=["params mismatch", "TVD exceeded"],
            baseline_run_id="BASE123",
            candidate_run_id="CAND456",
            duration_ms=200,
        )

        junit_path = tmp_path / "failures.xml"
        write_junit(result, str(junit_path))

        content = junit_path.read_text()
        assert 'failures="1"' in content
        assert "failure" in content.lower()

    def test_result_to_junit_string(self):
        """result_to_junit returns valid XML string."""
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
    """Tests for GitHub Actions annotation output."""

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
        """Failing verification uses ::error."""
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
