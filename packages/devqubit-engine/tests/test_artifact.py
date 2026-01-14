# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for artifact browsing utilities."""

from __future__ import annotations

from devqubit_engine.artifacts import (
    CountsInfo,
    diff_counts,
    find_artifact,
    format_counts_table,
    get_artifact,
    get_counts,
    list_artifacts,
)
from devqubit_engine.core.run import track


class TestFindArtifact:
    """Tests for find_artifact utility."""

    def test_find_by_role(self, store, registry, config):
        """Find artifact by role."""
        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="program.qasm",
                data=b"OPENQASM 3;",
                media_type="text/plain",
                role="program",
            )
            run.log_bytes(
                kind="results.counts",
                data=b'{"00": 500}',
                media_type="application/json",
                role="results",
            )
            run_id = run.run_id

        record = registry.load(run_id)

        art = find_artifact(record, role="program")
        assert art is not None
        assert art.role == "program"

        art = find_artifact(record, role="results")
        assert art is not None
        assert art.role == "results"

    def test_find_by_kind(self, store, registry, config):
        """Find artifact by kind substring."""
        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="openqasm3.canonical",
                data=b"qasm",
                media_type="text/plain",
                role="program",
            )
            run_id = run.run_id

        record = registry.load(run_id)

        art = find_artifact(record, kind_contains="openqasm")
        assert art is not None
        assert "openqasm" in art.kind

    def test_find_returns_none_if_missing(self, store, registry, config):
        """Returns None if no match."""
        with track(project="test", config=config) as run:
            run_id = run.run_id

        record = registry.load(run_id)
        art = find_artifact(record, role="nonexistent")
        assert art is None


class TestListArtifacts:
    """Tests for list_artifacts."""

    def test_list_with_filter(self, store, registry, config):
        """List artifacts with role filter."""
        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="prog.qasm",
                data=b"1",
                media_type="text/plain",
                role="program",
            )
            run.log_bytes(
                kind="res.counts",
                data=b"2",
                media_type="text/plain",
                role="results",
            )
            run_id = run.run_id

        record = registry.load(run_id)
        arts = list_artifacts(record, role="program")

        assert len(arts) == 1
        assert arts[0].role == "program"


class TestGetArtifact:
    """Tests for get_artifact selector."""

    def test_get_by_index(self, store, registry, config):
        """Get artifact by index."""
        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="first.artifact",
                data=b"1",
                media_type="text/plain",
                role="test",
            )
            run.log_bytes(
                kind="second.artifact",
                data=b"2",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        record = registry.load(run_id)

        art = get_artifact(record, 0)
        assert art is not None

        art = get_artifact(record, 1)
        assert art is not None

        art = get_artifact(record, 99)
        assert art is None

    def test_get_by_kind(self, store, registry, config):
        """Get artifact by kind substring."""
        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="openqasm3.circuit",
                data=b"qasm",
                media_type="text/plain",
                role="program",
            )
            run_id = run.run_id

        record = registry.load(run_id)

        art = get_artifact(record, "openqasm")
        assert art is not None
        assert "openqasm" in art.kind


class TestGetCounts:
    """Tests for measurement counts extraction."""

    def test_get_simple_counts(self, store, registry, config):
        """Extract simple counts format."""
        counts_data = '{"counts": {"00": 450, "11": 550}}'

        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="results.counts",
                data=counts_data.encode(),
                media_type="application/json",
                role="results",
            )
            run_id = run.run_id

        record = registry.load(run_id)
        counts = get_counts(record, store)

        assert counts is not None
        assert counts.total_shots == 1000
        assert counts.num_outcomes == 2
        assert counts.counts["00"] == 450
        assert counts.counts["11"] == 550

    def test_counts_probabilities(self, store, registry, config):
        """Counts computes probabilities."""
        counts_data = '{"counts": {"00": 250, "11": 750}}'

        with track(project="test", config=config) as run:
            run.log_bytes(
                kind="results.counts",
                data=counts_data.encode(),
                media_type="application/json",
                role="results",
            )
            run_id = run.run_id

        record = registry.load(run_id)
        counts = get_counts(record, store)

        probs = counts.probabilities
        assert probs["00"] == 0.25
        assert probs["11"] == 0.75

    def test_counts_top_k(self):
        """CountsInfo.top_k returns top outcomes."""
        counts = CountsInfo(
            counts={"00": 100, "01": 50, "10": 30, "11": 20},
            total_shots=200,
            num_outcomes=4,
        )

        top = counts.top_k(2)
        assert len(top) == 2
        assert top[0][0] == "00"  # Highest count
        assert top[0][1] == 100
        assert top[1][0] == "01"


class TestDiffCounts:
    """Tests for comparing count distributions."""

    def test_identical_counts(self, bell_state_counts):
        """Identical counts have zero TVD."""
        counts_a = CountsInfo(
            counts=bell_state_counts,
            total_shots=1000,
            num_outcomes=2,
        )
        counts_b = CountsInfo(
            counts=bell_state_counts,
            total_shots=1000,
            num_outcomes=2,
        )

        diff = diff_counts(counts_a, counts_b)

        assert diff["tvd"] == 0.0

    def test_different_counts(self):
        """Different counts have non-zero TVD."""
        counts_a = CountsInfo(counts={"00": 1000}, total_shots=1000, num_outcomes=1)
        counts_b = CountsInfo(counts={"11": 1000}, total_shots=1000, num_outcomes=1)

        diff = diff_counts(counts_a, counts_b)

        assert diff["tvd"] == 1.0  # Maximum TVD

    def test_diff_aligned_outcomes(self):
        """Diff provides aligned comparison."""
        counts_a = CountsInfo(
            counts={"00": 600, "11": 400},
            total_shots=1000,
            num_outcomes=2,
        )
        counts_b = CountsInfo(
            counts={"00": 400, "11": 600},
            total_shots=1000,
            num_outcomes=2,
        )

        diff = diff_counts(counts_a, counts_b)

        assert len(diff["aligned"]) == 2
        # Check TVD is correct: |0.6-0.4| + |0.4-0.6| = 0.4, TVD = 0.2
        assert abs(diff["tvd"] - 0.2) < 0.001


class TestFormatCountsTable:
    """Tests for counts table formatting."""

    def test_format_basic(self, bell_state_counts):
        """Format produces readable table."""
        counts = CountsInfo(
            counts=bell_state_counts,
            total_shots=1000,
            num_outcomes=2,
        )

        table = format_counts_table(counts)

        # Just check it contains the bitstrings
        assert "00" in table
        assert "11" in table
