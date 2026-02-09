# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for result processing (results.py)."""

from devqubit_cudaq.results import (
    _CUDAQ_COUNTS_FORMAT,
    build_result_snapshot,
    detect_result_type,
)


class TestDetectResultType:

    def test_sample_result(self, bell_sample_result):
        assert detect_result_type(bell_sample_result) == "sample"

    def test_observe_result(self, observe_result):
        assert detect_result_type(observe_result) == "observe"

    def test_none(self):
        assert detect_result_type(None) == "unknown"

    def test_non_result(self):
        assert detect_result_type("not a result") == "unknown"


class TestBuildResultSnapshotSample:

    def test_success(self, bell_sample_result):
        snap = build_result_snapshot(
            bell_sample_result,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
        )
        assert snap.success is True
        assert snap.status == "completed"
        assert len(snap.items) >= 1

    def test_counts_extracted(self, bell_sample_result):
        snap = build_result_snapshot(
            bell_sample_result,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
        )
        counts_item = snap.items[0]
        assert counts_item.counts["counts"]["00"] == 480
        assert counts_item.counts["counts"]["11"] == 520
        assert counts_item.counts["shots"] == 1000

    def test_quasi_probabilities_derived(self, bell_sample_result):
        snap = build_result_snapshot(
            bell_sample_result,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
        )
        assert len(snap.items) >= 2
        dist = snap.items[1].quasi_probability.distribution
        assert abs(dist["00"] - 0.48) < 0.01
        assert abs(dist["11"] - 0.52) < 0.01

    def test_uniform_counts(self, uniform_sample_result):
        snap = build_result_snapshot(
            uniform_sample_result,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
        )
        counts = snap.items[0].counts["counts"]
        assert len(counts) == 4
        assert sum(counts.values()) == 1000


class TestBuildResultSnapshotObserve:

    def test_expectation(self, observe_result):
        snap = build_result_snapshot(
            observe_result,
            result_type="observe",
            backend_name="nvidia",
        )
        assert snap.success is True
        exp = snap.items[0].expectation
        assert exp is not None
        assert abs(exp.value - (-0.42)) < 1e-10

    def test_observe_with_counts(self, observe_result_with_counts):
        snap = build_result_snapshot(
            observe_result_with_counts,
            result_type="observe",
            backend_name="qpp-cpu",
            shots=1000,
        )
        has_exp = any(it.expectation is not None for it in snap.items)
        has_counts = any(it.counts is not None for it in snap.items)
        assert has_exp
        assert has_counts


class TestBuildResultSnapshotErrors:

    def test_failed_execution(self):
        snap = build_result_snapshot(
            None,
            result_type="sample",
            backend_name="ionq",
            success=False,
            error_info={"type": "RuntimeError", "message": "Target offline"},
        )
        assert snap.success is False
        assert snap.status == "failed"
        assert snap.error.type == "RuntimeError"
        assert "offline" in snap.error.message

    def test_none_result_success(self):
        snap = build_result_snapshot(
            None,
            result_type="unknown",
            backend_name="qpp-cpu",
        )
        assert snap.success is True
        assert len(snap.items) == 0


class TestCountsFormat:

    def test_source_sdk(self):
        assert _CUDAQ_COUNTS_FORMAT["source_sdk"] == "cudaq"

    def test_bit_order(self):
        assert _CUDAQ_COUNTS_FORMAT["bit_order"] == "cbit0_left"

    def test_not_transformed(self):
        assert _CUDAQ_COUNTS_FORMAT["transformed"] is False
