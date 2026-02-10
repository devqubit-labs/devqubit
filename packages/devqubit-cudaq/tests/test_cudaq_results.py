# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for result processing."""

from devqubit_cudaq.results import (
    _CUDAQ_COUNTS_FORMAT,
    _CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER,
    _normalize_bitstrings_to_cbit0_right,
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

    def test_batch_sample(self, bell_sample_result, uniform_sample_result):
        assert (
            detect_result_type([bell_sample_result, uniform_sample_result])
            == "batch_sample"
        )

    def test_batch_observe(self, observe_result):
        assert detect_result_type([observe_result]) == "batch_observe"

    def test_empty_list(self):
        assert detect_result_type([]) == "unknown"


class TestNormalizeBitstrings:

    def test_palindromes_unchanged(self):
        counts = {"00": 100, "11": 200}
        result = _normalize_bitstrings_to_cbit0_right(counts)
        assert result == {"00": 100, "11": 200}

    def test_non_palindromes_reversed(self):
        counts = {"01": 100, "10": 200}
        result = _normalize_bitstrings_to_cbit0_right(counts)
        assert result == {"10": 100, "01": 200}

    def test_three_qubit(self):
        counts = {"100": 500}
        result = _normalize_bitstrings_to_cbit0_right(counts)
        assert result == {"001": 500}


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

    def test_counts_extracted_and_normalized(self, bell_sample_result):
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

    def test_non_palindrome_bitstrings_normalized(self, make_sample_result):
        raw = make_sample_result({"01": 300, "10": 700})
        snap = build_result_snapshot(
            raw, result_type="sample", backend_name="qpp-cpu", shots=1000
        )
        counts = snap.items[0].counts["counts"]
        assert counts["10"] == 300
        assert counts["01"] == 700


class TestExplicitMeasurements:

    def test_explicit_measurements_no_reversal(self, make_sample_result):
        """With explicit_measurements=True, bitstrings stay as-is."""
        raw = make_sample_result({"01": 300, "10": 700})
        snap = build_result_snapshot(
            raw,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
            call_kwargs={"explicit_measurements": True},
        )
        counts = snap.items[0].counts["counts"]
        # NOT reversed — kept in measurement order
        assert counts["01"] == 300
        assert counts["10"] == 700

    def test_explicit_measurements_format_marked(self, make_sample_result):
        raw = make_sample_result({"01": 500, "10": 500})
        snap = build_result_snapshot(
            raw,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
            call_kwargs={"explicit_measurements": True},
        )
        fmt = snap.items[0].counts["format"]
        assert fmt["bit_order"] == "measurement_order"
        assert fmt["transformed"] is False

    def test_explicit_measurements_metadata_flag(self, make_sample_result):
        raw = make_sample_result({"00": 1000})
        snap = build_result_snapshot(
            raw,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
            call_kwargs={"explicit_measurements": True},
        )
        assert snap.metadata.get("cudaq.explicit_measurements") is True

    def test_standard_mode_still_reverses(self, make_sample_result):
        """Without explicit_measurements, standard reversal applies."""
        raw = make_sample_result({"01": 300, "10": 700})
        snap = build_result_snapshot(
            raw,
            result_type="sample",
            backend_name="qpp-cpu",
            shots=1000,
            call_kwargs={},
        )
        counts = snap.items[0].counts["counts"]
        assert counts["10"] == 300
        assert counts["01"] == 700
        fmt = snap.items[0].counts["format"]
        assert fmt["bit_order"] == "cbit0_right"


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


class TestBuildResultSnapshotBroadcast:

    def test_batch_sample(self, make_sample_result):
        results = [
            make_sample_result({"00": 400, "11": 600}),
            make_sample_result({"01": 500, "10": 500}),
        ]
        snap = build_result_snapshot(
            results, result_type="batch_sample", backend_name="qpp-cpu", shots=1000
        )
        assert snap.success is True
        assert len(snap.items) >= 2

    def test_batch_observe(self, make_observe_result):
        results = [
            make_observe_result(-0.5),
            make_observe_result(0.3),
        ]
        snap = build_result_snapshot(
            results, result_type="batch_observe", backend_name="nvidia"
        )
        assert snap.success is True
        expectations = [it for it in snap.items if it.expectation is not None]
        assert len(expectations) == 2


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

    def test_bit_order_canonical(self):
        assert _CUDAQ_COUNTS_FORMAT["bit_order"] == "cbit0_right"

    def test_transformed(self):
        assert _CUDAQ_COUNTS_FORMAT["transformed"] is True

    def test_measurement_order_format(self):
        assert (
            _CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER["bit_order"] == "measurement_order"
        )
        assert _CUDAQ_COUNTS_FORMAT_MEASUREMENT_ORDER["transformed"] is False
