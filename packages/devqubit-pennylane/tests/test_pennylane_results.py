# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for PennyLane results processing, utils, and adapter internals.

Focuses on realistic end-to-end behavior with real PennyLane objects.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from devqubit_pennylane.results import (
    _PENNYLANE_COUNTS_FORMAT,
    _extract_expectation_values,
    _extract_probabilities,
    _extract_sample_counts,
    _sample_to_bitstring,
    build_result_snapshot,
    extract_result_type,
)


# =============================================================================
# Results: extract_result_type
# =============================================================================


class TestExtractResultType:
    def test_expval_tape(self):
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))
        assert "expectation" in extract_result_type([tape]).lower()

    def test_probs_tape(self):
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.probs(wires=[0])
        assert "prob" in extract_result_type([tape]).lower()

    def test_sample_tape(self):
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.sample(wires=[0])
        assert "sample" in extract_result_type([tape]).lower()

    def test_mixed_types_returns_mixed(self):
        t1 = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
        t2 = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.probs(wires=[0])])
        assert extract_result_type([t1, t2]) == "mixed"

    def test_empty_tapes(self):
        assert extract_result_type([]) == "unknown"


# =============================================================================
# Results: expectation extraction
# =============================================================================


class TestExtractExpectationValues:
    def test_scalar(self):
        exps = _extract_expectation_values(np.float64(0.42), num_circuits=1)
        assert len(exps) == 1
        assert abs(exps[0].value - 0.42) < 1e-10

    def test_array_multi_observable(self):
        exps = _extract_expectation_values(np.array([0.1, -0.3, 0.5]), num_circuits=1)
        assert len(exps) == 3
        assert exps[1].observable_index == 1
        assert abs(exps[1].value - (-0.3)) < 1e-10

    def test_batch(self):
        results = [np.float64(0.5), np.float64(-0.7)]
        exps = _extract_expectation_values(results, num_circuits=2)
        assert len(exps) == 2
        assert exps[0].circuit_index == 0
        assert exps[1].circuit_index == 1

    def test_batch_multi_observable(self):
        results = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        exps = _extract_expectation_values(results, num_circuits=2)
        assert len(exps) == 4
        assert exps[2].circuit_index == 1
        assert exps[2].observable_index == 0

    def test_none_returns_empty(self):
        assert _extract_expectation_values(None) == []


# =============================================================================
# Results: sample/counts extraction
# =============================================================================


class TestExtractSampleCounts:
    def test_dict_counts(self):
        counts = _extract_sample_counts({"00": 50, "11": 50}, num_circuits=1)
        assert len(counts) == 1
        assert counts[0].counts == {"00": 50, "11": 50}
        assert counts[0].shots == 100

    def test_2d_samples(self):
        # 10 shots, 2 wires
        samples = np.array([[0, 0]] * 6 + [[1, 1]] * 4)
        counts = _extract_sample_counts(samples, num_circuits=1)
        assert len(counts) == 1
        assert counts[0].counts["00"] == 6
        assert counts[0].counts["11"] == 4
        assert counts[0].shots == 10

    def test_batch_dict_counts(self):
        results = [{"0": 30, "1": 70}, {"00": 50, "11": 50}]
        counts = _extract_sample_counts(results, num_circuits=2)
        assert len(counts) == 2
        assert counts[0].circuit_index == 0
        assert counts[1].circuit_index == 1

    def test_none_returns_empty(self):
        assert _extract_sample_counts(None) == []


# =============================================================================
# Results: probability extraction
# =============================================================================


class TestExtractProbabilities:
    def test_single_qubit(self):
        probs = _extract_probabilities(np.array([0.5, 0.5]), num_circuits=1)
        assert len(probs) == 1
        assert abs(sum(probs[0].distribution.values()) - 1.0) < 1e-10

    def test_two_qubit(self):
        probs = _extract_probabilities(np.array([0.5, 0.0, 0.0, 0.5]), num_circuits=1)
        assert len(probs) == 1
        # Near-zero probs are filtered
        assert "01" not in probs[0].distribution
        assert "10" not in probs[0].distribution
        assert abs(probs[0].distribution["00"] - 0.5) < 1e-10

    def test_batch(self):
        results = [np.array([0.5, 0.5]), np.array([1.0, 0.0])]
        probs = _extract_probabilities(results, num_circuits=2)
        assert len(probs) == 2
        assert probs[0].circuit_index == 0
        assert probs[1].circuit_index == 1

    def test_none_returns_empty(self):
        assert _extract_probabilities(None) == []


# =============================================================================
# Results: build_result_snapshot
# =============================================================================


class TestBuildResultSnapshot:
    def test_success_with_expectations(self):
        snap = build_result_snapshot(
            np.float64(0.42),
            result_type="Expectation",
            num_circuits=1,
        )
        assert snap.success is True
        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].expectation is not None

    def test_success_with_counts(self):
        snap = build_result_snapshot(
            {"00": 50, "11": 50},
            result_type="Counts",
            num_circuits=1,
        )
        assert snap.success is True
        assert len(snap.items) == 1
        assert snap.items[0].counts is not None
        # Verify uses the hoisted format
        assert snap.items[0].counts["format"]["source_sdk"] == "pennylane"

    def test_success_with_probabilities(self):
        snap = build_result_snapshot(
            np.array([0.5, 0.5]),
            result_type="Probability",
            num_circuits=1,
        )
        assert snap.success is True
        assert len(snap.items) == 1
        assert snap.items[0].quasi_probability is not None

    def test_failed_execution(self):
        snap = build_result_snapshot(
            None,
            success=False,
            error_info={"type": "RuntimeError", "message": "Device offline"},
        )
        assert snap.success is False
        assert snap.status == "failed"
        assert snap.error is not None
        assert snap.error.type == "RuntimeError"

    def test_none_result_success(self):
        snap = build_result_snapshot(None, result_type="unknown")
        assert snap.success is True
        assert len(snap.items) == 0


# =============================================================================
# Results: sample_to_bitstring edge cases
# =============================================================================


class TestSampleToBitstring:
    def test_numpy_array(self):
        assert _sample_to_bitstring(np.array([0, 1, 1, 0])) == "0110"

    def test_list(self):
        assert _sample_to_bitstring([1, 0, 1]) == "101"

    def test_scalar(self):
        assert _sample_to_bitstring(0) == "0"
        assert _sample_to_bitstring(1) == "1"

    def test_fallback_returns_deterministic_hash(self):
        # An object that can't be iterated or converted
        result = _sample_to_bitstring(object())
        assert result.startswith("sample_")
        # Deterministic: same object type gives same prefix pattern
        assert len(result) > 7


# =============================================================================
# Results: hoisted CountsFormat constant
# =============================================================================


class TestCountsFormatConstant:
    def test_is_dict(self):
        assert isinstance(_PENNYLANE_COUNTS_FORMAT, dict)

    def test_has_required_keys(self):
        assert _PENNYLANE_COUNTS_FORMAT["source_sdk"] == "pennylane"
        assert _PENNYLANE_COUNTS_FORMAT["bit_order"] == "cbit0_left"
        assert _PENNYLANE_COUNTS_FORMAT["transformed"] is False
