# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for PennyLane result processing."""

import numpy as np
import pennylane as qml
from devqubit_pennylane.results import (
    _extract_probabilities,
    _extract_sample_counts,
    _result_type_for_tape,
    _sample_to_bitstring,
    build_result_snapshot,
    extract_result_type,
)


class TestResultTypeForTape:
    """Tests for single tape result type detection."""

    def test_expectation_type(self, expectation_tape):
        """Detects expectation value type."""
        result_type = _result_type_for_tape(expectation_tape)
        assert "Expectation" in result_type or "expval" in result_type.lower()

    def test_probability_type(self, probability_tape):
        """Detects probability type."""
        result_type = _result_type_for_tape(probability_tape)
        assert "Probability" in result_type or "prob" in result_type.lower()

    def test_sample_bitstring_type(self, sample_bitstring_tape):
        """Detects sample type for bitstring samples."""
        result_type = _result_type_for_tape(sample_bitstring_tape)
        assert "Sample" in result_type or "sample" in result_type.lower()

    def test_sample_eigenvalue_type(self, sample_eigenvalue_tape):
        """Detects sample type for eigenvalue samples."""
        result_type = _result_type_for_tape(sample_eigenvalue_tape)
        assert "Sample" in result_type or "sample" in result_type.lower()

    def test_counts_type(self, bell_tape):
        """Detects counts type."""
        result_type = _result_type_for_tape(bell_tape)
        assert "Counts" in result_type or "count" in result_type.lower()

    def test_empty_measurements(self):
        """Handles tape with no measurements."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)

        result_type = _result_type_for_tape(tape)
        assert result_type == "unknown"

    def test_fallback_to_class_name(self):
        """Falls back to class name when return_type unavailable."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        # Should still detect from class name
        result_type = _result_type_for_tape(tape)
        assert result_type != "unknown"


class TestExtractResultType:
    """Tests for batch result type extraction."""

    def test_single_expectation(self, expectation_tape):
        """Single expectation tape."""
        result_type = extract_result_type([expectation_tape])
        assert "Expectation" in result_type or "expval" in result_type.lower()

    def test_single_counts(self, bell_tape):
        """Single counts tape."""
        result_type = extract_result_type([bell_tape])
        assert "Counts" in result_type or "count" in result_type.lower()

    def test_mixed_types(self, expectation_tape, probability_tape):
        """Detects mixed result types across tapes."""
        result_type = extract_result_type([expectation_tape, probability_tape])
        assert result_type == "mixed"

    def test_empty_tapes(self):
        """Handles empty tape list."""
        result_type = extract_result_type([])
        assert result_type == "unknown"

    def test_multiple_same_type(self, bell_tape, ghz_tape):
        """Multiple tapes with same type."""
        result_type = extract_result_type([bell_tape, ghz_tape])
        assert "Counts" in result_type or "count" in result_type.lower()


class TestSampleToBitstring:
    """Tests for sample to bitstring conversion."""

    def test_numpy_array_1d(self):
        """Converts 1D numpy array to bitstring."""
        sample = np.array([0, 1, 1, 0])
        assert _sample_to_bitstring(sample) == "0110"

    def test_numpy_array_2d_flattens(self):
        """Flattens 2D array to single bitstring."""
        sample = np.array([[0, 1], [1, 0]])
        assert _sample_to_bitstring(sample) == "0110"

    def test_list_input(self):
        """Converts list to bitstring."""
        assert _sample_to_bitstring([0, 1, 0]) == "010"

    def test_tuple_input(self):
        """Converts tuple to bitstring."""
        assert _sample_to_bitstring((1, 0, 1)) == "101"

    def test_single_qubit(self):
        """Handles single qubit measurement."""
        assert _sample_to_bitstring(0) == "0"
        assert _sample_to_bitstring(1) == "1"

    def test_numpy_scalar(self):
        """Handles numpy scalar."""
        assert _sample_to_bitstring(np.int64(1)) == "1"


class TestBuildResultSnapshot:
    """Tests for result snapshot building (UEC 1.0 API)."""

    def test_expectations_two_circuits(self):
        """Builds expectation snapshot for multiple circuits."""
        snap = build_result_snapshot(
            [0.125, -0.5],
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=2,
        )

        assert snap.status == "completed"
        assert snap.success is True
        assert len(snap.items) == 2
        assert snap.items[0].item_index == 0
        assert snap.items[1].item_index == 1
        assert snap.items[0].expectation.value == 0.125
        assert snap.items[1].expectation.value == -0.5

    def test_expectations_single_circuit(self):
        """Builds expectation snapshot for single circuit."""
        snap = build_result_snapshot(
            0.5,
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].expectation.value == 0.5
        assert snap.items[0].expectation.circuit_index == 0

    def test_expectations_multiple_observables_single_circuit(self):
        """Handles multiple observables for single circuit."""
        # Single circuit with 3 expectation values
        snap = build_result_snapshot(
            np.array([0.1, 0.2, 0.3]),
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 3
        assert [item.expectation.observable_index for item in snap.items] == [0, 1, 2]

    def test_counts_dict(self):
        """Builds counts snapshot from dict."""
        snap = build_result_snapshot(
            {"0": 2, "1": 3},
            result_type="Counts",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].counts["shots"] == 5
        assert snap.items[0].counts["counts"]["0"] == 2
        assert snap.items[0].counts["counts"]["1"] == 3

    def test_samples_bitstring_single_circuit(self):
        """
        Converts bitstring samples to counts for single circuit.

        qml.sample(wires=[0,1]) returns 2D array: (shots, num_wires)
        Each row is a bitstring sample like [0, 1] -> "01"
        """
        # 4 shots, 2 wires: each row is a sample
        samples = np.array(
            [
                [0, 0],  # "00"
                [0, 1],  # "01"
                [0, 0],  # "00"
                [1, 1],  # "11"
            ]
        )

        snap = build_result_snapshot(
            samples,
            result_type="Sample",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].counts["shots"] == 4

        counts = snap.items[0].counts["counts"]
        assert counts["00"] == 2
        assert counts["01"] == 1
        assert counts["11"] == 1

    def test_samples_eigenvalue_single_circuit(self):
        """
        Converts eigenvalue samples to counts for single circuit.

        qml.sample(qml.PauliZ(0)) returns 1D array of eigenvalues: [1, -1, 1, ...]
        These should be converted to string keys like "1", "-1"
        """
        # Eigenvalue samples from PauliZ
        samples = np.array([1, -1, 1, 1, -1])

        snap = build_result_snapshot(
            samples,
            result_type="Sample",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].counts["shots"] == 5

        counts = snap.items[0].counts["counts"]
        assert counts["1"] == 3
        assert counts["-1"] == 2

    def test_samples_batch_circuits(self):
        """Handles batch of sample results."""
        # Two circuits, each with samples
        samples_batch = [
            np.array([[0, 0], [0, 1], [1, 1]]),  # Circuit 0: 3 shots
            np.array([[1, 0], [1, 0]]),  # Circuit 1: 2 shots
        ]

        snap = build_result_snapshot(
            samples_batch,
            result_type="Sample",
            backend_name="default.qubit",
            num_circuits=2,
        )

        assert len(snap.items) == 2
        assert snap.items[0].counts["shots"] == 3
        assert snap.items[1].counts["shots"] == 2

    def test_probabilities_single_circuit_flat(self):
        """
        Converts 1D probability array for single circuit.

        This is the critical P0 bug: qml.probs() returns flat 1D array
        for single circuit, not nested [[...]].
        """
        # Single circuit: flat 1D probability array
        probs = np.array([0.5, 0.5, 0.0, 0.0])  # 2 qubits: |00>, |01>, |10>, |11>

        snap = build_result_snapshot(
            probs,
            result_type="Probability",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.items[0].counts["shots"] is None  # Probabilities don't have shots

        dist = snap.items[0].counts["counts"]
        assert set(dist.keys()) == {"00", "01"}
        assert abs(dist["00"] - 0.5) < 1e-9
        assert abs(dist["01"] - 0.5) < 1e-9

    def test_probabilities_batch_circuits(self):
        """Converts probability arrays for batch of circuits."""
        # Two circuits with probability results
        probs_batch = [
            np.array([0.5, 0.5, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]

        snap = build_result_snapshot(
            probs_batch,
            result_type="Probability",
            backend_name="default.qubit",
            num_circuits=2,
        )

        assert snap.status == "completed"
        assert len(snap.items) == 2

        # First circuit
        assert abs(sum(snap.items[0].counts["counts"].values()) - 1.0) < 1e-9

        # Second circuit (uniform)
        dist1 = snap.items[1].counts["counts"]
        assert len(dist1) == 4
        for v in dist1.values():
            assert abs(v - 0.25) < 1e-9

    def test_unknown_type_goes_to_other(self):
        """Unknown result type stores in metadata."""
        snap = build_result_snapshot(
            "raw-result",
            result_type=None,
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert snap.metadata["pennylane_result_type"] is None
        assert len(snap.items) == 0  # No structured items for unknown type

    def test_failed_execution_snapshot(self):
        """Builds snapshot for failed execution."""
        snap = build_result_snapshot(
            None,
            result_type=None,
            backend_name="default.qubit",
            num_circuits=1,
            success=False,
            error_info={"type": "RuntimeError", "message": "Backend failed"},
        )

        assert snap.success is False
        assert snap.status == "failed"
        assert snap.error.type == "RuntimeError"
        assert snap.error.message == "Backend failed"
        assert len(snap.items) == 0


class TestExtractProbabilities:
    """Tests for probability extraction edge cases."""

    def test_single_circuit_1d_array(self):
        """Handles single circuit as flat 1D array."""
        probs = np.array([0.5, 0.5])

        counts = _extract_probabilities(probs, num_circuits=1)

        assert len(counts) == 1
        assert counts[0].circuit_index == 0
        assert "0" in counts[0].counts
        assert "1" in counts[0].counts

    def test_filters_near_zero_probs(self):
        """Filters out near-zero probabilities."""
        probs = np.array([0.5, 0.5, 1e-15, 1e-15])

        counts = _extract_probabilities(probs, num_circuits=1)

        assert len(counts) == 1
        assert len(counts[0].counts) == 2  # Only non-zero probs


class TestExtractSampleCounts:
    """Tests for sample count extraction edge cases."""

    def test_2d_samples_correct_shape(self):
        """2D samples array: (shots, num_wires)."""
        samples = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )

        counts = _extract_sample_counts(samples, num_circuits=1)

        assert len(counts) == 1
        assert counts[0].shots == 4
        assert len(counts[0].counts) == 4  # All unique bitstrings

    def test_1d_samples_single_wire(self):
        """1D samples array for single wire."""
        samples = np.array([0, 1, 0, 0, 1])

        counts = _extract_sample_counts(samples, num_circuits=1)

        assert len(counts) == 1
        assert counts[0].shots == 5
        assert counts[0].counts["0"] == 3
        assert counts[0].counts["1"] == 2


class TestEdgeCases:
    """Tests for edge cases in result extraction."""

    def test_variance_type(self):
        """Detects variance measurement type."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.var(qml.PauliZ(0))

        result_type = extract_result_type([tape])
        assert "Variance" in result_type or "var" in result_type.lower()

    def test_state_type(self):
        """Detects state measurement type."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.state()

        result_type = extract_result_type([tape])
        assert "State" in result_type or "state" in result_type.lower()

    def test_multi_measurement_tape(self, multi_measurement_tape):
        """Handles tape with multiple different measurements."""
        # First measurement determines type
        result_type = _result_type_for_tape(multi_measurement_tape)
        assert "Expectation" in result_type or "expval" in result_type.lower()

    def test_empty_results(self):
        """Handles empty results gracefully."""
        snap = build_result_snapshot(
            [],
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=0,
        )

        assert snap.success is True
        assert len(snap.items) == 0  # No items for empty results


class TestResultTypeMapping:
    """Tests for result type stored in metadata."""

    def test_sample_maps_correctly(self):
        """Sample type is stored in metadata."""
        snap = build_result_snapshot(
            np.array([0, 1, 0]),
            result_type="Sample",
            backend_name="default.qubit",
            num_circuits=1,
        )
        assert snap.metadata["pennylane_result_type"] == "Sample"
        assert snap.status == "completed"

    def test_counts_maps_correctly(self):
        """Counts type is stored in metadata."""
        snap = build_result_snapshot(
            {"0": 5, "1": 5},
            result_type="Counts",
            backend_name="default.qubit",
            num_circuits=1,
        )
        assert snap.metadata["pennylane_result_type"] == "Counts"
        assert snap.status == "completed"

    def test_state_maps_correctly(self):
        """State type is stored in metadata."""
        snap = build_result_snapshot(
            np.array([0.707, 0.707]),
            result_type="State",
            backend_name="default.qubit",
            num_circuits=1,
        )
        assert snap.metadata["pennylane_result_type"] == "State"
        assert snap.status == "completed"
