# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Braket result processing."""

import pytest
from devqubit_braket.results import extract_counts_payload, extract_measurement_counts


class TestExtractMeasurementCounts:
    """Tests for single result measurement counts extraction."""

    def test_extracts_from_real_result(self, local_simulator, bell_circuit):
        """Extracts counts from real Braket result."""
        circuit = bell_circuit.measure([0, 1])
        task = local_simulator.run(circuit, shots=100)
        result = task.result()

        counts = extract_measurement_counts(result)

        assert counts is not None
        assert isinstance(counts, dict)
        assert all(isinstance(k, str) for k in counts.keys())
        assert all(isinstance(v, int) for v in counts.values())
        assert sum(counts.values()) == 100

    def test_bell_state_counts_distribution(self, local_simulator, bell_circuit):
        """Bell state produces 00 and 11 outcomes."""
        circuit = bell_circuit.measure([0, 1])
        task = local_simulator.run(circuit, shots=1000)
        result = task.result()

        counts = extract_measurement_counts(result)

        # Bell state should only have 00 and 11
        assert len(counts) <= 2
        if "00" in counts:
            assert counts["00"] > 0
        if "11" in counts:
            assert counts["11"] > 0

    def test_returns_none_for_none_result(self):
        """Returns None when result is None."""
        assert extract_measurement_counts(None) is None

    def test_returns_none_for_invalid_result(self):
        """Returns None for objects without measurement_counts."""

        class FakeResult:
            pass

        assert extract_measurement_counts(FakeResult()) is None


class TestExtractCountsPayload:
    """Tests for devqubit-style counts payload extraction."""

    def test_creates_payload_from_single_result(self, local_simulator, bell_circuit):
        """Creates payload structure from single result."""
        circuit = bell_circuit.measure([0, 1])
        task = local_simulator.run(circuit, shots=100)
        result = task.result()

        payload = extract_counts_payload(result)

        assert payload is not None
        assert "experiments" in payload
        assert len(payload["experiments"]) == 1

        exp = payload["experiments"][0]
        assert exp["index"] == 0
        assert "counts" in exp
        assert isinstance(exp["counts"], dict)
        assert sum(exp["counts"].values()) == 100

    def test_returns_none_for_none_result(self):
        """Returns None when result is None."""
        assert extract_counts_payload(None) is None


class TestMockResults:
    """Tests with mock result objects."""

    def test_extracts_from_measurement_counts_attr(self):
        """Extracts from measurement_counts attribute."""

        class MockResult:
            measurement_counts = {"00": 50, "11": 50}

        counts = extract_measurement_counts(MockResult())
        assert counts == {"00": 50, "11": 50}

    def test_extracts_from_callable_measurement_counts(self):
        """Extracts from callable measurement_counts."""

        class MockResult:
            def measurement_counts(self):
                return {"00": 30, "11": 70}

        counts = extract_measurement_counts(MockResult())
        assert counts == {"00": 30, "11": 70}

    def test_normalizes_keys_to_strings(self):
        """Normalizes keys to strings."""

        class MockResult:
            measurement_counts = {0: 50, 1: 50}

        counts = extract_measurement_counts(MockResult())
        assert counts == {"0": 50, "1": 50}

    def test_handles_counter_like_objects(self):
        """Handles Counter-like objects."""
        from collections import Counter

        class MockResult:
            measurement_counts = Counter({"00": 50, "11": 50})

        counts = extract_measurement_counts(MockResult())
        assert counts == {"00": 50, "11": 50}


class TestProgramSetResults:
    """Tests for Program Set result structure."""

    def test_handles_nested_entries_structure(self):
        """Handles nested entries structure from Program Set results."""

        class MockMeasuredEntry:
            counts = {"00": 50, "11": 50}

        class MockCompositeEntry:
            entries = [MockMeasuredEntry(), MockMeasuredEntry()]

        class MockProgramSetResult:
            entries = [MockCompositeEntry(), MockCompositeEntry()]

        payload = extract_counts_payload(MockProgramSetResult())

        assert payload is not None
        assert len(payload["experiments"]) == 4

        # Check indices
        for i, exp in enumerate(payload["experiments"]):
            assert exp["index"] == i
            assert "program_index" in exp
            assert "executable_index" in exp
            assert exp["counts"] == {"00": 50, "11": 50}

    def test_handles_empty_entries(self):
        """Handles empty entries list."""

        class MockProgramSetResult:
            entries = []

        payload = extract_counts_payload(MockProgramSetResult())
        assert payload is None

    def test_handles_mixed_success_failure_entries(self):
        """Handles entries with some missing counts (partial failures)."""

        class MockMeasuredEntrySuccess:
            counts = {"00": 100}

        class MockMeasuredEntryFail:
            counts = None  # Failed entry

        class MockCompositeEntry:
            entries = [MockMeasuredEntrySuccess(), MockMeasuredEntryFail()]

        class MockProgramSetResult:
            entries = [MockCompositeEntry()]

        payload = extract_counts_payload(MockProgramSetResult())

        # Should extract successful entries only
        assert payload is not None
        assert len(payload["experiments"]) == 1
        assert payload["experiments"][0]["counts"] == {"00": 100}


class TestShotsZeroBehavior:
    """Tests for shots=0 (exact/analytical) mode behavior."""

    def test_shots_zero_requires_result_types(self, local_simulator):
        """
        With shots=0, Braket requires result_types to be specified.
        This documents the expected Braket behavior.
        """
        from braket.circuits import Circuit

        # Circuit without result types
        circuit = Circuit().h(0).cnot(0, 1)

        # shots=0 without result_types should raise ValueError
        with pytest.raises(ValueError, match="result types"):
            local_simulator.run(circuit, shots=0)

    def test_shots_zero_with_result_types(self, local_simulator):
        """
        shots=0 with result_types returns analytical results (no counts).
        """
        from braket.circuits import Circuit

        # Circuit with result type for analytical simulation
        circuit = Circuit().h(0).cnot(0, 1)
        circuit.state_vector()

        task = local_simulator.run(circuit, shots=0)
        result = task.result()

        # Analytical mode - measurement_counts may be empty/unavailable
        counts = extract_measurement_counts(result)

        # Either None or empty dict is acceptable for shots=0 analytical
        if counts is not None:
            assert isinstance(counts, dict)

    def test_shots_zero_with_measurements_requires_result_types(self, local_simulator):
        """
        shots=0 with measurement instructions still requires result_types.
        """
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1).measure([0, 1])

        # shots=0 without result_types should raise ValueError
        with pytest.raises(ValueError, match="result types"):
            local_simulator.run(circuit, shots=0)


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_handles_very_large_counts(self):
        """Handles large shot counts."""

        class MockResult:
            measurement_counts = {
                "00": 500000,
                "01": 250000,
                "10": 150000,
                "11": 100000,
            }

        counts = extract_measurement_counts(MockResult())
        assert sum(counts.values()) == 1000000

    def test_handles_many_bitstrings(self):
        """Handles results with many unique bitstrings."""
        # Simulate a 10-qubit result with many outcomes
        many_outcomes = {format(i, "010b"): 1 for i in range(100)}

        class MockResult:
            measurement_counts = many_outcomes

        counts = extract_measurement_counts(MockResult())
        assert len(counts) == 100

    def test_handles_measurement_counts_exception(self):
        """Handles exception when accessing measurement_counts."""

        class ExplodingResult:
            @property
            def measurement_counts(self):
                raise RuntimeError("Counts unavailable")

        # Should return None, not crash
        counts = extract_measurement_counts(ExplodingResult())
        assert counts is None

    def test_handles_non_dict_counts_attribute(self):
        """Handles cases where counts attribute exists but isn't dict-like."""

        class BadResult:
            measurement_counts = "not a dict"

        counts = extract_measurement_counts(BadResult())
        assert counts is None

    def test_payload_experiments_have_consistent_structure(
        self, local_simulator, bell_circuit
    ):
        """All experiments in payload have consistent structure."""
        circuit = bell_circuit.measure([0, 1])
        task = local_simulator.run(circuit, shots=100)
        result = task.result()

        payload = extract_counts_payload(result)

        required_keys = {"index", "counts"}
        for exp in payload["experiments"]:
            assert required_keys.issubset(set(exp.keys()))
            assert isinstance(exp["index"], int)
            assert isinstance(exp["counts"], dict)
