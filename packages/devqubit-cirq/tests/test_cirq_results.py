# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Cirq result processing."""

import cirq
import numpy as np
import pytest
from devqubit_cirq.results import (
    _extract_params,
    counts_from_measurements,
    get_result_measurements,
    normalize_counts_payload,
)


class TestGetResultMeasurements:
    """Tests for measurement extraction from results."""

    def test_extracts_measurements_dict(self):
        """Extracts measurements from result with dict attribute."""

        class MockResult:
            measurements = {"m": np.array([[0, 1], [1, 0]])}

        result = MockResult()
        meas = get_result_measurements(result)

        assert meas == result.measurements

    def test_returns_empty_for_missing_or_invalid(self):
        """Returns empty dict if measurements missing or invalid."""

        class MockResult:
            pass

        assert get_result_measurements(MockResult()) == {}

        class MockResultBadType:
            measurements = "not a dict"

        assert get_result_measurements(MockResultBadType()) == {}


class TestCountsFromMeasurements:
    """Tests for building counts from measurement arrays."""

    def test_single_key_measurements(self):
        """Builds counts from single measurement key."""
        measurements = {"m": np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]])}
        counts, keys, nbits = counts_from_measurements(measurements)

        assert keys == ["m"]
        assert nbits == 2
        assert counts["00"] == 2
        assert counts["01"] == 1
        assert counts["10"] == 1
        assert counts["11"] == 1

    def test_multiple_keys_sorted(self):
        """Concatenates multiple keys in sorted order."""
        measurements = {
            "b": np.array([[0], [1], [0], [1]]),
            "a": np.array([[0], [0], [1], [1]]),
        }
        counts, keys, nbits = counts_from_measurements(measurements)

        assert keys == ["a", "b"]  # Sorted order
        assert nbits == 2
        assert counts == {"00": 1, "01": 1, "10": 1, "11": 1}

    def test_empty_measurements(self):
        """Handles empty measurements gracefully."""
        counts, keys, nbits = counts_from_measurements({})

        assert counts == {}
        assert keys == []
        assert nbits == 0

    def test_1d_array_reshaped(self):
        """1D arrays are reshaped to 2D."""
        measurements = {"m": np.array([0, 1, 0, 1])}
        counts, keys, nbits = counts_from_measurements(measurements)

        assert nbits == 1
        assert counts["0"] == 2
        assert counts["1"] == 2

    def test_bool_dtype_measurements(self):
        """Handles boolean dtype measurements (common in Cirq).

        Cirq often returns measurement arrays with dtype=bool.
        This must be handled correctly.
        """
        measurements = {
            "m": np.array([[False, True], [True, False], [False, True]], dtype=bool)
        }
        counts, keys, nbits = counts_from_measurements(measurements)

        assert keys == ["m"]
        assert nbits == 2
        assert counts["01"] == 2
        assert counts["10"] == 1

    def test_mixed_bool_and_int(self):
        """Handles mixed bool and int measurement arrays."""
        measurements = {
            "a": np.array([[False], [True], [False]], dtype=bool),
            "b": np.array([[0], [1], [1]], dtype=int),
        }
        counts, keys, nbits = counts_from_measurements(measurements)

        assert keys == ["a", "b"]
        assert nbits == 2
        assert counts == {"00": 1, "11": 1, "01": 1}

    def test_raises_on_inconsistent_repetitions(self):
        """Raises ValueError when measurement arrays have different rep counts.

        This guards against silent data corruption when arrays are misaligned.
        """
        measurements = {
            "a": np.array([[0], [1], [0]]),  # 3 reps
            "b": np.array([[0], [1]]),  # 2 reps - inconsistent!
        }

        with pytest.raises(ValueError, match="Inconsistent repetition counts"):
            counts_from_measurements(measurements)


class TestNormalizeCountsPayload:
    """Tests for normalizing results to canonical format."""

    def test_single_result(self):
        """Normalizes single result object."""

        class MockResult:
            measurements = {"m": np.array([[0], [1], [0]])}

        payload = normalize_counts_payload(MockResult())

        assert len(payload["experiments"]) == 1
        exp = payload["experiments"][0]
        assert exp["index"] == 0
        assert exp["counts"]["0"] == 2
        assert exp["counts"]["1"] == 1
        assert exp["measurement_keys"] == ["m"]
        assert exp["num_bits"] == 1

        # P1 FIX: Verify format metadata is present
        assert "format" in payload
        assert payload["format"]["source_sdk"] == "cirq"
        assert payload["format"]["bit_order"] == "cbit0_left"

    def test_list_of_results(self):
        """Normalizes list of results (run_sweep output)."""

        class MockResult:
            def __init__(self, val):
                self.measurements = {"m": np.array([[val]])}

        results = [MockResult(0), MockResult(1), MockResult(0)]
        payload = normalize_counts_payload(results)

        assert len(payload["experiments"]) == 3
        assert payload["experiments"][0]["counts"] == {"0": 1}
        assert payload["experiments"][1]["counts"] == {"1": 1}
        assert payload["experiments"][2]["counts"] == {"0": 1}

        # P1 FIX: Format should be present
        assert payload["format"]["source_sdk"] == "cirq"

    def test_nested_results(self):
        """Normalizes nested results (run_batch output)."""

        class MockResult:
            def __init__(self, val):
                self.measurements = {"m": np.array([[val]])}

        results = [
            [MockResult(0), MockResult(1)],
            [MockResult(1), MockResult(0)],
        ]
        payload = normalize_counts_payload(results)

        assert len(payload["experiments"]) == 4
        assert payload["experiments"][0]["batch_index"] == 0
        assert payload["experiments"][0]["sweep_index"] == 0
        assert payload["experiments"][1]["batch_index"] == 0
        assert payload["experiments"][1]["sweep_index"] == 1
        assert payload["experiments"][2]["batch_index"] == 1

        # P1 FIX: Format should be present
        assert payload["format"]["source_sdk"] == "cirq"
        assert payload["format"]["bit_order"] == "cbit0_left"

    def test_empty_or_invalid(self):
        """Returns empty experiments for empty/invalid input."""
        empty_payload = normalize_counts_payload([])
        assert empty_payload["experiments"] == []
        assert "format" in empty_payload  # Format still present

        invalid_payload = normalize_counts_payload("not a result")
        assert invalid_payload["experiments"] == []
        assert "format" in invalid_payload

    def test_extracts_params_from_result(self):
        """Extracts params attribute from result objects."""

        class MockParamResolver:
            param_dict = {"theta": 0.5, "phi": 1.2}

            def items(self):
                return self.param_dict.items()

        class MockResultWithParams:
            measurements = {"m": np.array([[0], [1]])}
            params = MockParamResolver()

        payload = normalize_counts_payload(MockResultWithParams())

        exp = payload["experiments"][0]
        assert "params" in exp
        assert exp["params"]["theta"] == 0.5
        assert exp["params"]["phi"] == 1.2

    def test_params_in_sweep_results(self):
        """Preserves params for each result in sweep."""

        class MockParamResolver:
            def __init__(self, val):
                self.param_dict = {"theta": val}

            def items(self):
                return self.param_dict.items()

        class MockResult:
            def __init__(self, val):
                self.measurements = {"m": np.array([[0]])}
                self.params = MockParamResolver(val)

        results = [MockResult(0.0), MockResult(0.5), MockResult(1.0)]
        payload = normalize_counts_payload(results)

        assert len(payload["experiments"]) == 3
        assert payload["experiments"][0]["params"]["theta"] == 0.0
        assert payload["experiments"][1]["params"]["theta"] == 0.5
        assert payload["experiments"][2]["params"]["theta"] == 1.0

    def test_handles_missing_params(self):
        """Gracefully handles results without params attribute."""

        class MockResult:
            measurements = {"m": np.array([[0]])}
            # No params attribute

        payload = normalize_counts_payload(MockResult())
        exp = payload["experiments"][0]

        # params should not be present (not set to None)
        assert "params" not in exp or exp.get("params") is None


class TestExtractParams:
    """Tests for parameter extraction helper."""

    def test_extracts_from_param_dict(self):
        """Extracts params from ParamResolver-like object."""

        class MockResolver:
            param_dict = {"a": 1.0, "b": 2.5}

        params = _extract_params(type("R", (), {"params": MockResolver()})())
        assert params == {"a": 1.0, "b": 2.5}

    def test_handles_sympy_symbols(self):
        """Converts sympy symbol keys to strings."""
        import sympy

        theta = sympy.Symbol("theta")

        class MockResolver:
            param_dict = {theta: 0.5}

        params = _extract_params(type("R", (), {"params": MockResolver()})())
        assert "theta" in params
        assert params["theta"] == 0.5

    def test_returns_none_for_no_params(self):
        """Returns None when result has no params."""
        assert _extract_params(object()) is None

        class NoParams:
            params = None

        assert _extract_params(NoParams()) is None


class TestRealSimulatorResults:
    """Tests with actual Cirq simulator results."""

    def test_single_real_result(self, simulator):
        """Normalizes real simulator result."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, key="a"),
            cirq.measure(q1, key="b"),
        )
        result = simulator.run(circuit, repetitions=100)
        payload = normalize_counts_payload(result)

        exp = payload["experiments"][0]
        assert exp["measurement_keys"] == ["a", "b"]
        assert sum(exp["counts"].values()) == 100
