# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit calibration extraction."""

import pytest
from devqubit_qiskit.calibration import extract_calibration_from_properties
from devqubit_qiskit.utils import (
    as_int_tuple,
    convert_duration_to_ns,
    convert_time_to_us,
    to_float,
)


class TestUnitConversions:
    """Tests for unit conversion utilities."""

    def test_time_to_microseconds(self):
        """Converts various time units to microseconds."""
        # Seconds
        assert convert_time_to_us(1.0, "s") == 1e6
        assert convert_time_to_us(0.001, "sec") == 1e3

        # Milliseconds
        assert convert_time_to_us(1.0, "ms") == 1e3

        # Microseconds (pass-through)
        assert convert_time_to_us(100.0, "us") == 100.0
        assert convert_time_to_us(100.0, "µs") == 100.0

        # Nanoseconds
        assert convert_time_to_us(1000.0, "ns") == 1.0

        # None assumes microseconds
        assert convert_time_to_us(100.0, None) == 100.0

    def test_duration_to_nanoseconds(self):
        """Converts various duration units to nanoseconds."""
        # Seconds
        assert convert_duration_to_ns(1.0, "s") == 1e9

        # Microseconds
        assert convert_duration_to_ns(1.0, "us") == 1e3
        assert convert_duration_to_ns(35.5, "µs") == 35500.0

        # Nanoseconds (pass-through)
        assert convert_duration_to_ns(100.0, "ns") == 100.0

        # None assumes nanoseconds
        assert convert_duration_to_ns(100.0, None) == 100.0

    def test_to_float_various_inputs(self):
        """Converts various inputs to float."""
        assert to_float(3.14) == 3.14
        assert to_float(42) == 42.0
        assert to_float("3.14") == 3.14
        assert to_float("  42  ") == 42.0
        assert to_float(None) is None
        assert to_float("invalid") is None

    def test_as_int_tuple(self):
        """Converts sequences to tuple of ints."""
        assert as_int_tuple([0, 1]) == (0, 1)
        assert as_int_tuple([1, "2", 3.0]) == (1, 2, 3)
        assert as_int_tuple("invalid") is None
        assert as_int_tuple([1, "bad", 3]) is None


class TestCalibrationExtractionBasics:
    """Tests for basic calibration extraction behavior."""

    def test_empty_or_none_returns_none(self):
        """Empty dict or None returns None."""
        assert extract_calibration_from_properties({}) is None
        assert extract_calibration_from_properties(None) is None

    def test_no_useful_metrics_returns_none(self):
        """Properties without recognized metrics return None."""
        props = {"qubits": [[{"name": "unknown_property", "value": 42}]]}
        assert extract_calibration_from_properties(props) is None


class TestQubitCalibrationExtraction:
    """Tests for qubit calibration extraction."""

    def test_extracts_t1_t2_readout(self):
        """Extracts T1, T2, and readout error."""
        props = {
            "qubits": [
                [
                    {"name": "T1", "value": 150.0, "unit": "us"},
                    {"name": "T2", "value": 85.0, "unit": "us"},
                    {"name": "readout_error", "value": 0.015},
                ]
            ]
        }

        cal = extract_calibration_from_properties(props)

        assert cal is not None
        assert len(cal.qubits) == 1
        assert cal.qubits[0].t1_us == 150.0
        assert cal.qubits[0].t2_us == 85.0
        assert cal.qubits[0].readout_error == 0.015

    def test_derives_readout_error_from_assignment_probabilities(self):
        """Derives readout error from prob_meas0_prep1 / prob_meas1_prep0."""
        props = {
            "qubits": [
                [
                    {"name": "T1", "value": 100.0},
                    {"name": "prob_meas0_prep1", "value": 0.02},
                    {"name": "prob_meas1_prep0", "value": 0.03},
                ]
            ]
        }

        cal = extract_calibration_from_properties(props)

        # Average: (0.02 + 0.03) / 2 = 0.025
        assert cal.qubits[0].readout_error == pytest.approx(0.025)

    def test_extracts_multiple_qubits(self):
        """Extracts calibration for multiple qubits with correct indices."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us"}],
                [{"name": "T1", "value": 150.0, "unit": "us"}],
                [{"name": "T1", "value": 200.0, "unit": "us"}],
            ]
        }

        cal = extract_calibration_from_properties(props)

        assert len(cal.qubits) == 3
        assert cal.qubits[0].qubit == 0
        assert cal.qubits[1].qubit == 1
        assert cal.qubits[2].qubit == 2
        assert cal.qubits[0].t1_us == 100.0
        assert cal.qubits[1].t1_us == 150.0
        assert cal.qubits[2].t1_us == 200.0


class TestGateCalibrationExtraction:
    """Tests for gate calibration extraction."""

    def test_extracts_single_qubit_gate(self):
        """Extracts single-qubit gate error and duration."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.0002},
                        {"name": "gate_length", "value": 35.5, "unit": "ns"},
                    ],
                }
            ],
        }

        cal = extract_calibration_from_properties(props)

        assert len(cal.gates) == 1
        assert cal.gates[0].gate == "sx"
        assert cal.gates[0].qubits == (0,)
        assert cal.gates[0].error == 0.0002
        assert cal.gates[0].duration_ns == 35.5

    def test_extracts_two_qubit_gate(self):
        """Extracts two-qubit gate calibration."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": [
                        {"name": "gate_error", "value": 0.008},
                        {"name": "gate_length", "value": 300.0, "unit": "ns"},
                    ],
                }
            ],
        }

        cal = extract_calibration_from_properties(props)

        assert cal.gates[0].gate == "cx"
        assert cal.gates[0].qubits == (0, 1)
        assert cal.gates[0].error == 0.008
        assert cal.gates[0].duration_ns == 300.0


class TestDerivedValues:
    """Tests for derived calibration values (1Q gate error, medians)."""

    def test_derives_1q_gate_error_from_single_gate(self):
        """Derives 1Q gate error from single gate per qubit."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.0002}],
                }
            ],
        }

        cal = extract_calibration_from_properties(props)
        assert cal.qubits[0].gate_error_1q == 0.0002

    def test_derives_median_from_multiple_gates(self):
        """Derives median 1Q gate error from multiple gates per qubit."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.0001}],
                },
                {
                    "gate": "rz",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.0003}],
                },
                {
                    "gate": "x",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.0002}],
                },
            ],
        }

        cal = extract_calibration_from_properties(props)

        # Median of [0.0001, 0.0002, 0.0003] = 0.0002
        assert cal.qubits[0].gate_error_1q == pytest.approx(0.0002)

    def test_calculates_median_t1_across_qubits(self):
        """Calculates median T1 across all qubits."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us"}],
                [{"name": "T1", "value": 200.0, "unit": "us"}],
                [{"name": "T1", "value": 150.0, "unit": "us"}],
            ]
        }

        cal = extract_calibration_from_properties(props)

        # Median of [100, 150, 200] = 150
        assert cal.median_t1_us == pytest.approx(150.0)


class TestRealisticProviderData:
    """Tests with realistic provider properties (mock_properties fixture)."""

    def test_full_extraction_from_realistic_properties(self, mock_properties):
        """Full extraction from realistic IBM-like properties."""
        props = mock_properties.to_dict()
        cal = extract_calibration_from_properties(props)

        assert cal is not None
        assert cal.source == "provider"
        assert len(cal.qubits) == 2
        assert len(cal.gates) == 3

        # Qubit calibration
        assert cal.qubits[0].t1_us == pytest.approx(150.0)
        assert cal.qubits[0].t2_us == pytest.approx(85.0)
        assert cal.qubits[0].readout_error == pytest.approx(0.012)

        # Derived 1Q gate error
        assert cal.qubits[0].gate_error_1q == pytest.approx(0.0002)
        assert cal.qubits[1].gate_error_1q == pytest.approx(0.0003)

        # Gate calibration
        cx = next(g for g in cal.gates if g.gate == "cx")
        assert cx.qubits == (0, 1)
        assert cx.error == pytest.approx(0.008)
        assert cx.duration_ns == pytest.approx(300.0)

        # Median T1: median(150, 175) = 162.5
        assert cal.median_t1_us == pytest.approx(162.5)


class TestMalformedData:
    """Tests for handling malformed/partial data gracefully."""

    def test_handles_non_list_qubit_entry(self):
        """Skips non-list qubit entries."""
        props = {
            "qubits": [
                "not a list",  # Invalid
                [{"name": "T1", "value": 100.0}],  # Valid
            ],
        }

        cal = extract_calibration_from_properties(props)

        assert cal is not None
        assert len(cal.qubits) == 1
        assert cal.qubits[0].qubit == 1  # Index 1 since index 0 was skipped

    def test_handles_missing_property_fields(self):
        """Skips properties without name or value."""
        props = {
            "qubits": [
                [
                    {"value": 100.0},  # No name
                    {"name": "T1"},  # No value
                    {"name": "T2", "value": 85.0},  # Valid
                ]
            ],
        }

        cal = extract_calibration_from_properties(props)

        assert cal.qubits[0].t1_us is None
        assert cal.qubits[0].t2_us == 85.0

    def test_handles_gate_without_qubits(self):
        """Skips gates without qubits field."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {"gate": "sx", "parameters": [{"name": "gate_error", "value": 0.001}]},
            ],
        }

        cal = extract_calibration_from_properties(props)
        assert len(cal.gates) == 0

    def test_handles_gate_without_metrics(self):
        """Skips gates without error or duration."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
            "gates": [
                {"gate": "sx", "qubits": [0], "parameters": []},
            ],
        }

        cal = extract_calibration_from_properties(props)
        assert len(cal.gates) == 0


class TestSourceTracking:
    """Tests for calibration source tracking."""

    def test_default_source_is_provider(self):
        """Default source is 'provider'."""
        props = {"qubits": [[{"name": "T1", "value": 100.0}]]}
        cal = extract_calibration_from_properties(props)
        assert cal.source == "provider"

    def test_custom_source_preserved(self):
        """Custom source is preserved."""
        props = {"qubits": [[{"name": "T1", "value": 100.0}]]}
        cal = extract_calibration_from_properties(props, source="manual")
        assert cal.source == "manual"


class TestCalibrationSerialization:
    """Tests for Calibration serialization (UEC compliance)."""

    def test_serializes_to_expected_structure(self, mock_properties):
        """Calibration serializes to expected dict structure."""
        props = mock_properties.to_dict()
        cal = extract_calibration_from_properties(props)
        d = cal.to_dict()

        # Required field
        assert "source" in d
        assert isinstance(d["source"], str)

        # Qubit structure
        assert "qubits" in d
        q = d["qubits"][0]
        assert "qubit" in q
        assert isinstance(q["qubit"], int)

        # Gate structure
        assert "gates" in d
        g = d["gates"][0]
        assert "gate" in g
        assert "qubits" in g
        assert isinstance(g["gate"], str)
        assert isinstance(g["qubits"], (list, tuple))
