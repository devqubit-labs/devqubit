# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Braket calibration extraction."""

import pytest
from devqubit_braket.calibration import (
    _parse_qubit_key,
    extract_calibration_from_device,
)


class TestParseQubitKey:
    """Tests for Braket-style qubit key parsing."""

    def test_parses_common_formats(self):
        """Parses various qubit key formats."""
        assert _parse_qubit_key("0-1") == [0, 1]
        assert _parse_qubit_key("(0,1)") == [0, 1]
        assert _parse_qubit_key("[0, 1]") == [0, 1]
        assert _parse_qubit_key("0,1,2") == [0, 1, 2]

    def test_parses_formats_with_spaces(self):
        """Parses qubit keys with various spacing (real payloads can be messy)."""
        assert _parse_qubit_key("0 - 1") == [0, 1]
        assert _parse_qubit_key("( 0 , 1 )") == [0, 1]
        assert _parse_qubit_key("[ 0, 1 ]") == [0, 1]
        assert _parse_qubit_key("  0  ,  1  ") == [0, 1]

    def test_parses_tuple_style_formats(self):
        """Parses tuple-style formats from different providers."""
        assert _parse_qubit_key("(0, 1)") == [0, 1]
        assert _parse_qubit_key("(12, 13)") == [12, 13]

    def test_parses_single_qubit(self):
        """Parses single qubit key."""
        assert _parse_qubit_key("0") == [0]
        assert _parse_qubit_key("42") == [42]

    def test_invalid_returns_empty(self):
        """Returns empty list for invalid input."""
        assert _parse_qubit_key("invalid") == []
        assert _parse_qubit_key("") == []
        assert _parse_qubit_key("a-b") == []
        assert _parse_qubit_key("one,two") == []


class TestExtractCalibration:
    """Tests for extracting calibration from devices."""

    def test_returns_none_without_properties(self):
        """Returns None when device has no properties."""

        class NoPropsDevice:
            pass

        assert extract_calibration_from_device(NoPropsDevice()) is None

    def test_returns_none_without_standardized_section(self, device_factory):
        """Returns None without standardized properties."""
        device = device_factory(name="no_cal", qubit_count=2)
        assert extract_calibration_from_device(device) is None

    def test_extracts_single_qubit_gate_errors(self, device_factory):
        """Extracts 1Q gate errors from fidelities."""
        device = device_factory(
            name="1q_cal",
            qubit_count=2,
            one_qubit_fidelities={
                0: [
                    {"gateName": "x", "fidelity": 0.99},
                    {"gateName": "rz", "fidelity": 0.995},
                ],
                1: [{"gateName": "x", "fidelity": 0.98}],
            },
        )

        cal = extract_calibration_from_device(device)

        assert cal is not None
        assert len(cal.gates) == 3  # Two gates on q0, one on q1

        # Check error = 1 - fidelity
        x_q0 = [g for g in cal.gates if g.gate == "x" and g.qubits == (0,)]
        assert len(x_q0) == 1
        assert x_q0[0].error == pytest.approx(0.01)

    def test_extracts_two_qubit_gate_errors(self, device_factory):
        """Extracts 2Q gate errors from fidelities."""
        device = device_factory(
            name="2q_cal",
            qubit_count=2,
            two_qubit_fidelities={
                "0-1": [{"gateName": "cz", "fidelity": 0.90}],
            },
        )

        cal = extract_calibration_from_device(device)

        assert cal is not None
        assert len(cal.gates) == 1

        cz = cal.gates[0]
        assert cz.gate == "cz"
        assert cz.qubits == (0, 1)
        assert cz.error == pytest.approx(0.10)

    def test_extracts_two_qubit_with_various_key_formats(self, device_factory):
        """Extracts 2Q gates with various edge key formats."""
        # Test various key formats that might appear in real payloads
        test_cases = [
            ("0-1", (0, 1)),
            ("(0,1)", (0, 1)),
            ("(0, 1)", (0, 1)),
            ("[0,1]", (0, 1)),
            ("0,1", (0, 1)),
        ]

        for key_format, expected_qubits in test_cases:
            device = device_factory(
                name=f"test_{key_format}",
                qubit_count=2,
                two_qubit_fidelities={
                    key_format: [{"gateName": "cz", "fidelity": 0.95}]
                },
            )
            cal = extract_calibration_from_device(device)

            assert cal is not None, f"Failed for key format: {key_format}"
            assert len(cal.gates) == 1, f"Failed for key format: {key_format}"
            assert (
                cal.gates[0].qubits == expected_qubits
            ), f"Failed for key format: {key_format}"

    def test_computes_qubit_median_errors(self, device_factory):
        """Computes median 1Q errors per qubit."""
        device = device_factory(
            name="median_cal",
            qubit_count=2,
            one_qubit_fidelities={
                0: [
                    {"gateName": "x", "fidelity": 0.99},  # error 0.01
                    {"gateName": "rz", "fidelity": 0.98},  # error 0.02
                ],
                1: [{"gateName": "x", "fidelity": 0.97}],  # error 0.03
            },
        )

        cal = extract_calibration_from_device(device)
        assert cal is not None

        # q0 median([0.01, 0.02]) = 0.015
        q0 = [q for q in cal.qubits if q.qubit == 0]
        assert len(q0) == 1
        assert q0[0].gate_error_1q == pytest.approx(0.015)

        # q1 median([0.03]) = 0.03
        q1 = [q for q in cal.qubits if q.qubit == 1]
        assert len(q1) == 1
        assert q1[0].gate_error_1q == pytest.approx(0.03)

    def test_qpu_calibration(self, mock_device):
        """Extracts calibration from QPU properties."""
        cal = extract_calibration_from_device(mock_device)

        assert cal is not None
        assert cal.calibration_time == "2025-01-02T03:04:05Z"

        # Should have 1Q and 2Q gates
        assert len(cal.gates) > 0
        assert any(len(g.qubits) == 1 for g in cal.gates)
        assert any(len(g.qubits) == 2 for g in cal.gates)

        # Median 2Q error should be computed
        assert cal.median_2q_error is not None
        assert 0 < cal.median_2q_error < 1

    def test_ignores_missing_fidelity_values(self, device_factory):
        """Ignores entries without fidelity values."""
        device = device_factory(
            name="partial_cal",
            qubit_count=2,
            one_qubit_fidelities={
                0: [
                    {"gateName": "x"},  # No fidelity
                    {"gateName": "rz", "fidelity": 0.99},  # Has fidelity
                ]
            },
        )

        cal = extract_calibration_from_device(device)

        # Should only extract the valid entry
        assert cal is not None
        assert len(cal.gates) == 1
        assert cal.gates[0].gate == "rz"

    def test_handles_negative_errors_gracefully(self, device_factory):
        """Clamps negative errors to zero."""
        device = device_factory(
            name="high_fidelity",
            qubit_count=1,
            one_qubit_fidelities={
                0: [{"gateName": "x", "fidelity": 1.05}]  # > 1.0 fidelity
            },
        )

        cal = extract_calibration_from_device(device)

        assert cal is not None
        assert cal.gates[0].error == 0.0

    def test_handles_fidelity_exactly_one(self, device_factory):
        """Handles fidelity = 1.0 (perfect gate)."""
        device = device_factory(
            name="perfect_gate",
            qubit_count=1,
            one_qubit_fidelities={0: [{"gateName": "x", "fidelity": 1.0}]},
        )

        cal = extract_calibration_from_device(device)

        assert cal is not None
        assert cal.gates[0].error == 0.0

    def test_handles_alternative_field_names(self, device_factory):
        """Handles alternative field naming conventions."""
        # Test with standard field names (device_factory uses standard format)
        device = device_factory(
            name="alt_names",
            qubit_count=2,
            one_qubit_fidelities={0: [{"gateName": "x", "fidelity": 0.99}]},
        )
        cal = extract_calibration_from_device(device)

        # Should extract calibration
        assert cal is not None
        assert len(cal.gates) == 1

    def test_handles_exception_in_properties_access(self):
        """Handles exceptions during properties access gracefully."""

        class ExplodingDevice:
            @property
            def properties(self):
                raise RuntimeError("Calibration service unavailable")

        cal = extract_calibration_from_device(ExplodingDevice())
        assert cal is None  # Should not crash
