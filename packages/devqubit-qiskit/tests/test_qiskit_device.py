# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for device snapshot and calibration extraction."""

from datetime import datetime

import pytest
from devqubit_qiskit.device import (
    _extract_connectivity_from_coupling_map,
    _extract_from_target,
    create_device_snapshot,
    extract_calibration_from_properties,
)
from devqubit_qiskit.utils import (
    as_int_tuple,
    convert_duration_to_ns,
    convert_time_to_us,
    qiskit_version,
    to_float,
)


# =============================================================================
# Unit Conversion Tests
# =============================================================================


class TestUnitConversions:
    """Tests for unit conversion utilities."""

    def test_time_to_microseconds(self):
        """Converts various time units to microseconds."""
        assert convert_time_to_us(1.0, "s") == 1e6
        assert convert_time_to_us(0.001, "sec") == 1e3
        assert convert_time_to_us(1.0, "ms") == 1e3
        assert convert_time_to_us(100.0, "us") == 100.0
        assert convert_time_to_us(100.0, "µs") == 100.0
        assert convert_time_to_us(1000.0, "ns") == 1.0
        assert convert_time_to_us(100.0, None) == 100.0

    def test_duration_to_nanoseconds(self):
        """Converts various duration units to nanoseconds."""
        assert convert_duration_to_ns(1.0, "s") == 1e9
        assert convert_duration_to_ns(1.0, "us") == 1e3
        assert convert_duration_to_ns(35.5, "µs") == 35500.0
        assert convert_duration_to_ns(100.0, "ns") == 100.0
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


# =============================================================================
# Calibration Extraction Tests
# =============================================================================


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


class TestDerivedCalibrationValues:
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
        assert cal.median_t1_us == pytest.approx(150.0)


class TestRealisticCalibrationData:
    """Tests with realistic provider properties."""

    def test_full_extraction_from_realistic_properties(self, mock_properties):
        """Full extraction from realistic IBM-like properties."""
        props = mock_properties.to_dict()
        cal = extract_calibration_from_properties(props)

        assert cal is not None
        assert cal.source == "provider"
        assert len(cal.qubits) == 2
        assert len(cal.gates) == 3

        assert cal.qubits[0].t1_us == pytest.approx(150.0)
        assert cal.qubits[0].t2_us == pytest.approx(85.0)
        assert cal.qubits[0].readout_error == pytest.approx(0.012)
        assert cal.qubits[0].gate_error_1q == pytest.approx(0.0002)

        cx = next(g for g in cal.gates if g.gate == "cx")
        assert cx.qubits == (0, 1)
        assert cx.error == pytest.approx(0.008)


# =============================================================================
# Connectivity Extraction Tests
# =============================================================================


class TestConnectivityExtraction:
    """Tests for extracting connectivity from coupling maps."""

    def test_from_coupling_map(self, mock_coupling_map):
        """Extracts edges from CouplingMap-like object."""
        conn = _extract_connectivity_from_coupling_map(mock_coupling_map)
        assert conn == [(0, 1), (1, 2), (2, 3)]

    def test_from_coupling_map_no_method(self):
        """Object without get_edges returns None."""

        class NoGetEdges:
            pass

        assert _extract_connectivity_from_coupling_map(NoGetEdges()) is None

    def test_from_coupling_map_handles_exception(self):
        """Gracefully handles exception in get_edges."""

        class BrokenCouplingMap:
            def get_edges(self):
                raise RuntimeError("Failed")

        assert _extract_connectivity_from_coupling_map(BrokenCouplingMap()) is None


# =============================================================================
# Target Extraction Tests
# =============================================================================


class TestTargetExtraction:
    """Tests for extracting info from BackendV2 Target."""

    def test_extracts_all_fields(self, mock_target):
        """Extracts num_qubits, connectivity, and native gates."""
        nq, conn, gates, raw = _extract_from_target(mock_target)

        assert nq == 4
        assert conn == [(0, 1), (1, 2), (2, 3)]
        assert gates == ["cx", "rz", "sx", "x"]
        assert raw["num_qubits"] == 4

    def test_handles_none_target(self):
        """None target returns all None values."""
        nq, conn, gates, raw = _extract_from_target(None)

        assert nq is None
        assert conn is None
        assert gates is None
        assert raw == {}

    def test_handles_partial_target(self):
        """Handles target with only num_qubits."""

        class PartialTarget:
            num_qubits = 5

        nq, conn, gates, raw = _extract_from_target(PartialTarget())

        assert nq == 5
        assert conn is None
        assert gates is None


# =============================================================================
# Device Snapshot Creation Tests
# =============================================================================


class TestCreateDeviceSnapshot:
    """Tests for complete device snapshot creation."""

    def test_aer_simulator_basic_fields(self, aer_simulator):
        """Creates snapshot with basic fields from AerSimulator."""
        snapshot = create_device_snapshot(aer_simulator)

        assert snapshot.provider in ("aer", "qiskit")
        assert "aer_simulator" in snapshot.backend_name.lower()
        assert snapshot.backend_type == "simulator"
        assert snapshot.captured_at is not None

    def test_aer_simulator_sdk_version(self, aer_simulator):
        """Snapshot includes SDK version information."""
        snapshot = create_device_snapshot(aer_simulator)

        assert snapshot.sdk_versions is not None
        assert "qiskit" in snapshot.sdk_versions
        assert snapshot.sdk_versions["qiskit"] == qiskit_version()

    def test_backendv2_with_full_target(self, mock_backend):
        """Extracts all info from BackendV2 with complete Target."""
        snapshot = create_device_snapshot(mock_backend)

        assert snapshot.backend_name == "mock_backend"
        assert snapshot.num_qubits == 4
        assert snapshot.connectivity == [(0, 1), (1, 2), (2, 3)]
        assert snapshot.native_gates == ["cx", "rz", "sx", "x"]

    def test_backend_with_calibration(self, mock_backend_with_calibration):
        """Snapshot includes calibration from properties."""
        snapshot = create_device_snapshot(mock_backend_with_calibration)

        assert snapshot.calibration is not None
        assert snapshot.calibration.qubits[0].t1_us == 150.0
        assert snapshot.calibration.qubits[0].t2_us == 85.0

    def test_minimal_backend(self):
        """Handles backend with no extra attributes."""

        class MinimalBackend:
            pass

        snapshot = create_device_snapshot(MinimalBackend())

        assert snapshot.backend_name == "MinimalBackend"
        assert snapshot.provider == "local"
        assert snapshot.num_qubits is None

    def test_snapshot_without_tracker_has_no_ref(self, aer_simulator):
        """DeviceSnapshot without tracker has raw_properties_ref=None."""
        snapshot = create_device_snapshot(aer_simulator, tracker=None)
        assert snapshot.raw_properties_ref is None


# =============================================================================
# UEC Compliance Tests
# =============================================================================


class TestDeviceSnapshotUECCompliance:
    """Tests for DeviceSnapshot UEC compliance."""

    def test_required_fields_present(self, aer_simulator):
        """All required DeviceSnapshot fields are present."""
        snapshot = create_device_snapshot(aer_simulator)

        assert snapshot.captured_at is not None
        assert snapshot.backend_name is not None
        assert snapshot.backend_type is not None
        assert snapshot.provider is not None

    def test_captured_at_is_iso_format(self, aer_simulator):
        """captured_at is in ISO 8601 format."""
        snapshot = create_device_snapshot(aer_simulator)
        ts = snapshot.captured_at.replace("Z", "+00:00")
        datetime.fromisoformat(ts)

    def test_sdk_versions_format(self, aer_simulator):
        """sdk_versions is a dict mapping package names to versions."""
        snapshot = create_device_snapshot(aer_simulator)

        assert isinstance(snapshot.sdk_versions, dict)
        assert "qiskit" in snapshot.sdk_versions
        assert isinstance(snapshot.sdk_versions["qiskit"], str)

    def test_connectivity_format(self, mock_backend):
        """Connectivity is list of (int, int) tuples."""
        snapshot = create_device_snapshot(mock_backend)

        assert snapshot.connectivity is not None
        for edge in snapshot.connectivity:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert all(isinstance(q, int) for q in edge)

    def test_calibration_serializes_to_expected_structure(self, mock_properties):
        """Calibration serializes to expected dict structure."""
        props = mock_properties.to_dict()
        cal = extract_calibration_from_properties(props)
        d = cal.to_dict()

        assert "source" in d
        assert isinstance(d["source"], str)
        assert "qubits" in d
        assert "gates" in d

        q = d["qubits"][0]
        assert "qubit" in q
        assert isinstance(q["qubit"], int)

        g = d["gates"][0]
        assert "gate" in g
        assert "qubits" in g
