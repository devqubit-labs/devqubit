# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit device snapshot creation."""

from datetime import datetime

from devqubit_qiskit.snapshot import (
    _extract_calibration,
    _extract_connectivity_from_coupling_map,
    _extract_from_target,
    create_device_snapshot,
)
from devqubit_qiskit.utils import qiskit_version


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


class TestTargetExtraction:
    """Tests for extracting info from BackendV2 Target."""

    def test_extracts_all_fields(self, mock_target):
        """Extracts num_qubits, connectivity, and native gates."""
        nq, conn, gates, raw = _extract_from_target(mock_target)

        assert nq == 4
        assert conn == [(0, 1), (1, 2), (2, 3)]
        assert gates == ["cx", "rz", "sx", "x"]  # Sorted
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

    def test_handles_broken_coupling_map(self):
        """Handles target where build_coupling_map fails."""

        class BrokenTarget:
            num_qubits = 3
            operation_names = ["h", "cx"]

            def build_coupling_map(self):
                raise RuntimeError("Not supported")

        nq, conn, gates, _ = _extract_from_target(BrokenTarget())

        assert nq == 3
        assert conn is None
        assert gates == ["cx", "h"]


class TestCalibrationExtraction:
    """Tests for extracting calibration from backend.properties()."""

    def test_extracts_from_properties(self, mock_backend_with_calibration):
        """Extracts calibration from backend with properties()."""
        cal, raw = _extract_calibration(
            mock_backend_with_calibration,
            refresh_properties=False,
        )

        assert cal is not None
        assert cal.qubits[0].t1_us == 150.0
        assert cal.qubits[0].t2_us == 85.0
        assert "properties" in raw

    def test_extracts_gate_calibration(self, mock_backend_with_calibration):
        """Extracts gate calibration data."""
        cal, _ = _extract_calibration(
            mock_backend_with_calibration,
            refresh_properties=False,
        )

        assert len(cal.gates) == 3
        cx_gate = next((g for g in cal.gates if g.gate == "cx"), None)
        assert cx_gate is not None
        assert cx_gate.qubits == (0, 1)
        assert cx_gate.error == 0.008

    def test_handles_no_properties_method(self):
        """Handles backend without properties method."""

        class MockBackend:
            pass

        cal, raw = _extract_calibration(MockBackend(), refresh_properties=False)
        assert cal is None
        assert raw == {}

    def test_handles_properties_exception(self):
        """Handles exception in properties()."""

        class MockBackend:
            def properties(self):
                raise RuntimeError("Not available")

        cal, _ = _extract_calibration(MockBackend(), refresh_properties=False)
        assert cal is None


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

    def test_snapshot_serializes_to_dict(self, aer_simulator):
        """Snapshot serializes to dict correctly."""
        snapshot = create_device_snapshot(aer_simulator)
        d = snapshot.to_dict()

        assert d["provider"] in ("aer", "qiskit")
        assert d["backend_type"] == "simulator"
        assert "captured_at" in d
        assert "sdk_versions" in d

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
        assert snapshot.calibration.qubits[0].readout_error == 0.012

    def test_minimal_backend(self):
        """Handles backend with no extra attributes."""

        class MinimalBackend:
            pass

        snapshot = create_device_snapshot(MinimalBackend())

        assert snapshot.backend_name == "MinimalBackend"
        assert snapshot.provider == "qiskit"
        assert snapshot.num_qubits is None
        assert snapshot.calibration is None

    def test_backend_name_extraction(self):
        """Handles various backend name formats."""

        class CallableNameBackend:
            def name(self):
                return "callable_name"

        class PropertyNameBackend:
            name = "property_name"

        snapshot1 = create_device_snapshot(CallableNameBackend())
        assert snapshot1.backend_name == "callable_name"

        snapshot2 = create_device_snapshot(PropertyNameBackend())
        assert snapshot2.backend_name == "property_name"


class TestBackendTypeCompliance:
    """Tests that backend_type returns schema-compliant values."""

    VALID_BACKEND_TYPES = {"simulator", "hardware", "emulator", "sim", "hw", "qpu"}

    def test_aer_simulator_returns_simulator(self, aer_simulator):
        """AerSimulator returns 'simulator' backend_type."""
        snapshot = create_device_snapshot(aer_simulator)
        assert snapshot.backend_type == "simulator"
        assert snapshot.backend_type in self.VALID_BACKEND_TYPES

    def test_unknown_backend_defaults_to_valid(self):
        """Unknown backends default to a valid backend_type."""

        class UnknownBackend:
            name = "mystery_backend"

        snapshot = create_device_snapshot(UnknownBackend())
        assert snapshot.backend_type in self.VALID_BACKEND_TYPES


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
        datetime.fromisoformat(ts)  # Should not raise

    def test_sdk_versions_format(self, aer_simulator):
        """sdk_versions is a dict mapping package names to versions."""
        snapshot = create_device_snapshot(aer_simulator)

        assert isinstance(snapshot.sdk_versions, dict)
        assert "qiskit" in snapshot.sdk_versions
        assert isinstance(snapshot.sdk_versions["qiskit"], str)

    def test_calibration_source_is_provider(self, mock_backend_with_calibration):
        """Calibration source is 'provider' for backend-provided data."""
        snapshot = create_device_snapshot(mock_backend_with_calibration)

        assert snapshot.calibration is not None
        assert snapshot.calibration.source == "provider"

    def test_connectivity_format(self, mock_backend):
        """Connectivity is list of (int, int) tuples."""
        snapshot = create_device_snapshot(mock_backend)

        assert snapshot.connectivity is not None
        for edge in snapshot.connectivity:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert all(isinstance(q, int) for q in edge)

    def test_native_gates_format(self, mock_backend):
        """native_gates is list of gate name strings."""
        snapshot = create_device_snapshot(mock_backend)

        assert snapshot.native_gates is not None
        assert isinstance(snapshot.native_gates, list)
        for gate in snapshot.native_gates:
            assert isinstance(gate, str)

    def test_schema_compliance(self, aer_simulator):
        """DeviceSnapshot dict matches expected schema structure."""
        snapshot = create_device_snapshot(aer_simulator)
        d = snapshot.to_dict()

        # String fields
        for field in ["captured_at", "backend_name", "backend_type", "provider"]:
            assert isinstance(d.get(field), str)

        # Optional int field
        if d.get("num_qubits") is not None:
            assert isinstance(d["num_qubits"], int)

        # Optional list fields
        if d.get("native_gates") is not None:
            assert isinstance(d["native_gates"], list)
        if d.get("connectivity") is not None:
            assert isinstance(d["connectivity"], list)

        # Optional dict field
        if d.get("sdk_versions") is not None:
            assert isinstance(d["sdk_versions"], dict)
