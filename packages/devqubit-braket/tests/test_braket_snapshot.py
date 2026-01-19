# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Braket device snapshot creation."""

from devqubit_braket.snapshot import create_device_snapshot


class TestSnapshotWithRealDevice:
    """Tests using real Braket LocalSimulator."""

    def test_local_simulator_snapshot(self, local_simulator):
        """Creates snapshot from real LocalSimulator."""
        snap = create_device_snapshot(local_simulator)

        assert snap.provider == "local"
        assert snap.backend_name is not None
        assert snap.backend_type == "simulator"  # Schema-valid value
        assert snap.captured_at is not None
        assert snap.sdk_versions is not None
        assert "braket" in snap.sdk_versions

    def test_local_simulator_has_no_calibration(self, local_simulator):
        """LocalSimulator doesn't have calibration data."""
        snap = create_device_snapshot(local_simulator)
        assert snap.calibration is None

    def test_snapshot_to_dict_format(self, local_simulator):
        """Snapshot serializes to correct dict format."""
        snap = create_device_snapshot(local_simulator)
        d = snap.to_dict()

        assert d["schema"] == "devqubit.device_snapshot/1.0"
        assert d["provider"] == "local"
        assert d["backend_type"] == "simulator"
        assert "sdk_versions" in d
        assert isinstance(d["sdk_versions"], dict)


class TestSnapshotTopologyExtraction:
    """Tests for topology extraction from device properties."""

    def test_extracts_basic_topology(self, device_factory):
        """Extracts qubit count and connectivity."""
        device = device_factory(
            name="linear_3q",
            qubit_count=3,
            connectivity_graph={"0": [1], "1": [0, 2], "2": [1]},
        )

        snap = create_device_snapshot(device)

        assert snap.num_qubits == 3
        assert snap.connectivity is not None
        assert set(snap.connectivity) == {(0, 1), (1, 0), (1, 2), (2, 1)}

    def test_fully_connected_topology(self, device_factory):
        """Handles fully connected devices without expanding edges."""
        device = device_factory(
            name="fully_connected",
            qubit_count=10,
            fully_connected=True,
        )

        snap = create_device_snapshot(device)

        assert snap.num_qubits == 10
        # Don't expand O(n^2) edges
        assert snap.connectivity is None

    def test_realistic_qpu_topology(self, mock_device):
        """Extracts topology from realistic QPU."""
        snap = create_device_snapshot(mock_device)

        assert snap.num_qubits == 8
        assert snap.connectivity is not None
        assert len(snap.connectivity) > 0

    def test_connectivity_preserves_directionality(self, device_factory):
        """
        Connectivity graph preserves edge directions from device properties.

        Some devices have asymmetric coupling (e.g., one direction is better).
        """
        # Device with explicit directional edges
        device = device_factory(
            name="directional",
            qubit_count=3,
            connectivity_graph={
                "0": [1],  # 0->1 only
                "1": [2],  # 1->2 only
                # No reverse edges
            },
        )

        snap = create_device_snapshot(device)

        assert snap.connectivity is not None
        edges = set(snap.connectivity)

        # Should have the specified directed edges
        assert (0, 1) in edges
        assert (1, 2) in edges

    def test_connectivity_normalizes_to_sorted_list(self, device_factory):
        """Connectivity is normalized to a sorted list for determinism."""
        device = device_factory(
            name="sorted",
            qubit_count=4,
            connectivity_graph={
                "3": [2],
                "0": [1],
                "2": [1, 3],
                "1": [0, 2],
            },
        )

        snap = create_device_snapshot(device)

        assert snap.connectivity is not None
        # Should be sorted
        assert snap.connectivity == sorted(snap.connectivity)


class TestSnapshotNativeGates:
    """Tests for native gate extraction."""

    def test_extracts_native_gate_set(self, device_factory):
        """Extracts native gates from properties."""
        device = device_factory(
            name="native_gates",
            qubit_count=2,
            native_gates=["x", "rz", "cz"],
        )

        snap = create_device_snapshot(device)

        assert snap.native_gates == ["x", "rz", "cz"]

    def test_fallback_to_supported_operations(self, device_factory):
        """Falls back to supported operations when native gates not available."""
        device = device_factory(name="supported_ops", qubit_count=2)

        # Inject "action" into the raw properties dict
        device.properties._d["action"] = {
            "braket.ir.openqasm.program": {
                "supportedOperations": ["h", "cx", "rz", "measure"]
            }
        }

        snap = create_device_snapshot(device)

        assert snap.native_gates == ["h", "cx", "rz", "measure"]

    def test_no_native_gates_returns_none(self, device_factory):
        """Returns None when no gate information available."""
        device = device_factory(name="no_gates", qubit_count=2)

        snap = create_device_snapshot(device)

        assert snap.native_gates is None


class TestSnapshotCalibration:
    """Tests for calibration integration in snapshots."""

    def test_includes_calibration_when_available(self, mock_device):
        """Includes calibration data in snapshot."""
        snap = create_device_snapshot(mock_device)

        assert snap.calibration is not None
        assert snap.calibration.gates is not None
        assert len(snap.calibration.gates) > 0

    def test_calibration_summary_accessible(self, mock_device):
        """Calibration summary is accessible."""
        snap = create_device_snapshot(mock_device)

        summary = snap.get_calibration_summary()
        assert summary is not None
        assert "median_2q_error" in summary
        assert summary["median_2q_error"] is not None


class TestBackendType:
    """Tests for backend_type resolution."""

    def test_local_simulator_type(self, local_simulator):
        """LocalSimulator has backend_type='simulator'."""
        snap = create_device_snapshot(local_simulator)
        assert snap.backend_type == "simulator"

    def test_mock_qpu_type(self, mock_device):
        """Mock QPU device has backend_type='hardware'."""
        snap = create_device_snapshot(mock_device)
        # MockDevice has __module__ = "braket.aws.aws_device" and type="QPU"
        assert snap.backend_type == "hardware"

    def test_backend_type_in_dict(self, local_simulator):
        """backend_type is schema-valid in serialized dict."""
        snap = create_device_snapshot(local_simulator)
        d = snap.to_dict()

        # Must be one of the schema-valid values
        valid_types = {"simulator", "hardware", "emulator", "sim", "hw", "qpu"}
        assert d["backend_type"] in valid_types

    def test_backend_type_from_device_type_attribute(self, mock_device):
        """Resolves backend_type from device.type attribute."""
        # MockDevice has type="QPU" which should resolve to "hardware"
        snap = create_device_snapshot(mock_device)
        assert snap.backend_type == "hardware"


class TestBackendId:
    """Tests for backend_id extraction."""

    def test_extracts_arn_as_backend_id(self, mock_device):
        """Extracts ARN as backend_id for AWS devices."""
        snap = create_device_snapshot(mock_device)

        assert snap.backend_id is not None
        assert "arn:aws:braket" in snap.backend_id

    def test_local_simulator_no_backend_id(self, local_simulator):
        """LocalSimulator has no ARN/backend_id."""
        snap = create_device_snapshot(local_simulator)
        # LocalSimulator doesn't have an ARN
        assert snap.backend_id is None


class TestSdkVersions:
    """Tests for SDK version capture."""

    def test_captures_braket_version(self, local_simulator):
        """Captures Braket SDK version."""
        snap = create_device_snapshot(local_simulator)

        assert snap.sdk_versions is not None
        assert "braket" in snap.sdk_versions

    def test_sdk_versions_in_dict(self, local_simulator):
        """sdk_versions is dict in serialized output."""
        snap = create_device_snapshot(local_simulator)
        d = snap.to_dict()

        assert "sdk_versions" in d
        assert isinstance(d["sdk_versions"], dict)
        assert "braket" in d["sdk_versions"]

    def test_braket_version_format(self, local_simulator):
        """Braket version has reasonable format."""
        snap = create_device_snapshot(local_simulator)

        version = snap.sdk_versions.get("braket")
        assert version is not None
        # Version should be string like "1.70.0" or "unknown"
        assert isinstance(version, str)
        assert len(version) > 0


class TestErrorHandling:
    """Tests for robustness and error handling."""

    def test_handles_missing_properties_gracefully(self):
        """Handles devices without properties."""

        class NoPropsDevice:
            __module__ = "braket.devices"

            @property
            def name(self):
                return "NoPropsDevice"

        snap = create_device_snapshot(NoPropsDevice())

        assert snap.provider == "aws_braket"
        assert snap.backend_type == "simulator"  # Default for non-AWS
        assert snap.num_qubits is None
        assert snap.connectivity is None
        assert snap.calibration is None

    def test_handles_broken_properties_access(self):
        """Handles exceptions during property access."""

        class BrokenDevice:
            __module__ = "braket.devices"

            @property
            def name(self):
                return "BrokenDevice"

            @property
            def properties(self):
                raise RuntimeError("Properties unavailable")

        snap = create_device_snapshot(BrokenDevice())

        assert snap.provider == "aws_braket"
        assert snap.captured_at is not None
        # Should not crash, just return None for unavailable data
        assert snap.num_qubits is None

    def test_handles_none_property_values(self, device_factory):
        """Handles None values in property fields."""
        # device_factory without native_gates or connectivity creates device with None values
        device = device_factory(
            name="none_values",
            qubit_count=2,
            # No native_gates or connectivity_graph specified
        )
        snap = create_device_snapshot(device)

        # Should handle gracefully
        assert snap.num_qubits == 2
        assert snap.native_gates is None
        assert snap.connectivity is None

    def test_handles_empty_connectivity_graph(self, device_factory):
        """Handles empty connectivity graph."""
        # device_factory without connectivity creates device with no edges
        device = device_factory(name="empty_graph", qubit_count=2)
        snap = create_device_snapshot(device)

        assert snap.connectivity is None  # No graph -> None


class TestSnapshotDeterminism:
    """Tests for snapshot determinism (important for comparisons)."""

    def test_same_device_same_snapshot_structure(self, local_simulator):
        """Same device produces consistent snapshot structure."""
        snap1 = create_device_snapshot(local_simulator)
        snap2 = create_device_snapshot(local_simulator)

        # Key fields should match (captured_at will differ)
        assert snap1.provider == snap2.provider
        assert snap1.backend_name == snap2.backend_name
        assert snap1.backend_type == snap2.backend_type
        assert snap1.sdk_versions == snap2.sdk_versions

    def test_mock_device_snapshot_consistency(self, mock_device):
        """Mock device produces consistent snapshots."""
        snap1 = create_device_snapshot(mock_device)
        snap2 = create_device_snapshot(mock_device)

        # Topology should be identical
        assert snap1.num_qubits == snap2.num_qubits
        assert snap1.connectivity == snap2.connectivity
        assert snap1.native_gates == snap2.native_gates

        # Calibration should match
        assert snap1.calibration is not None
        assert snap2.calibration is not None
        assert len(snap1.calibration.gates) == len(snap2.calibration.gates)
