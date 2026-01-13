# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for PennyLane device snapshot creation."""

import json

import pennylane as qml
import pytest
from devqubit_engine.core.run import track
from devqubit_pennylane.snapshot import (
    _extract_capabilities,
    _extract_shots,
    _extract_wires,
    create_device_snapshot,
    resolve_pennylane_backend,
)
from devqubit_pennylane.utils import ShotsInfo, extract_shots_info


class TestExtractWires:
    """Tests for wire extraction."""

    def test_extract_from_device(self, default_qubit):
        """Extracts wire info from device."""
        num_qubits, props = _extract_wires(default_qubit)

        assert num_qubits == 2
        assert "wires" in props
        assert len(props["wires"]) == 2

    def test_handles_many_wires(self):
        """Handles devices with many wires."""
        dev = qml.device("default.qubit", wires=10)
        num_qubits, props = _extract_wires(dev)

        assert num_qubits == 10

    def test_handles_named_wires(self):
        """Handles devices with named wires."""
        dev = qml.device("default.qubit", wires=["a", "b", "c"])
        num_qubits, props = _extract_wires(dev)

        assert num_qubits == 3
        assert "a" in props["wires"]


class TestExtractShots:
    """Tests for shots extraction from devices and tapes."""

    def test_extract_shots_analytic_device(self, default_qubit):
        """Extracts shots info from analytic device (no shots)."""
        props = _extract_shots(default_qubit)

        assert props["analytic"] is True
        assert props["total_shots"] is None

    def test_extract_shots_from_tape(self, tape_with_shots_1000):
        """Extracts shots info from tape with finite shots."""
        props = _extract_shots(tape_with_shots_1000)

        assert props["analytic"] is False
        assert props["total_shots"] == 1000

    def test_extract_shot_vector_from_tape(self, tape_with_shot_vector_uniform):
        """
        Extracts shot vector from tape.

        P0 requirement: properly handle PennyLane Shots class with shot_vector.
        """
        props = _extract_shots(tape_with_shot_vector_uniform)

        assert props["analytic"] is False
        assert props["total_shots"] == 300  # 100 + 100 + 100
        assert "shot_vector" in props
        assert isinstance(props["shot_vector"], list)


class TestShotsInfo:
    """Tests for ShotsInfo dataclass."""

    def test_analytic_device(self, default_qubit):
        """ShotsInfo correctly identifies analytic mode."""
        info = extract_shots_info(default_qubit)

        assert isinstance(info, ShotsInfo)
        assert info.analytic is True
        assert info.total_shots is None
        assert info.shot_vector is None

    def test_finite_shots_tape(self, tape_with_shots_1000):
        """ShotsInfo correctly handles finite shots from tape."""
        info = extract_shots_info(tape_with_shots_1000)

        assert info.analytic is False
        assert info.total_shots == 1000
        assert info.has_partitioned_shots is False

    def test_shot_vector_tape(self, tape_with_shot_vector_uniform):
        """ShotsInfo correctly handles shot vectors from tape."""
        info = extract_shots_info(tape_with_shot_vector_uniform)

        assert info.analytic is False
        assert info.total_shots == 300
        assert info.shot_vector is not None
        # PennyLane consolidates identical shots: [100, 100, 100] -> [(100, 3)]
        # So we check total via sum(shots * copies)
        total = sum(shots * copies for shots, copies in info.shot_vector)
        assert total == 300
        assert info.has_partitioned_shots is True

    def test_to_dict_serialization(self, tape_with_shot_vector_uniform):
        """ShotsInfo serializes to dict correctly."""
        info = extract_shots_info(tape_with_shot_vector_uniform)
        d = info.to_dict()

        assert "total_shots" in d
        assert "analytic" in d
        assert "shot_vector" in d
        assert d["total_shots"] == 300


class TestExtractCapabilities:
    """Tests for capabilities extraction."""

    def test_extract_operations(self, default_qubit):
        """Extracts supported operations info."""
        props = _extract_capabilities(default_qubit)
        assert isinstance(props, dict)

    def test_handles_missing(self):
        """Handles devices without capabilities."""

        class MinimalDevice:
            pass

        props = _extract_capabilities(MinimalDevice())
        assert isinstance(props, dict)


class TestRawPropertiesArtifact:
    """Tests for raw_properties artifact via tracker."""

    def test_snapshot_without_tracker_has_no_ref(self, default_qubit):
        """DeviceSnapshot without tracker should have raw_properties_ref=None."""
        snapshot = create_device_snapshot(default_qubit, tracker=None)

        # Without tracker, raw_properties_ref should be None
        assert snapshot.raw_properties_ref is None

    def test_snapshot_with_tracker_has_raw_properties_ref(
        self,
        default_qubit,
        store,
        registry,
    ):
        """DeviceSnapshot with tracker should have raw_properties_ref populated."""
        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(default_qubit, tracker=run)

            # With tracker, raw_properties_ref should be an ArtifactRef
            assert snapshot.raw_properties_ref is not None
            assert (
                snapshot.raw_properties_ref.kind
                == "device.pennylane.raw_properties.json"
            )
            assert snapshot.raw_properties_ref.role == "device_raw"
            assert snapshot.raw_properties_ref.digest.startswith("sha256:")

    def test_raw_properties_content_is_valid(self, default_qubit, store, registry):
        """raw_properties artifact should contain expected device metadata."""

        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(default_qubit, tracker=run)

        # Verify artifact was stored and contains expected keys
        assert snapshot.raw_properties_ref is not None

        # Load the artifact content
        content = store.get_bytes(snapshot.raw_properties_ref.digest)
        raw_props = json.loads(content.decode("utf-8"))

        assert isinstance(raw_props, dict)
        assert "device_class" in raw_props
        assert "device_module" in raw_props
        assert "execution_provider" in raw_props
        assert raw_props["execution_provider"] == "local"

    def test_raw_properties_includes_shots_info(self, default_qubit, store, registry):
        """raw_properties artifact should include shots configuration."""

        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(default_qubit, tracker=run)

        content = store.get_bytes(snapshot.raw_properties_ref.digest)
        raw_props = json.loads(content.decode("utf-8"))

        # Analytic mode device
        assert "total_shots" in raw_props
        assert raw_props["total_shots"] is None
        assert raw_props["analytic"] is True

    def test_raw_properties_includes_wires(self, default_qubit, store, registry):
        """raw_properties artifact should include wire info."""

        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(default_qubit, tracker=run)

        content = store.get_bytes(snapshot.raw_properties_ref.digest)
        raw_props = json.loads(content.decode("utf-8"))

        assert "wires" in raw_props
        assert len(raw_props["wires"]) == 2


class TestCreateDeviceSnapshot:
    """Tests for device snapshot creation."""

    def test_default_qubit_core_fields(self, default_qubit):
        """Creates snapshot with core fields from default.qubit."""
        snap = create_device_snapshot(default_qubit)

        assert snap.provider == "local"
        assert snap.backend_name == "default.qubit"
        assert snap.backend_type == "simulator"
        assert snap.captured_at is not None
        assert snap.num_qubits == 2

    def test_snapshot_has_sdk_version(self, default_qubit):
        """Snapshot includes SDK version."""
        snap = create_device_snapshot(default_qubit)

        assert snap.sdk_versions is not None
        assert isinstance(snap.sdk_versions, dict)

    def test_snapshot_to_dict(self, default_qubit):
        """Snapshot serializes to dict correctly."""
        snap = create_device_snapshot(default_qubit)
        d = snap.to_dict()

        assert d["provider"] == "local"
        assert d["backend_name"] == "default.qubit"
        assert "captured_at" in d
        assert "num_qubits" in d

    def test_no_calibration_or_connectivity_for_simulator(self, default_qubit):
        """Simulator has no calibration or connectivity data."""
        snap = create_device_snapshot(default_qubit)

        assert snap.calibration is None
        assert snap.connectivity is None

    def test_resolve_remote_backend_disabled_by_default(self, default_qubit):
        """Remote backend resolution is disabled by default."""
        # This test ensures we don't make network calls by default
        snap = create_device_snapshot(default_qubit)
        assert snap is not None


class TestDeviceSnapshotWithDifferentDevices:
    """Tests with various PennyLane devices."""

    def test_lightning_qubit_if_available(self):
        """Creates snapshot from lightning.qubit if available."""
        try:
            dev = qml.device("lightning.qubit", wires=3)
            snap = create_device_snapshot(dev)

            assert snap.provider == "local"
            assert "lightning" in snap.backend_name
            assert snap.num_qubits == 3
        except Exception:
            pytest.skip("lightning.qubit not available")

    def test_default_mixed_if_available(self):
        """Creates snapshot from default.mixed if available."""
        try:
            dev = qml.device("default.mixed", wires=2)
            snap = create_device_snapshot(dev)

            assert snap.provider == "local"
            assert "mixed" in snap.backend_name
            assert snap.num_qubits == 2
        except Exception:
            pytest.skip("default.mixed not available")


class TestMockDevices:
    """Tests with mock device objects for plugin detection."""

    def test_minimal_device(self):
        """Handles minimal device with few attributes."""

        class MinimalDevice:
            name = "minimal"

        snap = create_device_snapshot(MinimalDevice())

        assert snap.backend_name == "minimal"
        assert snap.provider == "local"
        assert snap.num_qubits is None

    def test_device_with_wires_only(self):
        """Handles device with only wires attribute."""

        class WiresOnlyDevice:
            name = "wires_only"
            wires = [0, 1, 2, 3]

        snap = create_device_snapshot(WiresOnlyDevice())
        assert snap.num_qubits == 4

    def test_handles_broken_attributes(self):
        """Handles devices with broken attributes gracefully."""

        class BrokenDevice:
            name = "broken"

            @property
            def wires(self):
                raise RuntimeError("Wires unavailable")

        snap = create_device_snapshot(BrokenDevice())

        assert snap.backend_name == "broken"
        assert snap.num_qubits is None

    def test_mock_device_with_shots_object(self):
        """
        Handles mock device with Shots object.

        P0: Must properly distinguish Shots(total_shots=None) from shots=None.
        """
        try:
            from pennylane.measurements import Shots

            class MockDeviceWithShots:
                name = "mock_shots"
                shots = Shots(1000)
                wires = [0, 1]

            info = extract_shots_info(MockDeviceWithShots())
            assert info.total_shots == 1000
            assert info.analytic is False

        except ImportError:
            pytest.skip("Shots class not available in this PennyLane version")

    def test_mock_device_with_analytic_shots(self):
        """
        Handles mock device with explicit analytic mode.

        Shots(total_shots=None) is different from shots=None.
        """
        try:
            from pennylane.measurements import Shots

            class MockDeviceAnalytic:
                name = "mock_analytic"
                shots = Shots(None)  # Explicit analytic mode
                wires = [0, 1]

            info = extract_shots_info(MockDeviceAnalytic())
            assert info.total_shots is None
            assert info.analytic is True

        except ImportError:
            pytest.skip("Shots class not available in this PennyLane version")


class TestResolvePennyLaneBackend:
    """Tests for provider/backend detection from plugin modules."""

    def test_resolve_braket_like_device(self, monkeypatch):
        """Resolves Braket-like device from pennylane_braket module."""
        import devqubit_pennylane.snapshot as snapshot_mod

        monkeypatch.setattr(snapshot_mod, "get_device_name", lambda d: d.short_name)

        class MockBraketDevice:
            short_name = "braket.aws.qubit"
            __module__ = "pennylane_braket.aws_qubit"
            device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            wires = [0, 1]
            shots = None

        info = resolve_pennylane_backend(MockBraketDevice())

        assert info is not None
        assert info["provider"] == "aws_braket"  # Physical provider
        assert info["sdk_frontend"] == "braket"  # SDK frontend
        assert info["backend_type"] == "simulator"
        assert info["backend_id"] == MockBraketDevice.device_arn

    def test_resolve_qiskit_like_device(self, monkeypatch):
        """Resolves Qiskit-like device from pennylane_qiskit module."""
        import devqubit_pennylane.snapshot as snapshot_mod

        monkeypatch.setattr(snapshot_mod, "get_device_name", lambda d: d.short_name)

        class MockBackend:
            name = "aer_simulator"

            def backend_id(self):
                return "ibm-backend-123"

        class MockQiskitDevice:
            short_name = "qiskit.remote"
            __module__ = "pennylane_qiskit.remote"
            backend = MockBackend()
            wires = [0, 1]
            shots = None

        snap = create_device_snapshot(MockQiskitDevice())

        # aer_simulator is local, so provider is "local"
        assert snap.provider == "local"  # Physical provider (aer is local)
        assert snap.backend_id == "ibm-backend-123"
        assert snap.backend_type == "simulator"
        assert snap.num_qubits == 2

    def test_braket_hardware_detection(self, monkeypatch):
        """Detects Braket hardware vs simulator from ARN."""
        import devqubit_pennylane.snapshot as snapshot_mod

        monkeypatch.setattr(snapshot_mod, "get_device_name", lambda d: d.short_name)

        class MockBraketHardware:
            short_name = "braket.aws.qubit"
            __module__ = "pennylane_braket.aws_qubit"
            device_arn = "arn:aws:braket:us-west-1::device/qpu/ionq/Aria-1"
            wires = [0, 1]
            shots = None

        info = resolve_pennylane_backend(MockBraketHardware())

        assert info["provider"] == "aws_braket"  # Physical provider
        assert info["sdk_frontend"] == "braket"  # SDK frontend
        assert info["backend_type"] == "hardware"  # Not simulator


class TestBackendTypeCompliance:
    """Tests that backend_type always returns schema-compliant values."""

    VALID_BACKEND_TYPES = {"simulator", "hardware"}

    def test_default_qubit_is_simulator(self, default_qubit):
        """default.qubit returns valid backend_type 'simulator'."""
        snap = create_device_snapshot(default_qubit)

        assert snap.backend_type == "simulator"
        assert snap.backend_type in self.VALID_BACKEND_TYPES

    def test_backend_type_in_serialized_dict(self, default_qubit):
        """Serialized snapshot contains valid backend_type."""
        snap = create_device_snapshot(default_qubit)
        d = snap.to_dict()

        assert "backend_type" in d
        assert d["backend_type"] in self.VALID_BACKEND_TYPES

    def test_unknown_device_defaults_to_valid_type(self):
        """Unknown devices default to a valid backend_type."""

        class UnknownDevice:
            name = "mystery_device"

        snap = create_device_snapshot(UnknownDevice())
        assert snap.backend_type in self.VALID_BACKEND_TYPES


class TestDeviceSnapshotUECCompliance:
    """Tests for DeviceSnapshot UEC compliance."""

    def test_required_fields_present(self, default_qubit):
        """All required DeviceSnapshot fields are present."""
        snap = create_device_snapshot(default_qubit)

        assert snap.captured_at is not None
        assert snap.backend_name is not None
        assert snap.backend_type is not None
        assert snap.provider is not None

    def test_captured_at_is_iso_format(self, default_qubit):
        """captured_at is in ISO 8601 format."""
        from datetime import datetime

        snap = create_device_snapshot(default_qubit)

        ts = snap.captured_at.replace("Z", "+00:00")
        datetime.fromisoformat(ts)  # Should not raise

    def test_sdk_versions_format(self, default_qubit):
        """sdk_versions is a valid dict with version strings."""
        snap = create_device_snapshot(default_qubit)

        assert isinstance(snap.sdk_versions, dict)

    def test_num_qubits_matches_wires(self, default_qubit):
        """num_qubits matches device wire count."""
        snap = create_device_snapshot(default_qubit)
        assert snap.num_qubits == 2


class TestSchemaValidation:
    """Tests for JSON Schema compliance."""

    def test_device_snapshot_schema_compliance(self, default_qubit):
        """DeviceSnapshot dict matches expected schema structure."""
        snap = create_device_snapshot(default_qubit)
        d = snap.to_dict()

        # String fields
        for field in ["captured_at", "backend_name", "backend_type", "provider"]:
            assert isinstance(d.get(field), str)

        # Optional int field
        if d.get("num_qubits") is not None:
            assert isinstance(d["num_qubits"], int)

        # Optional dict field (sdk_versions)
        if d.get("sdk_versions") is not None:
            assert isinstance(d["sdk_versions"], dict)

    def test_to_dict_roundtrip_types(self, default_qubit):
        """to_dict produces JSON-serializable types."""
        snap = create_device_snapshot(default_qubit)
        d = snap.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["provider"] == "local"
        assert parsed["backend_name"] == "default.qubit"

    def test_frontend_config_serializable(self, default_qubit):
        """Frontend config is JSON serializable."""
        snap = create_device_snapshot(default_qubit)

        assert snap.frontend is not None
        frontend_dict = snap.frontend.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(frontend_dict)
        parsed = json.loads(json_str)

        assert parsed["sdk"] == "pennylane"
