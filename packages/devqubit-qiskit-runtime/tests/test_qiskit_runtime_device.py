# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime device snapshot and envelope utilities."""

from datetime import datetime

from devqubit_qiskit_runtime.device import (
    _extract_options,
    _extract_session_info,
    create_device_snapshot,
    resolve_runtime_backend,
)


class TestExtractOptions:
    """Tests for options extraction."""

    def test_extracts_standard_options(self):
        """Extracts optimization_level and default_shots from options."""

        class Options:
            optimization_level = 2
            default_shots = 1024
            resilience_level = 1

        class Primitive:
            options = Options()

        result = _extract_options(Primitive())

        assert result["optimization_level"] == 2
        assert result["default_shots"] == 1024
        assert result["options_resilience_level"] == 1

    def test_no_options(self):
        """Returns empty dict if no options."""

        class Primitive:
            pass

        assert _extract_options(Primitive()) == {}


class TestExtractSessionInfo:
    """Tests for session info extraction."""

    def test_extracts_session_info(self):
        """Extracts session_id and max_time from session."""

        class Session:
            session_id = "test-session-12345"
            max_time = 7200

        class Primitive:
            session = Session()

        result = _extract_session_info(Primitive())

        assert result["session_id"] == "test-session-12345"
        assert result["max_time"] == 7200

    def test_no_session(self):
        """Returns None if no session."""

        class Primitive:
            session = None

        assert _extract_session_info(Primitive()) is None


class TestCreateDeviceSnapshotWithFakeBackend:
    """Tests for device snapshot creation using real fake backends."""

    def test_snapshot_captures_topology(self, fake_sampler, fake_backend):
        """Snapshot captures qubit count and connectivity from fake backend."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.num_qubits == fake_backend.num_qubits
        assert snapshot.connectivity is not None
        assert len(snapshot.connectivity) > 0
        assert all(isinstance(edge, (list, tuple)) for edge in snapshot.connectivity)

    def test_snapshot_captures_native_gates(self, fake_sampler):
        """Snapshot captures native gates including entangling gate."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.native_gates is not None
        assert len(snapshot.native_gates) > 0
        gate_names = set(snapshot.native_gates)
        assert "cx" in gate_names or "ecr" in gate_names

    def test_snapshot_captures_calibration(self, fake_sampler):
        """Snapshot captures calibration with T1/T2 or gate error info."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.calibration is not None
        summary = snapshot.get_calibration_summary()
        assert summary is not None

        has_qubit_info = (
            summary.get("median_t1_us") is not None
            or summary.get("median_t2_us") is not None
        )
        has_gate_info = summary.get("median_2q_error") is not None
        assert has_qubit_info or has_gate_info

    def test_snapshot_provider_and_timestamp(self, fake_sampler):
        """Snapshot has correct provider and ISO timestamp."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.provider in ("fake", "ibm_quantum", "aer", "local")
        assert snapshot.captured_at is not None
        ts = snapshot.captured_at.replace("Z", "+00:00")
        datetime.fromisoformat(ts)

    def test_snapshot_to_dict(self, fake_sampler):
        """Snapshot serializes to dict with expected fields."""
        snapshot = create_device_snapshot(fake_sampler)
        d = snapshot.to_dict()

        assert d["provider"] in ("fake", "ibm_quantum", "aer", "local")
        assert "captured_at" in d
        assert "num_qubits" in d
        assert "connectivity" in d
        assert "native_gates" in d
        assert "backend_type" in d

    def test_estimator_snapshot(self, fake_estimator):
        """Snapshot works for Estimator primitive."""
        snapshot = create_device_snapshot(fake_estimator)

        assert snapshot.provider in ("fake", "ibm_quantum", "aer", "local")
        assert snapshot.num_qubits is not None

    def test_raw_properties_ref_when_tracker_provided(
        self, fake_sampler, store, registry
    ):
        """raw_properties_ref is set when tracker is provided."""
        from devqubit_engine.tracking.run import track

        with track(project="raw_props_test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(fake_sampler, tracker=run)

        d = snapshot.to_dict()
        assert d.get("raw_properties_ref") is not None


class TestBackendTypeCompliance:
    """Tests that backend_type returns schema-compliant values."""

    VALID_BACKEND_TYPES = {"simulator", "hardware", "emulator", "sim", "hw", "qpu"}

    def test_sampler_backend_type_is_valid(self, fake_sampler):
        """Sampler primitive returns valid backend_type."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.backend_type in self.VALID_BACKEND_TYPES

    def test_estimator_backend_type_is_valid(self, fake_estimator):
        """Estimator primitive returns valid backend_type."""
        snapshot = create_device_snapshot(fake_estimator)

        assert snapshot.backend_type in self.VALID_BACKEND_TYPES

    def test_backend_type_not_primitive_class(self, fake_sampler):
        """backend_type should NOT be the primitive class name."""
        snapshot = create_device_snapshot(fake_sampler)

        invalid_values = {
            "SamplerV2",
            "EstimatorV2",
            "FakeSamplerV2",
            "FakeEstimatorV2",
        }
        assert snapshot.backend_type not in invalid_values

    def test_fake_backend_is_simulator_type(self, fake_sampler):
        """Fake backends should return 'simulator' as backend_type."""
        snapshot = create_device_snapshot(fake_sampler)
        assert snapshot.backend_type == "simulator"


class TestResolveRuntimeBackend:
    """Tests for backend resolution helper."""

    def test_returns_stable_info(self, fake_sampler):
        """Backend resolution returns consistent, expected structure."""
        info = resolve_runtime_backend(fake_sampler)

        assert info is not None
        assert info["provider"] in ("fake", "ibm_quantum", "aer", "local")
        assert info["backend_name"]
        assert info["backend_obj"] is not None
        assert info["primitive_type"] in ("sampler", "estimator")
        assert info["backend_type"] in ("hardware", "simulator")


class TestDeviceSnapshotUECCompliance:
    """Tests for DeviceSnapshot UEC compliance."""

    def test_required_fields_present(self, fake_sampler):
        """All required DeviceSnapshot fields are present."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.captured_at is not None
        assert snapshot.backend_name is not None
        assert snapshot.backend_type is not None
        assert snapshot.provider is not None

    def test_sdk_versions_format(self, fake_sampler):
        """sdk_versions is a dict with package versions."""
        snapshot = create_device_snapshot(fake_sampler)

        assert isinstance(snapshot.sdk_versions, dict)
        assert len(snapshot.sdk_versions) > 0

    def test_calibration_source_is_provider(self, fake_sampler):
        """Calibration source is 'provider' for backend-provided data."""
        snapshot = create_device_snapshot(fake_sampler)

        if snapshot.calibration is not None:
            assert snapshot.calibration.source == "provider"

    def test_connectivity_and_gates_format(self, fake_sampler):
        """Connectivity and native_gates have correct formats."""
        snapshot = create_device_snapshot(fake_sampler)

        for edge in snapshot.connectivity:
            assert isinstance(edge, (tuple, list))
            assert len(edge) == 2

        for gate in snapshot.native_gates:
            assert isinstance(gate, str)

    def test_frontend_config_present(self, fake_sampler):
        """Frontend config is present for primitive layer."""
        snapshot = create_device_snapshot(fake_sampler)

        assert snapshot.frontend is not None
        assert snapshot.frontend.sdk == "qiskit-ibm-runtime"
        assert "primitive_type" in (snapshot.frontend.config or {})
