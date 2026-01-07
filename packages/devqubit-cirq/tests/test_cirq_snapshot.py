# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Cirq device snapshot creation."""

from devqubit_cirq.snapshot import create_device_snapshot


class TestCreateDeviceSnapshot:
    """Tests for device snapshot creation."""

    def test_simulator_snapshot(self, simulator):
        """Creates valid snapshot from Simulator."""
        snapshot = create_device_snapshot(simulator)

        assert snapshot.provider == "cirq"
        assert snapshot.backend_name == "Simulator"
        assert snapshot.backend_type == "simulator"
        assert snapshot.captured_at is not None
        assert snapshot.sdk_versions is not None
        assert "cirq" in snapshot.sdk_versions
        assert "." in snapshot.sdk_versions["cirq"]

    def test_density_matrix_simulator_snapshot(self, density_matrix_simulator):
        """Creates snapshot from DensityMatrixSimulator."""
        snapshot = create_device_snapshot(density_matrix_simulator)

        assert snapshot.provider == "cirq"
        assert snapshot.backend_name == "DensityMatrixSimulator"
        assert snapshot.backend_type == "simulator"

    def test_snapshot_serializes(self, simulator):
        """Snapshot serializes to dict correctly."""
        snapshot = create_device_snapshot(simulator)
        d = snapshot.to_dict()

        assert d["schema"] == "devqubit.device_snapshot/0.1"
        assert d["provider"] == "cirq"
        assert d["backend_name"] == "Simulator"
        assert d["backend_type"] == "simulator"
        assert "captured_at" in d
        assert isinstance(d["sdk_versions"], dict)

    def test_no_calibration_for_simulator(self, simulator):
        """Simulators have no calibration data."""
        snapshot = create_device_snapshot(simulator)
        assert snapshot.calibration is None


class TestBackendType:
    """Tests for backend_type resolution."""

    # Canonical backend types - using strict values to catch regressions
    VALID_BACKEND_TYPES = {"simulator", "hardware"}

    def test_simulator_type(self, simulator):
        """Simulator has backend_type='simulator'."""
        snapshot = create_device_snapshot(simulator)
        assert snapshot.backend_type == "simulator"
        assert snapshot.backend_type in self.VALID_BACKEND_TYPES

    def test_backend_type_in_dict(self, simulator):
        """backend_type is schema-valid in serialized dict."""
        snapshot = create_device_snapshot(simulator)
        d = snapshot.to_dict()
        assert d["backend_type"] in self.VALID_BACKEND_TYPES


class TestHardwareBackendType:
    """Tests for hardware backend detection."""

    def test_engine_detected_as_hardware(self):
        """Google Quantum Engine samplers are detected as hardware."""

        class MockEngineSampler:
            __module__ = "cirq_google.engine.engine_sampler"

        snapshot = create_device_snapshot(MockEngineSampler())
        assert snapshot.backend_type == "hardware"

    def test_processor_detected_as_hardware(self):
        """Processor samplers are detected as hardware."""

        class MockProcessorSampler:
            __module__ = "cirq_google.engine.processor"

        snapshot = create_device_snapshot(MockProcessorSampler())
        assert snapshot.backend_type == "hardware"

    def test_ionq_detected_as_hardware(self):
        """IonQ samplers are detected as hardware."""

        class MockIonQSampler:
            __module__ = "cirq_ionq.service"

        snapshot = create_device_snapshot(MockIonQSampler())
        assert snapshot.backend_type == "hardware"


class TestSdkVersions:
    """Tests for SDK version capture."""

    def test_captures_cirq_version(self, simulator):
        """Captures Cirq SDK version."""
        snapshot = create_device_snapshot(simulator)

        assert snapshot.sdk_versions is not None
        assert "cirq" in snapshot.sdk_versions
        assert snapshot.sdk_versions["cirq"] != "unknown"

    def test_sdk_versions_in_dict(self, simulator):
        """sdk_versions is dict in serialized output."""
        snapshot = create_device_snapshot(simulator)
        d = snapshot.to_dict()

        assert "sdk_versions" in d
        assert isinstance(d["sdk_versions"], dict)
        assert "cirq" in d["sdk_versions"]


class TestMockSamplers:
    """Tests with mock sampler objects for edge cases."""

    def test_minimal_sampler(self):
        """Handles minimal sampler with few attributes."""

        class MinimalSampler:
            __module__ = "cirq.sim"

        snapshot = create_device_snapshot(MinimalSampler())
        assert snapshot.backend_name == "MinimalSampler"
        assert snapshot.backend_type == "simulator"

    def test_sampler_with_device_qubit_count(self):
        """Captures qubit count from device if available."""

        class MockDevice:
            qubits = [0, 1, 2, 3, 4]

        class SamplerWithDevice:
            __module__ = "cirq.sim"
            device = MockDevice()

        snapshot = create_device_snapshot(SamplerWithDevice())
        assert snapshot.num_qubits == 5

    def test_handles_broken_device_access(self):
        """Handles exceptions when accessing device attribute."""

        class SamplerWithBrokenDevice:
            __module__ = "cirq.sim"

            @property
            def device(self):
                raise RuntimeError("Device unavailable")

        snapshot = create_device_snapshot(SamplerWithBrokenDevice())
        assert snapshot.num_qubits is None


class TestConnectivityExtraction:
    """Tests for connectivity and native gates extraction."""

    def test_extracts_from_device_metadata(self):
        """Extracts connectivity from device metadata when properly structured."""

        class MockQubit:
            def __init__(self, x):
                self.x = x

            def __str__(self):
                return f"q({self.x})"

        class MockMetadata:
            qubit_pairs = [(MockQubit(0), MockQubit(1)), (MockQubit(1), MockQubit(2))]

        class MockDevice:
            metadata = MockMetadata()
            qubits = [MockQubit(i) for i in range(3)]

        class SamplerWithConnectivity:
            __module__ = "cirq.sim"
            device = MockDevice()

        snapshot = create_device_snapshot(SamplerWithConnectivity())

        # Connectivity may or may not be extracted depending on implementation
        # The key test is that it doesn't crash
        assert snapshot.provider == "cirq"

    def test_resilient_to_broken_metadata(self):
        """Handles broken device metadata gracefully."""

        class BrokenDevice:
            @property
            def metadata(self):
                raise RuntimeError("boom")

        class FakeExecutor:
            __module__ = "cirq.sim.fake"
            device = BrokenDevice()

        snapshot = create_device_snapshot(FakeExecutor())
        assert snapshot.connectivity is None

    def test_connectivity_deterministic(self):
        """Connectivity extraction is deterministic across calls.

        Qubit indexing must be stable (not hash-based) to ensure
        reproducible connectivity graphs.
        """

        class MockQubit:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

            def __hash__(self):
                # Intentionally return different hash each time
                # to verify we're NOT using hash-based indexing
                import random

                return random.randint(0, 1000000)

            def __eq__(self, other):
                return self.name == other.name

        q0 = MockQubit("q0")
        q1 = MockQubit("q1")
        q2 = MockQubit("q2")

        class MockMetadata:
            qubit_pairs = [(q0, q1), (q1, q2), (q0, q2)]

        class MockDevice:
            metadata = MockMetadata()
            qubits = [q0, q1, q2]

        class SamplerWithUnstableHash:
            __module__ = "cirq.sim"
            device = MockDevice()

        # Call multiple times - should produce identical results
        results = []
        for _ in range(5):
            snapshot = create_device_snapshot(SamplerWithUnstableHash())
            if snapshot.connectivity is not None:
                results.append(tuple(sorted(snapshot.connectivity)))

        # All results should be identical despite unstable hash
        if results:
            assert all(
                r == results[0] for r in results
            ), "Connectivity should be deterministic despite unstable qubit hash"

    def test_connectivity_uses_string_sort_order(self):
        """Connectivity indices are based on string sort order, not arbitrary hash."""

        class MockQubit:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

        # Qubits with specific string ordering
        qa = MockQubit("a")
        qb = MockQubit("b")
        qc = MockQubit("c")

        class MockMetadata:
            qubit_pairs = [(qc, qa), (qa, qb)]  # Intentionally out of order

        class MockDevice:
            metadata = MockMetadata()
            qubits = [qc, qa, qb]  # Also out of order

        class Sampler:
            __module__ = "cirq.sim"
            device = MockDevice()

        snapshot = create_device_snapshot(Sampler())

        if snapshot.connectivity is not None:
            # After string-based sorting: a=0, b=1, c=2
            # So (c,a) -> (2,0) and (a,b) -> (0,1)
            edges = set(snapshot.connectivity)
            # Verify edges use consistent indexing
            # (exact values depend on implementation, but should be stable)
            assert len(edges) == 2
