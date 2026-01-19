# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for results processing, device snapshots, and utilities."""

import json

import numpy as np
from devqubit_engine.tracking.run import track
from devqubit_pennylane.results import build_result_snapshot, extract_result_type
from devqubit_pennylane.snapshot import (
    create_device_snapshot,
    resolve_pennylane_backend,
)
from devqubit_pennylane.utils import (
    collect_sdk_versions,
    extract_shots_info,
    get_device_name,
    get_num_wires,
    is_pennylane_device,
    pennylane_version,
)


class TestResultTypeDetection:
    """Tests for result type detection."""

    def test_expectation_type(self, expectation_tape):
        """Detects expectation type."""
        result_type = extract_result_type([expectation_tape])
        assert "Expectation" in result_type or "expval" in result_type.lower()

    def test_counts_type(self, bell_tape):
        """Detects counts type."""
        result_type = extract_result_type([bell_tape])
        assert "Counts" in result_type or "count" in result_type.lower()

    def test_mixed_types(self, expectation_tape, probability_tape):
        """Detects mixed types."""
        assert extract_result_type([expectation_tape, probability_tape]) == "mixed"


class TestResultSnapshot:
    """Tests for ResultSnapshot building."""

    def test_expectations_single(self):
        """Builds expectation snapshot."""
        snap = build_result_snapshot(
            0.5,
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.status == "completed"
        assert snap.items[0].expectation.value == 0.5

    def test_expectations_multiple(self):
        """Builds multi-circuit expectation snapshot."""
        snap = build_result_snapshot(
            [0.1, -0.5],
            result_type="Expectation",
            backend_name="default.qubit",
            num_circuits=2,
        )

        assert len(snap.items) == 2
        assert snap.items[1].expectation.value == -0.5

    def test_counts_from_dict(self):
        """Builds counts snapshot."""
        snap = build_result_snapshot(
            {"0": 2, "1": 3},
            result_type="Counts",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.items[0].counts["shots"] == 5

    def test_samples_bitstring(self):
        """Converts samples to counts."""
        samples = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
        snap = build_result_snapshot(
            samples,
            result_type="Sample",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert snap.items[0].counts["counts"]["00"] == 2

    def test_probabilities(self):
        """Converts probabilities."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        snap = build_result_snapshot(
            probs,
            result_type="Probability",
            backend_name="default.qubit",
            num_circuits=1,
        )

        assert set(snap.items[0].quasi_probability.distribution.keys()) == {"00", "01"}

    def test_failed_execution(self):
        """Builds failed snapshot."""
        snap = build_result_snapshot(
            None,
            result_type=None,
            backend_name="default.qubit",
            num_circuits=1,
            success=False,
            error_info={"type": "RuntimeError", "message": "Failed"},
        )

        assert snap.success is False
        assert snap.error.type == "RuntimeError"


class TestDeviceSnapshot:
    """Tests for device snapshot."""

    def test_default_qubit_core_fields(self, default_qubit):
        """Creates snapshot with core fields."""
        snap = create_device_snapshot(default_qubit)

        assert snap.provider == "local"
        assert snap.backend_name == "default.qubit"
        assert snap.backend_type == "simulator"
        assert snap.num_qubits == 2

    def test_snapshot_serializable(self, default_qubit):
        """Snapshot is JSON serializable."""
        snap = create_device_snapshot(default_qubit)
        parsed = json.loads(json.dumps(snap.to_dict()))

        assert parsed["provider"] == "local"

    def test_frontend_config(self, default_qubit):
        """Snapshot has frontend config."""
        snap = create_device_snapshot(default_qubit)
        assert snap.frontend.sdk == "pennylane"


class TestShotsExtraction:
    """Tests for shots extraction."""

    def test_analytic_device(self, default_qubit):
        """Extracts analytic mode."""
        info = extract_shots_info(default_qubit)
        assert info.analytic is True

    def test_finite_shots(self, tape_with_shots_1000):
        """Extracts finite shots."""
        info = extract_shots_info(tape_with_shots_1000)
        assert info.total_shots == 1000

    def test_shot_vector(self, tape_with_shot_vector_uniform):
        """Extracts shot vector."""
        info = extract_shots_info(tape_with_shot_vector_uniform)
        assert info.total_shots == 300
        assert info.has_partitioned_shots is True


class TestBackendResolution:
    """Tests for backend resolution."""

    def test_native_device(self, default_qubit):
        """Resolves native device."""
        info = resolve_pennylane_backend(default_qubit)

        assert info["provider"] == "local"
        assert info["sdk_frontend"] == "pennylane"

    def test_braket_device(self, monkeypatch):
        """Resolves Braket device."""
        import devqubit_pennylane.snapshot as mod

        monkeypatch.setattr(mod, "get_device_name", lambda d: d.short_name)

        class MockBraket:
            short_name = "braket.aws.qubit"
            __module__ = "pennylane_braket.aws_qubit"
            device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            wires = [0, 1]
            shots = None

        info = resolve_pennylane_backend(MockBraket())
        assert info["provider"] == "aws_braket"


class TestRawPropertiesArtifact:
    """Tests for raw_properties artifact."""

    def test_without_tracker(self, default_qubit):
        """No ref without tracker."""
        snap = create_device_snapshot(default_qubit, tracker=None)
        assert snap.raw_properties_ref is None

    def test_with_tracker(self, default_qubit, store, registry):
        """Has ref with tracker."""
        with track(project="test", store=store, registry=registry) as run:
            snap = create_device_snapshot(default_qubit, tracker=run)
            assert snap.raw_properties_ref is not None


class TestUtilities:
    """Tests for utilities."""

    def test_pennylane_version(self):
        """Gets version."""
        assert "." in pennylane_version()

    def test_collect_sdk_versions(self):
        """Collects versions."""
        assert "pennylane" in collect_sdk_versions()

    def test_is_pennylane_device(self, default_qubit):
        """Detects device."""
        assert is_pennylane_device(default_qubit) is True
        assert is_pennylane_device(None) is False

    def test_get_device_name(self, default_qubit):
        """Gets name."""
        assert get_device_name(default_qubit) == "default.qubit"

    def test_get_num_wires(self, default_qubit):
        """Gets wires."""
        assert get_num_wires(default_qubit) == 2
