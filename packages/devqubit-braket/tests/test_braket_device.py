# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for device snapshot and calibration extraction.

These tests verify that device properties are correctly captured
in DeviceSnapshot objects with topology, native gates, and calibration.
"""

from __future__ import annotations

import pytest
from devqubit_braket.device import create_device_snapshot, extract_calibration


# =============================================================================
# Device Snapshot with Real Device
# =============================================================================


class TestDeviceSnapshotReal:
    """Tests using real Braket LocalSimulator."""

    def test_local_simulator_snapshot(self, local_simulator):
        """Creates valid snapshot from LocalSimulator."""
        snap = create_device_snapshot(local_simulator)

        assert snap.provider == "local"
        assert snap.backend_name is not None
        assert snap.backend_type == "simulator"
        assert snap.captured_at is not None
        assert "braket" in snap.sdk_versions

    def test_local_simulator_has_no_calibration(self, local_simulator):
        """LocalSimulator doesn't have calibration data."""
        snap = create_device_snapshot(local_simulator)
        assert snap.calibration is None

    def test_snapshot_serializes_to_valid_dict(self, local_simulator):
        """Snapshot serializes to schema-compliant dict."""
        snap = create_device_snapshot(local_simulator)
        d = snap.to_dict()

        assert d["schema"] == "devqubit.device_snapshot/1.0"
        assert d["provider"] == "local"
        assert d["backend_type"] == "simulator"
        assert isinstance(d["sdk_versions"], dict)


# =============================================================================
# Topology Extraction
# =============================================================================


class TestTopologyExtraction:
    """Tests for device topology extraction."""

    def test_extracts_linear_topology(self, device_factory):
        """Extracts qubit count and linear connectivity."""
        device = device_factory(
            name="linear_3q",
            qubit_count=3,
            connectivity_graph={"0": [1], "1": [0, 2], "2": [1]},
        )

        snap = create_device_snapshot(device)

        assert snap.num_qubits == 3
        assert snap.connectivity is not None
        assert (0, 1) in snap.connectivity
        assert (1, 2) in snap.connectivity

    def test_fully_connected_omits_edges(self, device_factory):
        """Fully connected devices don't expand O(n²) edges."""
        device = device_factory(
            name="fully_connected",
            qubit_count=10,
            fully_connected=True,
        )

        snap = create_device_snapshot(device)

        assert snap.num_qubits == 10
        assert snap.connectivity is None  # Not expanded

    def test_connectivity_is_sorted(self, device_factory):
        """Connectivity list is sorted for determinism."""
        device = device_factory(
            name="sorted",
            qubit_count=4,
            connectivity_graph={"3": [2], "0": [1], "2": [1, 3], "1": [0, 2]},
        )

        snap = create_device_snapshot(device)

        assert snap.connectivity == sorted(snap.connectivity)


# =============================================================================
# Native Gates Extraction
# =============================================================================


class TestNativeGatesExtraction:
    """Tests for native gate set extraction."""

    def test_extracts_native_gates(self, device_factory):
        """Extracts native gates from properties."""
        device = device_factory(
            name="native_gates",
            qubit_count=2,
            native_gates=["x", "rz", "cz"],
        )

        snap = create_device_snapshot(device)

        assert snap.native_gates == ["x", "rz", "cz"]

    def test_no_native_gates_returns_none(self, device_factory):
        """Returns None when no gate info available."""
        device = device_factory(name="no_gates", qubit_count=2)

        snap = create_device_snapshot(device)

        assert snap.native_gates is None


# =============================================================================
# Calibration Extraction
# =============================================================================


class TestCalibrationExtraction:
    """Tests for device calibration extraction."""

    def test_extracts_one_qubit_gate_errors(self, device_factory):
        """Extracts 1Q gate errors from fidelities (error = 1 - fidelity)."""
        device = device_factory(
            name="1q_cal",
            qubit_count=2,
            one_qubit_fidelities={
                0: [{"gateName": "x", "fidelity": 0.99}],
                1: [{"gateName": "x", "fidelity": 0.98}],
            },
        )

        cal = extract_calibration(device)

        assert cal is not None
        assert len(cal.gates) == 2

        x_q0 = [g for g in cal.gates if g.qubits == (0,)][0]
        assert x_q0.error == pytest.approx(0.01)

    def test_extracts_two_qubit_gate_errors(self, device_factory):
        """Extracts 2Q gate errors from fidelities."""
        device = device_factory(
            name="2q_cal",
            qubit_count=2,
            two_qubit_fidelities={
                "0-1": [{"gateName": "cz", "fidelity": 0.90}],
            },
        )

        cal = extract_calibration(device)

        assert cal is not None
        cz = cal.gates[0]
        assert cz.gate == "cz"
        assert cz.qubits == (0, 1)
        assert cz.error == pytest.approx(0.10)

    def test_computes_median_errors(self, device_factory):
        """Computes per-qubit median 1Q errors."""
        device = device_factory(
            name="median_cal",
            qubit_count=2,
            one_qubit_fidelities={
                0: [
                    {"gateName": "x", "fidelity": 0.99},  # error 0.01
                    {"gateName": "rz", "fidelity": 0.98},  # error 0.02
                ],
            },
        )

        cal = extract_calibration(device)

        q0 = [q for q in cal.qubits if q.qubit == 0][0]
        assert q0.gate_error_1q == pytest.approx(0.015)  # median([0.01, 0.02])

    def test_handles_various_edge_key_formats(self, device_factory):
        """Parses various 2Q edge key formats from different providers."""
        formats_and_expected = [
            ("0-1", (0, 1)),
            ("(0,1)", (0, 1)),
            ("[0,1]", (0, 1)),
        ]

        for key_format, expected in formats_and_expected:
            device = device_factory(
                name=f"test_{key_format}",
                qubit_count=2,
                two_qubit_fidelities={
                    key_format: [{"gateName": "cz", "fidelity": 0.95}]
                },
            )
            cal = extract_calibration(device)

            assert cal is not None, f"Failed for {key_format}"
            assert cal.gates[0].qubits == expected, f"Failed for {key_format}"

    def test_returns_none_without_calibration(self, device_factory):
        """Returns None when no calibration data available."""
        device = device_factory(name="no_cal", qubit_count=2)

        cal = extract_calibration(device)

        assert cal is None


# =============================================================================
# Calibration Integration in Snapshot
# =============================================================================


class TestSnapshotWithCalibration:
    """Tests for calibration data in device snapshots."""

    def test_snapshot_includes_calibration(self, mock_device):
        """Snapshot includes calibration when available."""
        snap = create_device_snapshot(mock_device)

        assert snap.calibration is not None
        assert len(snap.calibration.gates) > 0

    def test_calibration_summary_accessible(self, mock_device):
        """Calibration summary is accessible via snapshot."""
        snap = create_device_snapshot(mock_device)

        summary = snap.get_calibration_summary()

        assert summary is not None
        assert "median_2q_error" in summary


# =============================================================================
# Backend Type Resolution
# =============================================================================


class TestBackendTypeResolution:
    """Tests for backend_type resolution logic."""

    def test_local_simulator_is_simulator(self, local_simulator):
        """LocalSimulator resolves to 'simulator'."""
        snap = create_device_snapshot(local_simulator)
        assert snap.backend_type == "simulator"

    def test_mock_qpu_is_hardware(self, mock_device):
        """Mock QPU device resolves to 'hardware'."""
        snap = create_device_snapshot(mock_device)
        assert snap.backend_type == "hardware"


# =============================================================================
# Provider Detection
# =============================================================================


class TestProviderDetection:
    """Tests for physical provider detection."""

    def test_local_simulator_provider_is_local(self, local_simulator):
        """LocalSimulator has provider='local'."""
        snap = create_device_snapshot(local_simulator)
        assert snap.provider == "local"

    def test_mock_aws_device_provider_is_aws(self, mock_device):
        """Mock AWS device has provider='aws_braket'."""
        snap = create_device_snapshot(mock_device)
        assert snap.provider == "aws_braket"


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_handles_missing_properties(self):
        """Handles devices without properties attribute."""

        class NoPropsDevice:
            __module__ = "braket.devices"

            @property
            def name(self):
                return "NoPropsDevice"

        snap = create_device_snapshot(NoPropsDevice())

        assert snap.captured_at is not None
        assert snap.num_qubits is None
        assert snap.calibration is None

    def test_handles_broken_properties(self):
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

        # Should not crash, just return None for unavailable data
        assert snap.captured_at is not None
        assert snap.num_qubits is None


# =============================================================================
# Snapshot Determinism
# =============================================================================


class TestSnapshotDeterminism:
    """Tests for snapshot consistency (important for comparisons)."""

    def test_same_device_consistent_snapshots(self, mock_device):
        """Same device produces consistent snapshots."""
        snap1 = create_device_snapshot(mock_device)
        snap2 = create_device_snapshot(mock_device)

        # captured_at will differ, but structure should match
        assert snap1.provider == snap2.provider
        assert snap1.backend_name == snap2.backend_name
        assert snap1.num_qubits == snap2.num_qubits
        assert snap1.connectivity == snap2.connectivity
        assert snap1.native_gates == snap2.native_gates
