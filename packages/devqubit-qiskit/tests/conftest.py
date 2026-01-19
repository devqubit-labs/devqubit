# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for Qiskit adapter tests."""

from __future__ import annotations

import pytest
from devqubit_engine.storage.factory import create_registry, create_store
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator


# =============================================================================
# Infrastructure Fixtures
# =============================================================================


@pytest.fixture
def tracking_root(tmp_path):
    """Create temporary tracking directory."""
    return tmp_path / ".devqubit"


@pytest.fixture
def store(tracking_root):
    """Create a temporary store."""
    return create_store(f"file://{tracking_root}/objects")


@pytest.fixture
def registry(tracking_root):
    """Create a temporary registry."""
    return create_registry(f"file://{tracking_root}")


# =============================================================================
# Backend Fixtures
# =============================================================================


@pytest.fixture
def aer_simulator():
    """Create an AerSimulator backend."""
    return AerSimulator()


# =============================================================================
# Circuit Fixtures
# =============================================================================


@pytest.fixture
def bell_circuit():
    """Create a Bell state circuit."""
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def ghz_circuit():
    """Create a 3-qubit GHZ circuit."""
    qc = QuantumCircuit(3, name="ghz")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized circuit."""
    theta = Parameter("θ")
    phi = Parameter("φ")
    qc = QuantumCircuit(2, name="parameterized")
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def batch_circuits():
    """Create a batch of 3 different circuits."""
    circuits = []
    for n in [2, 3, 4]:
        qc = QuantumCircuit(n, name=f"ghz_{n}")
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        circuits.append(qc)
    return circuits


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_coupling_map():
    """Create a mock coupling map."""

    class MockCouplingMap:
        def get_edges(self):
            return [(0, 1), (1, 2), (2, 3)]

    return MockCouplingMap()


@pytest.fixture
def mock_target(mock_coupling_map):
    """Create a mock BackendV2 Target."""

    class MockTarget:
        num_qubits = 4
        operation_names = ["cx", "sx", "rz", "x"]

        def build_coupling_map(self):
            return mock_coupling_map

    return MockTarget()


@pytest.fixture
def mock_backend(mock_target):
    """Create a mock BackendV2 backend."""

    class MockBackend:
        name = "mock_backend"
        target = mock_target

    return MockBackend()


@pytest.fixture
def mock_properties():
    """Create mock backend properties with calibration data."""

    class MockProperties:
        def to_dict(self):
            return {
                "last_update_date": "2024-01-15T10:00:00Z",
                "qubits": [
                    [
                        {"name": "T1", "value": 150.0, "unit": "us"},
                        {"name": "T2", "value": 85.0, "unit": "us"},
                        {"name": "readout_error", "value": 0.012},
                    ],
                    [
                        {"name": "T1", "value": 175.0, "unit": "us"},
                        {"name": "T2", "value": 95.0, "unit": "us"},
                        {"name": "readout_error", "value": 0.015},
                    ],
                ],
                "gates": [
                    {
                        "gate": "sx",
                        "qubits": [0],
                        "parameters": [
                            {"name": "gate_error", "value": 0.0002},
                            {"name": "gate_length", "value": 35.5, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "sx",
                        "qubits": [1],
                        "parameters": [
                            {"name": "gate_error", "value": 0.0003},
                            {"name": "gate_length", "value": 35.5, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "cx",
                        "qubits": [0, 1],
                        "parameters": [
                            {"name": "gate_error", "value": 0.008},
                            {"name": "gate_length", "value": 300.0, "unit": "ns"},
                        ],
                    },
                ],
            }

    return MockProperties()


@pytest.fixture
def mock_backend_with_calibration(mock_properties):
    """Create a mock backend with calibration properties."""

    class MockBackend:
        name = "calibrated_backend"

        def properties(self):
            return mock_properties

    return MockBackend()
