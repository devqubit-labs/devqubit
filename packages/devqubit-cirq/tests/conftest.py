# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for Cirq adapter tests."""

from __future__ import annotations

import cirq
import pytest
import sympy
from devqubit_engine.storage.factory import create_registry, create_store


@pytest.fixture
def tracking_root(tmp_path):
    """Create temporary tracking directory."""
    return tmp_path / ".devqubit"


@pytest.fixture
def store(tracking_root):
    """Create a temporary object store."""
    return create_store(f"file://{tracking_root}/objects")


@pytest.fixture
def registry(tracking_root):
    """Create a temporary run registry."""
    return create_registry(f"file://{tracking_root}")


@pytest.fixture
def bell_circuit():
    """Create a 2-qubit Bell state circuit."""
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        [
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="result"),
        ]
    )


@pytest.fixture
def ghz_circuit():
    """Create a 3-qubit GHZ circuit."""
    qubits = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.measure(*qubits, key="result"),
        ]
    )


@pytest.fixture
def non_clifford_circuit():
    """Create a circuit with non-Clifford gates (T gate)."""
    qubit = cirq.LineQubit(0)
    return cirq.Circuit(
        [
            cirq.H(qubit),
            cirq.T(qubit),
            cirq.measure(qubit, key="m"),
        ]
    )


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized circuit."""
    theta = sympy.Symbol("theta")
    qubit = cirq.LineQubit(0)
    return cirq.Circuit(
        [
            cirq.rz(theta).on(qubit),
            cirq.measure(qubit, key="m"),
        ]
    )


@pytest.fixture
def simulator():
    """Create a Cirq Simulator."""
    return cirq.Simulator()


@pytest.fixture
def density_matrix_simulator():
    """Create a Cirq DensityMatrixSimulator."""
    return cirq.DensityMatrixSimulator()
