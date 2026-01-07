# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared fixtures for PennyLane adapter tests."""

from __future__ import annotations

import pennylane as qml
import pytest
from devqubit_engine.storage.factory import create_registry, create_store


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


@pytest.fixture
def default_qubit():
    """Create a default.qubit device with 2 wires."""
    return qml.device("default.qubit", wires=2)


@pytest.fixture
def default_qubit_3wire():
    """Create a default.qubit device with 3 wires."""
    return qml.device("default.qubit", wires=3)


@pytest.fixture
def bell_tape():
    """Create a Bell state tape with counts measurement."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.counts(wires=[0, 1])
    return tape


@pytest.fixture
def bell_tape_with_shots():
    """Create a Bell state tape with counts measurement and 1000 shots."""
    with qml.tape.QuantumTape(shots=1000) as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.counts(wires=[0, 1])
    return tape


@pytest.fixture
def ghz_tape():
    """Create a 3-qubit GHZ tape with counts measurement."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.counts(wires=[0, 1, 2])
    return tape


@pytest.fixture
def expectation_tape():
    """Create a tape with expectation value measurement."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))
    return tape


@pytest.fixture
def probability_tape():
    """Create a tape with probability measurement."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0, 1])
    return tape


@pytest.fixture
def sample_bitstring_tape():
    """
    Create a tape with sample measurement returning bitstrings (0/1).

    qml.sample(wires=...) returns computational basis samples as 0/1 values.
    This is different from qml.sample(observable) which returns eigenvalues.
    """
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.sample(wires=[0, 1])
    return tape


@pytest.fixture
def sample_eigenvalue_tape():
    """
    Create a tape with sample measurement returning eigenvalues (±1).

    qml.sample(qml.PauliZ(0)) returns eigenvalues of the observable,
    typically ±1 for Pauli observables. This is semantically different
    from bitstring sampling.
    """
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.sample(qml.PauliZ(0))
    return tape


@pytest.fixture
def non_clifford_tape():
    """Create a tape with non-Clifford gates (T gate)."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.T(wires=0)
        qml.counts(wires=0)
    return tape


@pytest.fixture
def parameterized_tape():
    """Create a tape with trainable parameters."""
    with qml.tape.QuantumTape() as tape:
        qml.RX(0.5, wires=0)
        qml.RY(0.3, wires=0)
        qml.expval(qml.PauliZ(0))
    return tape


@pytest.fixture
def multi_measurement_tape():
    """Create a tape with multiple different measurements."""
    with qml.tape.QuantumTape() as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))
        qml.expval(qml.PauliX(1))
        qml.probs(wires=[0, 1])
    return tape


@pytest.fixture
def quantum_script_fixture():
    """
    Create a QuantumScript.

    PennyLane recommends QuantumScript when queueing context is not needed.
    It's more memory efficient and is the preferred internal representation.
    """
    ops = [
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
    ]
    measurements = [qml.expval(qml.PauliZ(0))]

    if hasattr(qml.tape, "QuantumScript"):
        return qml.tape.QuantumScript(ops, measurements)
    else:
        # Fallback for older PennyLane versions
        with qml.tape.QuantumTape() as tape:
            for op in ops:
                qml.apply(op)
            for m in measurements:
                qml.apply(m)
        return tape


@pytest.fixture
def tape_with_shots():
    """Create a tape with explicit shots configuration (500 shots)."""
    with qml.tape.QuantumTape(shots=500) as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.counts(wires=[0, 1])
    return tape


@pytest.fixture
def tape_with_shots_1000():
    """Create a tape with 1000 shots for finite shots testing."""
    with qml.tape.QuantumTape(shots=1000) as tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.counts(wires=[0, 1])
    return tape


@pytest.fixture
def tape_with_shot_vector():
    """
    Create a tape with shot vector (partitioned shots).

    This tests proper handling of PennyLane's Shots class with shot_vector.
    Total: 50 + 50 + 100 = 200 shots.
    """
    with qml.tape.QuantumTape(shots=[50, 50, 100]) as tape:
        qml.Hadamard(wires=0)
        qml.counts(wires=0)
    return tape


@pytest.fixture
def tape_with_shot_vector_uniform():
    """
    Create a tape with uniform shot vector [100, 100, 100].

    PennyLane consolidates identical shots: [100, 100, 100] -> [(100, 3)].
    Total: 300 shots.
    """
    with qml.tape.QuantumTape(shots=[100, 100, 100]) as tape:
        qml.Hadamard(wires=0)
        qml.counts(wires=0)
    return tape
