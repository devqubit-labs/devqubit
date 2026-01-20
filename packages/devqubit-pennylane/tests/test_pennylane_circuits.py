# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit hashing and serialization."""

import json
import math

import pennylane as qml
import pytest
from devqubit_engine.circuit.models import SDK, CircuitFormat
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_pennylane.circuits import (
    _deterministic_wire_hash,
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)
from devqubit_pennylane.serialization import (
    PennyLaneCircuitLoader,
    PennyLaneCircuitSerializer,
    is_pennylane_tape,
    serialize_tape,
    serialize_tapes,
    summarize_pennylane_tape,
)


class TestCircuitHashing:
    """Tests for tape hashing functionality."""

    def test_identical_tapes_same_hash(self):
        """Identical tapes produce identical structural hash."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        assert compute_structural_hash(tape1) == compute_structural_hash(tape2)

    def test_different_gates_different_hash(self):
        """Different gate sequences produce different hashes."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        with qml.tape.QuantumTape() as tape2:
            qml.PauliX(wires=0)
            qml.CZ(wires=[0, 1])

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)

    def test_none_returns_none(self):
        """None input returns None for both hashes."""
        structural, parametric = compute_circuit_hashes(None)
        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash follows sha256:<64hex> format."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)

        h = compute_structural_hash(tape)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64

    def test_cnot_direction_matters(self):
        """CNOT(0,1) and CNOT(1,0) hash differently."""
        with qml.tape.QuantumTape() as tape1:
            qml.CNOT(wires=[0, 1])

        with qml.tape.QuantumTape() as tape2:
            qml.CNOT(wires=[1, 0])

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)

    def test_structural_ignores_param_values(self):
        """Structural hash ignores parameter values."""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.5, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.5, wires=0)

        assert compute_structural_hash(tape1) == compute_structural_hash(tape2)

    def test_parametric_differs_for_different_values(self):
        """Parametric hash differs for different values."""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.5, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.5, wires=0)

        assert compute_parametric_hash(tape1) != compute_parametric_hash(tape2)

    def test_no_params_structural_equals_parametric(self):
        """For tapes without parameters, structural == parametric."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        structural, parametric = compute_circuit_hashes(tape)
        assert structural == parametric

    def test_float_encoding_deterministic(self):
        """Same IEEE-754 float produces same hash."""
        vals = [math.pi / 4, 0.7853981633974483, math.atan(1)]
        hashes = []
        for v in vals:
            with qml.tape.QuantumTape() as tape:
                qml.RX(v, wires=0)
            hashes.append(compute_parametric_hash(tape))

        assert hashes[0] == hashes[1] == hashes[2]


class TestTapeSerialization:
    """Tests for tape serialization and loading."""

    def test_serialize_bell_tape(self, bell_tape):
        """Serializes Bell state tape to JSON."""
        data = serialize_tape(bell_tape, name="bell", index=0)

        assert data.format == CircuitFormat.TAPE_JSON
        assert data.sdk == SDK.PENNYLANE

        parsed = json.loads(data.data)
        assert parsed["num_operations"] == 2

    def test_serialize_includes_shots(self, tape_with_shots):
        """Includes shots in serialization."""
        data = serialize_tape(tape_with_shots)
        parsed = json.loads(data.data)
        assert parsed["shots"] == 500

    def test_serialize_batch(self, bell_tape, ghz_tape):
        """Batch serialization includes indices."""
        data = serialize_tapes([bell_tape, ghz_tape])
        parsed = json.loads(data.data)

        assert parsed["num_tapes"] == 2
        assert [t["index"] for t in parsed["tapes"]] == [0, 1]

    def test_roundtrip_preserves_operations(self, bell_tape):
        """Roundtrip preserves operations."""
        loader = PennyLaneCircuitLoader()
        serialized = serialize_tape(bell_tape)
        loaded = loader.load(serialized)

        assert len(loaded.circuit.operations) == len(bell_tape.operations)

    def test_unsupported_format_raises(self, bell_tape):
        """Raises for unsupported format."""
        with pytest.raises(SerializerError):
            PennyLaneCircuitSerializer().serialize(bell_tape, CircuitFormat.OPENQASM3)

    def test_loader_unsupported_format_raises(self):
        """Loader raises for unsupported format."""
        from devqubit_engine.circuit.models import CircuitData

        data = CircuitData(
            data="{}",
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.PENNYLANE,
            name="x",
            index=0,
        )
        with pytest.raises(LoaderError):
            PennyLaneCircuitLoader().load(data)


class TestTapeSummary:
    """Tests for tape summary."""

    def test_bell_tape_summary(self, bell_tape):
        """Summarizes Bell tape correctly."""
        summary = summarize_pennylane_tape(bell_tape)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1
        assert summary.gate_count_2q == 1

    def test_clifford_detection(self, bell_tape, non_clifford_tape):
        """Detects Clifford vs non-Clifford."""
        assert summarize_pennylane_tape(bell_tape).is_clifford is True
        assert summarize_pennylane_tape(non_clifford_tape).is_clifford is False

    def test_parameterized_detection(self, parameterized_tape):
        """Detects parameterized tapes."""
        summary = summarize_pennylane_tape(parameterized_tape)
        assert summary.has_parameters is True


class TestTapeTypeDetection:
    """Tests for tape type detection."""

    def test_detects_quantum_tape(self, bell_tape):
        """Detects QuantumTape."""
        assert is_pennylane_tape(bell_tape) is True

    def test_rejects_non_tapes(self):
        """Rejects non-tapes."""
        assert is_pennylane_tape(None) is False
        assert is_pennylane_tape("tape") is False


class TestWireHashDeterminism:
    """Tests for deterministic wire hashing (audit fix)."""

    def test_string_wires_hash_deterministically(self):
        """String wire labels produce consistent hashes across calls."""

        # Same input should always produce same output
        wire_str = "qubit_alpha"
        hash1 = _deterministic_wire_hash(wire_str)
        hash2 = _deterministic_wire_hash(wire_str)

        assert hash1 == hash2
        assert isinstance(hash1, int)
        assert 0 <= hash1 < 2**31

    def test_different_wires_different_hashes(self):
        """Different wire labels produce different hashes."""

        hash_a = _deterministic_wire_hash("wire_a")
        hash_b = _deterministic_wire_hash("wire_b")

        assert hash_a != hash_b

    def test_string_wire_tapes_hash_consistently(self):
        """Tapes with string wires produce consistent structural hashes."""
        # Create tape with string wire labels
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["a", "b"])

        with qml.tape.QuantumTape() as tape2:
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["a", "b"])

        h1 = compute_structural_hash(tape1)
        h2 = compute_structural_hash(tape2)

        assert h1 == h2
        assert h1 is not None
