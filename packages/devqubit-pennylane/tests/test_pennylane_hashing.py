# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for PennyLane tape hashing.

These tests verify that the PennyLane adapter's hashing correctly delegates
to the engine and produces UEC-compliant hashes.
"""

import math

import pennylane as qml
from devqubit_pennylane.circuits import (
    _get_tapes,
    _is_tape_like,
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
    tape_to_op_stream,
)


class TestPennyLaneHashingBasics:
    """Basic hashing functionality tests."""

    def test_identical_tapes_same_hash(self):
        """Identical tapes must produce identical structural hash."""
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
        """Different gate sequences must produce different hashes."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        with qml.tape.QuantumTape() as tape2:
            qml.PauliX(wires=0)
            qml.CZ(wires=[0, 1])

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)

    def test_none_returns_none(self):
        """None input must return None for both hashes."""
        structural, parametric = compute_circuit_hashes(None)

        assert structural is None
        assert parametric is None

    def test_empty_list_returns_none(self):
        """Empty list must return None."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash must follow sha256:<64hex> format."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        h = compute_structural_hash(tape)

        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
        # Verify hex chars
        hex_part = h[7:]
        assert all(char in "0123456789abcdef" for char in hex_part)


class TestPennyLaneWireOrderPreservation:
    """Tests that wire order is preserved (critical for directional gates)."""

    def test_cnot_direction_matters(self):
        """CNOT(0,1) and CNOT(1,0) must hash differently."""
        with qml.tape.QuantumTape() as tape1:
            qml.CNOT(wires=[0, 1])  # control=0, target=1

        with qml.tape.QuantumTape() as tape2:
            qml.CNOT(wires=[1, 0])  # control=1, target=0

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)

    def test_toffoli_control_order_matters(self):
        """Toffoli with different control order must hash differently."""
        with qml.tape.QuantumTape() as tape1:
            qml.Toffoli(wires=[0, 1, 2])

        with qml.tape.QuantumTape() as tape2:
            qml.Toffoli(wires=[1, 0, 2])  # Swapped controls

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)


class TestPennyLaneTapeDimensions:
    """Tests that tape dimensions (num_wires) affect hash."""

    def test_different_wire_counts_different_hash(self):
        """Same gates on different wire indices must hash differently."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

        with qml.tape.QuantumTape() as tape2:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=2)  # Wire 2 instead of 1

        assert compute_structural_hash(tape1) != compute_structural_hash(tape2)


class TestPennyLaneParameterHandling:
    """Tests for parameter handling in hashes."""

    def test_structural_ignores_param_values(self):
        """Structural hash must ignore bound parameter values."""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.5, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.5, wires=0)

        with qml.tape.QuantumTape() as tape3:
            qml.RX(math.pi, wires=0)

        h1 = compute_structural_hash(tape1)
        h2 = compute_structural_hash(tape2)
        h3 = compute_structural_hash(tape3)

        assert h1 == h2 == h3, "Structural hash must be same regardless of values"

    def test_parametric_differs_for_different_values(self):
        """Parametric hash must differ for different parameter values."""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.5, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(1.5, wires=0)

        h1 = compute_parametric_hash(tape1)
        h2 = compute_parametric_hash(tape2)

        assert h1 != h2, "Different values must produce different parametric hashes"

    def test_multi_param_gate(self):
        """Multi-parameter gates must hash all parameters."""
        with qml.tape.QuantumTape() as tape1:
            qml.Rot(0.1, 0.2, 0.3, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.Rot(0.1, 0.2, 0.4, wires=0)  # Different last param

        # Structural should be same (same gate type)
        assert compute_structural_hash(tape1) == compute_structural_hash(tape2)

        # Parametric should differ (different values)
        assert compute_parametric_hash(tape1) != compute_parametric_hash(tape2)


class TestPennyLaneHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_no_params_structural_equals_parametric(self):
        """CRITICAL: For tapes without parameters, structural == parametric."""
        test_cases = []

        # Empty tape
        with qml.tape.QuantumTape() as tape_empty:
            pass
        test_cases.append(("empty", tape_empty))

        # Single gate
        with qml.tape.QuantumTape() as tape_single:
            qml.Hadamard(wires=0)
        test_cases.append(("single_gate", tape_single))

        # Bell state
        with qml.tape.QuantumTape() as tape_bell:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
        test_cases.append(("bell", tape_bell))

        # With measurement
        with qml.tape.QuantumTape() as tape_meas:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
        test_cases.append(("with_meas", tape_meas))

        for name, tape in test_cases:
            structural, parametric = compute_circuit_hashes(tape)
            assert structural == parametric, (
                f"Contract violated for '{name}': "
                f"structural={structural[:20]}... != parametric={parametric[:20]}..."
            )

    def test_float_encoding_deterministic(self):
        """Float encoding must be deterministic across representations."""
        # Same value computed different ways
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # math.pi/4 as float literal
        val3 = math.atan(1)  # Another way to get pi/4

        with qml.tape.QuantumTape() as tape1:
            qml.RX(val1, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(val2, wires=0)

        with qml.tape.QuantumTape() as tape3:
            qml.RX(val3, wires=0)

        h1 = compute_parametric_hash(tape1)
        h2 = compute_parametric_hash(tape2)
        h3 = compute_parametric_hash(tape3)

        assert h1 == h2 == h3, "Same IEEE-754 value must produce same hash"

    def test_negative_zero_normalized(self):
        """Negative zero must be normalized to positive zero."""
        with qml.tape.QuantumTape() as tape_pos:
            qml.RX(0.0, wires=0)

        with qml.tape.QuantumTape() as tape_neg:
            qml.RX(-0.0, wires=0)

        h_pos = compute_parametric_hash(tape_pos)
        h_neg = compute_parametric_hash(tape_neg)

        assert h_pos == h_neg, "-0.0 must be normalized to 0.0"

    def test_trainable_params_not_in_hash(self):
        """trainable_params should NOT affect hash (it's training metadata)."""
        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.5, wires=0)
            qml.RY(0.3, wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.5, wires=0)
            qml.RY(0.3, wires=0)

        # Modify trainable_params on one tape
        tape1.trainable_params = [0]  # Only first param trainable
        tape2.trainable_params = [0, 1]  # Both params trainable

        # Hash should be the same (trainable_params is metadata, not semantics)
        assert compute_parametric_hash(tape1) == compute_parametric_hash(tape2)


class TestPennyLaneBatchHashing:
    """Tests for multi-tape batch hashing."""

    def test_batch_order_matters(self):
        """Tape order in batch must affect hash."""
        with qml.tape.QuantumTape() as tape_h:
            qml.Hadamard(wires=0)

        with qml.tape.QuantumTape() as tape_x:
            qml.PauliX(wires=0)

        h1 = compute_structural_hash([tape_h, tape_x])
        h2 = compute_structural_hash([tape_x, tape_h])

        assert h1 != h2, "Different order must produce different hash"

    def test_batch_consistent(self):
        """Same batch must produce same hash."""
        tapes = []
        for _ in range(3):
            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
            tapes.append(tape)

        h1 = compute_structural_hash(tapes)
        h2 = compute_structural_hash(tapes)

        assert h1 == h2

    def test_batch_boundaries_respected(self):
        """Tape boundaries must be properly delimited."""
        # Single tape with 2 H gates
        with qml.tape.QuantumTape() as tape_single:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

        # Two tapes with 1 H gate each
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.Hadamard(wires=0)

        h_single = compute_structural_hash([tape_single])
        h_batch = compute_structural_hash([tape1, tape2])

        # Must be different due to tape boundaries
        assert h_single != h_batch, "Batch boundaries must be preserved"


class TestPennyLaneMeasurementHashing:
    """Tests for measurement hashing."""

    def test_different_measurement_types_different_hash(self):
        """Different measurement types must produce different hashes."""
        with qml.tape.QuantumTape() as tape_expval:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape_var:
            qml.Hadamard(wires=0)
            qml.var(qml.PauliZ(0))

        assert compute_structural_hash(tape_expval) != compute_structural_hash(tape_var)

    def test_different_observables_different_hash(self):
        """Different observables must produce different hashes."""
        with qml.tape.QuantumTape() as tape_z:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape_x:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliX(0))

        assert compute_structural_hash(tape_z) != compute_structural_hash(tape_x)


class TestPennyLaneOpStreamConversion:
    """Tests for tape_to_op_stream conversion."""

    def test_op_stream_gate_names_lowercase(self):
        """Gate names in op_stream must be lowercase."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(0.5, wires=0)

        ops = tape_to_op_stream(tape)

        for op in ops:
            if not op["gate"].startswith("__"):  # Skip special markers
                assert (
                    op["gate"] == op["gate"].lower()
                ), f"Gate name not lowercase: {op['gate']}"

    def test_op_stream_wires_as_integers(self):
        """Wire indices must be integers."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        ops = tape_to_op_stream(tape)

        for op in ops:
            for w in op["qubits"]:
                assert isinstance(w, int), f"Wire index not int: {type(w)}"


class TestPennyLaneHelperFunctions:
    """Tests for helper functions."""

    def test_is_tape_like_with_tape(self):
        """_is_tape_like should return True for tapes."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)

        assert _is_tape_like(tape) is True

    def test_is_tape_like_with_non_tape(self):
        """_is_tape_like should return False for non-tapes."""
        assert _is_tape_like(None) is False
        assert _is_tape_like([1, 2, 3]) is False
        assert _is_tape_like("not a tape") is False

    def test_get_tapes_with_single_tape(self):
        """_get_tapes should wrap single tape in list."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)

        tapes = _get_tapes(tape)
        assert len(tapes) == 1
        assert tapes[0] is tape

    def test_get_tapes_with_list(self):
        """_get_tapes should return list as-is."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.PauliX(wires=0)

        tapes = _get_tapes([tape1, tape2])
        assert len(tapes) == 2
