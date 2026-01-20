# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Cirq circuit hashing.

These tests verify that the Cirq adapter's hashing correctly delegates
to the engine and produces UEC-compliant hashes.
"""

import math

import cirq
import sympy
from devqubit_cirq.circuits import (
    _get_num_qubits,
    _qubit_to_index,
    circuit_to_op_stream,
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)


class TestCirqHashingBasics:
    """Basic hashing functionality tests."""

    def test_identical_circuits_same_hash(self):
        """Identical circuits must produce identical structural hash."""
        q0, q1 = cirq.LineQubit.range(2)

        circuit1 = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.measure(q0, q1, key="m"),
            ]
        )

        circuit2 = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.measure(q0, q1, key="m"),
            ]
        )

        assert compute_structural_hash([circuit1]) == compute_structural_hash(
            [circuit2]
        )

    def test_different_gates_different_hash(self):
        """Different gate sequences must produce different hashes."""
        q0, q1 = cirq.LineQubit.range(2)

        circuit1 = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
            ]
        )

        circuit2 = cirq.Circuit(
            [
                cirq.X(q0),
                cirq.CZ(q0, q1),
            ]
        )

        assert compute_structural_hash([circuit1]) != compute_structural_hash(
            [circuit2]
        )

    def test_empty_list_returns_none(self):
        """Empty list must return None for both hashes."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash must follow sha256:<64hex> format."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])

        h = compute_structural_hash([circuit])

        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
        # Verify hex chars
        hex_part = h[7:]
        assert all(char in "0123456789abcdef" for char in hex_part)


class TestCirqQubitOrderPreservation:
    """Tests that qubit order is preserved (critical for directional gates)."""

    def test_cnot_direction_matters(self):
        """CNOT(0,1) and CNOT(1,0) must hash differently."""
        q0, q1 = cirq.LineQubit.range(2)

        circuit1 = cirq.Circuit([cirq.CNOT(q0, q1)])  # control=0, target=1
        circuit2 = cirq.Circuit([cirq.CNOT(q1, q0)])  # control=1, target=0

        assert compute_structural_hash([circuit1]) != compute_structural_hash(
            [circuit2]
        )

    def test_ccnot_control_order_matters(self):
        """CCX with different control order must hash differently."""
        q0, q1, q2 = cirq.LineQubit.range(3)

        circuit1 = cirq.Circuit([cirq.TOFFOLI(q0, q1, q2)])
        circuit2 = cirq.Circuit([cirq.TOFFOLI(q1, q0, q2)])  # Swapped controls

        assert compute_structural_hash([circuit1]) != compute_structural_hash(
            [circuit2]
        )


class TestCirqCircuitDimensions:
    """Tests that circuit dimensions (num_qubits) affect hash."""

    def test_different_qubit_indices_different_hash(self):
        """Same gates on different qubit indices must hash differently."""
        q0, q1, q2 = cirq.LineQubit.range(3)

        circuit1 = cirq.Circuit([cirq.H(q0), cirq.H(q1)])
        circuit2 = cirq.Circuit([cirq.H(q0), cirq.H(q2)])  # q2 instead of q1

        assert compute_structural_hash([circuit1]) != compute_structural_hash(
            [circuit2]
        )


class TestCirqParameterHandling:
    """Tests for parameter handling in hashes."""

    def test_structural_ignores_param_values(self):
        """Structural hash must ignore bound parameter values."""
        q = cirq.LineQubit(0)

        circuit1 = cirq.Circuit([cirq.Rx(rads=0.5)(q)])
        circuit2 = cirq.Circuit([cirq.Rx(rads=1.5)(q)])
        circuit3 = cirq.Circuit([cirq.Rx(rads=math.pi)(q)])

        h1 = compute_structural_hash([circuit1])
        h2 = compute_structural_hash([circuit2])
        h3 = compute_structural_hash([circuit3])

        assert h1 == h2 == h3, "Structural hash must be same regardless of values"

    def test_parametric_differs_for_different_values(self):
        """Parametric hash must differ for different parameter values."""
        q = cirq.LineQubit(0)

        circuit1 = cirq.Circuit([cirq.Rx(rads=0.5)(q)])
        circuit2 = cirq.Circuit([cirq.Rx(rads=1.5)(q)])

        h1 = compute_parametric_hash([circuit1])
        h2 = compute_parametric_hash([circuit2])

        assert h1 != h2, "Different values must produce different parametric hashes"

    def test_symbolic_parameter_resolved(self):
        """Symbolic parameters resolved via ParamResolver must affect hash."""
        q = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")

        circuit = cirq.Circuit([cirq.Rx(rads=theta)(q)])

        resolver1 = cirq.ParamResolver({"theta": 0.5})
        resolver2 = cirq.ParamResolver({"theta": 1.5})

        h1 = compute_parametric_hash([circuit], resolver1)
        h2 = compute_parametric_hash([circuit], resolver2)

        assert h1 != h2, "Different resolver values must produce different hashes"


class TestCirqHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_no_params_structural_equals_parametric(self):
        """CRITICAL: For circuits without parameters, structural == parametric."""
        q0, q1 = cirq.LineQubit.range(2)

        test_cases = []

        # Empty circuit
        test_cases.append(("empty", cirq.Circuit()))

        # Single gate
        test_cases.append(("single_gate", cirq.Circuit([cirq.H(q0)])))

        # Bell state
        test_cases.append(
            (
                "bell",
                cirq.Circuit(
                    [
                        cirq.H(q0),
                        cirq.CNOT(q0, q1),
                    ]
                ),
            )
        )

        # With measurement
        test_cases.append(
            (
                "with_meas",
                cirq.Circuit(
                    [
                        cirq.H(q0),
                        cirq.CNOT(q0, q1),
                        cirq.measure(q0, q1, key="m"),
                    ]
                ),
            )
        )

        for name, circuit in test_cases:
            structural, parametric = compute_circuit_hashes([circuit])
            assert structural == parametric, (
                f"Contract violated for '{name}': "
                f"structural={structural[:20]}... != parametric={parametric[:20]}..."
            )

    def test_float_encoding_deterministic(self):
        """Float encoding must be deterministic across representations."""
        q = cirq.LineQubit(0)

        # Same value computed different ways
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # math.pi/4 as float literal
        val3 = math.atan(1)  # Another way to get pi/4

        circuit1 = cirq.Circuit([cirq.Rx(rads=val1)(q)])
        circuit2 = cirq.Circuit([cirq.Rx(rads=val2)(q)])
        circuit3 = cirq.Circuit([cirq.Rx(rads=val3)(q)])

        h1 = compute_parametric_hash([circuit1])
        h2 = compute_parametric_hash([circuit2])
        h3 = compute_parametric_hash([circuit3])

        assert h1 == h2 == h3, "Same IEEE-754 value must produce same hash"

    def test_negative_zero_normalized(self):
        """Negative zero must be normalized to positive zero."""
        q = cirq.LineQubit(0)

        circuit_pos = cirq.Circuit([cirq.Rx(rads=0.0)(q)])
        circuit_neg = cirq.Circuit([cirq.Rx(rads=-0.0)(q)])

        h_pos = compute_parametric_hash([circuit_pos])
        h_neg = compute_parametric_hash([circuit_neg])

        assert h_pos == h_neg, "-0.0 must be normalized to 0.0"


class TestCirqBatchHashing:
    """Tests for multi-circuit batch hashing."""

    def test_batch_order_matters(self):
        """Circuit order in batch must affect hash."""
        q = cirq.LineQubit(0)

        circuit_h = cirq.Circuit([cirq.H(q)])
        circuit_x = cirq.Circuit([cirq.X(q)])

        h1 = compute_structural_hash([circuit_h, circuit_x])
        h2 = compute_structural_hash([circuit_x, circuit_h])

        assert h1 != h2, "Different order must produce different hash"

    def test_batch_consistent(self):
        """Same batch must produce same hash."""
        q0, q1 = cirq.LineQubit.range(2)

        circuits = [
            cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)]),
            cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)]),
            cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)]),
        ]

        h1 = compute_structural_hash(circuits)
        h2 = compute_structural_hash(circuits)

        assert h1 == h2

    def test_batch_boundaries_respected(self):
        """Circuit boundaries must be properly delimited."""
        q0, q1 = cirq.LineQubit.range(2)

        # Single circuit with 2 H gates
        circuit_single = cirq.Circuit([cirq.H(q0), cirq.H(q1)])

        # Two circuits with 1 H gate each
        circuit1 = cirq.Circuit([cirq.H(q0)])
        circuit2 = cirq.Circuit([cirq.H(q0)])

        h_single = compute_structural_hash([circuit_single])
        h_batch = compute_structural_hash([circuit1, circuit2])

        # Must be different due to circuit boundaries
        assert h_single != h_batch, "Batch boundaries must be preserved"


class TestCirqMeasurementHashing:
    """Tests for measurement hashing."""

    def test_different_measurement_keys_different_hash(self):
        """Different measurement keys must produce different hashes."""
        q = cirq.LineQubit(0)

        circuit1 = cirq.Circuit([cirq.H(q), cirq.measure(q, key="a")])
        circuit2 = cirq.Circuit([cirq.H(q), cirq.measure(q, key="b")])

        assert compute_structural_hash([circuit1]) != compute_structural_hash(
            [circuit2]
        )


class TestCirqOpStreamConversion:
    """Tests for circuit_to_op_stream conversion."""

    def test_op_stream_gate_names_lowercase(self):
        """Gate names in op_stream must be lowercase."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.Rx(rads=0.5)(q0),
            ]
        )

        ops = circuit_to_op_stream(circuit)

        for op in ops:
            if not op["gate"].startswith("__"):  # Skip special markers
                assert (
                    op["gate"] == op["gate"].lower()
                ), f"Gate name not lowercase: {op['gate']}"

    def test_op_stream_qubits_as_integers(self):
        """Qubit indices must be integers."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

        ops = circuit_to_op_stream(circuit)

        for op in ops:
            for q in op["qubits"]:
                assert isinstance(q, int), f"Qubit index not int: {type(q)}"


class TestCirqMomentDeterminism:
    """Tests for deterministic ordering within moments."""

    def test_moment_operations_sorted(self):
        """Operations within a moment must be sorted deterministically."""
        # Create qubits in non-sequential order
        q3 = cirq.LineQubit(3)
        q1 = cirq.LineQubit(1)
        q0 = cirq.LineQubit(0)

        # Build circuit with operations in same moment (different order)
        circuit1 = cirq.Circuit([cirq.Moment([cirq.H(q3), cirq.H(q1), cirq.H(q0)])])
        circuit2 = cirq.Circuit([cirq.Moment([cirq.H(q0), cirq.H(q3), cirq.H(q1)])])

        h1 = compute_structural_hash([circuit1])
        h2 = compute_structural_hash([circuit2])

        assert (
            h1 == h2
        ), "Same operations in different order within moment must hash same"


class TestCirqQubitTypes:
    """Tests for different qubit types."""

    def test_line_qubit_indexing(self):
        """LineQubit indices must be correctly extracted."""
        q5 = cirq.LineQubit(5)
        assert _qubit_to_index(q5) == 5

    def test_grid_qubit_indexing(self):
        """GridQubit must produce unique indices."""
        g00 = cirq.GridQubit(0, 0)
        g01 = cirq.GridQubit(0, 1)
        g10 = cirq.GridQubit(1, 0)

        idx00 = _qubit_to_index(g00)
        idx01 = _qubit_to_index(g01)
        idx10 = _qubit_to_index(g10)

        # All must be different
        assert len({idx00, idx01, idx10}) == 3

    def test_named_qubit_indexing(self):
        """NamedQubit must produce deterministic indices."""
        na = cirq.NamedQubit("a")
        nb = cirq.NamedQubit("b")

        idx_a1 = _qubit_to_index(na)
        idx_a2 = _qubit_to_index(na)
        idx_b = _qubit_to_index(nb)

        # Same qubit = same index
        assert idx_a1 == idx_a2

        # Different qubits = different indices
        assert idx_a1 != idx_b


class TestCirqHelperFunctions:
    """Tests for helper functions."""

    def test_get_num_qubits(self):
        """_get_num_qubits returns correct count."""
        q0, q1, q2 = cirq.LineQubit.range(3)

        circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q1, q2)])
        nq = _get_num_qubits(circuit)

        # Should be max index + 1 = 3
        assert nq == 3

    def test_get_num_qubits_empty(self):
        """_get_num_qubits returns 0 for empty circuit."""
        circuit = cirq.Circuit()
        assert _get_num_qubits(circuit) == 0


class TestCirqHashingDeterminism:
    """Tests for deterministic hashing across process invocations.

    These tests verify that hashing does not use Python's built-in hash()
    which is randomized per process since Python 3.3.
    """

    def test_named_qubit_hash_stable_without_builtin_hash(self):
        """NamedQubit hashing must not depend on Python's hash().

        Python's hash() is randomized per process (PYTHONHASHSEED).
        Qubit indexing must use deterministic methods like sum(ord()).
        """
        from devqubit_cirq.circuits import _build_qubit_map

        qa = cirq.NamedQubit("alice")
        qb = cirq.NamedQubit("bob")
        qc = cirq.NamedQubit("charlie")

        circuit = cirq.Circuit(
            [
                cirq.H(qa),
                cirq.CNOT(qa, qb),
                cirq.CNOT(qb, qc),
                cirq.measure(qa, qb, qc, key="m"),
            ]
        )

        # Build qubit map multiple times - must be identical
        map1 = _build_qubit_map(circuit)
        map2 = _build_qubit_map(circuit)

        assert map1 == map2, "Qubit map must be deterministic"

        # Verify sorted string order is used (alice < bob < charlie)
        assert (
            map1["alice"] < map1["bob"] < map1["charlie"]
        ), "NamedQubit indices should follow sorted string order"

    def test_circuit_hash_deterministic_with_named_qubits(self):
        """Circuit with NamedQubits must hash identically on repeated calls."""
        qa = cirq.NamedQubit("q_alpha")
        qb = cirq.NamedQubit("q_beta")

        circuit = cirq.Circuit(
            [
                cirq.H(qa),
                cirq.CNOT(qa, qb),
                cirq.measure(qa, qb, key="result"),
            ]
        )

        # Hash 10 times - all must be identical
        hashes = [compute_structural_hash([circuit]) for _ in range(10)]

        assert (
            len(set(hashes)) == 1
        ), "NamedQubit circuit hash must be deterministic across calls"
