# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Braket circuit hashing."""

import math

from braket.circuits import Circuit, FreeParameter
from devqubit_braket.circuits import (
    circuit_to_op_stream,
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)


class TestBraketHashingBasics:
    """Basic hashing functionality tests."""

    def test_identical_circuits_same_hash(self):
        """Identical circuits must produce identical structural hash."""
        c1 = Circuit().h(0).cnot(0, 1)
        c2 = Circuit().h(0).cnot(0, 1)

        assert compute_structural_hash([c1]) == compute_structural_hash([c2])

    def test_different_gates_different_hash(self):
        """Different gate sequences must produce different hashes."""
        c1 = Circuit().h(0).cnot(0, 1)
        c2 = Circuit().x(0).cz(0, 1)

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_empty_list_returns_none(self):
        """Empty circuit list must return None for both hashes."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash must follow sha256:<64hex> format."""
        c = Circuit().h(0).cnot(0, 1)

        h = compute_structural_hash([c])

        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
        # Verify hex chars
        hex_part = h[7:]
        assert all(char in "0123456789abcdef" for char in hex_part)


class TestBraketQubitOrderPreservation:
    """Tests that qubit order is preserved (critical for directional gates)."""

    def test_cnot_direction_matters(self):
        """CNOT(0,1) and CNOT(1,0) must hash differently."""
        c1 = Circuit().cnot(0, 1)  # control=0, target=1
        c2 = Circuit().cnot(1, 0)  # control=1, target=0

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_ccnot_control_order_matters(self):
        """CCNot with different control order must hash differently."""
        c1 = Circuit().ccnot(0, 1, 2)
        c2 = Circuit().ccnot(1, 0, 2)  # Swapped controls

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_multi_qubit_gate_order(self):
        """Multi-qubit gates must preserve qubit order."""
        c1 = Circuit().cnot(0, 1).cnot(1, 2)
        c2 = Circuit().cnot(1, 0).cnot(2, 1)  # Swapped

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])


class TestBraketCircuitDimensions:
    """Tests that circuit dimensions (nq) affect hash."""

    def test_different_qubit_counts_different_hash(self):
        """Same gates on different qubit counts must hash differently."""
        # These circuits have different qubit counts due to different max indices
        c2 = Circuit().h(0).h(1)  # 2 qubits
        c3 = Circuit().h(0).h(2)  # 3 qubits (indices 0, 2)
        c5 = Circuit().h(0).h(4)  # 5 qubits (indices 0, 4)

        h2 = compute_structural_hash([c2])
        h3 = compute_structural_hash([c3])
        h5 = compute_structural_hash([c5])

        assert len({h2, h3, h5}) == 3, "Each dimension must produce unique hash"

    def test_idle_qubits_affect_hash(self):
        """Idle qubits must affect hash (prevents false deduplication)."""
        # Circuit with gate on q0, q1 used but idle
        c_2q = Circuit().h(0).i(1)  # Identity on q1 to include it

        # Circuit with gate on q0, q2 used
        c_3q = Circuit().h(0).i(2)  # Identity on q2

        assert compute_structural_hash([c_2q]) != compute_structural_hash([c_3q])


class TestBraketParameterHandling:
    """Tests for parameter handling in hashes."""

    def test_structural_ignores_param_values(self):
        """Structural hash must ignore bound parameter values."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        bound_05 = c.make_bound_circuit({"theta": 0.5})
        bound_15 = c.make_bound_circuit({"theta": 1.5})
        bound_pi = c.make_bound_circuit({"theta": math.pi})

        h1 = compute_structural_hash([bound_05])
        h2 = compute_structural_hash([bound_15])
        h3 = compute_structural_hash([bound_pi])

        assert h1 == h2 == h3, "Structural hash must be same regardless of values"

    def test_parametric_differs_for_different_values(self):
        """Parametric hash must differ for different parameter values."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        # Use inputs dict to bind parameters
        inputs_05 = {"theta": 0.5}
        inputs_15 = {"theta": 1.5}

        h1 = compute_parametric_hash([c], inputs_05)
        h2 = compute_parametric_hash([c], inputs_15)

        assert h1 != h2, "Different values must produce different parametric hashes"

    def test_multi_param_gate(self):
        """Multi-parameter gates must hash all parameters."""
        # Braket doesn't have a direct U gate, but we can use multiple rotations
        c1 = Circuit().rx(0, 0.1).ry(0, 0.2).rz(0, 0.3)
        c2 = Circuit().rx(0, 0.1).ry(0, 0.2).rz(0, 0.4)  # Different last param

        # Structural should be same (same gate types)
        assert compute_structural_hash([c1]) == compute_structural_hash([c2])

        # Parametric should differ (different values)
        assert compute_parametric_hash([c1]) != compute_parametric_hash([c2])

    def test_unbound_parameter_in_structural(self):
        """Unbound parameters should work in structural hash."""
        theta = FreeParameter("theta")
        phi = FreeParameter("phi")

        c1 = Circuit().rx(0, theta)
        c2 = Circuit().rx(0, phi)  # Different parameter name

        # Structural hash should be same (same structure)
        assert compute_structural_hash([c1]) == compute_structural_hash([c2])


class TestBraketHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_no_params_structural_equals_parametric(self):
        """CRITICAL: For circuits without parameters, structural == parametric."""
        test_cases = [
            ("empty", Circuit()),
            ("single_gate", Circuit().h(0)),
            ("bell", Circuit().h(0).cnot(0, 1)),
            ("ghz_3", Circuit().h(0).cnot(0, 1).cnot(1, 2)),
        ]

        for name, c in test_cases:
            structural, parametric = compute_circuit_hashes([c])
            assert structural == parametric, (
                f"Contract violated for '{name}': "
                f"structural={structural[:20]}... != parametric={parametric[:20]}..."
            )

    def test_bound_params_structural_equals_parametric_same_values(self):
        """Bound params with same values: structural same, parametric same."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        inputs = {"theta": 1.234}

        s1, p1 = compute_circuit_hashes([c], inputs)
        s2, p2 = compute_circuit_hashes([c], inputs)

        assert s1 == s2, "Same structure must have same structural hash"
        assert p1 == p2, "Same values must have same parametric hash"

    def test_float_encoding_deterministic(self):
        """Float encoding must be deterministic across representations."""
        # Same value computed different ways
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # math.pi/4 as float literal
        val3 = math.atan(1)  # Another way to get pi/4

        _ = Circuit().rx(0, val1)

        # Create circuits with same structure but potentially different float paths
        h1 = compute_parametric_hash([Circuit().rx(0, val1)])
        h2 = compute_parametric_hash([Circuit().rx(0, val2)])
        h3 = compute_parametric_hash([Circuit().rx(0, val3)])

        assert h1 == h2 == h3, "Same IEEE-754 value must produce same hash"

    def test_negative_zero_normalized(self):
        """Negative zero must be normalized to positive zero."""
        h_pos = compute_parametric_hash([Circuit().rx(0, 0.0)])
        h_neg = compute_parametric_hash([Circuit().rx(0, -0.0)])

        assert h_pos == h_neg, "-0.0 must be normalized to 0.0"


class TestBraketBatchHashing:
    """Tests for multi-circuit batch hashing."""

    def test_batch_order_matters(self):
        """Circuit order in batch must affect hash."""
        c_h = Circuit().h(0)
        c_x = Circuit().x(0)

        h1 = compute_structural_hash([c_h, c_x])
        h2 = compute_structural_hash([c_x, c_h])

        assert h1 != h2, "Different order must produce different hash"

    def test_batch_consistent(self):
        """Same batch must produce same hash."""
        circuits = [
            Circuit().h(0).cnot(0, 1),
            Circuit().h(0).cnot(0, 1).cnot(1, 2),
            Circuit().x(0).y(1).z(2),
        ]

        h1 = compute_structural_hash(circuits)
        h2 = compute_structural_hash(circuits)

        assert h1 == h2

    def test_batch_boundaries_respected(self):
        """Circuit boundaries must be properly delimited."""
        # Single 2-qubit circuit with 2 H gates
        c_single = Circuit().h(0).h(1)

        # Two 1-qubit circuits
        c1 = Circuit().h(0)
        c2 = Circuit().h(0)

        h_single = compute_structural_hash([c_single])
        h_batch = compute_structural_hash([c1, c2])

        # Must be different due to circuit boundaries
        assert h_single != h_batch, "Batch boundaries must be preserved"


class TestBraketOpStreamConversion:
    """Tests for circuit_to_op_stream conversion."""

    def test_op_stream_gate_names_lowercase(self):
        """Gate names in op_stream must be lowercase."""
        c = Circuit().h(0).cnot(0, 1).rx(0, 0.5)

        ops = circuit_to_op_stream(c)

        for op in ops:
            assert (
                op["gate"] == op["gate"].lower()
            ), f"Gate name not lowercase: {op['gate']}"

    def test_op_stream_qubits_as_integers(self):
        """Qubit indices must be integers, not strings."""
        c = Circuit().h(0).cnot(0, 1)

        ops = circuit_to_op_stream(c)

        for op in ops:
            for q in op["qubits"]:
                assert isinstance(q, int), f"Qubit index not int: {type(q)}"

    def test_op_stream_params_format(self):
        """Parameters must follow p0, p0_name format."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        ops = circuit_to_op_stream(c)

        # Find the rx operation
        rx_op = next(op for op in ops if op["gate"] == "rx")
        assert "params" in rx_op
        assert "p0" in rx_op["params"]
        assert "p0_name" in rx_op["params"]
        assert rx_op["params"]["p0"] is None  # Unbound
        assert rx_op["params"]["p0_name"] == "theta"


class TestBraketSpecialCases:
    """Tests for edge cases and special circuits."""

    def test_identity_gate(self):
        """Identity gates must be included in hash."""
        c1 = Circuit().h(0)
        c2 = Circuit().h(0).i(0)

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_deep_circuit(self):
        """Deep circuits must hash correctly."""
        c = Circuit()
        for _ in range(100):
            c.h(0).cnot(0, 1).rz(1, 0.1)

        h1 = compute_structural_hash([c])
        h2 = compute_structural_hash([c])

        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_wide_circuit(self):
        """Wide circuits must hash correctly."""
        n = 20
        c = Circuit()
        for i in range(n):
            c.h(i)
        for i in range(n - 1):
            c.cnot(i, i + 1)

        h1 = compute_structural_hash([c])
        h2 = compute_structural_hash([c])

        assert h1 == h2
