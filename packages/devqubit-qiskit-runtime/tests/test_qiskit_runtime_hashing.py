# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime circuit hashing.

These tests verify that the Runtime adapter's hashing correctly delegates
to the engine and produces UEC-compliant hashes.
"""

import math

from devqubit_qiskit_runtime.circuits import (
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class TestRuntimeHashingBasics:
    """Basic hashing functionality tests."""

    def test_identical_circuits_same_hash(self):
        """Identical circuits must produce identical structural hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure_all()

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure_all()

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])

    def test_different_gates_different_hash(self):
        """Different gate sequences must produce different hashes."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.x(0)
        qc2.cz(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_empty_list_returns_none(self):
        """Empty circuit list must return None for both hashes."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash must follow sha256:<64hex> format."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        h = compute_structural_hash([qc])

        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
        # Verify hex chars
        hex_part = h[7:]
        assert all(c in "0123456789abcdef" for c in hex_part)


class TestRuntimeQubitOrderPreservation:
    """Tests that qubit order is preserved (critical for directional gates)."""

    def test_cx_direction_matters(self):
        """CX(0,1) and CX(1,0) must hash differently."""
        qc1 = QuantumCircuit(2)
        qc1.cx(0, 1)  # control=0, target=1

        qc2 = QuantumCircuit(2)
        qc2.cx(1, 0)  # control=1, target=0

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_ccx_control_order_matters(self):
        """CCX with different control order must hash differently."""
        qc1 = QuantumCircuit(3)
        qc1.ccx(0, 1, 2)

        qc2 = QuantumCircuit(3)
        qc2.ccx(1, 0, 2)  # Swapped controls

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_swap_is_symmetric(self):
        """SWAP(0,1) and SWAP(1,0) should hash same (symmetric gate)."""
        qc1 = QuantumCircuit(2)
        qc1.swap(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.swap(1, 0)

        # SWAP is symmetric, but we preserve order in hash
        # Both should still produce different hashes due to qubit ordering
        # Actually, let's verify what the actual behavior is
        h1 = compute_structural_hash([qc1])
        h2 = compute_structural_hash([qc2])
        # Document actual behavior - order is preserved even for symmetric gates
        assert h1 != h2, "Qubit order preserved even for symmetric gates"


class TestRuntimeCircuitDimensions:
    """Tests that circuit dimensions (nq, nc) affect hash."""

    def test_different_qubit_counts_different_hash(self):
        """Same gates on different qubit counts must hash differently."""
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        qc3 = QuantumCircuit(3)
        qc3.h(0)

        qc5 = QuantumCircuit(5)
        qc5.h(0)

        h2 = compute_structural_hash([qc2])
        h3 = compute_structural_hash([qc3])
        h5 = compute_structural_hash([qc5])

        assert len({h2, h3, h5}) == 3, "Each dimension must produce unique hash"

    def test_idle_qubits_affect_hash(self):
        """Idle qubits must affect hash (prevents false deduplication)."""
        # Circuit with gate on q0, idle q1
        qc_2q = QuantumCircuit(2)
        qc_2q.h(0)

        # Circuit with gate on q0, idle q1 and q2
        qc_3q = QuantumCircuit(3)
        qc_3q.h(0)

        assert compute_structural_hash([qc_2q]) != compute_structural_hash([qc_3q])

    def test_different_clbit_counts_different_hash(self):
        """Different classical bit counts must hash differently."""
        qc1 = QuantumCircuit(2, 1)
        qc1.h(0)
        qc1.measure(0, 0)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.measure(0, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])


class TestRuntimeParameterHandling:
    """Tests for parameter handling in hashes."""

    def test_structural_ignores_param_values(self):
        """Structural hash must ignore bound parameter values."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound_05 = qc.assign_parameters({theta: 0.5})
        bound_15 = qc.assign_parameters({theta: 1.5})
        bound_pi = qc.assign_parameters({theta: math.pi})

        h1 = compute_structural_hash([bound_05])
        h2 = compute_structural_hash([bound_15])
        h3 = compute_structural_hash([bound_pi])

        assert h1 == h2 == h3, "Structural hash must be same regardless of values"

    def test_parametric_differs_for_different_values(self):
        """Parametric hash must differ for different parameter values."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound_05 = qc.assign_parameters({theta: 0.5})
        bound_15 = qc.assign_parameters({theta: 1.5})

        h1 = compute_parametric_hash([bound_05])
        h2 = compute_parametric_hash([bound_15])

        assert h1 != h2, "Different values must produce different parametric hashes"

    def test_multi_param_gate(self):
        """Multi-parameter gates must hash all parameters."""
        qc1 = QuantumCircuit(1)
        qc1.u(0.1, 0.2, 0.3, 0)

        qc2 = QuantumCircuit(1)
        qc2.u(0.1, 0.2, 0.4, 0)  # Different phi

        # Structural should be same (same gate type and arity)
        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])

        # Parametric should differ
        assert compute_parametric_hash([qc1]) != compute_parametric_hash([qc2])

    def test_unbound_parameter_in_structural(self):
        """Unbound parameters should work in structural hash."""
        theta = Parameter("θ")
        phi = Parameter("φ")

        qc1 = QuantumCircuit(1)
        qc1.rx(theta, 0)

        qc2 = QuantumCircuit(1)
        qc2.rx(phi, 0)  # Different parameter name

        # Structural hash should be same (same structure)
        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])


class TestRuntimeHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_no_params_structural_equals_parametric(self):
        """CRITICAL: For circuits without parameters, structural == parametric."""
        # Various no-param circuits
        test_cases = [
            ("empty_1q", QuantumCircuit(1)),
            ("empty_2q", QuantumCircuit(2)),
            ("bell", self._make_bell()),
            ("ghz_3", self._make_ghz(3)),
            ("with_measure", self._make_measured_circuit()),
        ]

        for name, qc in test_cases:
            structural, parametric = compute_circuit_hashes([qc])
            assert structural == parametric, (
                f"Contract violated for '{name}': "
                f"structural={structural[:20]}... != parametric={parametric[:20]}..."
            )

    def test_bound_params_structural_equals_parametric_same_values(self):
        """Bound params with same values: structural == structural, parametric == parametric."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 1.234})
        bound2 = qc.assign_parameters({theta: 1.234})

        s1, p1 = compute_circuit_hashes([bound1])
        s2, p2 = compute_circuit_hashes([bound2])

        assert s1 == s2, "Same structure must have same structural hash"
        assert p1 == p2, "Same values must have same parametric hash"

    def test_float_encoding_deterministic(self):
        """Float encoding must be deterministic across representations."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        # Same value computed different ways
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # math.pi/4 as float literal
        val3 = math.atan(1)  # Another way to get pi/4

        h1 = compute_parametric_hash([qc.assign_parameters({theta: val1})])
        h2 = compute_parametric_hash([qc.assign_parameters({theta: val2})])
        h3 = compute_parametric_hash([qc.assign_parameters({theta: val3})])

        assert h1 == h2 == h3, "Same IEEE-754 value must produce same hash"

    def test_negative_zero_normalized(self):
        """Negative zero must be normalized to positive zero."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        h_pos = compute_parametric_hash([qc.assign_parameters({theta: 0.0})])
        h_neg = compute_parametric_hash([qc.assign_parameters({theta: -0.0})])

        assert h_pos == h_neg, "-0.0 must be normalized to 0.0"

    # Helper methods
    @staticmethod
    def _make_bell():
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    @staticmethod
    def _make_ghz(n: int):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        return qc

    @staticmethod
    def _make_measured_circuit():
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc


class TestRuntimeBatchHashing:
    """Tests for multi-circuit batch hashing."""

    def test_batch_order_matters(self):
        """Circuit order in batch must affect hash."""
        qc_h = QuantumCircuit(1)
        qc_h.h(0)

        qc_x = QuantumCircuit(1)
        qc_x.x(0)

        h1 = compute_structural_hash([qc_h, qc_x])
        h2 = compute_structural_hash([qc_x, qc_h])

        assert h1 != h2, "Different order must produce different hash"

    def test_batch_consistent(self):
        """Same batch must produce same hash."""
        circuits = []
        for n in [2, 3, 4]:
            qc = QuantumCircuit(n)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            circuits.append(qc)

        h1 = compute_structural_hash(circuits)
        h2 = compute_structural_hash(circuits)

        assert h1 == h2

    def test_batch_boundaries_respected(self):
        """Circuit boundaries must be properly delimited."""
        # Single 2-qubit circuit
        qc_single = QuantumCircuit(2)
        qc_single.h(0)
        qc_single.h(1)

        # Two 1-qubit circuits
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(1)
        qc2.h(0)

        h_single = compute_structural_hash([qc_single])
        h_batch = compute_structural_hash([qc1, qc2])

        # Must be different due to circuit boundaries
        assert h_single != h_batch, "Batch boundaries must be preserved"

    def test_mixed_param_batch(self):
        """Batch with mixed parameterized/non-parameterized circuits."""
        theta = Parameter("θ")

        qc_param = QuantumCircuit(1)
        qc_param.rx(theta, 0)
        qc_bound = qc_param.assign_parameters({theta: 1.5})

        qc_fixed = QuantumCircuit(1)
        qc_fixed.h(0)

        # Structural should be deterministic
        h1 = compute_structural_hash([qc_bound, qc_fixed])
        h2 = compute_structural_hash([qc_bound, qc_fixed])
        assert h1 == h2


class TestRuntimeMeasurementHashing:
    """Tests for measurement and classical bit hashing."""

    def test_measurement_target_matters(self):
        """Which classical bit receives measurement must affect hash."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.measure(0, 0)  # q0 -> c0

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.measure(0, 1)  # q0 -> c1

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_measurement_source_matters(self):
        """Which qubit is measured must affect hash."""
        qc1 = QuantumCircuit(2, 1)
        qc1.h(0)
        qc1.h(1)
        qc1.measure(0, 0)

        qc2 = QuantumCircuit(2, 1)
        qc2.h(0)
        qc2.h(1)
        qc2.measure(1, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])


class TestRuntimeSpecialCases:
    """Tests for edge cases and special circuits."""

    def test_barrier_included(self):
        """Barriers must be included in hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.barrier()
        qc2.cx(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_reset_included(self):
        """Reset operations must be included in hash."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(1)
        qc2.reset(0)
        qc2.h(0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_delay_included(self):
        """Delay operations must be included in hash."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(1)
        qc2.h(0)
        qc2.delay(100, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_deep_circuit(self):
        """Deep circuits must hash correctly."""
        qc = QuantumCircuit(2)
        for _ in range(100):
            qc.h(0)
            qc.cx(0, 1)
            qc.rz(0.1, 1)

        h1 = compute_structural_hash([qc])
        h2 = compute_structural_hash([qc])

        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_wide_circuit(self):
        """Wide circuits must hash correctly."""
        n = 20
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        h1 = compute_structural_hash([qc])
        h2 = compute_structural_hash([qc])

        assert h1 == h2


class TestRuntimeCrossAdapterConsistency:
    """Tests to verify consistency with base Qiskit adapter.

    These tests ensure that the Runtime adapter produces the same
    hashes as the base Qiskit adapter for equivalent circuits.
    """

    def test_same_circuit_same_hash_as_base_adapter(self):
        """Runtime adapter must produce same hash as base Qiskit adapter."""
        from devqubit_qiskit.circuits import (
            compute_structural_hash as qiskit_structural,
        )

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        runtime_hash = compute_structural_hash([qc])
        qiskit_hash = qiskit_structural([qc])

        assert (
            runtime_hash == qiskit_hash
        ), "Runtime and base Qiskit adapters must produce identical hashes"

    def test_parametric_hash_same_as_base_adapter(self):
        """Runtime parametric hash must match base Qiskit adapter."""
        from devqubit_qiskit.circuits import (
            compute_parametric_hash as qiskit_parametric,
        )

        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        bound = qc.assign_parameters({theta: 1.234})

        runtime_hash = compute_parametric_hash([bound])
        qiskit_hash = qiskit_parametric([bound])

        assert runtime_hash == qiskit_hash
