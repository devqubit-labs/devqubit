# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit hashing."""

import math

from devqubit_qiskit.circuits import (
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class TestQiskitHashing:
    """Tests for Qiskit-specific hashing integration."""

    def test_same_structure_same_hash(self):
        """Identical circuits must produce same structural hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])

    def test_different_gates_different_hash(self):
        """Different gates must produce different hashes."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.x(0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_parameter_values_ignored_structural(self):
        """Bound param values must not affect structural hash."""

        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        assert compute_structural_hash([bound1]) == compute_structural_hash([bound2])

    def test_cx_direction_matters(self):
        """CX(0,1) and CX(1,0) must have different hashes."""
        qc1 = QuantumCircuit(2)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.cx(1, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_different_qubit_counts(self):
        """Same gates on different qubit counts must hash differently."""
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        qc3 = QuantumCircuit(3)
        qc3.h(0)

        assert compute_structural_hash([qc2]) != compute_structural_hash([qc3])

    def test_no_params_equal_hashes(self):
        """structural == parametric for no-param circuits."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        structural, parametric = compute_circuit_hashes([qc])

        assert structural == parametric

    def test_different_param_values_different_parametric(self):
        """Different param values must produce different parametric hashes."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        assert compute_parametric_hash([bound1]) != compute_parametric_hash([bound2])

    def test_batch_consistent(self):
        """Batch hashing must be consistent."""
        circuits = []
        for n in [2, 3]:
            qc = QuantumCircuit(n)
            qc.h(0)
            circuits.append(qc)

        h1 = compute_structural_hash(circuits)
        h2 = compute_structural_hash(circuits)

        assert h1 == h2

    def test_empty_returns_none(self):
        """Empty circuit list must return None."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format(self):
        """Hash format must be sha256:<hex>."""
        qc = QuantumCircuit(2)
        qc.h(0)

        h = compute_structural_hash([qc])

        assert h.startswith("sha256:")
        assert len(h) == 7 + 64

    def test_measurement_clbits(self):
        """Measurements must include classical bit mapping."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.measure(0, 0)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.measure(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_barrier_handling(self):
        """Barriers must be included in hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.barrier()
        qc2.cx(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_multi_param_gate(self):
        """Multi-param gates must hash correctly."""
        qc1 = QuantumCircuit(1)
        qc1.u(0.1, 0.2, 0.3, 0)

        qc2 = QuantumCircuit(1)
        qc2.u(0.1, 0.2, 0.3, 0)

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])


class TestHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_contract_no_params_structural_equals_parametric(self):
        """no params => structural == parametric."""
        # Various no-param circuits
        circuits = [
            QuantumCircuit(1),  # Empty
            QuantumCircuit(2),  # Empty 2q
        ]

        # Add gates
        circuits[0].h(0)
        circuits[1].h(0)
        circuits[1].cx(0, 1)

        for qc in circuits:
            structural, parametric = compute_circuit_hashes([qc])
            assert structural == parametric, f"Contract violated for {qc.name}"

    def test_contract_qubit_order_preserved(self):
        """qubit order must be preserved."""
        # CX with different directions
        qc1 = QuantumCircuit(3)
        qc1.cx(0, 1)
        qc1.cx(1, 2)

        qc2 = QuantumCircuit(3)
        qc2.cx(1, 0)  # Swapped
        qc2.cx(2, 1)  # Swapped

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_contract_float_encoding_deterministic(self):
        """float encoding must be deterministic."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        # Same value computed different ways
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # Explicit
        val3 = math.atan(1)  # Computed

        h1 = compute_parametric_hash([qc.assign_parameters({theta: val1})])
        h2 = compute_parametric_hash([qc.assign_parameters({theta: val2})])
        h3 = compute_parametric_hash([qc.assign_parameters({theta: val3})])

        assert h1 == h2 == h3, "Same float value must produce same hash"

    def test_contract_circuit_dimensions_in_hash(self):
        """circuit dimensions must affect hash."""
        # Same gate, different circuit sizes
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        qc3 = QuantumCircuit(3)
        qc3.h(0)

        qc4 = QuantumCircuit(4)
        qc4.h(0)

        hashes = [
            compute_structural_hash([qc2]),
            compute_structural_hash([qc3]),
            compute_structural_hash([qc4]),
        ]

        # All must be unique
        assert (
            len(set(hashes)) == 3
        ), "Different dimensions must produce different hashes"
