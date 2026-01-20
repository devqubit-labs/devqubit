# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit hashing and materialization."""

import math

from devqubit_qiskit.circuits import (
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
    materialize_circuits,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# =============================================================================
# Circuit Materialization Tests
# =============================================================================


class TestMaterializeCircuits:
    """Tests for generator/iterator handling."""

    def test_single_circuit(self, bell_circuit):
        """Single circuit returns (list, was_single=True)."""
        result, was_single = materialize_circuits(bell_circuit)
        assert len(result) == 1
        assert was_single is True

    def test_list_of_circuits(self, bell_circuit, ghz_circuit):
        """List of circuits passes through."""
        result, was_single = materialize_circuits([bell_circuit, ghz_circuit])
        assert len(result) == 2
        assert was_single is False

    def test_generator_consumed_once(self):
        """Generator is fully consumed once."""

        def circuit_gen():
            for _ in range(3):
                qc = QuantumCircuit(1)
                qc.h(0)
                qc.measure_all()
                yield qc

        gen = circuit_gen()
        result, _ = materialize_circuits(gen)
        assert len(result) == 3


# =============================================================================
# Structural Hashing Tests
# =============================================================================


class TestStructuralHashing:
    """Tests for circuit structural hashing."""

    def test_same_structure_same_hash(self):
        """Identical circuits produce same structural hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])

    def test_different_gates_different_hash(self):
        """Different gates produce different hashes."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.x(0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_parameter_values_ignored(self):
        """Bound param values don't affect structural hash."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        assert compute_structural_hash([bound1]) == compute_structural_hash([bound2])

    def test_cx_direction_matters(self):
        """CX(0,1) and CX(1,0) have different hashes."""
        qc1 = QuantumCircuit(2)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.cx(1, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_different_qubit_counts(self):
        """Same gates on different qubit counts hash differently."""
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        qc3 = QuantumCircuit(3)
        qc3.h(0)

        assert compute_structural_hash([qc2]) != compute_structural_hash([qc3])

    def test_measurement_clbits(self):
        """Measurements include classical bit mapping."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.measure(0, 0)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.measure(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_barrier_handling(self):
        """Barriers included in hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.barrier()
        qc2.cx(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])


# =============================================================================
# Parametric Hashing Tests
# =============================================================================


class TestParametricHashing:
    """Tests for circuit parametric hashing."""

    def test_no_params_equal_hashes(self):
        """structural == parametric for no-param circuits."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        structural, parametric = compute_circuit_hashes([qc])
        assert structural == parametric

    def test_different_param_values_different_hash(self):
        """Different param values produce different parametric hashes."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        assert compute_parametric_hash([bound1]) != compute_parametric_hash([bound2])

    def test_float_encoding_deterministic(self):
        """Float encoding is deterministic."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        val1 = math.pi / 4
        val2 = 0.7853981633974483
        val3 = math.atan(1)

        h1 = compute_parametric_hash([qc.assign_parameters({theta: val1})])
        h2 = compute_parametric_hash([qc.assign_parameters({theta: val2})])
        h3 = compute_parametric_hash([qc.assign_parameters({theta: val3})])

        assert h1 == h2 == h3


# =============================================================================
# Batch & Format Tests
# =============================================================================


class TestBatchAndFormat:
    """Tests for batch hashing and hash format."""

    def test_batch_consistent(self):
        """Batch hashing is consistent."""
        circuits = []
        for n in [2, 3]:
            qc = QuantumCircuit(n)
            qc.h(0)
            circuits.append(qc)

        h1 = compute_structural_hash(circuits)
        h2 = compute_structural_hash(circuits)

        assert h1 == h2

    def test_empty_returns_none(self):
        """Empty circuit list returns None."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format(self):
        """Hash format is sha256:<hex>."""
        qc = QuantumCircuit(2)
        qc.h(0)

        h = compute_structural_hash([qc])

        assert h.startswith("sha256:")
        assert len(h) == 7 + 64

    def test_multi_param_gate(self):
        """Multi-param gates hash correctly."""
        qc1 = QuantumCircuit(1)
        qc1.u(0.1, 0.2, 0.3, 0)

        qc2 = QuantumCircuit(1)
        qc2.u(0.1, 0.2, 0.3, 0)

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])


# =============================================================================
# Parameter Binds Hashing Tests
# =============================================================================


class TestParameterBindsHashing:
    """Tests for parameter_binds incorporation into parametric hash."""

    def test_different_binds_different_parametric_hash(self):
        """Different parameter_binds produce different parametric hashes."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()

        binds1 = [{theta: 0.5}]
        binds2 = [{theta: 1.5}]

        structural1, parametric1 = compute_circuit_hashes([qc], binds1)
        structural2, parametric2 = compute_circuit_hashes([qc], binds2)

        assert structural1 == structural2  # Same structure
        assert parametric1 != parametric2  # Different binds

    def test_same_binds_same_parametric_hash(self):
        """Same parameter_binds produce same parametric hash."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.measure_all()

        binds = [{theta: 0.5}]

        _, parametric1 = compute_circuit_hashes([qc], binds)
        _, parametric2 = compute_circuit_hashes([qc], binds)

        assert parametric1 == parametric2

    def test_no_binds_uses_circuit_params(self):
        """Without parameter_binds, hash uses circuit's bound values."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        _, parametric1 = compute_circuit_hashes([bound1], None)
        _, parametric2 = compute_circuit_hashes([bound2], None)

        assert parametric1 != parametric2


# =============================================================================
# UEC Hashing Contract Tests
# =============================================================================


class TestHashingContract:
    """Tests for UEC hashing contract compliance."""

    def test_contract_no_params_structural_equals_parametric(self):
        """No params => structural == parametric."""
        circuits = [
            QuantumCircuit(1),
            QuantumCircuit(2),
        ]
        circuits[0].h(0)
        circuits[1].h(0)
        circuits[1].cx(0, 1)

        for qc in circuits:
            structural, parametric = compute_circuit_hashes([qc])
            assert structural == parametric

    def test_contract_qubit_order_preserved(self):
        """Qubit order is preserved."""
        qc1 = QuantumCircuit(3)
        qc1.cx(0, 1)
        qc1.cx(1, 2)

        qc2 = QuantumCircuit(3)
        qc2.cx(1, 0)
        qc2.cx(2, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_contract_circuit_dimensions_in_hash(self):
        """Circuit dimensions affect hash."""
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

        assert len(set(hashes)) == 3
