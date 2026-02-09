# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for circuit processing: hashing, serialization, and result extraction.

These tests verify the core circuit handling functionality that enables
artifact deduplication and cross-SDK comparison.
"""

from __future__ import annotations

import json
import math

from braket.circuits import Circuit, FreeParameter
from devqubit_braket.circuits import (
    compute_circuit_hashes,
    compute_parametric_hash,
    compute_structural_hash,
)
from devqubit_braket.results import extract_counts_payload, extract_measurement_counts
from devqubit_braket.serialization import (
    BraketCircuitLoader,
    BraketCircuitSerializer,
    is_braket_circuit,
    serialize_jaqcd,
    serialize_openqasm,
    summarize_braket_circuit,
)
from devqubit_engine.circuit.models import SDK, CircuitFormat


# =============================================================================
# Circuit Hashing - Core Contract
# =============================================================================


class TestHashingContract:
    """
    Tests for UEC hashing contract compliance.

    CRITICAL: These tests verify the hashing guarantees that enable
    circuit deduplication and caching.
    """

    def test_identical_circuits_same_hash(self):
        """Identical circuits produce identical structural hash."""
        c1 = Circuit().h(0).cnot(0, 1)
        c2 = Circuit().h(0).cnot(0, 1)

        assert compute_structural_hash([c1]) == compute_structural_hash([c2])

    def test_different_circuits_different_hash(self):
        """Different gate sequences produce different hashes."""
        c1 = Circuit().h(0).cnot(0, 1)
        c2 = Circuit().x(0).cz(0, 1)

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_qubit_order_preserved(self):
        """Directional gates must have qubit order preserved (CNOT direction matters)."""
        c1 = Circuit().cnot(0, 1)  # control=0, target=1
        c2 = Circuit().cnot(1, 0)  # control=1, target=0

        assert compute_structural_hash([c1]) != compute_structural_hash([c2])

    def test_no_params_structural_equals_parametric(self):
        """For circuits without parameters, structural == parametric."""
        test_cases = [
            Circuit(),
            Circuit().h(0),
            Circuit().h(0).cnot(0, 1),
        ]

        for circuit in test_cases:
            structural, parametric = compute_circuit_hashes([circuit])
            assert structural == parametric

    def test_empty_list_returns_none(self):
        """Empty circuit list returns None for both hashes."""
        structural, parametric = compute_circuit_hashes([])

        assert structural is None
        assert parametric is None

    def test_hash_format_sha256(self):
        """Hash follows sha256:<64hex> format."""
        h = compute_structural_hash([Circuit().h(0)])

        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


# =============================================================================
# Parameter Handling
# =============================================================================


class TestParameterHandling:
    """Tests for parametric circuit hashing."""

    def test_structural_ignores_param_values(self):
        """Structural hash ignores bound parameter values."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        bound_1 = c.make_bound_circuit({"theta": 0.5})
        bound_2 = c.make_bound_circuit({"theta": 1.5})

        assert compute_structural_hash([bound_1]) == compute_structural_hash([bound_2])

    def test_parametric_differs_for_different_values(self):
        """Parametric hash differs for different parameter values."""
        theta = FreeParameter("theta")
        c = Circuit().rx(0, theta)

        h1 = compute_parametric_hash([c], {"theta": 0.5})
        h2 = compute_parametric_hash([c], {"theta": 1.5})

        assert h1 != h2

    def test_float_encoding_deterministic(self):
        """Same IEEE-754 value produces same hash regardless of computation path."""
        val1 = math.pi / 4
        val2 = 0.7853981633974483  # Same value as literal

        h1 = compute_parametric_hash([Circuit().rx(0, val1)])
        h2 = compute_parametric_hash([Circuit().rx(0, val2)])

        assert h1 == h2

    def test_negative_zero_normalized(self):
        """-0.0 is normalized to 0.0."""
        h_pos = compute_parametric_hash([Circuit().rx(0, 0.0)])
        h_neg = compute_parametric_hash([Circuit().rx(0, -0.0)])

        assert h_pos == h_neg


# =============================================================================
# Batch Hashing
# =============================================================================


class TestBatchHashing:
    """Tests for multi-circuit batch hashing."""

    def test_batch_order_matters(self):
        """Circuit order in batch affects hash."""
        c_h = Circuit().h(0)
        c_x = Circuit().x(0)

        h1 = compute_structural_hash([c_h, c_x])
        h2 = compute_structural_hash([c_x, c_h])

        assert h1 != h2

    def test_batch_boundaries_preserved(self):
        """Circuit boundaries are properly delimited."""
        # Single 2-qubit circuit
        c_single = Circuit().h(0).h(1)

        # Two separate circuits
        c1 = Circuit().h(0)
        c2 = Circuit().h(0)

        assert compute_structural_hash([c_single]) != compute_structural_hash([c1, c2])


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization:
    """Tests for circuit serialization."""

    def test_jaqcd_serialization(self, bell_circuit):
        """Serializes circuit to valid JAQCD."""
        data = serialize_jaqcd(bell_circuit, name="bell", index=0)

        assert data.format == CircuitFormat.JAQCD
        assert data.sdk == SDK.BRAKET

        # Verify valid JSON with instructions
        parsed = json.loads(data.data)
        assert "instructions" in parsed

    def test_openqasm_serialization(self, bell_circuit):
        """Serializes circuit to valid OpenQASM."""
        data = serialize_openqasm(bell_circuit, name="bell", index=0)

        assert data.format == CircuitFormat.OPENQASM3
        assert "OPENQASM" in data.data
        assert "qubit[2]" in data.data

    def test_serialization_deterministic(self, bell_circuit):
        """Same circuit produces identical serialization."""
        data1 = serialize_jaqcd(bell_circuit)
        data2 = serialize_jaqcd(bell_circuit)

        assert data1.data == data2.data

    def test_roundtrip_jaqcd(self, bell_circuit):
        """Serialize + load preserves circuit structure."""
        serialized = serialize_jaqcd(bell_circuit)

        loader = BraketCircuitLoader()
        loaded = loader.load(serialized)

        assert loaded.sdk == SDK.BRAKET
        assert loaded.circuit.qubit_count == bell_circuit.qubit_count


# =============================================================================
# Circuit Detection and Serializer Interface
# =============================================================================


class TestCircuitDetection:
    """Tests for circuit type detection."""

    def test_detects_braket_circuit(self, bell_circuit):
        """Recognizes Braket circuits."""
        assert is_braket_circuit(bell_circuit) is True

    def test_rejects_non_circuits(self):
        """Rejects non-circuit objects."""
        assert is_braket_circuit("string") is False
        assert is_braket_circuit(None) is False
        assert is_braket_circuit([]) is False


class TestSerializerInterface:
    """Tests for BraketCircuitSerializer interface."""

    def test_serializer_properties(self):
        """Serializer reports correct SDK and formats."""
        serializer = BraketCircuitSerializer()

        assert serializer.sdk == SDK.BRAKET
        assert CircuitFormat.JAQCD in serializer.supported_formats
        assert CircuitFormat.OPENQASM3 in serializer.supported_formats

    def test_can_serialize_braket(self, bell_circuit):
        """Serializer accepts Braket circuits."""
        serializer = BraketCircuitSerializer()
        assert serializer.can_serialize(bell_circuit) is True
        assert serializer.can_serialize("not a circuit") is False


# =============================================================================
# Circuit Summary
# =============================================================================


class TestCircuitSummary:
    """Tests for circuit summary generation."""

    def test_bell_circuit_summary(self, bell_circuit):
        """Correctly summarizes Bell circuit."""
        summary = summarize_braket_circuit(bell_circuit)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_2q == 1  # CNOT
        assert summary.is_clifford is True

    def test_non_clifford_detection(self):
        """Detects non-Clifford circuits."""
        circuit = Circuit().h(0).t(0).cnot(0, 1)

        summary = summarize_braket_circuit(circuit)

        assert summary.is_clifford is False


# =============================================================================
# Result Extraction
# =============================================================================


class TestResultExtraction:
    """Tests for measurement result extraction."""

    def test_extracts_from_real_result(self, local_simulator, measured_bell_circuit):
        """Extracts counts from real Braket execution."""
        task = local_simulator.run(measured_bell_circuit, shots=100)
        result = task.result()

        counts = extract_measurement_counts(result)

        assert counts is not None
        assert sum(counts.values()) == 100
        assert all(isinstance(k, str) for k in counts)

    def test_bell_state_distribution(self, local_simulator, measured_bell_circuit):
        """Bell state produces correlated 00/11 outcomes."""
        task = local_simulator.run(measured_bell_circuit, shots=1000)
        result = task.result()

        counts = extract_measurement_counts(result)

        # Bell state should only have 00 and 11
        assert set(counts.keys()) <= {"00", "11"}

    def test_returns_none_for_none(self):
        """Returns None for None result."""
        assert extract_measurement_counts(None) is None

    def test_payload_structure(self, local_simulator, measured_bell_circuit):
        """Creates correct payload structure."""
        task = local_simulator.run(measured_bell_circuit, shots=100)
        result = task.result()

        payload = extract_counts_payload(result)

        assert "experiments" in payload
        assert len(payload["experiments"]) == 1

        exp = payload["experiments"][0]
        assert exp["index"] == 0
        assert "counts" in exp


# =============================================================================
# Result Extraction - Mock Objects
# =============================================================================


class TestResultExtractionMock:
    """Tests for result extraction with mock objects."""

    def test_extracts_from_measurement_counts_attr(self):
        """Extracts from measurement_counts attribute."""

        class MockResult:
            measurement_counts = {"00": 50, "11": 50}

        counts = extract_measurement_counts(MockResult())
        assert counts == {"00": 50, "11": 50}

    def test_handles_callable_measurement_counts(self):
        """Handles callable measurement_counts."""

        class MockResult:
            def measurement_counts(self):
                return {"00": 30, "11": 70}

        counts = extract_measurement_counts(MockResult())
        assert counts == {"00": 30, "11": 70}

    def test_handles_program_set_results(self):
        """Handles nested Program Set result structure."""

        class MockMeasuredEntry:
            counts = {"00": 50, "11": 50}

        class MockCompositeEntry:
            entries = [MockMeasuredEntry(), MockMeasuredEntry()]

        class MockProgramSetResult:
            entries = [MockCompositeEntry()]

        payload = extract_counts_payload(MockProgramSetResult())

        assert payload is not None
        assert len(payload["experiments"]) == 2

        for i, exp in enumerate(payload["experiments"]):
            assert exp["index"] == i
            assert "program_index" in exp
            assert "executable_index" in exp


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_deep_circuit_hashes(self):
        """Deep circuits hash correctly."""
        c = Circuit()
        for _ in range(100):
            c.h(0).cnot(0, 1)

        h1 = compute_structural_hash([c])
        h2 = compute_structural_hash([c])

        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_wide_circuit_hashes(self):
        """Wide circuits hash correctly."""
        c = Circuit()
        for i in range(20):
            c.h(i)

        h1 = compute_structural_hash([c])
        h2 = compute_structural_hash([c])

        assert h1 == h2

    def test_handles_broken_result(self):
        """Handles exceptions in result access gracefully."""

        class ExplodingResult:
            @property
            def measurement_counts(self):
                raise RuntimeError("Unavailable")

        counts = extract_measurement_counts(ExplodingResult())
        assert counts is None
