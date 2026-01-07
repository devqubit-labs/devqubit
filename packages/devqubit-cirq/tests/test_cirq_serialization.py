# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Cirq circuit serialization and summarization."""

import json

import cirq
import pytest
from devqubit_cirq.serialization import (
    CirqCircuitLoader,
    CirqCircuitSerializer,
    is_cirq_circuit,
    serialize_json,
    serialize_openqasm3,
    summarize_cirq_circuit,
)
from devqubit_engine.circuit.models import SDK, CircuitFormat
from devqubit_engine.circuit.registry import SerializerError


class TestCircuitDetection:
    """Tests for circuit type detection."""

    def test_detects_cirq_circuit(self, bell_circuit):
        """Detects Cirq Circuit objects."""
        assert is_cirq_circuit(bell_circuit) is True

    def test_detects_frozen_circuit(self, bell_circuit):
        """Detects FrozenCircuit objects."""
        frozen = cirq.FrozenCircuit(bell_circuit)
        assert is_cirq_circuit(frozen) is True

    def test_rejects_non_circuits(self):
        """Rejects non-circuit objects."""
        assert is_cirq_circuit(None) is False
        assert is_cirq_circuit("circuit") is False
        assert is_cirq_circuit([]) is False


class TestSerializeJson:
    """Tests for Cirq JSON serialization."""

    def test_serialize_bell_circuit(self, bell_circuit):
        """Serializes Bell circuit to valid JSON."""
        data = serialize_json(bell_circuit, name="bell", index=0)

        assert data.format == CircuitFormat.CIRQ_JSON
        assert data.sdk == SDK.CIRQ
        assert data.name == "bell"
        assert data.index == 0

        parsed = json.loads(data.data)
        assert parsed.get("cirq_type") == "Circuit"

    def test_default_name(self, bell_circuit):
        """Uses default name when not provided."""
        data = serialize_json(bell_circuit, index=5)
        assert data.name == "circuit_5"


class TestSerializeOpenQASM3:
    """Tests for OpenQASM serialization (3.0 with 2.0 fallback)."""

    def test_serialize_simple_circuit(self):
        """Serializes simple circuit to OpenQASM."""
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            [
                cirq.H(qubit),
                cirq.measure(qubit, key="m"),
            ]
        )

        data = serialize_openqasm3(circuit, name="simple")

        assert data.sdk == SDK.CIRQ
        # Format should be either QASM3 or QASM2 (fallback)
        assert data.format in (CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2)

        # Verify QASM header is present
        assert "OPENQASM" in data.data

    def test_detects_qasm3_vs_qasm2(self):
        """Distinguishes QASM 3.0 from 2.0 based on syntax.

        QASM 2.0 uses: qreg, creg, OPENQASM 2.0
        QASM 3.0 uses: qubit, bit, OPENQASM 3.0
        """
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            [
                cirq.H(qubit),
                cirq.measure(qubit, key="m"),
            ]
        )

        data = serialize_openqasm3(circuit)

        if data.format == CircuitFormat.OPENQASM3:
            # QASM 3.0 should have "3" in header or use 'qubit' keyword
            assert (
                "OPENQASM 3" in data.data or "qubit" in data.data.lower()
            ), "QASM3 format but no QASM3 markers found"
        else:
            # QASM 2.0 should have "2.0" in header or use 'qreg' keyword
            assert (
                "OPENQASM 2" in data.data or "qreg" in data.data.lower()
            ), "QASM2 format but no QASM2 markers found"

    def test_format_matches_content(self):
        """CircuitData.format accurately reflects actual QASM version."""
        qubit = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.X(qubit), cirq.measure(qubit, key="m"))

        data = serialize_openqasm3(circuit)

        # Check that format and content are consistent
        if "OPENQASM 3" in data.data:
            assert data.format == CircuitFormat.OPENQASM3
        elif "OPENQASM 2" in data.data:
            assert data.format == CircuitFormat.OPENQASM2

    def test_raises_for_unsupported_circuit(self):
        """Raises error for circuits without to_qasm support."""

        class FakeCircuit:
            pass

        with pytest.raises(SerializerError, match="does not support to_qasm"):
            serialize_openqasm3(FakeCircuit())


class TestCirqCircuitLoader:
    """Tests for circuit loading."""

    def test_loader_properties(self):
        """Loader has correct SDK and formats."""
        loader = CirqCircuitLoader()

        assert loader.sdk == SDK.CIRQ
        assert loader.name == "cirq"
        assert CircuitFormat.CIRQ_JSON in loader.supported_formats

    def test_json_roundtrip(self, bell_circuit):
        """Roundtrip through JSON preserves circuit structure."""
        data = serialize_json(bell_circuit, name="test")
        loader = CirqCircuitLoader()
        loaded = loader.load(data)

        assert loaded.sdk == SDK.CIRQ
        assert loaded.source_format == CircuitFormat.CIRQ_JSON
        assert loaded.name == "test"

        original_qubits = len(list(bell_circuit.all_qubits()))
        loaded_qubits = len(list(loaded.circuit.all_qubits()))
        assert loaded_qubits == original_qubits

    def test_openqasm_roundtrip(self):
        """Roundtrip through OpenQASM preserves circuit structure."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m")
        )
        data = serialize_openqasm3(circuit)

        loader = CirqCircuitLoader()
        loaded = loader.load(data)

        assert loaded.circuit is not None
        assert len(list(loaded.circuit.all_qubits())) == 2


class TestCirqCircuitSerializer:
    """Tests for circuit serializer protocol."""

    def test_serializer_properties(self):
        """Serializer has correct SDK and formats."""
        serializer = CirqCircuitSerializer()

        assert serializer.sdk == SDK.CIRQ
        assert serializer.name == "cirq"
        assert CircuitFormat.CIRQ_JSON in serializer.supported_formats
        assert CircuitFormat.OPENQASM3 in serializer.supported_formats

    def test_can_serialize(self, bell_circuit):
        """Recognizes Cirq circuits."""
        serializer = CirqCircuitSerializer()
        assert serializer.can_serialize(bell_circuit) is True
        assert serializer.can_serialize("not a circuit") is False

    def test_unsupported_format_raises(self, bell_circuit):
        """Raises error for unsupported formats."""
        serializer = CirqCircuitSerializer()

        with pytest.raises(SerializerError):
            serializer.serialize(bell_circuit, CircuitFormat.JAQCD)


class TestSummarizeCirqCircuit:
    """Tests for circuit summary generation."""

    def test_bell_circuit_summary(self, bell_circuit):
        """Summarizes Bell state circuit correctly."""
        summary = summarize_cirq_circuit(bell_circuit)

        assert summary.sdk == SDK.CIRQ
        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1  # H gate
        assert summary.gate_count_2q == 1  # CNOT gate
        assert summary.gate_count_measure == 1
        assert summary.depth > 0

    def test_clifford_detection(self, bell_circuit, non_clifford_circuit):
        """Detects Clifford vs non-Clifford circuits."""
        assert summarize_cirq_circuit(bell_circuit).is_clifford is True
        assert summarize_cirq_circuit(non_clifford_circuit).is_clifford is False

    def test_parameterized_and_gate_types(self, parameterized_circuit, bell_circuit):
        """Detects parameters and normalizes gate types."""
        assert summarize_cirq_circuit(parameterized_circuit).has_parameters is True

        summary = summarize_cirq_circuit(bell_circuit)
        assert "h" in summary.gate_types
        assert "cnot" in summary.gate_types

    def test_empty_circuit(self):
        """Handles empty circuit gracefully."""
        summary = summarize_cirq_circuit(cirq.Circuit())
        assert summary.num_qubits == 0
        assert summary.is_clifford is None
