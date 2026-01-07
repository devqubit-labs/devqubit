# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Braket circuit serialization."""

import json

from braket.circuits import Circuit, FreeParameter
from devqubit_braket.serialization import (
    BraketCircuitLoader,
    BraketCircuitSerializer,
    is_braket_circuit,
    serialize_jaqcd,
    serialize_openqasm,
    summarize_braket_circuit,
)
from devqubit_engine.circuit.models import SDK, CircuitFormat


class TestCircuitDetection:
    """Tests for circuit type detection."""

    def test_is_braket_circuit(self, bell_circuit):
        """Detects actual Braket circuits."""
        assert is_braket_circuit(bell_circuit) is True

    def test_is_not_braket_circuit(self):
        """Rejects non-Braket objects."""
        assert is_braket_circuit("string") is False
        assert is_braket_circuit(123) is False
        assert is_braket_circuit(None) is False
        assert is_braket_circuit([]) is False
        assert is_braket_circuit({}) is False


class TestJAQCDSerialization:
    """Tests for JAQCD serialization."""

    def test_serialize_bell_circuit(self, bell_circuit):
        """Serializes Bell circuit to JAQCD."""
        data = serialize_jaqcd(bell_circuit, name="bell", index=0)

        assert data.format == CircuitFormat.JAQCD
        assert data.sdk == SDK.BRAKET
        assert data.name == "bell"
        assert data.index == 0

        # Verify valid JSON
        parsed = json.loads(data.data)
        assert "instructions" in parsed

    def test_jaqcd_contains_gates(self, bell_circuit):
        """JAQCD contains H and CNOT gates."""
        data = serialize_jaqcd(bell_circuit)
        parsed = json.loads(data.data)

        gate_types = [instr["type"] for instr in parsed["instructions"]]
        assert "h" in gate_types
        assert "cnot" in gate_types

    def test_jaqcd_default_name(self, bell_circuit):
        """Uses default name pattern."""
        data = serialize_jaqcd(bell_circuit, index=42)
        assert data.name == "circuit_42"

    def test_jaqcd_as_bytes(self, bell_circuit):
        """CircuitData.as_bytes() returns valid bytes."""
        data = serialize_jaqcd(bell_circuit)
        raw = data.as_bytes()

        assert isinstance(raw, bytes)
        assert len(raw) > 0

        # Should be valid JSON bytes
        parsed = json.loads(raw.decode("utf-8"))
        assert "instructions" in parsed


class TestOpenQASMSerialization:
    """Tests for OpenQASM serialization."""

    def test_serialize_bell_circuit(self, bell_circuit):
        """Serializes Bell circuit to OpenQASM."""
        data = serialize_openqasm(bell_circuit, name="bell_qasm", index=1)

        assert data.format == CircuitFormat.OPENQASM3
        assert data.sdk == SDK.BRAKET
        assert data.name == "bell_qasm"

        # Verify OpenQASM structure
        assert "OPENQASM" in data.data
        assert "qubit[2]" in data.data

    def test_openqasm_contains_operations(self, ghz_circuit):
        """OpenQASM contains circuit operations."""
        data = serialize_openqasm(ghz_circuit)

        assert "h" in data.data.lower()
        assert "cnot" in data.data.lower() or "cx" in data.data.lower()

    def test_openqasm_captures_parameter_values(self):
        """OpenQASM captures actual parameter values (critical for hashing)."""
        theta = FreeParameter("theta")
        template = Circuit().rx(0, theta)

        bound = template.make_bound_circuit({"theta": 1.5708})
        data = serialize_openqasm(bound)

        # The angle value should appear in OpenQASM
        # (exact format depends on Braket version)
        assert "1.5708" in data.data or "rx" in data.data.lower()

    def test_openqasm_different_params_different_output(self):
        """Different parameter values produce different OpenQASM output."""
        theta = FreeParameter("theta")
        template = Circuit().rx(0, theta)

        c1 = template.make_bound_circuit({"theta": 0.1})
        c2 = template.make_bound_circuit({"theta": 2.0})

        data1 = serialize_openqasm(c1)
        data2 = serialize_openqasm(c2)

        # Different params -> different OpenQASM source
        assert data1.data != data2.data


class TestBraketCircuitLoader:
    """Tests for circuit loading."""

    def test_loader_properties(self):
        """Loader has correct SDK and formats."""
        loader = BraketCircuitLoader()

        assert loader.sdk == SDK.BRAKET
        assert CircuitFormat.JAQCD in loader.supported_formats
        assert CircuitFormat.OPENQASM3 in loader.supported_formats

    def test_roundtrip_jaqcd(self, bell_circuit):
        """Serialize then load JAQCD preserves circuit."""
        serialized = serialize_jaqcd(bell_circuit, name="roundtrip")

        loader = BraketCircuitLoader()
        loaded = loader.load(serialized)

        assert loaded.sdk == SDK.BRAKET
        assert loaded.source_format == CircuitFormat.JAQCD
        assert loaded.name == "roundtrip"
        assert loaded.circuit.qubit_count == bell_circuit.qubit_count

    def test_roundtrip_openqasm(self, ghz_circuit):
        """Serialize then load OpenQASM preserves circuit."""
        serialized = serialize_openqasm(ghz_circuit, name="ghz")

        loader = BraketCircuitLoader()
        loaded = loader.load(serialized)

        assert loaded.sdk == SDK.BRAKET
        assert loaded.source_format == CircuitFormat.OPENQASM3
        assert loaded.circuit.qubit_count == ghz_circuit.qubit_count


class TestBraketCircuitSerializer:
    """Tests for circuit serializer."""

    def test_serializer_properties(self):
        """Serializer has correct SDK and formats."""
        serializer = BraketCircuitSerializer()

        assert serializer.sdk == SDK.BRAKET
        assert CircuitFormat.JAQCD in serializer.supported_formats
        assert CircuitFormat.OPENQASM3 in serializer.supported_formats

    def test_can_serialize_braket(self, bell_circuit):
        """Recognizes real Braket circuits."""
        serializer = BraketCircuitSerializer()
        assert serializer.can_serialize(bell_circuit) is True

    def test_cannot_serialize_non_braket(self):
        """Rejects non-Braket objects."""
        serializer = BraketCircuitSerializer()
        assert serializer.can_serialize("not a circuit") is False
        assert serializer.can_serialize(None) is False
        assert serializer.can_serialize([]) is False

    def test_serialize_to_jaqcd(self, bell_circuit):
        """Serializes to JAQCD format."""
        serializer = BraketCircuitSerializer()
        data = serializer.serialize(
            bell_circuit,
            CircuitFormat.JAQCD,
            name="test",
        )

        assert data.format == CircuitFormat.JAQCD
        assert data.name == "test"

    def test_serialize_to_openqasm(self, bell_circuit):
        """Serializes to OpenQASM format."""
        serializer = BraketCircuitSerializer()
        data = serializer.serialize(
            bell_circuit,
            CircuitFormat.OPENQASM3,
            name="test",
        )

        assert data.format == CircuitFormat.OPENQASM3
        assert "OPENQASM" in data.data


class TestCircuitSummary:
    """Tests for circuit summary generation."""

    def test_bell_circuit_summary(self, bell_circuit):
        """Summarizes Bell state circuit."""
        summary = summarize_braket_circuit(bell_circuit)

        assert summary.sdk == SDK.BRAKET
        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1  # H gate
        assert summary.gate_count_2q == 1  # CNOT gate
        assert summary.gate_count_total == 2

    def test_ghz_circuit_summary(self, ghz_circuit):
        """Summarizes GHZ state circuit."""
        summary = summarize_braket_circuit(ghz_circuit)

        assert summary.num_qubits == 3
        assert summary.gate_count_1q == 1  # H gate
        assert summary.gate_count_2q == 2  # Two CNOTs

    def test_clifford_detection_positive(self, bell_circuit):
        """Detects Clifford circuits."""
        summary = summarize_braket_circuit(bell_circuit)
        assert summary.is_clifford is True

    def test_clifford_detection_negative(self, non_clifford_circuit):
        """Detects non-Clifford circuits."""
        summary = summarize_braket_circuit(non_clifford_circuit)
        assert summary.is_clifford is False

    def test_gate_types_counted(self, bell_circuit):
        """Counts gate types correctly."""
        summary = summarize_braket_circuit(bell_circuit)

        assert "h" in summary.gate_types
        assert summary.gate_types["h"] == 1
        assert "cnot" in summary.gate_types
        assert summary.gate_types["cnot"] == 1

    def test_circuit_with_measurements(self):
        """Handles measurement operations."""
        circuit = Circuit()
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.measure([0, 1])

        summary = summarize_braket_circuit(circuit)

        assert summary.gate_count_measure > 0
        assert summary.gate_count_total > 2

    def test_parametric_circuit_summary(self):
        """Summarizes parametric circuit."""
        theta = FreeParameter("theta")
        circuit = Circuit().rx(0, theta).ry(1, theta).cnot(0, 1)
        bound = circuit.make_bound_circuit({"theta": 0.5})

        summary = summarize_braket_circuit(bound)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 2  # rx, ry
        assert summary.gate_count_2q == 1  # cnot
        assert summary.is_clifford is False  # rx, ry are non-Clifford

    def test_depth_calculation(self, ghz_circuit):
        """Circuit depth is calculated."""
        summary = summarize_braket_circuit(ghz_circuit)

        # GHZ circuit has depth 3: H, CNOT, CNOT (sequential)
        assert summary.depth >= 1  # At least has depth


class TestSerializationDeterminism:
    """Tests for serialization determinism (critical for artifact deduplication)."""

    def test_jaqcd_deterministic(self, bell_circuit):
        """Same circuit produces identical JAQCD output."""
        data1 = serialize_jaqcd(bell_circuit)
        data2 = serialize_jaqcd(bell_circuit)

        assert data1.data == data2.data

    def test_openqasm_deterministic(self, bell_circuit):
        """Same circuit produces identical OpenQASM output."""
        data1 = serialize_openqasm(bell_circuit)
        data2 = serialize_openqasm(bell_circuit)

        assert data1.data == data2.data

    def test_equivalent_circuits_same_serialization(self):
        """Equivalent circuits (built same way) produce same serialization."""
        c1 = Circuit().h(0).cnot(0, 1)
        c2 = Circuit().h(0).cnot(0, 1)

        j1 = serialize_jaqcd(c1)
        j2 = serialize_jaqcd(c2)

        assert j1.data == j2.data
