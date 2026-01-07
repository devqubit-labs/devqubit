# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit circuit serialization."""

import io

import pytest
from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_qiskit.serialization import (
    LoadedCircuitBatch,
    QiskitCircuitLoader,
    QiskitCircuitSerializer,
    circuit_to_text,
    serialize_circuit_artifacts,
    serialize_circuits_text,
    serialize_qasm3,
    serialize_qpy,
    summarize_qiskit_circuit,
)
from qiskit import QuantumCircuit, qpy


class TestSerializeQpy:
    """Tests for QPY serialization."""

    def test_serialize_returns_valid_data(self, bell_circuit):
        """Serializes circuit to QPY with correct metadata."""
        data = serialize_qpy(bell_circuit, name="bell", index=0)

        assert data.format == CircuitFormat.QPY
        assert data.sdk == SDK.QISKIT
        assert data.name == "bell"
        assert data.index == 0
        assert isinstance(data.data, bytes)
        assert len(data.data) > 0

    def test_serialize_multiple_circuits(self, bell_circuit, ghz_circuit):
        """Serializes multiple circuits to single QPY blob."""
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch", index=0)

        # Verify both load back
        loaded = qpy.load(io.BytesIO(data.as_bytes()))
        assert len(loaded) == 2

    def test_roundtrip_preserves_structure(self, bell_circuit):
        """QPY roundtrip preserves circuit structure and gates."""
        data = serialize_qpy(bell_circuit)
        circuits = qpy.load(io.BytesIO(data.as_bytes()))
        loaded = circuits[0]

        assert loaded.num_qubits == bell_circuit.num_qubits
        assert loaded.num_clbits == bell_circuit.num_clbits
        assert len(loaded.data) == len(bell_circuit.data)

        # Gate names preserved
        original_gates = [instr.operation.name for instr in bell_circuit.data]
        loaded_gates = [instr.operation.name for instr in loaded.data]
        assert original_gates == loaded_gates

    def test_preserves_name_and_parameters(self, parameterized_circuit):
        """QPY preserves circuit name and unbound parameters."""
        qc = QuantumCircuit(2, name="my_special_circuit")
        qc.h(0)
        data = serialize_qpy(qc)
        loaded = qpy.load(io.BytesIO(data.as_bytes()))[0]
        assert loaded.name == "my_special_circuit"

        # Parameters
        data = serialize_qpy(parameterized_circuit)
        loaded = qpy.load(io.BytesIO(data.as_bytes()))[0]
        assert loaded.num_parameters == parameterized_circuit.num_parameters


class TestSerializeQasm3:
    """Tests for OpenQASM 3.0 serialization."""

    def test_serialize_single_circuit(self, bell_circuit):
        """Serializes circuit to QASM3 with correct format."""
        results = serialize_qasm3(bell_circuit)

        assert len(results) == 1
        data = results[0]
        assert data.format == CircuitFormat.OPENQASM3
        assert data.sdk == SDK.QISKIT
        assert "OPENQASM" in data.data
        assert "qubit" in data.data.lower()

    def test_serialize_multiple_circuits(self, bell_circuit, ghz_circuit):
        """Serializes multiple circuits to separate QASM3 strings."""
        results = serialize_qasm3([bell_circuit, ghz_circuit])

        assert len(results) == 2
        for i, data in enumerate(results):
            assert data.format == CircuitFormat.OPENQASM3
            assert data.index == i
            assert "OPENQASM" in data.data

    def test_contains_expected_gates(self, bell_circuit):
        """QASM3 output contains H and CX gates."""
        results = serialize_qasm3(bell_circuit)
        qasm = results[0].data.lower()

        assert "h " in qasm or "h(" in qasm or "h q" in qasm
        assert "cx " in qasm or "cx(" in qasm or "cnot" in qasm

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        assert serialize_qasm3([]) == []


class TestQiskitCircuitLoader:
    """Tests for circuit loading."""

    def test_loader_properties(self):
        """Loader has correct SDK and supported formats."""
        loader = QiskitCircuitLoader()

        assert loader.name == "qiskit"
        assert loader.sdk == SDK.QISKIT
        assert CircuitFormat.QPY in loader.supported_formats
        assert CircuitFormat.OPENQASM3 in loader.supported_formats
        assert CircuitFormat.OPENQASM2 in loader.supported_formats

    def test_load_qpy(self, bell_circuit):
        """Loads circuit from QPY format."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy(bell_circuit, name="test_bell")

        loaded = loader.load(data)

        assert loaded.sdk == SDK.QISKIT
        assert loaded.source_format == CircuitFormat.QPY
        assert loaded.circuit.num_qubits == bell_circuit.num_qubits

    def test_load_qasm3(self, bell_circuit):
        """Loads circuit from QASM3 format."""
        loader = QiskitCircuitLoader()
        results = serialize_qasm3(bell_circuit)

        loaded = loader.load(results[0])

        assert loaded.sdk == SDK.QISKIT
        assert loaded.source_format == CircuitFormat.OPENQASM3
        assert loaded.circuit.num_qubits == bell_circuit.num_qubits

    def test_load_qasm2(self):
        """Loads circuit from QASM2 format."""
        loader = QiskitCircuitLoader()
        qasm2 = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """

        data = CircuitData(
            data=qasm2,
            format=CircuitFormat.OPENQASM2,
            sdk=SDK.QISKIT,
            name="bell_qasm2",
            index=0,
        )

        loaded = loader.load(data)

        assert loaded.source_format == CircuitFormat.OPENQASM2
        assert loaded.circuit.num_qubits == 2

    def test_unsupported_format_raises(self):
        """Raises LoaderError for unsupported format."""
        loader = QiskitCircuitLoader()
        data = CircuitData(
            data="{}",
            format=CircuitFormat.JAQCD,
            sdk=SDK.QISKIT,
            name="test",
            index=0,
        )

        with pytest.raises(LoaderError):
            loader.load(data)


class TestQPYBatchLoading:
    """Tests for QPY batch loading - MAJOR from review."""

    def test_load_batch_returns_all_circuits(self, bell_circuit, ghz_circuit):
        """load_batch() returns all circuits from QPY file."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        batch = loader.load_batch(data)

        assert isinstance(batch, LoadedCircuitBatch)
        assert len(batch) == 2
        assert batch.count == 2

    def test_load_batch_preserves_circuit_structure(self, bell_circuit, ghz_circuit):
        """load_batch() preserves structure of all circuits."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        batch = loader.load_batch(data)

        # First circuit (bell) - 2 qubits
        assert batch[0].num_qubits == 2
        # Second circuit (ghz) - 3 qubits
        assert batch[1].num_qubits == 3

    def test_load_batch_to_loaded_circuits(self, bell_circuit, ghz_circuit):
        """to_loaded_circuits() returns list with proper indices."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        batch = loader.load_batch(data)
        loaded_circuits = batch.to_loaded_circuits()

        assert len(loaded_circuits) == 2
        assert loaded_circuits[0].index == 0
        assert loaded_circuits[1].index == 1
        assert loaded_circuits[0].sdk == SDK.QISKIT
        assert loaded_circuits[1].sdk == SDK.QISKIT

    def test_load_batch_iterable(self, bell_circuit, ghz_circuit):
        """LoadedCircuitBatch is iterable."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        batch = loader.load_batch(data)

        circuits = list(batch)
        assert len(circuits) == 2

    def test_load_batch_rejects_non_qpy(self, bell_circuit):
        """load_batch() rejects non-QPY formats."""
        loader = QiskitCircuitLoader()
        results = serialize_qasm3(bell_circuit)

        with pytest.raises(LoaderError, match="only supports QPY"):
            loader.load_batch(results[0])

    def test_load_single_warns_on_multi_circuit(
        self, bell_circuit, ghz_circuit, caplog
    ):
        """load() logs warning when QPY has multiple circuits."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        import logging

        with caplog.at_level(logging.WARNING):
            loaded = loader.load(data)

        # Should return first circuit
        assert loaded.circuit.num_qubits == 2

        # Should log warning about multiple circuits
        assert any(
            "multiple circuits" in msg.lower() or "load_batch" in msg
            for msg in caplog.messages
        )

    def test_load_batch_three_circuits(self, batch_circuits):
        """load_batch() handles 3+ circuits correctly."""
        loader = QiskitCircuitLoader()
        data = serialize_qpy(batch_circuits, name="batch")

        batch = loader.load_batch(data)

        assert len(batch) == 3
        assert batch[0].num_qubits == 2
        assert batch[1].num_qubits == 3
        assert batch[2].num_qubits == 4


class TestQiskitCircuitSerializer:
    """Tests for circuit serialization."""

    def test_serializer_properties(self):
        """Serializer has correct SDK and formats."""
        serializer = QiskitCircuitSerializer()

        assert serializer.name == "qiskit"
        assert serializer.sdk == SDK.QISKIT
        assert CircuitFormat.QPY in serializer.supported_formats
        assert CircuitFormat.OPENQASM3 in serializer.supported_formats

    def test_serialize_qpy_format(self, bell_circuit):
        """Serializes to QPY format."""
        serializer = QiskitCircuitSerializer()
        data = serializer.serialize(bell_circuit, CircuitFormat.QPY)

        assert data.format == CircuitFormat.QPY
        assert isinstance(data.data, bytes)

    def test_serialize_qasm3_format(self, bell_circuit):
        """Serializes to QASM3 format."""
        serializer = QiskitCircuitSerializer()
        data = serializer.serialize(bell_circuit, CircuitFormat.OPENQASM3)

        assert data.format == CircuitFormat.OPENQASM3
        assert "OPENQASM" in data.data

    def test_unsupported_format_raises(self, bell_circuit):
        """Raises SerializerError for unsupported format."""
        serializer = QiskitCircuitSerializer()

        with pytest.raises(SerializerError):
            serializer.serialize(bell_circuit, CircuitFormat.JAQCD)


class TestSummarizeQiskitCircuit:
    """Tests for circuit summarization."""

    def test_basic_summary(self, bell_circuit):
        """Summarizes basic circuit properties."""
        summary = summarize_qiskit_circuit(bell_circuit)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q >= 1  # H gate
        assert summary.gate_count_2q >= 1  # CX gate
        assert summary.depth == bell_circuit.depth()

    def test_gate_types_counted(self, bell_circuit):
        """Counts gate types correctly."""
        summary = summarize_qiskit_circuit(bell_circuit)

        assert "h" in summary.gate_types
        assert "cx" in summary.gate_types
        assert summary.gate_types["h"] == 1
        assert summary.gate_types["cx"] == 1

    def test_parameterized_circuit_detection(self, parameterized_circuit):
        """Detects parameterized circuits."""
        summary = summarize_qiskit_circuit(parameterized_circuit)

        assert summary.has_parameters is True
        assert summary.parameter_count == 2

    def test_empty_circuit(self):
        """Handles empty circuit."""
        qc = QuantumCircuit(2)
        summary = summarize_qiskit_circuit(qc)

        assert summary.num_qubits == 2
        assert summary.gate_count_total == 0
        assert summary.is_clifford is None

    def test_multi_qubit_gates(self):
        """Counts multi-qubit gates (Toffoli)."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.ccx(0, 1, 2)
        qc.measure_all()

        summary = summarize_qiskit_circuit(qc)

        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_multi == 1  # CCX

    def test_clifford_detection(self):
        """Detects Clifford vs non-Clifford circuits."""
        # All Clifford gates
        clifford_qc = QuantumCircuit(2)
        clifford_qc.h(0)
        clifford_qc.s(0)
        clifford_qc.x(0)
        clifford_qc.cx(0, 1)

        summary = summarize_qiskit_circuit(clifford_qc)
        assert summary.is_clifford is True

        # T gate breaks Clifford
        non_clifford_qc = QuantumCircuit(1)
        non_clifford_qc.h(0)
        non_clifford_qc.t(0)

        summary = summarize_qiskit_circuit(non_clifford_qc)
        assert summary.is_clifford is False


class TestCircuitToText:
    """Tests for human-readable circuit text."""

    def test_basic_format(self, bell_circuit):
        """Produces readable text format with key info."""
        text = circuit_to_text(bell_circuit, index=0)

        assert "=== Circuit 0 ===" in text
        assert "Qubits:" in text
        assert "Depth:" in text

    def test_multiple_circuits(self, bell_circuit, ghz_circuit):
        """Serializes multiple circuits to text."""
        text = serialize_circuits_text([bell_circuit, ghz_circuit])

        assert "=== Circuit 0 ===" in text
        assert "=== Circuit 1 ===" in text

    def test_none_returns_empty(self):
        """None input returns empty string."""
        assert serialize_circuits_text(None) == ""


class TestSerializeCircuitArtifacts:
    """Tests for multi-format artifact serialization."""

    def test_generates_all_formats(self, bell_circuit):
        """Generates QPY, OpenQASM, and diagram artifacts."""
        artifacts = serialize_circuit_artifacts(bell_circuit)

        assert "qpy" in artifacts
        assert "openqasm" in artifacts
        assert "diagram" in artifacts

    def test_qpy_artifact_valid(self, bell_circuit):
        """QPY artifact is valid and loadable."""
        artifacts = serialize_circuit_artifacts(bell_circuit)
        qpy_bytes, media_type = artifacts["qpy"]

        assert isinstance(qpy_bytes, bytes)
        assert "qiskit.qpy" in media_type

        # Verify loadable
        circuits = qpy.load(io.BytesIO(qpy_bytes))
        assert len(circuits) > 0

    def test_openqasm_artifact_valid(self, bell_circuit):
        """OpenQASM artifact is valid text."""
        artifacts = serialize_circuit_artifacts(bell_circuit)
        openqasm_bytes, media_type = artifacts["openqasm"]

        text = openqasm_bytes.decode("utf-8")
        assert "OPENQASM" in text
        assert "text/plain" in media_type


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_circuit_with_barriers(self):
        """Handles circuits with barriers."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.measure_all()

        summary = summarize_qiskit_circuit(qc)

        assert summary.gate_count_1q == 1
        assert summary.gate_count_2q == 1
        assert "barrier" in summary.gate_types

    def test_large_circuit(self):
        """Handles larger circuits efficiently."""
        qc = QuantumCircuit(10)
        for i in range(10):
            qc.h(i)
        for i in range(9):
            qc.cx(i, i + 1)
        qc.measure_all()

        summary = summarize_qiskit_circuit(qc)

        assert summary.num_qubits == 10
        assert summary.gate_count_1q == 10
        assert summary.gate_count_2q == 9

    def test_deep_circuit(self):
        """Handles deep circuits."""
        qc = QuantumCircuit(2)
        for _ in range(100):
            qc.h(0)
            qc.cx(0, 1)

        summary = summarize_qiskit_circuit(qc)

        assert summary.depth == qc.depth()
        assert summary.gate_count_1q == 100
        assert summary.gate_count_2q == 100
