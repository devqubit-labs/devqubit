# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for circuit module.

Tests circuit data models, SDK detection, loaders, and summarization.
SDK-specific tests are skipped when the corresponding SDK is not installed.
"""

from __future__ import annotations

import io

import pytest
from devqubit_engine.circuit.extractors import detect_sdk
from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import (
    LoaderError,
    get_loader,
    list_available,
)
from devqubit_engine.circuit.summary import (
    CircuitSummary,
    summarize_circuit_data,
)
from devqubit_engine.storage.types import ArtifactRef


def sdk_available(sdk_name: str) -> bool:
    """Check if SDK loader is available."""
    available = list_available()
    return sdk_name in available.get("loaders", [])


requires_qiskit = pytest.mark.skipif(
    not sdk_available("qiskit"),
    reason="Qiskit not installed",
)

requires_braket = pytest.mark.skipif(
    not sdk_available("braket"),
    reason="Braket not installed",
)

requires_cirq = pytest.mark.skipif(
    not sdk_available("cirq"),
    reason="Cirq not installed",
)


class TestCircuitData:
    """Tests for CircuitData dataclass."""

    def test_circuit_data_binary(self, circuit_data_factory):
        """CircuitData stores binary data correctly."""
        data = circuit_data_factory(
            data=b"binary data",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        assert data.as_bytes() == b"binary data"
        assert data.format == CircuitFormat.QPY
        assert data.sdk == SDK.QISKIT

    def test_circuit_data_text(self, circuit_data_factory):
        """CircuitData stores text data correctly."""
        data = circuit_data_factory(
            data="OPENQASM 3.0;",
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.QISKIT,
        )

        assert data.as_text() == "OPENQASM 3.0;"
        assert data.format == CircuitFormat.OPENQASM3

    def test_circuit_data_with_metadata(self):
        """CircuitData can include metadata."""
        data = CircuitData(
            data=b"test",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
            name="my_circuit",
            index=0,
            metadata={"version": "1.0"},
        )

        assert data.name == "my_circuit"
        assert data.index == 0
        assert data.metadata == {"version": "1.0"}

    def test_as_bytes_from_text(self, circuit_data_factory):
        """as_bytes() converts text to bytes."""
        data = circuit_data_factory(
            data="hello",
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.QISKIT,
        )

        assert data.as_bytes() == b"hello"

    def test_as_text_from_bytes(self, circuit_data_factory):
        """as_text() converts bytes to text."""
        data = circuit_data_factory(
            data=b"hello",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        assert data.as_text() == "hello"

    def test_as_text_binary_raises(self, circuit_data_factory):
        """as_text() raises for non-UTF8 binary."""
        data = circuit_data_factory(
            data=b"\xff\xfe",  # Invalid UTF-8
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        with pytest.raises(UnicodeDecodeError):
            data.as_text()


class TestLoadedCircuit:
    """Tests for LoadedCircuit dataclass."""

    def test_loaded_circuit_fields(self):
        """LoadedCircuit has expected fields."""
        loaded = LoadedCircuit(
            circuit="mock_circuit_object",
            sdk=SDK.QISKIT,
            source_format=CircuitFormat.QPY,
            name="test_circuit",
            index=0,
        )

        assert loaded.circuit == "mock_circuit_object"
        assert loaded.sdk == SDK.QISKIT
        assert loaded.source_format == CircuitFormat.QPY
        assert loaded.name == "test_circuit"
        assert loaded.index == 0


class TestCircuitSummary:
    """Tests for CircuitSummary dataclass."""

    def test_summary_to_dict(self):
        """CircuitSummary serializes to dict."""
        summary = CircuitSummary(
            num_qubits=2,
            depth=3,
            gate_count_1q=2,
            gate_count_2q=1,
            gate_types={"h": 1, "cx": 1},
        )

        d = summary.to_dict()

        assert d["num_qubits"] == 2
        assert d["depth"] == 3
        assert d["gate_types"] == {"h": 1, "cx": 1}

    def test_summary_gate_count_total(self):
        """gate_count_total is computed correctly."""
        summary = CircuitSummary(
            gate_count_1q=5,
            gate_count_2q=3,
            gate_count_total=8,
        )

        assert summary.gate_count_total == 8


class TestDetectSDK:
    """Tests for SDK detection from run records."""

    def test_detect_qiskit_from_adapter(self, run_factory):
        """Detect Qiskit from adapter name."""
        record = run_factory(adapter="devqubit-qiskit")
        assert detect_sdk(record) == SDK.QISKIT

    def test_detect_braket_from_adapter(self, run_factory):
        """Detect Braket from adapter name."""
        record = run_factory(adapter="devqubit-braket")
        assert detect_sdk(record) == SDK.BRAKET

    def test_detect_cirq_from_adapter(self, run_factory):
        """Detect Cirq from adapter name."""
        record = run_factory(adapter="devqubit-cirq")
        assert detect_sdk(record) == SDK.CIRQ

    def test_detect_pennylane_from_adapter(self, run_factory):
        """Detect PennyLane from adapter name."""
        record = run_factory(adapter="devqubit-pennylane")
        assert detect_sdk(record) == SDK.PENNYLANE

    def test_detect_from_qiskit_artifact(self, run_factory):
        """Detect Qiskit from QPY artifact kind."""
        valid_digest = "sha256:" + "a" * 64
        artifact = ArtifactRef(
            role="program",
            kind="qiskit.qpy.circuits",
            digest=valid_digest,
            media_type="application/octet-stream",
        )
        record = run_factory(adapter="unknown", artifacts=[artifact])
        assert detect_sdk(record) == SDK.QISKIT

    def test_detect_from_braket_artifact(self, run_factory):
        """Detect Braket from JAQCD artifact kind."""
        valid_digest = "sha256:" + "a" * 64
        artifact = ArtifactRef(
            role="program",
            kind="braket.ir.jaqcd",
            digest=valid_digest,
            media_type="application/json",
        )
        record = run_factory(adapter="unknown", artifacts=[artifact])
        assert detect_sdk(record) == SDK.BRAKET

    def test_detect_from_cirq_artifact(self, run_factory):
        """Detect Cirq from circuit.json artifact kind."""
        valid_digest = "sha256:" + "a" * 64
        artifact = ArtifactRef(
            role="program",
            kind="cirq.circuit.json",
            digest=valid_digest,
            media_type="application/json",
        )
        record = run_factory(adapter="unknown", artifacts=[artifact])
        assert detect_sdk(record) == SDK.CIRQ

    def test_detect_unknown_returns_unknown(self, run_factory):
        """Unknown adapter returns SDK.UNKNOWN."""
        record = run_factory(adapter="some-unknown-adapter")
        assert detect_sdk(record) == SDK.UNKNOWN

    def test_detect_case_insensitive(self, run_factory):
        """SDK detection is case-insensitive."""
        record = run_factory(adapter="devqubit-Qiskit")
        assert detect_sdk(record) == SDK.QISKIT


class TestQiskitLoader:
    """Tests for Qiskit circuit loader."""

    @requires_qiskit
    def test_qiskit_loader_qpy(self):
        """QiskitLoader loads QPY format."""
        from qiskit import QuantumCircuit, qpy

        qc = QuantumCircuit(2, 2, name="test")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        buffer = io.BytesIO()
        qpy.dump(qc, buffer)

        data = CircuitData(
            data=buffer.getvalue(),
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        loader = get_loader(SDK.QISKIT)
        loaded = loader.load(data)

        assert loaded.sdk == SDK.QISKIT
        assert loaded.source_format == CircuitFormat.QPY
        assert loaded.circuit.num_qubits == 2

    @requires_qiskit
    def test_qiskit_loader_qasm2(self, bell_qasm2):
        """QiskitLoader loads QASM2 format."""
        data = CircuitData(
            data=bell_qasm2,
            format=CircuitFormat.OPENQASM2,
            sdk=SDK.QISKIT,
        )

        loader = get_loader(SDK.QISKIT)
        loaded = loader.load(data)

        assert loaded.sdk == SDK.QISKIT
        assert loaded.circuit.num_qubits == 2

    @requires_qiskit
    def test_qiskit_loader_preserves_name(self):
        """QiskitLoader preserves circuit name."""
        from qiskit import QuantumCircuit, qpy

        qc = QuantumCircuit(2, name="my_named_circuit")
        qc.h(0)

        buffer = io.BytesIO()
        qpy.dump(qc, buffer)

        data = CircuitData(
            data=buffer.getvalue(),
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        loader = get_loader(SDK.QISKIT)
        loaded = loader.load(data)

        assert loaded.circuit.name == "my_named_circuit"


class TestBraketLoader:
    """Tests for Braket circuit loader."""

    @requires_braket
    def test_braket_loader_jaqcd(self):
        """BraketLoader loads JAQCD format."""
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1)

        try:
            from braket.circuits.serialization import IRType

            ir_program = circuit.to_ir(ir_type=IRType.JAQCD)
        except ImportError:
            ir_program = circuit.to_ir()

        data = CircuitData(
            data=ir_program.json(),
            format=CircuitFormat.JAQCD,
            sdk=SDK.BRAKET,
        )

        loader = get_loader(SDK.BRAKET)
        loaded = loader.load(data)

        assert loaded.sdk == SDK.BRAKET
        assert loaded.source_format == CircuitFormat.JAQCD


class TestCirqLoader:
    """Tests for Cirq circuit loader."""

    @requires_cirq
    def test_cirq_loader_json(self):
        """CirqLoader loads Cirq JSON format."""
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.measure(q0, q1, key="m"),
            ]
        )

        data = CircuitData(
            data=cirq.to_json(circuit),
            format=CircuitFormat.CIRQ_JSON,
            sdk=SDK.CIRQ,
        )

        loader = get_loader(SDK.CIRQ)
        loaded = loader.load(data)

        assert loaded.sdk == SDK.CIRQ
        assert loaded.source_format == CircuitFormat.CIRQ_JSON


class TestLoaderErrors:
    """Tests for loader error handling."""

    def test_get_loader_unknown_raises(self):
        """get_loader raises for unknown SDK."""
        with pytest.raises(LoaderError):
            get_loader(SDK.UNKNOWN)

    @requires_qiskit
    def test_qiskit_loader_invalid_qpy_raises(self):
        """QiskitLoader raises for invalid QPY data."""
        data = CircuitData(
            data=b"not valid qpy data",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        loader = get_loader(SDK.QISKIT)
        with pytest.raises(Exception):  # Could be various exceptions
            loader.load(data)


class TestSummarizeCircuitData:
    """Tests for circuit data summarization."""

    @requires_qiskit
    def test_summarize_qasm2_data(self, bell_qasm2):
        """Summarize CircuitData with OpenQASM2 via loader."""
        data = CircuitData(
            data=bell_qasm2,
            format=CircuitFormat.OPENQASM2,
            sdk=SDK.QISKIT,
        )

        summary = summarize_circuit_data(data)

        assert summary is not None
        assert summary.num_qubits == 2

    @requires_qiskit
    def test_summarize_qpy_data(self):
        """Summarize CircuitData with QPY (requires loader)."""
        from qiskit import QuantumCircuit, qpy

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        buffer = io.BytesIO()
        qpy.dump(qc, buffer)

        data = CircuitData(
            data=buffer.getvalue(),
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        summary = summarize_circuit_data(data)

        assert summary is not None
        assert summary.num_qubits == 3
        assert summary.gate_count_1q == 1
        assert summary.gate_count_2q == 2

    @requires_braket
    def test_summarize_braket_jaqcd_data(self):
        """Summarize CircuitData with JAQCD."""
        from braket.circuits import Circuit
        from braket.circuits.serialization import IRType

        circuit = Circuit().h(0).cnot(0, 1).h(1)
        ir_program = circuit.to_ir(ir_type=IRType.JAQCD)

        data = CircuitData(
            data=ir_program.json(),
            format=CircuitFormat.JAQCD,
            sdk=SDK.BRAKET,
        )

        summary = summarize_circuit_data(data)

        assert summary is not None
        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 2
        assert summary.gate_count_2q == 1

    @requires_cirq
    def test_summarize_cirq_json_data(self):
        """Summarize CircuitData with Cirq JSON."""
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

        data = CircuitData(
            data=cirq.to_json(circuit),
            format=CircuitFormat.CIRQ_JSON,
            sdk=SDK.CIRQ,
        )

        summary = summarize_circuit_data(data)

        assert summary is not None
        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1
        assert summary.gate_count_2q == 1

    @requires_qiskit
    def test_summarize_ghz_circuit(self, ghz_qasm2):
        """Summarize 3-qubit GHZ circuit."""
        data = CircuitData(
            data=ghz_qasm2,
            format=CircuitFormat.OPENQASM2,
            sdk=SDK.QISKIT,
        )

        summary = summarize_circuit_data(data)

        assert summary.num_qubits == 3
        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_2q == 2  # 2x CX
