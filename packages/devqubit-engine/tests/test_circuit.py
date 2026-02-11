# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit module."""

from __future__ import annotations

import io

import pytest
from devqubit_engine.circuit.extractors import (
    extract_circuit,
    extract_circuit_from_refs,
)
from devqubit_engine.circuit.hashing import (
    hash_parametric,
    hash_structural,
)
from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.circuit.registry import LoaderError, get_loader, list_available
from devqubit_engine.circuit.summary import (
    CircuitSummary,
    diff_summaries,
    summarize_circuit_data,
)


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

requires_pennylane = pytest.mark.skipif(
    not sdk_available("pennylane"),
    reason="Pennylane not installed",
)


class TestCircuitPipeline:
    """End-to-end tests for circuit extraction and summarization."""

    @requires_qiskit
    def test_qiskit_qpy_full_pipeline(self, run_factory, store, artifact_factory):
        """Full pipeline: QPY artifact => extract => load => summarize."""
        from qiskit import QuantumCircuit, qpy

        # Create a realistic circuit
        qc = QuantumCircuit(3, 3, name="ghz_state")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        # Serialize to QPY
        buffer = io.BytesIO()
        qpy.dump(qc, buffer)
        qpy_bytes = buffer.getvalue()

        # Create artifact and run record
        artifact = artifact_factory(
            data=qpy_bytes,
            kind="qiskit.qpy.circuits",
            role="program",
            media_type="application/x-qpy",
        )
        record = run_factory(adapter="devqubit-qiskit", artifacts=[artifact])

        # Extract circuit from run
        circuit_data = extract_circuit(record, store)

        assert circuit_data is not None
        assert circuit_data.format == CircuitFormat.QPY
        assert circuit_data.sdk == SDK.QISKIT

        # Summarize
        summary = summarize_circuit_data(circuit_data)

        assert summary.num_qubits == 3
        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_2q == 2  # 2x CX
        assert summary.gate_count_measure == 3

    @requires_qiskit
    def test_qasm2_full_pipeline(
        self,
        run_factory,
        store,
        artifact_factory,
        bell_qasm2,
    ):
        """Full pipeline: OpenQASM2 artifact => extract => load => summarize."""
        artifact = artifact_factory(
            data=bell_qasm2.encode("utf-8"),
            kind="source.openqasm2",
            role="program",
            media_type="text/plain",
        )
        record = run_factory(adapter="devqubit-qiskit", artifacts=[artifact])

        circuit_data = extract_circuit(record, store)

        assert circuit_data is not None
        assert circuit_data.format == CircuitFormat.OPENQASM2

        summary = summarize_circuit_data(circuit_data)

        assert summary.num_qubits == 2
        assert summary.depth > 0

    @requires_braket
    def test_braket_jaqcd_full_pipeline(self, run_factory, store, artifact_factory):
        """Full pipeline: JAQCD artifact => extract => load => summarize."""
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1).h(1)

        try:
            from braket.circuits.serialization import IRType

            ir_program = circuit.to_ir(ir_type=IRType.JAQCD)
        except ImportError:
            ir_program = circuit.to_ir()

        artifact = artifact_factory(
            data=ir_program.json().encode("utf-8"),
            kind="braket.jaqcd",
            role="program",
            media_type="application/json",
        )
        record = run_factory(adapter="devqubit-braket", artifacts=[artifact])

        circuit_data = extract_circuit(record, store)

        assert circuit_data is not None
        assert circuit_data.format == CircuitFormat.JAQCD
        assert circuit_data.sdk == SDK.BRAKET

        summary = summarize_circuit_data(circuit_data)

        assert summary.num_qubits == 2
        assert summary.gate_count_2q == 1  # CNOT

    @requires_cirq
    def test_cirq_json_full_pipeline(self, run_factory, store, artifact_factory):
        """Full pipeline: Cirq JSON artifact => extract => load => summarize."""
        import cirq

        # Create a Bell state circuit
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.measure(q0, q1, key="result"),
            ]
        )

        # Serialize to JSON
        circuit_json = cirq.to_json(circuit)

        artifact = artifact_factory(
            data=circuit_json.encode("utf-8"),
            kind="cirq.json",
            role="program",
            media_type="application/json",
        )
        record = run_factory(adapter="devqubit-cirq", artifacts=[artifact])

        circuit_data = extract_circuit(record, store)

        assert circuit_data is not None
        assert circuit_data.format == CircuitFormat.CIRQ_JSON
        assert circuit_data.sdk == SDK.CIRQ

        summary = summarize_circuit_data(circuit_data)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_2q == 1  # CNOT
        assert summary.gate_count_measure == 1

    @requires_pennylane
    def test_pennylane_json_full_pipeline(self, run_factory, store, artifact_factory):
        """Full pipeline: PennyLane JSON artifact => extract => load => summarize."""
        import json

        import pennylane as qml

        # Create a simple circuit and serialize tape structure
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # Execute to build tape
        circuit()

        # Serialize tape to JSON (adapter-specific format)
        tape_json = json.dumps(
            {
                "operations": [
                    {"name": "Hadamard", "wires": [0], "params": []},
                    {"name": "CNOT", "wires": [0, 1], "params": []},
                ],
                "observables": [
                    {"name": "PauliZ", "wires": [0]},
                ],
                "num_wires": 2,
            }
        )

        artifact = artifact_factory(
            data=tape_json.encode("utf-8"),
            kind="pennylane.tape.json",
            role="program",
            media_type="application/json",
        )
        record = run_factory(adapter="devqubit-pennylane", artifacts=[artifact])

        circuit_data = extract_circuit(record, store)

        assert circuit_data is not None
        assert circuit_data.sdk == SDK.PENNYLANE

        summary = summarize_circuit_data(circuit_data)

        assert summary.num_qubits == 2
        assert summary.gate_count_1q >= 1  # H
        assert summary.gate_count_2q >= 1  # CNOT


class TestCircuitHashing:
    """Test circuit structural and parametric hashing."""

    def test_empty_ops_hash(self):
        """Empty circuit produces deterministic hash."""
        h = hash_structural([], 0, 0)
        assert h.startswith("sha256:")
        assert len(h) == 71  # sha256: + 64 hex chars

    def test_op_order_matters(self):
        """Different operation order => different hash."""
        ops1 = [
            {"gate": "h", "qubits": [0]},
            {"gate": "x", "qubits": [1]},
        ]
        ops2 = [
            {"gate": "x", "qubits": [1]},
            {"gate": "h", "qubits": [0]},
        ]
        assert hash_structural(ops1, 2, 0) != hash_structural(ops2, 2, 0)

    def test_num_qubits_matters(self):
        """Same ops, different num_qubits => different hash."""
        ops = [{"gate": "h", "qubits": [0]}]
        h2 = hash_structural(ops, 2, 0)
        h3 = hash_structural(ops, 3, 0)
        assert h2 != h3

    def test_num_clbits_matters(self):
        """Same ops, different num_clbits => different hash."""
        ops = [{"gate": "measure", "qubits": [0], "clbits": [0]}]
        h1 = hash_structural(ops, 1, 1)
        h2 = hash_structural(ops, 1, 2)
        assert h1 != h2

    def test_condition_details_matter(self):
        """Different condition values => different hash."""
        ops1 = [
            {
                "gate": "x",
                "qubits": [0],
                "condition": {"register": "c", "value": 1},
            }
        ]
        ops2 = [
            {
                "gate": "x",
                "qubits": [0],
                "condition": {"register": "c", "value": 2},
            }
        ]
        assert hash_structural(ops1, 1, 1) != hash_structural(ops2, 1, 1)

    def test_param_values_affect_parametric(self):
        """Different param values => different parametric hash."""
        h1 = hash_parametric(
            [{"gate": "rz", "qubits": [0], "params": {"t": 0.5}}],
            1,
            0,
        )
        h2 = hash_parametric(
            [{"gate": "rz", "qubits": [0], "params": {"t": 1.5}}],
            1,
            0,
        )
        assert h1 != h2

    def test_param_values_ignored_structural(self):
        """Param values ignored in structural hash."""
        h1 = hash_structural(
            [{"gate": "rz", "qubits": [0], "params": {"t": 0.5}}],
            1,
            0,
        )
        h2 = hash_structural(
            [{"gate": "rz", "qubits": [0], "params": {"t": 1.5}}],
            1,
            0,
        )
        assert h1 == h2

    def test_hash_format(self):
        """Hash format is sha256:<64 hex>."""
        h = hash_structural([{"gate": "h", "qubits": [0]}], 1, 0)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64

    def test_deterministic(self):
        """Same input => same hash."""
        ops = [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]
        h1 = hash_structural(ops, 2, 0)
        h2 = hash_structural(ops, 2, 0)
        assert h1 == h2

    def test_large_int_preserves_precision(self):
        """Large integers (>2^53) hash deterministically without precision loss."""
        large_int = 2**63
        ops = [{"gate": "custom", "qubits": [0], "params": {"n": large_int}}]

        h1 = hash_parametric(ops, 1, 0)
        h2 = hash_parametric(ops, 1, 0)

        # Same large int => same hash
        assert h1 == h2

        # Different large ints => different hashes
        ops_different = [{"gate": "custom", "qubits": [0], "params": {"n": 2**63 + 1}}]
        h3 = hash_parametric(ops_different, 1, 0)
        assert h1 != h3

    def test_int_float_distinguished(self):
        """Integer 1 and float 1.0 produce different hashes."""
        ops_int = [{"gate": "rz", "qubits": [0], "params": {"t": 1}}]
        ops_float = [{"gate": "rz", "qubits": [0], "params": {"t": 1.0}}]

        h_int = hash_parametric(ops_int, 1, 0)
        h_float = hash_parametric(ops_float, 1, 0)

        assert h_int != h_float


class TestExtractFromRefs:
    """Test circuit extraction from artifact refs (UEC envelope flow)."""

    def test_extract_from_refs_prefers_qasm3(self, store):
        """When multiple formats available, prefer OpenQASM3."""
        qasm3_content = "OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];"
        qasm2_content = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"

        from devqubit_engine.storage.types import ArtifactRef

        digest_qasm3 = store.put_bytes(qasm3_content.encode())
        digest_qasm2 = store.put_bytes(qasm2_content.encode())

        refs = [
            ArtifactRef(
                kind="source.openqasm2",
                digest=digest_qasm2,
                media_type="text/plain",
                role="program",
            ),
            ArtifactRef(
                kind="source.openqasm3",
                digest=digest_qasm3,
                media_type="text/plain",
                role="program",
            ),
        ]

        result = extract_circuit_from_refs(refs, store)

        assert result is not None
        assert result.format == CircuitFormat.OPENQASM3

    def test_extract_from_refs_empty_returns_none(self, store):
        """Empty refs list returns None."""
        result = extract_circuit_from_refs([], store)
        assert result is None


class TestCircuitDiff:
    """Test circuit summary comparison for drift detection."""

    def test_diff_identical_summaries(self):
        """Identical summaries should match."""
        summary = CircuitSummary(
            num_qubits=2,
            depth=3,
            gate_count_1q=1,
            gate_count_2q=1,
            gate_count_total=2,
        )

        diff = diff_summaries(summary, summary)

        assert diff.match is True
        assert len(diff.changed) == 0
        assert len(diff.added_gates) == 0
        assert len(diff.removed_gates) == 0

    def test_diff_detects_changes(self):
        """Diff should detect structural changes."""
        summary_a = CircuitSummary(
            num_qubits=2,
            depth=3,
            gate_count_1q=1,
            gate_count_2q=1,
            gate_count_total=2,
            gate_types={"h": 1, "cx": 1},
        )
        summary_b = CircuitSummary(
            num_qubits=3,  # changed
            depth=5,  # changed
            gate_count_1q=2,  # changed
            gate_count_2q=2,  # changed
            gate_count_total=4,
            gate_types={"h": 2, "cx": 2},
        )

        diff = diff_summaries(summary_a, summary_b)

        assert diff.match is False
        assert len(diff.changed) > 0
        assert "num_qubits" in diff.changed
        assert diff.changed["num_qubits"]["a"] == 2
        assert diff.changed["num_qubits"]["b"] == 3
        assert diff.changed["num_qubits"]["delta"] == 1

    def test_diff_detects_gate_type_changes(self):
        """Diff should detect added/removed gate types."""
        summary_a = CircuitSummary(
            num_qubits=2,
            gate_types={"h": 1, "cx": 1, "swap": 1},
        )
        summary_b = CircuitSummary(
            num_qubits=2,
            gate_types={"h": 1, "cx": 1, "rz": 1},
        )

        diff = diff_summaries(summary_a, summary_b)

        assert diff.match is False
        assert "rz" in diff.added_gates
        assert "swap" in diff.removed_gates

    def test_diff_detects_clifford_change(self):
        """Diff should detect is_clifford status change."""
        summary_a = CircuitSummary(num_qubits=2, is_clifford=True)
        summary_b = CircuitSummary(num_qubits=2, is_clifford=False)

        diff = diff_summaries(summary_a, summary_b)

        assert diff.match is False
        assert diff.is_clifford_changed is True
        assert diff.is_clifford_a is True
        assert diff.is_clifford_b is False


class TestErrorHandling:
    """Test error handling in circuit operations."""

    def test_loader_unknown_sdk_raises(self):
        """Getting loader for unknown SDK raises LoaderError."""
        with pytest.raises(LoaderError) as exc_info:
            get_loader(SDK.UNKNOWN)

        assert "No loader" in str(exc_info.value)

    def test_extract_no_artifacts_returns_none(self, run_factory, store):
        """Extraction from run without program artifacts returns None."""
        record = run_factory(artifacts=[])

        result = extract_circuit(record, store)

        assert result is None

    @requires_qiskit
    def test_loader_invalid_data_raises(self):
        """Loading invalid data raises exception."""
        data = CircuitData(
            data=b"not valid qpy data at all",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        loader = get_loader(SDK.QISKIT)

        with pytest.raises(Exception):
            loader.load(data)


class TestCircuitData:
    """Test CircuitData model basics."""

    def test_binary_text_conversion(self):
        """CircuitData converts between bytes and text."""
        text_data = CircuitData(
            data="OPENQASM 3.0;",
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.QISKIT,
        )

        assert text_data.as_bytes() == b"OPENQASM 3.0;"
        assert text_data.as_text() == "OPENQASM 3.0;"
        assert text_data.is_binary is False

        binary_data = CircuitData(
            data=b"\x00\x01\x02",
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

        assert binary_data.as_bytes() == b"\x00\x01\x02"
        assert binary_data.is_binary is True

    def test_circuit_summary_serialization(self):
        """CircuitSummary round-trips through dict."""
        summary = CircuitSummary(
            num_qubits=5,
            depth=10,
            gate_count_1q=8,
            gate_count_2q=4,
            gate_count_total=12,
            gate_types={"h": 5, "cx": 4, "rz": 3},
            is_clifford=False,
            sdk=SDK.QISKIT,
        )

        d = summary.to_dict()
        restored = CircuitSummary.from_dict(d)

        assert restored.num_qubits == summary.num_qubits
        assert restored.depth == summary.depth
        assert restored.gate_types == summary.gate_types
        assert restored.sdk == summary.sdk
