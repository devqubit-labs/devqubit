# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit module."""

from __future__ import annotations

import io

import pytest
from devqubit_engine.circuit.extractors import (
    detect_sdk,
    extract_circuit,
    extract_circuit_from_refs,
)
from devqubit_engine.circuit.hashing import (
    hash_circuit_pair,
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


class TestCircuitPipeline:
    """End-to-end tests for circuit extraction and summarization."""

    @requires_qiskit
    def test_qiskit_qpy_full_pipeline(self, run_factory, store, artifact_factory):
        """
        Full pipeline: QPY artifact → extract → load → summarize.

        This is the realistic flow when processing a Qiskit run.
        """
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
        self, run_factory, store, artifact_factory, bell_qasm2
    ):
        """
        Full pipeline: OpenQASM2 artifact → extract → load → summarize.
        """
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
        """
        Full pipeline: JAQCD artifact → extract → load → summarize.
        """
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


class TestSDKDetection:
    """Test SDK detection from run records."""

    @pytest.mark.parametrize(
        "adapter,expected_sdk",
        [
            ("devqubit-qiskit", SDK.QISKIT),
            ("devqubit-braket", SDK.BRAKET),
            ("devqubit-cirq", SDK.CIRQ),
            ("devqubit-pennylane", SDK.PENNYLANE),
            ("devqubit-Qiskit", SDK.QISKIT),  # case insensitive
            ("unknown-adapter", SDK.UNKNOWN),
            ("manual", SDK.UNKNOWN),
        ],
    )
    def test_detect_sdk_from_adapter(self, run_factory, adapter, expected_sdk):
        """SDK detection based on adapter name."""
        record = run_factory(adapter=adapter)
        assert detect_sdk(record) == expected_sdk


class TestCircuitHashing:
    """Test circuit hashing for reproducibility tracking."""

    def test_structural_hash_ignores_param_values(self):
        """
        Structural hash should be the same for circuits with
        different parameter values but same structure.
        """
        ops_v1 = [
            {"gate": "h", "qubits": [0]},
            {"gate": "rz", "qubits": [0], "params": {"theta": 0.5}},
            {"gate": "cx", "qubits": [0, 1]},
        ]
        ops_v2 = [
            {"gate": "h", "qubits": [0]},
            {"gate": "rz", "qubits": [0], "params": {"theta": 1.57}},  # different value
            {"gate": "cx", "qubits": [0, 1]},
        ]

        hash_v1 = hash_structural(ops_v1)
        hash_v2 = hash_structural(ops_v2)

        assert hash_v1 == hash_v2
        assert hash_v1.startswith("sha256:")

    def test_parametric_hash_differs_with_param_values(self):
        """
        Parametric hash should differ when parameter values differ.
        """
        ops = [
            {"gate": "h", "qubits": [0]},
            {"gate": "rz", "qubits": [0], "params": {"theta": None}},
        ]

        hash_v1 = hash_parametric(ops, {"theta": 0.5})
        hash_v2 = hash_parametric(ops, {"theta": 1.57})

        assert hash_v1 != hash_v2

    def test_hash_circuit_pair_convenience(self):
        """hash_circuit_pair returns both hashes."""
        ops = [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]},
        ]

        structural, parametric = hash_circuit_pair(ops)

        assert structural.startswith("sha256:")
        assert parametric.startswith("sha256:")

    def test_hash_deterministic(self):
        """Same input always produces same hash."""
        ops = [
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]},
            {"gate": "measure", "qubits": [0, 1], "clbits": [0, 1]},
        ]

        # Multiple calls
        hashes = [hash_structural(ops) for _ in range(5)]

        assert len(set(hashes)) == 1  # All identical

    def test_cx_direction_matters_structural(self):
        """cx(0,1) and cx(1,0) must have different structural hashes."""
        ops_01 = [{"gate": "cx", "qubits": [0, 1]}]
        ops_10 = [{"gate": "cx", "qubits": [1, 0]}]

        assert hash_structural(ops_01) != hash_structural(ops_10)

    def test_cx_direction_matters_parametric(self):
        """cx(0,1) and cx(1,0) must have different parametric hashes."""
        ops_01 = [{"gate": "cx", "qubits": [0, 1]}]
        ops_10 = [{"gate": "cx", "qubits": [1, 0]}]

        assert hash_parametric(ops_01) != hash_parametric(ops_10)

    def test_negative_zero_equals_zero(self):
        """-0.0 and 0.0 must produce identical hash."""
        ops_pos = [{"gate": "rz", "qubits": [0], "params": {"theta": 0.0}}]
        ops_neg = [{"gate": "rz", "qubits": [0], "params": {"theta": -0.0}}]

        assert hash_parametric(ops_pos) == hash_parametric(ops_neg)

    def test_nan_consistent(self):
        """NaN values produce consistent hash."""
        import math

        ops = [{"gate": "rz", "qubits": [0], "params": {"theta": math.nan}}]

        h1 = hash_parametric(ops)
        h2 = hash_parametric(ops)

        assert h1 == h2

    def test_unused_bound_params_ignored(self):
        """Extra keys in bound_params don't change hash."""
        ops = [{"gate": "rz", "qubits": [0], "params": {"theta": None}}]

        h1 = hash_parametric(ops, {"theta": 1.0})
        h2 = hash_parametric(ops, {"theta": 1.0, "unused": 999.0})

        assert h1 == h2

    def test_list_param_names_matter_structural(self):
        """List params with different names have different structural hash."""
        ops_ab = [{"gate": "u3", "qubits": [0], "params": ["alpha", "beta"]}]
        ops_xy = [{"gate": "u3", "qubits": [0], "params": ["x", "y"]}]

        assert hash_structural(ops_ab) != hash_structural(ops_xy)


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
        assert len(diff.changes) == 0

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
        assert len(diff.changes) > 0
        assert "qubits" in str(diff.changes)
        assert diff.metrics["num_qubits_delta"] == 1


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
