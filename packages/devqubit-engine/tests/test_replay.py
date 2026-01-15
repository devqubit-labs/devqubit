# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for bundle replay functionality."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from devqubit_engine.bundle.replay import (
    list_available_backends,
    replay,
)


def create_qiskit_bundle(tmp_path: Path, shots: int = 1024) -> Path:
    """Create a Qiskit bundle with Bell circuit."""
    try:
        from qiskit import QuantumCircuit, qpy

        qc = QuantumCircuit(2, 2, name="bell")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        qpy_buffer = io.BytesIO()
        qpy.dump(qc, qpy_buffer)
        qpy_bytes = qpy_buffer.getvalue()
        qpy_digest = f"sha256:{hashlib.sha256(qpy_bytes).hexdigest()}"
    except ImportError:
        pytest.skip("Qiskit not installed")

    return _create_bundle(
        tmp_path,
        run_id="test_qiskit",
        adapter="devqubit-qiskit",
        circuit_kind="qiskit.qpy.circuits",
        circuit_bytes=qpy_bytes,
        circuit_digest=qpy_digest,
        shots=shots,
    )


def create_braket_bundle(tmp_path: Path, shots: int = 1024) -> Path:
    """Create a Braket bundle with Bell circuit."""
    try:
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1)
        try:
            from braket.circuits.serialization import IRType

            ir_program = circuit.to_ir(ir_type=IRType.JAQCD)
        except ImportError:
            ir_program = circuit.to_ir()

        jaqcd_bytes = ir_program.json().encode("utf-8")
        jaqcd_digest = f"sha256:{hashlib.sha256(jaqcd_bytes).hexdigest()}"
    except ImportError:
        pytest.skip("Braket not installed")

    return _create_bundle(
        tmp_path,
        run_id="test_braket",
        adapter="devqubit-braket",
        circuit_kind="braket.ir.jaqcd",
        circuit_bytes=jaqcd_bytes,
        circuit_digest=jaqcd_digest,
        shots=shots,
    )


def create_cirq_bundle(tmp_path: Path, shots: int = 1024) -> Path:
    """Create a Cirq bundle with Bell circuit."""
    try:
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            [
                cirq.H(q0),
                cirq.CNOT(q0, q1),
                cirq.measure(q0, q1, key="m"),
            ]
        )
        cirq_bytes = cirq.to_json(circuit).encode("utf-8")
        cirq_digest = f"sha256:{hashlib.sha256(cirq_bytes).hexdigest()}"
    except ImportError:
        pytest.skip("Cirq not installed")

    return _create_bundle(
        tmp_path,
        run_id="test_cirq",
        adapter="devqubit-cirq",
        circuit_kind="cirq.circuit.json",
        circuit_bytes=cirq_bytes,
        circuit_digest=cirq_digest,
        shots=shots,
    )


def _create_bundle(
    tmp_path: Path,
    run_id: str,
    adapter: str,
    circuit_kind: str,
    circuit_bytes: bytes,
    circuit_digest: str,
    shots: int,
) -> Path:
    """Create bundle ZIP file with circuit and counts artifacts."""
    counts = {"00": shots // 2, "11": shots // 2}
    counts_bytes = json.dumps(
        {"experiments": [{"index": 0, "counts": counts}]}
    ).encode()
    counts_digest = f"sha256:{hashlib.sha256(counts_bytes).hexdigest()}"

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": adapter,
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": shots}, "metrics": {}, "tags": {}},
        "artifacts": [
            {
                "kind": circuit_kind,
                "digest": circuit_digest,
                "media_type": "application/octet-stream",
                "role": "program",
            },
            {
                "kind": "result.counts.json",
                "digest": counts_digest,
                "media_type": "application/json",
                "role": "results",
            },
        ],
        "backend": {"name": "simulator", "type": "simulator"},
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record))
        zf.writestr(
            "manifest.json",
            json.dumps(
                {"format": "devqubit.bundle/0.1", "run_id": run_id, "adapter": adapter}
            ),
        )

        for digest, data in [
            (circuit_digest, circuit_bytes),
            (counts_digest, counts_bytes),
        ]:
            hex_part = digest[7:]
            zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", data)

    return bundle_path


class TestReplayQiskit:
    """Qiskit replay tests."""

    @pytest.fixture(autouse=True)
    def require_qiskit(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

    def test_replay_bell_circuit(self, tmp_path):
        """Replay executes Bell circuit and returns valid counts."""
        bundle_path = create_qiskit_bundle(tmp_path, shots=1024)

        result = replay(
            bundle_path,
            ack_experimental=True,
            backend="aer_simulator",
        )

        assert result.ok is True
        assert result.circuit_source == "qpy"
        assert sum(result.counts.values()) == 1024
        assert all(k in ("00", "11") for k in result.counts)

    def test_replay_with_seed_reproducible(self, tmp_path):
        """Replay with seed produces reproducible results."""
        bundle_path = create_qiskit_bundle(tmp_path, shots=100)

        result1 = replay(bundle_path, ack_experimental=True, seed=42)
        result2 = replay(bundle_path, ack_experimental=True, seed=42)

        assert result1.counts == result2.counts

    def test_replay_rejects_hardware_backend(self, tmp_path):
        """Replay rejects non-simulator backends."""
        bundle_path = create_qiskit_bundle(tmp_path)

        result = replay(
            bundle_path,
            ack_experimental=True,
            backend="ibm_brisbane",
        )

        assert result.ok is False
        assert (
            "simulator" in result.message.lower()
            or "unsupported" in result.message.lower()
        )


class TestReplayBraket:
    """Braket replay tests."""

    @pytest.fixture(autouse=True)
    def require_braket(self):
        pytest.importorskip("braket")

    def test_replay_bell_circuit(self, tmp_path):
        """Replay executes Braket circuit and returns valid counts."""
        bundle_path = create_braket_bundle(tmp_path, shots=1024)

        result = replay(bundle_path, ack_experimental=True, backend="local")

        assert result.ok is True
        assert result.circuit_source == "jaqcd"
        assert sum(result.counts.values()) == 1024


class TestReplayCirq:
    """Cirq replay tests."""

    @pytest.fixture(autouse=True)
    def require_cirq(self):
        pytest.importorskip("cirq")

    def test_replay_bell_circuit(self, tmp_path):
        """Replay executes Cirq circuit and returns valid counts."""
        bundle_path = create_cirq_bundle(tmp_path, shots=1024)

        result = replay(bundle_path, ack_experimental=True, backend="simulator")

        assert result.ok is True
        assert result.circuit_source == "cirq_json"
        assert sum(result.counts.values()) == 1024


class TestReplayErrors:
    """Tests for replay error handling."""

    def test_missing_bundle_fails(self, tmp_path):
        """Replay fails for missing bundle file."""
        result = replay(tmp_path / "nonexistent.zip", ack_experimental=True)

        assert result.ok is False
        assert "Failed to load" in result.message

    def test_invalid_bundle_fails(self, tmp_path):
        """Replay fails for invalid bundle format."""
        bundle_path = tmp_path / "invalid.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("random.txt", "not a valid bundle")

        result = replay(bundle_path, ack_experimental=True)

        assert result.ok is False

    def test_no_circuit_fails(self, tmp_path):
        """Replay fails when no circuit artifact found."""
        pytest.importorskip("qiskit")

        run_record = {
            "run_id": "no_circuit",
            "adapter": "devqubit-qiskit",
            "backend": {"name": "aer_simulator"},
            "artifacts": [],
        }

        bundle_path = tmp_path / "no_circuit.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("run.json", json.dumps(run_record))
            zf.writestr("manifest.json", json.dumps({"format": "devqubit.bundle/0.1"}))

        result = replay(bundle_path, ack_experimental=True)

        assert result.ok is False


class TestListAvailableBackends:
    """Tests for listing available simulator backends."""

    def test_returns_dict_of_simulators(self):
        """Returns dictionary with only simulator backends."""
        backends = list_available_backends()

        assert isinstance(backends, dict)
        for sdk, backend_list in backends.items():
            for backend in backend_list:
                # Should not contain hardware backend names
                assert "ibm_" not in backend.lower()
                assert "ionq" not in backend.lower()
