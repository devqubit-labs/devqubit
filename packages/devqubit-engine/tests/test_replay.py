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
    ReplayResult,
    _run_circuit,
    list_available_backends,
    replay,
)
from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import get_loader


def create_qiskit_qpy_bundle(
    tmp_path: Path,
    run_id: str = "test_run_qiskit",
    shots: int = 1024,
    counts: dict[str, int] | None = None,
) -> Path:
    """Create a realistic Qiskit bundle with QPY circuit."""
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

    if counts is None:
        counts = {"00": shots // 2, "11": shots // 2}

    counts_obj = {"experiments": [{"index": 0, "counts": counts}]}
    counts_bytes = json.dumps(counts_obj).encode("utf-8")
    counts_digest = f"sha256:{hashlib.sha256(counts_bytes).hexdigest()}"

    artifacts = [
        {
            "kind": "qiskit.qpy.circuits",
            "digest": qpy_digest,
            "media_type": "application/x-qiskit-qpy",
            "role": "program",
        },
        {
            "kind": "result.counts.json",
            "digest": counts_digest,
            "media_type": "application/json",
            "role": "results",
        },
    ]

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": "devqubit-qiskit",
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": shots}, "metrics": {}, "tags": {}},
        "artifacts": artifacts,
        "backend": {"name": "aer_simulator", "type": "simulator"},
    }

    manifest = {
        "format": "devqubit.bundle/0.1",
        "run_id": run_id,
        "adapter": "devqubit-qiskit",
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        for digest, data in [(qpy_digest, qpy_bytes), (counts_digest, counts_bytes)]:
            hex_part = digest[7:]
            zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", data)

    return bundle_path


def create_braket_bundle(
    tmp_path: Path,
    run_id: str = "test_run_braket",
    shots: int = 1024,
    counts: dict[str, int] | None = None,
) -> Path:
    """Create a realistic Braket bundle with JAQCD."""
    try:
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1)

        try:
            from braket.circuits.serialization import IRType

            ir_program = circuit.to_ir(ir_type=IRType.JAQCD)
        except ImportError:
            ir_program = circuit.to_ir()

        jaqcd_str = ir_program.json()
        jaqcd_bytes = jaqcd_str.encode("utf-8")
        circuit_digest = f"sha256:{hashlib.sha256(jaqcd_bytes).hexdigest()}"

    except ImportError:
        pytest.skip("Braket not installed")

    if counts is None:
        counts = {"00": shots // 2, "11": shots // 2}

    counts_obj = {"experiments": [{"index": 0, "counts": counts}]}
    counts_bytes = json.dumps(counts_obj).encode("utf-8")
    counts_digest = f"sha256:{hashlib.sha256(counts_bytes).hexdigest()}"

    artifacts = [
        {
            "kind": "braket.ir.jaqcd",
            "digest": circuit_digest,
            "media_type": "application/json",
            "role": "program",
        },
        {
            "kind": "result.counts.json",
            "digest": counts_digest,
            "media_type": "application/json",
            "role": "results",
        },
    ]

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": "devqubit-braket",
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": shots}, "metrics": {}, "tags": {}},
        "artifacts": artifacts,
        "backend": {"name": "local", "type": "simulator"},
    }

    manifest = {
        "format": "devqubit.bundle/0.1",
        "run_id": run_id,
        "adapter": "devqubit-braket",
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        for digest, data in [
            (circuit_digest, jaqcd_bytes),
            (counts_digest, counts_bytes),
        ]:
            hex_part = digest[7:]
            zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", data)

    return bundle_path


def create_cirq_bundle(
    tmp_path: Path,
    run_id: str = "test_run_cirq",
    shots: int = 1024,
    counts: dict[str, int] | None = None,
) -> Path:
    """Create a realistic Cirq bundle with Cirq JSON."""
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
        cirq_json = cirq.to_json(circuit)
        cirq_bytes = cirq_json.encode("utf-8")
        circuit_digest = f"sha256:{hashlib.sha256(cirq_bytes).hexdigest()}"

    except ImportError:
        pytest.skip("Cirq not installed")

    if counts is None:
        counts = {"00": shots // 2, "11": shots // 2}

    counts_obj = {"experiments": [{"index": 0, "counts": counts}]}
    counts_bytes = json.dumps(counts_obj).encode("utf-8")
    counts_digest = f"sha256:{hashlib.sha256(counts_bytes).hexdigest()}"

    artifacts = [
        {
            "kind": "cirq.circuit.json",
            "digest": circuit_digest,
            "media_type": "application/json",
            "role": "program",
        },
        {
            "kind": "result.counts.json",
            "digest": counts_digest,
            "media_type": "application/json",
            "role": "results",
        },
    ]

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": "devqubit-cirq",
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": shots}, "metrics": {}, "tags": {}},
        "artifacts": artifacts,
        "backend": {"name": "simulator", "type": "simulator"},
    }

    manifest = {
        "format": "devqubit.bundle/0.1",
        "run_id": run_id,
        "adapter": "devqubit-cirq",
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        for digest, data in [
            (circuit_digest, cirq_bytes),
            (counts_digest, counts_bytes),
        ]:
            hex_part = digest[7:]
            zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", data)

    return bundle_path


def create_openqasm_bundle(
    tmp_path: Path,
    run_id: str = "test_run_qasm",
    shots: int = 1024,
) -> Path:
    """Create a bundle with OpenQASM circuit (should be rejected)."""
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    qasm_bytes = qasm_code.encode("utf-8")
    circuit_digest = f"sha256:{hashlib.sha256(qasm_bytes).hexdigest()}"

    counts_obj = {"experiments": [{"index": 0, "counts": {"00": 512, "11": 512}}]}
    counts_bytes = json.dumps(counts_obj).encode("utf-8")
    counts_digest = f"sha256:{hashlib.sha256(counts_bytes).hexdigest()}"

    artifacts = [
        {
            "kind": "source.openqasm",
            "digest": circuit_digest,
            "media_type": "application/openqasm",
            "role": "program",
        },
        {
            "kind": "result.counts.json",
            "digest": counts_digest,
            "media_type": "application/json",
            "role": "results",
        },
    ]

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": "devqubit-qiskit",
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": shots}, "metrics": {}, "tags": {}},
        "artifacts": artifacts,
        "backend": {"name": "aer_simulator", "type": "simulator"},
    }

    manifest = {
        "format": "devqubit.bundle/0.1",
        "run_id": run_id,
        "adapter": "devqubit-qiskit",
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        for digest, data in [
            (circuit_digest, qasm_bytes),
            (counts_digest, counts_bytes),
        ]:
            hex_part = digest[7:]
            zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", data)

    return bundle_path


def create_unknown_sdk_bundle(tmp_path: Path, run_id: str = "test_run_unknown") -> Path:
    """Create a bundle with unknown SDK."""
    circuit_bytes = b"some circuit data"
    circuit_digest = f"sha256:{hashlib.sha256(circuit_bytes).hexdigest()}"

    artifacts = [
        {
            "kind": "unknown.circuit",
            "digest": circuit_digest,
            "media_type": "application/octet-stream",
            "role": "program",
        },
    ]

    run_record = {
        "schema": "devqubit.run/0.1",
        "run_id": run_id,
        "created_at": "2024-01-01T00:00:00Z",
        "project": {"name": "test"},
        "adapter": "unknown-adapter",  # Not a recognized adapter
        "info": {"status": "FINISHED"},
        "data": {"params": {"shots": 1024}, "metrics": {}, "tags": {}},
        "artifacts": artifacts,
        "backend": {"name": "unknown", "type": "unknown"},
    }

    manifest = {
        "format": "devqubit.bundle/0.1",
        "run_id": run_id,
        "adapter": "unknown-adapter",
    }

    bundle_path = tmp_path / f"{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.json", json.dumps(run_record, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        hex_part = circuit_digest[7:]
        zf.writestr(f"objects/sha256/{hex_part[:2]}/{hex_part}", circuit_bytes)

    return bundle_path


class TestBraketNoAutoMeasurements:
    """Test that Braket replay does not add measurements."""

    @pytest.fixture(autouse=True)
    def require_braket(self):
        """Skip if Braket not installed."""
        pytest.importorskip("braket")

    def test_braket_jaqcd_without_measure_succeeds(self, tmp_path):
        bundle_path = create_braket_bundle(tmp_path, run_id="braket_jaqcd_no_measure")
        result = replay(bundle_path, ack_experimental=True)
        assert result.ok is True
        assert len(result.counts) > 0


class TestSeedSupport:
    """Test seed parameter for reproducibility."""

    def test_seed_in_result(self, tmp_path):
        """Seed is included in result."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        bundle_path = create_qiskit_qpy_bundle(tmp_path)
        result = replay(bundle_path, ack_experimental=True, seed=42)

        assert result.seed == 42

    def test_seed_none_in_result(self, tmp_path):
        """Seed is None when not provided."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        bundle_path = create_qiskit_qpy_bundle(tmp_path)
        result = replay(bundle_path, ack_experimental=True)

        assert result.seed is None

    def test_cirq_seed_reproducibility(self, tmp_path):
        """Cirq replay with same seed produces same results."""
        pytest.importorskip("cirq")

        bundle_path = create_cirq_bundle(tmp_path, shots=1000)

        result1 = replay(bundle_path, ack_experimental=True, seed=12345)
        result2 = replay(bundle_path, ack_experimental=True, seed=12345)

        assert result1.ok is True
        assert result2.ok is True
        # Cirq fully supports seed - results should be identical
        assert result1.counts == result2.counts

    def test_braket_seed_warning(self, tmp_path):
        """Braket replay with seed logs warning."""
        pytest.importorskip("braket")

        bundle_path = create_braket_bundle(tmp_path)
        result = replay(bundle_path, ack_experimental=True, seed=42)

        # Should have warning about seed not being supported
        assert result.ok is True
        assert any("seed" in e.lower() for e in result.errors)


class TestReplayQiskit:
    """Qiskit replay tests."""

    @pytest.fixture(autouse=True)
    def require_qiskit(self):
        """Skip if Qiskit not installed."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

    def test_replay_executes_circuit(self, tmp_path):
        """Replay executes circuit and returns counts."""
        bundle_path = create_qiskit_qpy_bundle(tmp_path, shots=1024)

        result = replay(bundle_path, ack_experimental=True, backend="aer_simulator")

        assert result.ok is True
        assert result.circuit_source == "qpy"
        assert result.backend_used == "aer_simulator"
        assert len(result.counts) > 0
        assert sum(result.counts.values()) == 1024

    def test_replay_returns_original_metadata(self, tmp_path):
        """Replay includes original run metadata."""
        bundle_path = create_qiskit_qpy_bundle(
            tmp_path,
            run_id="original_run_123",
            shots=512,
        )

        result = replay(bundle_path, ack_experimental=True)

        assert result.original_run_id == "original_run_123"
        assert result.original_adapter == "devqubit-qiskit"
        assert result.original_backend == "aer_simulator"
        assert result.shots == 512

    def test_replay_override_shots(self, tmp_path):
        """Replay can override shot count."""
        bundle_path = create_qiskit_qpy_bundle(tmp_path, shots=1000)

        result = replay(bundle_path, ack_experimental=True, shots=500)

        assert result.shots == 500
        assert sum(result.counts.values()) == 500

    def test_replay_bell_state_distribution(self, tmp_path):
        """Bell state replay produces ~50/50 distribution."""
        bundle_path = create_qiskit_qpy_bundle(tmp_path, shots=4096)

        result = replay(bundle_path, ack_experimental=True)

        # Bell state should only produce 00 and 11
        for key in result.counts:
            assert key in ("00", "11"), f"Unexpected outcome: {key}"

        # Roughly equal distribution (within statistical noise)
        total = sum(result.counts.values())
        for count in result.counts.values():
            assert 0.3 < count / total < 0.7

    def test_replay_rejects_non_simulator(self, tmp_path):
        """Replay rejects non-simulator backends."""
        bundle_path = create_qiskit_qpy_bundle(tmp_path)

        result = replay(bundle_path, ack_experimental=True, backend="ibm_brisbane")

        assert result.ok is False
        assert (
            "simulator" in result.message.lower()
            or "unsupported" in result.message.lower()
        )

    def test_replay_message_includes_experimental(self, tmp_path):
        """Successful replay message indicates experimental status."""
        bundle_path = create_qiskit_qpy_bundle(tmp_path)

        result = replay(bundle_path, ack_experimental=True)

        assert result.ok is True
        assert "EXPERIMENTAL" in result.message


class TestReplayBraket:
    """Braket replay tests."""

    @pytest.fixture(autouse=True)
    def require_braket(self):
        """Skip if Braket not installed."""
        pytest.importorskip("braket")

    def test_replay_executes_circuit(self, tmp_path):
        """Replay executes circuit and returns counts."""
        bundle_path = create_braket_bundle(tmp_path, shots=1024)

        result = replay(bundle_path, ack_experimental=True, backend="local")

        assert result.ok is True
        assert result.circuit_source == "jaqcd"
        assert result.backend_used == "local"
        assert len(result.counts) > 0
        assert sum(result.counts.values()) == 1024


class TestReplayCirq:
    """Cirq replay tests."""

    @pytest.fixture(autouse=True)
    def require_cirq(self):
        """Skip if Cirq not installed."""
        pytest.importorskip("cirq")

    def test_replay_executes_circuit(self, tmp_path):
        """Replay executes circuit and returns counts."""
        bundle_path = create_cirq_bundle(tmp_path, shots=1024)

        result = replay(bundle_path, ack_experimental=True, backend="simulator")

        assert result.ok is True
        assert result.circuit_source == "cirq_json"
        assert result.backend_used == "simulator"
        assert len(result.counts) > 0
        assert sum(result.counts.values()) == 1024

    def test_replay_density_matrix_backend(self, tmp_path):
        """Replay works with density_matrix backend."""
        bundle_path = create_cirq_bundle(tmp_path, shots=512)

        result = replay(bundle_path, ack_experimental=True, backend="density_matrix")

        assert result.ok is True
        assert result.backend_used == "density_matrix"
        assert sum(result.counts.values()) == 512


class TestRunCircuit:
    """Test _run_circuit dispatcher."""

    def test_run_qiskit_circuit(self):
        """Run Qiskit circuit via _run_circuit."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from qiskit import QuantumCircuit, qpy

        qc = QuantumCircuit(2, 2)
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

        errors = []
        counts = _run_circuit(loaded, 100, "aer_simulator", seed=42, errors=errors)

        assert sum(counts.values()) == 100
        for key in counts:
            assert key in ("00", "11")

    def test_run_cirq_circuit_with_seed(self):
        """Run Cirq circuit with seed via _run_circuit."""
        cirq = pytest.importorskip("cirq")

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

        errors = []
        counts1 = _run_circuit(loaded, 100, "simulator", seed=123, errors=errors)
        counts2 = _run_circuit(loaded, 100, "simulator", seed=123, errors=errors)

        assert counts1 == counts2  # Same seed = same results

    def test_run_unknown_sdk_raises(self):
        """Running with unknown SDK raises error."""
        loaded = LoadedCircuit(
            circuit="dummy",
            sdk=SDK.UNKNOWN,
            source_format=CircuitFormat.TEXT,
        )
        with pytest.raises(ValueError, match="No runner for SDK"):
            _run_circuit(loaded, 100, "backend", seed=None, errors=[])


class TestReplayErrors:
    """Test error handling in replay."""

    def test_replay_missing_bundle_fails(self, tmp_path):
        """Replay fails for missing bundle."""
        result = replay(tmp_path / "nonexistent.zip", ack_experimental=True)

        assert result.ok is False
        assert "Failed to load" in result.message

    def test_replay_no_circuit_fails(self, tmp_path):
        """Replay fails when no circuit artifact found."""
        pytest.importorskip("qiskit")  # For SDK detection

        run_record = {
            "run_id": "test_no_circuit",
            "adapter": "devqubit-qiskit",
            "backend": {"name": "aer_simulator"},
            "artifacts": [],  # No circuit
        }

        bundle_path = tmp_path / "no_circuit.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("run.json", json.dumps(run_record))
            zf.writestr("manifest.json", json.dumps({"format": "devqubit.bundle/0.1"}))

        result = replay(bundle_path, ack_experimental=True)

        assert result.ok is False
        assert "circuit" in result.message.lower() or any(
            "circuit" in e.lower() for e in result.errors
        )

    def test_replay_invalid_bundle_fails(self, tmp_path):
        """Replay fails for invalid bundle format."""
        bundle_path = tmp_path / "invalid.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("random.txt", "not a valid bundle")

        result = replay(bundle_path, ack_experimental=True)

        assert result.ok is False


# =============================================================================
# Test: list_available_backends
# =============================================================================


class TestListAvailableBackends:
    """Test list_available_backends function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        backends = list_available_backends()
        assert isinstance(backends, dict)

    def test_only_simulators(self):
        """Only simulator backends are listed."""
        backends = list_available_backends()

        for sdk, backend_list in backends.items():
            for backend in backend_list:
                # Should not contain hardware backend names
                assert "ibm_" not in backend.lower()
                assert "ionq" not in backend.lower()
                assert "rigetti" not in backend.lower()


class TestReplayResult:
    """Test ReplayResult dataclass."""

    def test_to_dict_includes_seed(self):
        """to_dict includes seed field."""
        result = ReplayResult(
            ok=True,
            original_run_id="test123",
            counts={"00": 50, "11": 50},
            seed=42,
        )

        d = result.to_dict()
        assert d["seed"] == 42

    def test_to_dict_seed_none(self):
        """to_dict includes seed=None when not provided."""
        result = ReplayResult(
            ok=True,
            original_run_id="test123",
        )

        d = result.to_dict()
        assert d["seed"] is None

    def test_repr(self):
        """repr is informative."""
        result = ReplayResult(
            ok=True,
            original_run_id="test123",
            shots=1000,
        )

        repr_str = repr(result)
        assert "ok" in repr_str
        assert "test123" in repr_str
        assert "1000" in repr_str
