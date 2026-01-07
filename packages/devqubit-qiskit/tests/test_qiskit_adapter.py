# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit adapter."""

import json
import logging
from unittest.mock import patch

from devqubit_engine.core.tracker import track
from devqubit_qiskit.adapter import (
    QiskitAdapter,
    TrackedBackend,
    TrackedJob,
    _compute_circuit_hash,
    _materialize_circuits,
)
from devqubit_qiskit.serialization import (
    LoadedCircuitBatch,
    QiskitCircuitLoader,
    serialize_qpy,
)
from devqubit_qiskit.snapshot import create_device_snapshot
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def _load_envelope(run_id, store, registry):
    """Load run and extract first envelope."""
    loaded = registry.load(run_id)
    env_artifacts = [a for a in loaded.artifacts if a.kind == "devqubit.envelope.json"]
    if env_artifacts:
        raw = store.get_bytes(env_artifacts[0].digest)
        return loaded, json.loads(raw.decode("utf-8"))
    return loaded, None


def _count_kind(loaded, kind: str) -> int:
    """Count artifacts of a specific kind."""
    return sum(1 for a in getattr(loaded, "artifacts", []) if a.kind == kind)


def _kinds(loaded) -> set[str]:
    """Extract artifact kinds from loaded run."""
    return {a.kind for a in getattr(loaded, "artifacts", [])}


def _normalize_bitstring(key: str) -> str:
    """Normalize bitstring by removing spaces and underscores."""
    return key.replace(" ", "").replace("_", "")


class TestQiskitAdapter:
    """Tests for adapter registration and backend detection."""

    def test_adapter_name(self):
        """Adapter has correct identifier."""
        assert QiskitAdapter().name == "qiskit"

    def test_supports_aer_simulator(self, aer_simulator):
        """Adapter supports AerSimulator."""
        assert QiskitAdapter().supports_executor(aer_simulator) is True

    def test_rejects_non_backends(self):
        """Adapter rejects non-backend objects."""
        adapter = QiskitAdapter()
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("backend") is False
        assert adapter.supports_executor(QuantumCircuit(2)) is False

    def test_wrap_executor_returns_tracked_backend(
        self, store, registry, aer_simulator
    ):
        """Wrapping returns a TrackedBackend."""
        with track(project="test", store=store, registry=registry) as run:
            wrapped = run.wrap(aer_simulator)
            assert isinstance(wrapped, TrackedBackend)
            assert wrapped.backend is aer_simulator


class TestMaterializeCircuits:
    """Tests for generator/iterator handling."""

    def test_single_circuit(self, bell_circuit):
        """Single circuit returns (list, was_single=True)."""
        result, was_single = _materialize_circuits(bell_circuit)
        assert len(result) == 1
        assert was_single is True

    def test_list_of_circuits(self, bell_circuit, ghz_circuit):
        """List of circuits passes through."""
        result, was_single = _materialize_circuits([bell_circuit, ghz_circuit])
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
        result, _ = _materialize_circuits(gen)
        assert len(result) == 3


class TestCircuitHash:
    """Tests for circuit structure hashing."""

    def test_same_structure_same_hash(self):
        """Identical structure produces same hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        assert _compute_circuit_hash([qc1]) == _compute_circuit_hash([qc2])

    def test_different_gates_different_hash(self):
        """Different gates produce different hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.x(0)

        assert _compute_circuit_hash([qc1]) != _compute_circuit_hash([qc2])

    def test_parameter_values_dont_change_hash(self):
        """Parameter values don't affect structure hash."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        bound1 = qc.assign_parameters({theta: 0.5})
        bound2 = qc.assign_parameters({theta: 1.5})

        assert _compute_circuit_hash([bound1]) == _compute_circuit_hash([bound2])


class TestTrackedBackendExecution:
    """Tests for backend wrapping and execution."""

    def test_run_returns_tracked_job(
        self, bell_circuit, aer_simulator, store, registry
    ):
        """run() returns a TrackedJob with results."""
        with track(project="test", store=store, registry=registry) as run:
            tracked = TrackedBackend(backend=aer_simulator, tracker=run)
            job = tracked.run(bell_circuit, shots=100)

            assert isinstance(job, TrackedJob)
            counts = job.result().get_counts()
            assert sum(counts.values()) == 100

    def test_bell_state_correlated_results(
        self, bell_circuit, aer_simulator, store, registry
    ):
        """Bell state produces correlated 00/11 outcomes."""
        with track(project="test", store=store, registry=registry) as run:
            tracked = TrackedBackend(backend=aer_simulator, tracker=run)
            counts = tracked.run(bell_circuit, shots=1000).result().get_counts()

            valid = {"00", "11"}
            for key in counts:
                clean = _normalize_bitstring(key)
                assert clean in valid or clean[::-1] in valid

    def test_batch_execution(self, aer_simulator, store, registry):
        """Multiple circuits executed correctly."""
        circuits = []
        for n in [2, 3, 4]:
            qc = QuantumCircuit(n, name=f"ghz_{n}")
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            circuits.append(qc)

        with track(project="test", store=store, registry=registry) as run:
            tracked = TrackedBackend(backend=aer_simulator, tracker=run)
            result = tracked.run(circuits, shots=500).result()

            assert len(result.results) == 3
            for i in range(3):
                assert sum(result.get_counts(i).values()) == 500


class TestSamplingBehavior:
    """Tests for execution sampling."""

    def test_log_every_n_zero_skips_repeats(self, aer_simulator, store, registry):
        """log_every_n=0 logs only unique circuits."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(aer_simulator, log_every_n=0, log_new_circuits=True)
            tracked.run(qc, shots=10).result()
            tracked.run(qc, shots=10).result()

        loaded = registry.load(run.run_id)
        assert _count_kind(loaded, "devqubit.envelope.json") == 1

    def test_log_every_n_periodic(self, aer_simulator, store, registry):
        """log_every_n=2 with 3 executions logs 2 (1st and 2nd)."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(aer_simulator, log_every_n=2, log_new_circuits=True)
            tracked.run(qc, shots=10).result()  # 1: logged (first)
            tracked.run(qc, shots=10).result()  # 2: logged (periodic)
            tracked.run(qc, shots=10).result()  # 3: not logged

        loaded = registry.load(run.run_id)
        assert _count_kind(loaded, "devqubit.envelope.json") == 2

    def test_log_new_circuits_on_structure_change(self, aer_simulator, store, registry):
        """log_new_circuits=True logs when structure changes."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.measure_all()

        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(aer_simulator, log_every_n=0, log_new_circuits=True)
            tracked.run(qc1, shots=10).result()
            tracked.run(qc1, shots=10).result()  # same - not logged
            tracked.run(qc2, shots=10).result()  # new - logged

        loaded = registry.load(run.run_id)
        assert _count_kind(loaded, "devqubit.envelope.json") == 2


class TestEnvelopeStructure:
    """Tests for UEC ExecutionEnvelope structure."""

    def test_envelope_created(self, bell_circuit, aer_simulator, store, registry):
        """Execution creates proper envelope."""
        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(bell_circuit, shots=100).result()

        _, envelope = _load_envelope(run.run_id, store, registry)

        assert envelope is not None
        assert envelope["schema"] == "devqubit.envelope/0.1"
        assert "device" in envelope
        assert "program" in envelope
        assert "execution" in envelope
        assert "result" in envelope

    def test_envelope_device_section(
        self, bell_circuit, aer_simulator, store, registry
    ):
        """Envelope device section has required fields."""
        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(bell_circuit, shots=100).result()

        _, envelope = _load_envelope(run.run_id, store, registry)

        device = envelope["device"]
        assert device["backend_type"] == "simulator"
        assert "captured_at" in device

    def test_envelope_program_section(
        self, bell_circuit, aer_simulator, store, registry
    ):
        """Envelope program section has circuit info."""
        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(bell_circuit, shots=100).result()

        _, envelope = _load_envelope(run.run_id, store, registry)

        program = envelope["program"]
        assert program["num_circuits"] == 1
        assert program["program_hash"].startswith("sha256:")


class TestBatchQASM3Artifacts:
    """Multi-circuit batches must produce QASM3 artifact per circuit."""

    def test_batch_3_circuits_produces_3_qasm3_refs(
        self, aer_simulator, store, registry
    ):
        """Batch of 3 circuits should produce 3 QASM3 artifacts."""
        circuits = []
        for n in [2, 3, 4]:
            qc = QuantumCircuit(n, name=f"ghz_{n}")
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            circuits.append(qc)

        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(circuits, shots=100).result()

        _, envelope = _load_envelope(run.run_id, store, registry)

        assert envelope["program"]["num_circuits"] == 3

        logical = envelope["program"].get("logical", [])
        qasm3 = [a for a in logical if a.get("format") == "openqasm3"]

        assert len(qasm3) == 3
        indices = {a.get("index") for a in qasm3}
        assert indices == {0, 1, 2}


class TestRawPropertiesArtifact:
    """raw_properties must be stored via raw_properties_ref when tracker is provided."""

    def test_snapshot_without_tracker_has_no_ref(self, aer_simulator):
        """DeviceSnapshot without tracker should have raw_properties_ref=None."""
        snapshot = create_device_snapshot(aer_simulator, tracker=None)

        # Without tracker, raw_properties_ref should be None
        assert snapshot.raw_properties_ref is None

    def test_snapshot_with_tracker_has_raw_properties_ref(
        self, aer_simulator, store, registry
    ):
        """DeviceSnapshot with tracker should have raw_properties_ref populated."""
        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(aer_simulator, tracker=run)

            # With tracker, raw_properties_ref should be an ArtifactRef
            assert snapshot.raw_properties_ref is not None
            assert (
                snapshot.raw_properties_ref.kind == "device.qiskit.raw_properties.json"
            )
            assert snapshot.raw_properties_ref.role == "device_raw"
            assert snapshot.raw_properties_ref.digest.startswith("sha256:")

    def test_raw_properties_logged_as_artifact(self, aer_simulator, store, registry):
        """raw_properties should be logged as artifact via adapter."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()

        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(qc, shots=100).result()

        loaded = registry.load(run.run_id)
        assert "device.qiskit.raw_properties.json" in _kinds(loaded)

    def test_raw_properties_content_is_valid(self, aer_simulator, store, registry):
        """raw_properties artifact should contain expected backend metadata."""
        with track(project="test", store=store, registry=registry) as run:
            snapshot = create_device_snapshot(aer_simulator, tracker=run)

        # Verify artifact was stored and contains expected keys
        assert snapshot.raw_properties_ref is not None

        # Load the artifact content
        content = store.get_bytes(snapshot.raw_properties_ref.digest)
        import json

        raw_props = json.loads(content.decode("utf-8"))

        assert isinstance(raw_props, dict)
        assert "backend_class" in raw_props
        assert "backend_module" in raw_props


class TestQPYBatchLoading:
    """QPY loader must support multi-circuit batches."""

    def test_load_batch_returns_all_circuits(self, bell_circuit, ghz_circuit):
        """load_batch() returns all circuits from QPY."""
        data = serialize_qpy([bell_circuit, ghz_circuit], name="batch")

        loader = QiskitCircuitLoader()
        batch = loader.load_batch(data)

        assert isinstance(batch, LoadedCircuitBatch)
        assert batch.count == 2
        assert batch.circuits[0].num_qubits == 2
        assert batch.circuits[1].num_qubits == 3

    def test_load_single_warns_on_multi(self, bell_circuit, ghz_circuit, caplog):
        """load() on multi-circuit QPY warns about load_batch()."""
        data = serialize_qpy([bell_circuit, ghz_circuit])
        loader = QiskitCircuitLoader()

        with caplog.at_level(logging.WARNING):
            loaded = loader.load(data)

        assert loaded.circuit.num_qubits == 2  # returns first
        assert any("load_batch" in r.message.lower() for r in caplog.records)


class TestEnvelopeErrorPath:
    """Envelope must be created even if snapshot fails."""

    def test_envelope_created_on_snapshot_error(self, aer_simulator, store, registry):
        """Envelope created with minimal snapshot on error."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()

        with patch(
            "devqubit_qiskit.adapter.create_device_snapshot",
            side_effect=RuntimeError("Snapshot failed"),
        ):
            with track(project="test", store=store, registry=registry) as run:
                backend = run.wrap(aer_simulator)
                result = backend.run(qc, shots=100).result()
                assert sum(result.get_counts().values()) == 100

        _, envelope = _load_envelope(run.run_id, store, registry)

        assert envelope is not None
        assert "device" in envelope
        assert "program" in envelope
        assert "result" in envelope


class TestIdempotentResultLogging:
    """job.result() called twice should not duplicate artifacts."""

    def test_result_twice_no_duplicates(
        self, bell_circuit, aer_simulator, store, registry
    ):
        """job.result() called twice creates only 1 envelope."""
        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            job = backend.run(bell_circuit, shots=100)

            result1 = job.result()
            result2 = job.result()

            assert result1.get_counts() == result2.get_counts()

        loaded = registry.load(run.run_id)
        assert _count_kind(loaded, "devqubit.envelope.json") == 1
        assert _count_kind(loaded, "result.counts.json") == 1


class TestArtifactTracking:
    """Tests for artifact creation."""

    def test_core_artifacts_created(self, bell_circuit, aer_simulator, store, registry):
        """Core artifacts are created on execution."""
        with track(project="test", store=store, registry=registry) as run:
            backend = run.wrap(aer_simulator)
            backend.run(bell_circuit, shots=100).result()

        loaded = registry.load(run.run_id)
        kinds = _kinds(loaded)

        assert "qiskit.qpy.circuits" in kinds
        assert "result.counts.json" in kinds
        assert "devqubit.envelope.json" in kinds
