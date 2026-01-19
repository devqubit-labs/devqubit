# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""End-to-end tests for Qiskit Runtime execution and result tracking."""

from __future__ import annotations

import json

from devqubit_engine.tracking.run import track
from devqubit_qiskit_runtime.adapter import QiskitRuntimeAdapter
from devqubit_qiskit_runtime.circuits import compute_structural_hash


# =============================================================================
# Test Helpers
# =============================================================================


def _load_envelopes(run_id, store, registry):
    """Load run and extract envelope artifacts."""
    loaded = registry.load(run_id)
    env_artifacts = [a for a in loaded.artifacts if a.kind == "devqubit.envelope.json"]
    envelopes = []
    for a in env_artifacts:
        raw = store.get_bytes(a.digest)
        envelopes.append(json.loads(raw.decode("utf-8")))
    return loaded, envelopes


def _kinds(loaded) -> set[str]:
    """Extract artifact kinds from loaded run."""
    return {a.kind for a in getattr(loaded, "artifacts", [])}


def _count_kind(loaded, kind: str) -> int:
    """Count artifacts of a specific kind."""
    return sum(1 for a in getattr(loaded, "artifacts", []) if a.kind == kind)


# =============================================================================
# Sampler Execution Tests
# =============================================================================


class TestSamplerExecution:
    """End-to-end tests for Sampler primitive."""

    def test_sampler_produces_envelope_and_artifacts(
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Sampler execution produces complete UEC envelope with counts."""
        adapter = QiskitRuntimeAdapter()

        with track(project="sampler_e2e", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], shots=256)
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)
        kinds = _kinds(loaded)

        # Core artifacts produced
        assert "devqubit.envelope.json" in kinds
        assert "qiskit_runtime.pubs.json" in kinds
        assert "result.qiskit_runtime.output.json" in kinds
        assert "result.counts.json" in kinds

        # Record populated
        assert loaded.record["execute"]["primitive_type"] == "sampler"
        assert loaded.record["execute"]["num_pubs"] == 1
        assert loaded.record["results"]["result_type"] == "counts"

        # Single envelope
        assert len(envelopes) == 1

    def test_sampler_envelope_structure(
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Sampler envelope has all required UEC sections."""
        adapter = QiskitRuntimeAdapter()

        with track(project="envelope_struct", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)
        envelope = envelopes[0]

        # Schema and ID
        assert envelope["schema"] == "devqubit.envelope/1.0"
        assert "envelope_id" in envelope

        # Producer
        assert envelope["producer"]["adapter"] == "devqubit-qiskit-runtime"
        assert "qiskit-ibm-runtime" in envelope["producer"]["frontends"]

        # Device
        device = envelope["device"]
        assert device["provider"] in ("fake", "ibm_quantum", "aer", "local")
        assert device["backend_type"] in {"simulator", "hardware"}
        assert device["num_qubits"] is not None

        # Program
        assert envelope["program"]["num_circuits"] >= 1

        # Execution
        assert "transpilation" in envelope["execution"]

        # Result
        result = envelope["result"]
        assert result["success"] is True
        assert result["status"] == "completed"
        assert len(result["items"]) >= 1
        assert result["metadata"]["primitive_type"] == "sampler"

    def test_sampler_multi_pub_execution(
        self, store, registry, fake_sampler, bell_circuit, ghz_circuit
    ):
        """Multiple PUBs in single job are tracked correctly."""
        adapter = QiskitRuntimeAdapter()
        pubs = [(bell_circuit,), (ghz_circuit,)]

        with track(project="multi_pub", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs)
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)

        assert loaded.record["execute"]["num_pubs"] == 2
        assert len(envelopes[0]["result"]["items"]) == 2
        assert envelopes[0]["program"]["num_circuits"] == 2

    def test_sampler_per_pub_shots(
        self, store, registry, fake_sampler, bell_circuit, ghz_circuit
    ):
        """Per-PUB shots override global shots."""
        adapter = QiskitRuntimeAdapter()
        pubs = [
            (bell_circuit, None, 100),  # 100 shots
            (ghz_circuit,),  # Uses global
        ]

        with track(project="per_pub_shots", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs, shots=1024)
            result = job.result()

        pub_results = list(result)
        assert pub_results[0].metadata.get("shots") == 100
        assert pub_results[1].metadata.get("shots") == 1024


# =============================================================================
# Estimator Execution Tests
# =============================================================================


class TestEstimatorExecution:
    """End-to-end tests for Estimator primitive."""

    def test_estimator_produces_expectations(
        self, store, registry, fake_estimator, estimator_pub
    ):
        """Estimator produces expectation values, not counts."""
        adapter = QiskitRuntimeAdapter()

        with track(project="estimator_e2e", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_estimator, run)
            job = wrapped.run([estimator_pub])
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)
        kinds = _kinds(loaded)

        # Produces estimator artifacts, not counts
        assert "devqubit.envelope.json" in kinds
        assert "result.qiskit_runtime.output.json" in kinds
        assert "result.counts.json" not in kinds

        assert loaded.record["execute"]["primitive_type"] == "estimator"

        # Envelope result
        result = envelopes[0]["result"]
        assert result["success"] is True
        assert result["metadata"]["primitive_type"] == "estimator"


# =============================================================================
# Sampling Behavior Tests
# =============================================================================


class TestSamplingBehavior:
    """Tests for execution sampling to prevent logging explosion."""

    def test_default_logs_first_only(self, store, registry, fake_sampler, bell_circuit):
        """Default (log_every_n=0): only first execution logged."""
        adapter = QiskitRuntimeAdapter()

        with track(project="sampling_default", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run, log_every_n=0)

            wrapped.run([(bell_circuit,)]).result()
            wrapped.run([(bell_circuit,)]).result()
            wrapped.run([(bell_circuit,)]).result()

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "devqubit.envelope.json") == 1
        assert _count_kind(loaded, "result.counts.json") == 1

    def test_log_every_n_periodic(self, store, registry, fake_sampler, bell_circuit):
        """log_every_n=2: logs on executions 1, 2, 4."""
        adapter = QiskitRuntimeAdapter()

        with track(project="sampling_periodic", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(
                fake_sampler, run, log_every_n=2, log_new_circuits=False
            )

            wrapped.run([(bell_circuit,)]).result()  # 1 -> logged
            wrapped.run([(bell_circuit,)]).result()  # 2 -> logged
            wrapped.run([(bell_circuit,)]).result()  # 3 -> skip
            wrapped.run([(bell_circuit,)]).result()  # 4 -> logged

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "devqubit.envelope.json") == 3

    def test_log_new_circuits_on_structure_change(
        self, store, registry, fake_sampler, bell_circuit, ghz_circuit
    ):
        """log_new_circuits=True: logs when circuit structure changes."""
        adapter = QiskitRuntimeAdapter()

        with track(project="new_circuits", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(
                fake_sampler, run, log_every_n=0, log_new_circuits=True
            )

            wrapped.run([(bell_circuit,)]).result()  # New -> logged
            wrapped.run([(bell_circuit,)]).result()  # Same -> skip
            wrapped.run([(ghz_circuit,)]).result()  # New -> logged

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "devqubit.envelope.json") == 2


# =============================================================================
# Result Idempotency Tests
# =============================================================================


class TestResultIdempotency:
    """Tests for job.result() idempotency."""

    def test_result_called_twice_no_duplicates(
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Calling job.result() twice does NOT duplicate artifacts."""
        adapter = QiskitRuntimeAdapter()

        with track(project="idempotency", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])

            result1 = job.result()
            result2 = job.result()

            assert result1 is result2

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "devqubit.envelope.json") == 1
        assert _count_kind(loaded, "result.counts.json") == 1

    def test_result_snapshot_cached(self, store, registry, fake_sampler, bell_circuit):
        """TrackedRuntimeJob caches result_snapshot after first call."""
        adapter = QiskitRuntimeAdapter()

        with track(project="snapshot_cache", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])

            assert job.result_snapshot is None
            job.result()
            snapshot1 = job.result_snapshot
            job.result()
            snapshot2 = job.result_snapshot

            assert snapshot1 is snapshot2
            assert snapshot1 is not None


# =============================================================================
# Circuit Hashing Tests
# =============================================================================


class TestCircuitHashing:
    """Tests for circuit hash computation and deduplication."""

    def test_identical_circuits_same_hash(self, bell_circuit):
        """Identical circuits produce identical structural hash."""
        from qiskit import QuantumCircuit

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure_all()

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure_all()

        assert compute_structural_hash([qc1]) == compute_structural_hash([qc2])

    def test_different_circuits_different_hash(self):
        """Different circuits produce different hashes."""
        from qiskit import QuantumCircuit

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.x(0)
        qc2.cz(0, 1)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_qubit_order_matters(self):
        """CX(0,1) and CX(1,0) produce different hashes."""
        from qiskit import QuantumCircuit

        qc1 = QuantumCircuit(2)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(2)
        qc2.cx(1, 0)

        assert compute_structural_hash([qc1]) != compute_structural_hash([qc2])

    def test_hash_format(self, bell_circuit):
        """Hash follows sha256:<hex> format."""
        h = compute_structural_hash([bell_circuit])

        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


# =============================================================================
# Device Snapshot Tests
# =============================================================================


class TestDeviceSnapshot:
    """Tests for device snapshot capture."""

    def test_snapshot_captures_backend_info(
        self, store, registry, fake_sampler, bell_circuit, fake_backend
    ):
        """Device snapshot captures backend topology and gates."""
        adapter = QiskitRuntimeAdapter()

        with track(project="device_snapshot", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)
        device = envelopes[0]["device"]

        assert device["num_qubits"] == fake_backend.num_qubits
        assert device["connectivity"] is not None
        assert len(device["connectivity"]) > 0
        assert device["native_gates"] is not None
        assert "cx" in device["native_gates"] or "ecr" in device["native_gates"]

    def test_backend_type_is_simulator_for_fake(
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Fake backends return 'simulator' as backend_type."""
        adapter = QiskitRuntimeAdapter()

        with track(project="backend_type", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)

        assert envelopes[0]["device"]["backend_type"] == "simulator"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_primitive_without_backend_produces_envelope(
        self, store, registry, bell_circuit
    ):
        """Primitive without backend still produces envelope."""
        adapter = QiskitRuntimeAdapter()

        class MockPubResult:
            def __init__(self, counts):
                self.metadata = {"shots": sum(counts.values())}

                class BitArray:
                    def __init__(ba_self, c):
                        ba_self._counts = c

                    def get_counts(ba_self):
                        return ba_self._counts

                class DataBin:
                    def __init__(db_self, c):
                        db_self.meas = BitArray(c)

                self.data = DataBin(counts)

        class MockResult:
            def __init__(self, pub_results):
                self._pub_results = pub_results

            def __iter__(self):
                return iter(self._pub_results)

            def __len__(self):
                return len(self._pub_results)

        class MockJob:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

            def job_id(self):
                return "mock-job"

        class MinimalPrimitive:
            __module__ = "qiskit_ibm_runtime.sampler"

            def run(self, pubs, **kwargs):
                return MockJob(MockResult([MockPubResult({"00": 50, "11": 50})]))

        with track(project="no_backend", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(MinimalPrimitive(), run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)
        kinds = _kinds(loaded)

        assert "devqubit.envelope.json" in kinds
        assert "result.counts.json" in kinds

    def test_empty_pubs_list(self, store, registry, fake_sampler):
        """Empty PUBs list is handled gracefully."""
        adapter = QiskitRuntimeAdapter()

        with track(project="empty_pubs", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([])
            job.result()

        loaded = registry.load(run.run_id)

        assert loaded.record["execute"]["num_pubs"] == 0
