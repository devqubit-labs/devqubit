# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for Qiskit IBM Runtime adapter with UEC ExecutionEnvelope.

Uses real fake backends from qiskit_ibm_runtime.fake_provider.
Only mocks job execution since that requires network access.
"""

from __future__ import annotations

import json

from devqubit_engine.core.run import track
from devqubit_qiskit_runtime.adapter import QiskitRuntimeAdapter


def _load_envelopes(run_id, store, registry):
    """Helper to load run and extract envelopes."""
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


class TestQiskitRuntimeAdapter:
    """Tests for adapter registration and primitive detection."""

    def test_adapter_name(self):
        """Adapter has correct identifier."""
        assert QiskitRuntimeAdapter().name == "qiskit-runtime"

    def test_supports_sampler(self, fake_sampler):
        """Adapter supports Sampler primitives."""
        adapter = QiskitRuntimeAdapter()
        assert adapter.supports_executor(fake_sampler) is True

    def test_supports_estimator(self, fake_estimator):
        """Adapter supports Estimator primitives."""
        adapter = QiskitRuntimeAdapter()
        assert adapter.supports_executor(fake_estimator) is True

    def test_rejects_non_runtime(self):
        """Adapter rejects non-Runtime executors."""
        adapter = QiskitRuntimeAdapter()

        class FakeExecutor:
            __module__ = "some.other.module"

            def run(self):
                pass

        assert adapter.supports_executor(FakeExecutor()) is False
        assert adapter.supports_executor(None) is False

    def test_describe_sampler(self, fake_sampler):
        """Describes Sampler primitive correctly."""
        desc = QiskitRuntimeAdapter().describe_executor(fake_sampler)

        assert desc["provider"] == "fake"
        assert desc["primitive_type"] == "sampler"
        assert "name" in desc
        assert "type" in desc

    def test_describe_estimator(self, fake_estimator):
        """Describes Estimator primitive correctly."""
        desc = QiskitRuntimeAdapter().describe_executor(fake_estimator)

        assert desc["provider"] == "fake"  # For FakeEstimatorV2
        assert desc["primitive_type"] == "estimator"

    def test_wrap_executor_returns_tracked_primitive(
        self,
        store,
        registry,
        fake_sampler,
    ):
        """Wrapping returns a TrackedRuntimePrimitive."""
        adapter = QiskitRuntimeAdapter()

        with track(project="test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            assert hasattr(wrapped, "run")
            assert wrapped.primitive is fake_sampler


class TestSamplerExecution:
    """Tests for Sampler primitive execution and artifacts."""

    def test_first_run_logs_envelope_and_core_artifacts(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """First Sampler run produces complete UEC envelope + core artifacts."""
        adapter = QiskitRuntimeAdapter()

        with track(project="rt_sampler_smoke", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], shots=256)
            job.result()

        loaded, _ = _load_envelopes(run.run_id, store, registry)
        kinds = _kinds(loaded)

        # Core UEC + Runtime artifacts
        assert "qiskit_runtime.pubs.json" in kinds
        assert "result.qiskit_runtime.output.json" in kinds
        assert "result.counts.json" in kinds
        assert "devqubit.envelope.json" in kinds

        # Record reflects the logged execution
        assert loaded.record["execute"]["primitive_type"] == "sampler"
        assert loaded.record["execute"]["num_pubs"] == 1
        assert loaded.record["results"]["result_type"] == "counts"

    def test_envelope_structure_complete(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Envelope has all required sections with correct structure."""
        adapter = QiskitRuntimeAdapter()

        with track(project="envelope_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)
        assert len(envelopes) == 1
        envelope = envelopes[0]

        # UEC 1.0 schema and producer
        assert envelope["schema"] == "devqubit.envelope/1.0"
        assert "envelope_id" in envelope
        assert "producer" in envelope
        assert envelope["producer"]["adapter"] == "devqubit-qiskit-runtime"
        assert "qiskit-ibm-runtime" in envelope["producer"]["frontends"]

        # Device section - provider is physical (fake for fake backends)
        device = envelope["device"]
        assert device["provider"] in ("fake", "ibm_quantum", "aer", "local")
        valid_types = {"simulator", "hardware", "emulator", "sim", "hw", "qpu"}
        assert device["backend_type"] in valid_types
        assert "captured_at" in device
        assert device["num_qubits"] is not None

        # Program section
        assert "program" in envelope
        assert envelope["program"]["num_circuits"] >= 1

        # Execution section
        assert "execution" in envelope
        assert "transpilation" in envelope["execution"]

        # Result section (UEC 1.0)
        result = envelope["result"]
        assert result["success"] is True
        assert result["status"] == "completed"
        assert len(result.get("items", [])) >= 1
        # Primitive type is in metadata
        assert result["metadata"]["primitive_type"] == "sampler"


class TestEstimatorExecution:
    """Tests for Estimator primitive execution."""

    def test_estimator_logs_expectations_path(
        self,
        store,
        registry,
        fake_estimator,
        estimator_pub,
    ):
        """Estimator produces expectation values, not counts."""
        adapter = QiskitRuntimeAdapter()

        with track(project="rt_estimator_smoke", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_estimator, run)
            job = wrapped.run([estimator_pub])
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)
        kinds = _kinds(loaded)

        assert "result.qiskit_runtime.output.json" in kinds
        assert "devqubit.envelope.json" in kinds

        # Estimator should not produce counts artifact
        assert "result.counts.json" not in kinds
        assert loaded.record["execute"]["primitive_type"] == "estimator"

        # Verify expectation values are in the envelope result
        assert len(envelopes) == 1
        envelope = envelopes[0]
        result = envelope["result"]
        assert result["success"] is True
        # Estimator results have expectation items or metadata
        assert result["metadata"]["primitive_type"] == "estimator"


class TestSamplingBehavior:
    """Tests for execution sampling to prevent logging explosion."""

    def test_log_every_n_zero_skips_subsequent_identical_runs(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """
        Default behavior (log_every_n=0, log_new_circuits=True): only first execution
        of a circuit structure is logged. This protects training loops.
        """
        adapter = QiskitRuntimeAdapter()

        with track(project="rt_sampling_zero", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(
                fake_sampler,
                run,
                log_every_n=0,
                log_new_circuits=True,
            )

            wrapped.run([(bell_circuit,)], shots=128).result()
            wrapped.run([(bell_circuit,)], shots=128).result()

        loaded = registry.load(run.run_id)

        # Only one logged result/envelope despite two executions
        assert _count_kind(loaded, "result.qiskit_runtime.output.json") == 1
        assert _count_kind(loaded, "result.counts.json") == 1
        assert _count_kind(loaded, "devqubit.envelope.json") == 1
        assert _count_kind(loaded, "qiskit_runtime.pubs.json") == 1

    def test_log_every_n_two_logs_periodically(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """With log_every_n=2: logs on execution 1, 2, 4 (not 3)."""

        adapter = QiskitRuntimeAdapter()

        with track(project="rt_sampling_two", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(
                fake_sampler,
                run,
                log_every_n=2,
                log_new_circuits=False,
            )

            wrapped.run(
                [(bell_circuit,)], shots=64
            ).result()  # exec 1 -> logged (first)
            wrapped.run(
                [(bell_circuit,)], shots=64
            ).result()  # exec 2 -> logged (2%2=0)
            wrapped.run([(bell_circuit,)], shots=64).result()  # exec 3 -> NOT logged
            wrapped.run(
                [(bell_circuit,)], shots=64
            ).result()  # exec 4 -> logged (4%2=0)

        loaded = registry.load(run.run_id)

        # Should have 3 result artifacts (exec 1, 2, 4)
        assert _count_kind(loaded, "result.qiskit_runtime.output.json") == 3
        assert _count_kind(loaded, "devqubit.envelope.json") == 3
        # Structure logged only for first execution
        assert _count_kind(loaded, "qiskit_runtime.pubs.json") == 1

    def test_log_new_circuits_triggers_on_new_structure(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
        ghz_circuit,
    ):
        """log_new_circuits=True logs when circuit structure changes."""
        adapter = QiskitRuntimeAdapter()

        with track(project="rt_new_circuits", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(
                fake_sampler,
                run,
                log_every_n=0,
                log_new_circuits=True,
            )

            wrapped.run([(bell_circuit,)], shots=64).result()
            wrapped.run([(bell_circuit,)], shots=64).result()  # Same - not logged
            wrapped.run([(ghz_circuit,)], shots=64).result()  # New - logged

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "devqubit.envelope.json") == 2
        assert _count_kind(loaded, "qiskit_runtime.pubs.json") == 2


class TestResultIdempotency:
    """Tests for job.result() idempotency (fix M1)."""

    def test_result_called_twice_no_duplicate_artifacts(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Calling job.result() twice should NOT duplicate artifacts."""

        adapter = QiskitRuntimeAdapter()

        with track(project="idempotency_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])

            # Call result() twice
            result1 = job.result()
            result2 = job.result()

            # Should return same result object
            assert result1 is result2

        loaded = registry.load(run.run_id)

        # Should have exactly ONE of each artifact type
        assert _count_kind(loaded, "devqubit.envelope.json") == 1
        assert _count_kind(loaded, "result.qiskit_runtime.output.json") == 1
        assert _count_kind(loaded, "result.counts.json") == 1

    def test_result_snapshot_cached(self, store, registry, fake_sampler, bell_circuit):
        """TrackedRuntimeJob caches result_snapshot after first result() call."""
        adapter = QiskitRuntimeAdapter()

        with track(project="snapshot_cache", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])

            assert job.result_snapshot is None  # Before result()

            job.result()
            snapshot1 = job.result_snapshot

            job.result()  # Second call
            snapshot2 = job.result_snapshot

            assert snapshot1 is snapshot2  # Same cached snapshot


class TestMultiplePubs:
    """Tests for multi-PUB execution."""

    def test_batch_pubs_single_job(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
        ghz_circuit,
    ):
        """Multiple PUBs in single job are tracked correctly."""
        adapter = QiskitRuntimeAdapter()
        pubs = [(bell_circuit,), (ghz_circuit,)]

        with track(project="batch_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs)
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)

        assert loaded.record["execute"]["num_pubs"] == 2
        # UEC 1.0: check items count instead of num_experiments
        assert len(envelopes[0]["result"]["items"]) == 2

    def test_multi_circuit_qasm3_artifacts(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
        ghz_circuit,
        simple_circuit,
    ):
        """Each circuit gets its own QASM3 artifact with correct index."""

        adapter = QiskitRuntimeAdapter()
        pubs = [(bell_circuit,), (ghz_circuit,), (simple_circuit,)]

        with track(project="multi_qasm3", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs)
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)

        # Check envelope has correct number of circuits
        assert envelopes[0]["program"]["num_circuits"] == 3

        # Check that program.logical has artifacts for each circuit
        logical = envelopes[0]["program"].get("logical", [])

        # Should have at least 3 QASM3 artifacts (one per circuit)
        qasm3_artifacts = [a for a in logical if a.get("format") == "openqasm3"]
        assert (
            len(qasm3_artifacts) >= 3
        ), f"Expected at least 3 QASM3 artifacts (one per circuit), got {len(qasm3_artifacts)}"

        # Each should have a unique index 0, 1, 2
        indices = {a.get("index") for a in qasm3_artifacts}
        assert indices == {
            0,
            1,
            2,
        }, f"QASM3 artifacts have incorrect indices: {indices}"


class TestPerPubShots:
    """Tests for per-PUB shots handling."""

    def test_per_pub_shots_respected(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
        ghz_circuit,
    ):
        """Per-PUB shots override global shots."""

        adapter = QiskitRuntimeAdapter()

        # PUB with explicit shots=100, another with default (uses global 1024)
        pubs = [
            (bell_circuit, None, 100),  # Explicit 100 shots
            (ghz_circuit,),  # Uses global shots
        ]

        with track(project="per_pub_shots", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs, shots=1024)
            result = job.result()

        # Check that results reflect different shot counts
        pub_results = list(result)

        # First PUB should have ~100 shots
        pub0_shots = pub_results[0].metadata.get("shots", 0)
        assert pub0_shots == 100, f"First PUB should have 100 shots, got {pub0_shots}"

        # Second PUB should have global shots (1024)
        pub1_shots = pub_results[1].metadata.get("shots", 0)
        assert (
            pub1_shots == 1024
        ), f"Second PUB should have 1024 shots, got {pub1_shots}"


class TestTranspilationModes:
    """Tests for transpilation mode handling."""

    def test_manual_mode_logged(self, store, registry, fake_sampler, bell_circuit):
        """Manual transpilation mode is logged correctly."""
        adapter = QiskitRuntimeAdapter()

        with track(project="manual_mode", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], devqubit_transpilation_mode="manual")
            job.result()

        loaded = registry.load(run.run_id)
        assert loaded.record["execute"]["transpilation_mode"] == "manual"
        assert loaded.record["execute"]["transpiled_by_devqubit"] is False

    def test_auto_mode_default(self, store, registry, fake_sampler, bell_circuit):
        """Auto mode is the default."""
        adapter = QiskitRuntimeAdapter()

        with track(project="auto_default", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)
        assert loaded.record["execute"]["transpilation_mode"] == "auto"

    def test_transpilation_options_logged(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Transpilation options are captured in record."""
        adapter = QiskitRuntimeAdapter()

        with track(project="options_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(
                [(bell_circuit,)],
                devqubit_transpilation_mode="manual",
                devqubit_transpilation_options={"optimization_level": 2},
            )
            job.result()

        loaded = registry.load(run.run_id)
        opts = loaded.record["execute"].get("transpilation_options", {})
        assert opts.get("optimization_level") == 2

    def test_transpilation_in_envelope(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Transpilation info is captured in envelope."""
        adapter = QiskitRuntimeAdapter()

        with track(project="transpile_env", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], devqubit_transpilation_mode="manual")
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)
        execution = envelopes[0]["execution"]

        assert "transpilation" in execution
        assert execution["transpilation"]["mode"] == "manual"

    def test_auto_transpilation_for_non_isa(
        self,
        store,
        registry,
        fake_sampler,
        non_isa_circuit,
    ):
        """Auto mode transpiles non-ISA circuits and records metadata."""
        adapter = QiskitRuntimeAdapter()

        with track(project="rt_auto_transpile", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(non_isa_circuit,)], shots=32)
            job.result()

        loaded = registry.load(run.run_id)

        exec_rec = loaded.record["execute"]
        assert exec_rec["transpilation_mode"] == "auto"
        assert exec_rec["transpiled_by_devqubit"] is True
        assert exec_rec["transpilation_reason"] == "transpiled"
        # When devqubit transpiles, executed hashes are recorded
        assert exec_rec.get("executed_structural_hash") or exec_rec.get(
            "executed_parametric_hash"
        )


class TestNoBackendPrimitive:
    """Tests for primitives without backend (graceful degradation)."""

    def test_no_backend_produces_envelope(self, store, registry, bell_circuit):
        """
        Primitive without backend should still produce envelope with minimal info.

        This tests graceful degradation when backend resolution fails.
        """
        adapter = QiskitRuntimeAdapter()

        # Define mock classes inline (can't import from conftest at runtime)
        class MockPubResult:
            def __init__(self, counts):
                self._counts = counts
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

            def __getitem__(self, idx):
                return self._pub_results[idx]

        class MockJob:
            def __init__(self, result):
                self._result = result
                self._job_id = "mock-minimal-job"

            def result(self):
                return self._result

            def job_id(self):
                return self._job_id

            def status(self):
                return "DONE"

        class MinimalPrimitive:
            __module__ = "qiskit_ibm_runtime.sampler"

            def run(self, pubs, **kwargs):
                pub_results = [MockPubResult({"00": 50, "11": 50})]
                return MockJob(MockResult(pub_results))

        minimal = MinimalPrimitive()

        with track(project="no_backend", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(minimal, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)
        kinds = _kinds(loaded)

        # Should still produce core artifacts
        assert "devqubit.envelope.json" in kinds
        assert "result.counts.json" in kinds


class TestJoinDataFallback:
    """Tests for Sampler result extraction using join_data()."""

    def test_join_data_used_when_available(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Sampler result extraction uses join_data() when available."""

        adapter = QiskitRuntimeAdapter()

        with track(project="join_data", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            result = job.result()

        # Verify result was extracted successfully
        pub_result = list(result)[0]
        assert hasattr(pub_result, "join_data")

        # join_data() should return object with get_counts()
        joined = pub_result.join_data()
        assert hasattr(joined, "get_counts")
        counts = joined.get_counts()
        assert isinstance(counts, dict)
        assert sum(counts.values()) > 0


class TestExecutionSnapshotUECCompliance:
    """Tests for ExecutionSnapshot UEC compliance."""

    def test_submitted_at_captured(self, store, registry, fake_sampler, bell_circuit):
        """Execution captures submitted_at timestamp."""
        adapter = QiskitRuntimeAdapter()

        with track(project="timestamp_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)
        assert "submitted_at" in loaded.record["execute"]

    def test_transpilation_fields_captured(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Transpilation fields are captured in execution record."""
        adapter = QiskitRuntimeAdapter()

        with track(project="transpile_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)

        assert "transpilation_mode" in loaded.record["execute"]
        assert "transpiled_by_devqubit" in loaded.record["execute"]


class TestResultSnapshotUECCompliance:
    """Tests for ResultSnapshot UEC compliance."""

    def test_sampler_result_has_counts(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
    ):
        """Sampler result contains counts artifact."""
        adapter = QiskitRuntimeAdapter()

        with track(project="counts_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)

        assert "result.counts.json" in _kinds(loaded)
        assert loaded.record["results"]["num_experiments"] >= 1

    def test_estimator_result_has_expectations(
        self,
        store,
        registry,
        fake_estimator,
        estimator_pub,
    ):
        """Estimator result contains expectation values in envelope."""
        adapter = QiskitRuntimeAdapter()

        with track(project="expval_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_estimator, run)
            job = wrapped.run([estimator_pub])
            job.result()

        loaded, envelopes = _load_envelopes(run.run_id, store, registry)

        # Verify envelope has estimator result
        assert len(envelopes) == 1
        envelope = envelopes[0]
        result = envelope["result"]
        assert result["success"] is True
        assert result["metadata"]["primitive_type"] == "estimator"


class TestProgramSnapshotUECCompliance:
    """Tests for ProgramSnapshot UEC compliance."""

    def test_pubs_artifact_captured(self, store, registry, fake_sampler, bell_circuit):
        """PUBs are captured as program artifact."""
        adapter = QiskitRuntimeAdapter()

        with track(project="pubs_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)
        assert "qiskit_runtime.pubs.json" in _kinds(loaded)

    def test_num_pubs_tracked(
        self,
        store,
        registry,
        fake_sampler,
        bell_circuit,
        ghz_circuit,
    ):
        """Number of PUBs is tracked correctly."""
        adapter = QiskitRuntimeAdapter()
        pubs = [(bell_circuit,), (ghz_circuit,)]

        with track(project="num_pubs_test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run(pubs)
            job.result()

        loaded = registry.load(run.run_id)
        assert loaded.record["execute"]["num_pubs"] == 2
