# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Contract-focused tests for the Braket adapter."""

from __future__ import annotations

import json

from braket.circuits import Circuit
from devqubit_braket.adapter import (
    BraketAdapter,
    TrackedDevice,
    TrackedTaskBatch,
    _is_program_set,
    _materialize_task_spec,
)
from devqubit_engine.tracking.run import track


def _artifact_kinds(run_loaded) -> list[str]:
    return [a.kind for a in run_loaded.artifacts]


def _artifacts_of_kind(run_loaded, kind: str):
    return [a for a in run_loaded.artifacts if a.kind == kind]


def _read_artifact_json(store, artifact) -> dict:
    payload = store.get_bytes(artifact.digest)
    return json.loads(payload.decode("utf-8"))


def _measured_bell() -> Circuit:
    """Bell circuit with explicit measurements (more robust for counts)."""
    return Circuit().h(0).cnot(0, 1).measure([0, 1])


class TestBraketAdapterUECContract:
    """Tests for UEC envelope contract compliance."""

    def test_supports_and_wraps_local_simulator(self, store, registry, local_simulator):
        """Adapter has correct name and wraps LocalSimulator."""
        adapter = BraketAdapter()
        assert adapter.name == "braket"
        assert adapter.supports_executor(local_simulator) is True

        with track(
            project="braket_wrap",
            store=store,
            registry=registry,
        ) as run:
            wrapped = adapter.wrap_executor(local_simulator, run)
            assert isinstance(wrapped, TrackedDevice)
            assert wrapped.device is local_simulator

    def test_logged_execution(self, store, registry, local_simulator):
        """
        One logged execution should produce:
        - program artifacts (JAQCD + OpenQASM + diagram)
        - raw result artifact + normalized counts artifact
        - a complete devqubit.envelope.json artifact referencing the program ref(s)
        """
        adapter = BraketAdapter()
        circuit = _measured_bell()
        shots = 25

        with track(
            project="braket_envelope_contract",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(local_simulator, run)
            task = device.run(circuit, shots=shots)
            result = task.result()

            # Sanity: real execution happened
            counts = dict(result.measurement_counts)
            assert sum(counts.values()) == shots

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"

        # Tags/params are part of the adapter contract (only set on logged executions)
        assert loaded.record["data"]["tags"]["provider"] == "aws_braket"
        assert loaded.record["data"]["tags"]["adapter"] == "devqubit-braket"
        assert loaded.record["data"]["params"]["shots"] == shots
        assert loaded.record["data"]["params"]["num_circuits"] == 1

        kinds = _artifact_kinds(loaded)
        assert "braket.ir.jaqcd" in kinds
        assert "braket.ir.openqasm" in kinds
        assert "braket.circuits.diagram" in kinds
        assert "result.braket.raw.json" in kinds
        assert "result.counts.json" in kinds
        assert "devqubit.envelope.json" in kinds

        # Validate envelope structure + key fields (UEC 1.0 contract)
        envelope_art = _artifacts_of_kind(loaded, "devqubit.envelope.json")[0]
        envelope = _read_artifact_json(store, envelope_art)

        assert envelope["schema"] == "devqubit.envelope/1.0"
        assert envelope["producer"]["adapter"] == "devqubit-braket"
        assert envelope["producer"]["sdk"] == "braket"
        assert envelope["device"]["provider"] == "local"
        assert envelope["execution"]["shots"] == shots
        assert envelope["execution"]["sdk"] == "braket"
        assert envelope["execution"]["transpilation"]["mode"] == "managed"

        # Program snapshot should reference all logged artifacts
        logical = envelope["program"]["logical"]
        assert len(logical) == 3  # JAQCD + OpenQASM + diagram

        # Verify all formats are present
        formats = {art["format"] for art in logical}
        assert "jaqcd" in formats
        assert "openqasm3" in formats
        assert "diagram" in formats

        # JAQCD artifact should be properly referenced
        jaqcd_artifacts = _artifacts_of_kind(loaded, "braket.ir.jaqcd")
        jaqcd_digests = {a.digest for a in jaqcd_artifacts}
        jaqcd_in_logical = [art for art in logical if art["format"] == "jaqcd"]
        assert len(jaqcd_in_logical) == 1
        assert jaqcd_in_logical[0]["ref"]["digest"] in jaqcd_digests

    def test_default_logs_first_only(self, store, registry, local_simulator):
        """Default behavior (log_every_n=0) logs the first execution only."""
        adapter = BraketAdapter()
        circuit = _measured_bell()

        with track(
            project="braket_first_only",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(
                local_simulator, run
            )  # default log_every_n=0
            for _ in range(3):
                task = device.run(circuit, shots=10)
                res = task.result()
                assert sum(dict(res.measurement_counts).values()) == 10

        loaded = registry.load(run.run_id)

        # Only one logged envelope + one logged raw-result + one logged counts
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 1
        assert len(_artifacts_of_kind(loaded, "result.braket.raw.json")) == 1
        assert len(_artifacts_of_kind(loaded, "result.counts.json")) == 1

    def test_log_every_n_sampling(self, store, registry, local_simulator):
        """
        log_every_n=N logs: first execution + every Nth after that.
        For 5 executions with N=2 => exec 1, 2, 4 are logged => 3 envelopes.
        """
        adapter = BraketAdapter()
        circuit = _measured_bell()

        with track(
            project="braket_every_2",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(
                local_simulator,
                run,
                log_every_n=2,
            )
            for _ in range(5):
                device.run(circuit, shots=5).result()

        loaded = registry.load(run.run_id)
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 3

    def test_log_new_circuits_additional_envelope(
        self,
        store,
        registry,
        local_simulator,
    ):
        """
        With log_every_n=0 (first only) + log_new_circuits=True:
        - first circuit logs
        - repeated same circuit does not
        - a structurally different circuit logs once when first seen
        """
        adapter = BraketAdapter()
        c1 = Circuit().h(0).cnot(0, 1).measure([0, 1])
        c2 = Circuit().x(0).cnot(0, 1).measure([0, 1])  # different structure

        with track(
            project="braket_new_circuits", store=store, registry=registry
        ) as run:
            device = adapter.wrap_executor(
                local_simulator,
                run,
                log_every_n=0,
                log_new_circuits=True,
            )
            device.run(c1, shots=5).result()
            device.run(c1, shots=5).result()
            device.run(c2, shots=5).result()

        loaded = registry.load(run.run_id)
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 2


class TestShotsHandling:
    """Tests for shots parameter handling."""

    def test_shots_none_not_passed_to_device(self, store, registry, device_factory):
        """
        When shots=None, shots should NOT be passed to device.run().
        This allows Braket to use its default (1000 for QPU, 0 for simulator).
        """
        mock_device = device_factory(name="shots_test", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="shots_none", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuit = Circuit().h(0).measure(0)

            # Call with shots=None (default)
            device.run(circuit)

        # Verify shots was NOT in kwargs
        assert len(mock_device._run_calls) == 1
        call = mock_device._run_calls[0]
        assert "shots" not in call["kwargs"]

    def test_shots_explicit_passed_to_device(self, store, registry, device_factory):
        """When shots is explicitly provided, it should be passed to device.run()."""
        mock_device = device_factory(name="shots_explicit", qubit_count=2)
        adapter = BraketAdapter()

        with track(
            project="shots_explicit",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuit = Circuit().h(0).measure(0)

            device.run(circuit, shots=500)

        assert len(mock_device._run_calls) == 1
        call = mock_device._run_calls[0]
        assert call["kwargs"].get("shots") == 500


class TestProgramSetHandling:
    """Tests for ProgramSet task specification handling."""

    def test_is_program_set_detection(self, mock_program_set):
        """Correctly detects ProgramSet objects."""
        assert _is_program_set(mock_program_set) is True
        assert _is_program_set(Circuit().h(0)) is False
        assert _is_program_set([Circuit().h(0)]) is False
        assert _is_program_set(None) is False

    def test_program_set_sent_as_is(
        self,
        store,
        registry,
        device_factory,
        mock_program_set,
    ):
        """
        ProgramSet should be sent to device.run() as-is, NOT converted to list.

        This is critical because ProgramSet is a special task specification
        that Braket handles differently from a list of circuits.
        """
        mock_device = device_factory(
            name="program_set_test",
            qubit_count=2,
        )
        adapter = BraketAdapter()

        with track(
            project="program_set",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(mock_device, run)
            device.run(mock_program_set, shots=100)

        # Verify ProgramSet was sent as-is (check marker)
        assert len(mock_device._run_calls) == 1
        call = mock_device._run_calls[0]
        task_spec = call["task_spec"]

        # Should be the original ProgramSet, not a list
        assert hasattr(task_spec, "marker")
        assert task_spec.marker == "test_program_set"

    def test_materialize_task_spec_preserves_program_set(self, mock_program_set):
        """_materialize_task_spec preserves ProgramSet for submission."""
        run_payload, circuits_for_logging, was_single, extra_meta = (
            _materialize_task_spec(mock_program_set)
        )

        # run_payload should be the original ProgramSet
        assert run_payload is mock_program_set

        # circuits_for_logging should be extracted circuits
        assert len(circuits_for_logging) == 2

        # Not a single circuit
        assert was_single is False

        # Extra metadata should include ProgramSet info
        assert extra_meta is not None
        assert extra_meta.get("is_program_set") is True
        assert extra_meta.get("total_executables") == 2


class TestRunBatch:
    """Tests for run_batch support."""

    def test_run_batch_returns_tracked_batch(self, store, registry, device_factory):
        """run_batch returns TrackedTaskBatch wrapper."""
        mock_device = device_factory(name="batch_test", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="run_batch", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuits = [Circuit().h(0).measure(0), Circuit().x(0).measure(0)]

            batch = device.run_batch(circuits, shots=100)

            assert isinstance(batch, TrackedTaskBatch)

    def test_run_batch_calls_device_run_batch(self, store, registry, device_factory):
        """run_batch delegates to device.run_batch, not device.run."""
        mock_device = device_factory(name="batch_delegate", qubit_count=2)
        adapter = BraketAdapter()

        with track(
            project="run_batch_delegate",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuits = [Circuit().h(0).measure(0), Circuit().x(0).measure(0)]

            device.run_batch(circuits, shots=100)

        # run_batch should be called, not run
        assert len(mock_device._run_batch_calls) == 1
        assert len(mock_device._run_calls) == 0

        call = mock_device._run_batch_calls[0]
        assert len(call["task_specs"]) == 2
        assert call["kwargs"].get("shots") == 100

    def test_run_batch_logs_envelope(self, store, registry, device_factory):
        """run_batch creates execution envelope."""
        mock_device = device_factory(name="batch_envelope", qubit_count=2)
        adapter = BraketAdapter()

        with track(
            project="run_batch_envelope",
            store=store,
            registry=registry,
        ) as run:
            device = adapter.wrap_executor(mock_device, run, log_every_n=-1)
            circuits = [Circuit().h(0).measure(0), Circuit().x(0).measure(0)]

            batch = device.run_batch(circuits, shots=100)
            batch.results()

        loaded = registry.load(run.run_id)
        envelopes = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(envelopes) == 1


class TestDeterministicArtifacts:
    """Tests for artifact digest determinism."""

    def test_same_circuit_same_jaqcd_digest(self, store, registry, local_simulator):
        """Same circuit produces same JAQCD artifact digest across runs."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).cnot(0, 1).measure([0, 1])

        digests = []
        for i in range(2):
            with track(
                project=f"deterministic_{i}",
                store=store,
                registry=registry,
            ) as run:
                device = adapter.wrap_executor(local_simulator, run)
                device.run(circuit, shots=10).result()

            loaded = registry.load(run.run_id)
            jaqcd = _artifacts_of_kind(loaded, "braket.ir.jaqcd")[0]
            digests.append(jaqcd.digest)

        # Same circuit -> same digest
        assert digests[0] == digests[1]

    def test_different_circuits_different_digests(
        self,
        store,
        registry,
        local_simulator,
    ):
        """Different circuits produce different artifact digests."""
        adapter = BraketAdapter()
        c1 = Circuit().h(0).measure(0)
        c2 = Circuit().x(0).measure(0)

        with track(
            project="diff_c1",
            store=store,
            registry=registry,
        ) as run1:
            device = adapter.wrap_executor(local_simulator, run1)
            device.run(c1, shots=10).result()

        with track(
            project="diff_c2",
            store=store,
            registry=registry,
        ) as run2:
            device = adapter.wrap_executor(local_simulator, run2)
            device.run(c2, shots=10).result()

        loaded1 = registry.load(run1.run_id)
        loaded2 = registry.load(run2.run_id)

        digest1 = _artifacts_of_kind(loaded1, "braket.ir.jaqcd")[0].digest
        digest2 = _artifacts_of_kind(loaded2, "braket.ir.jaqcd")[0].digest

        assert digest1 != digest2
