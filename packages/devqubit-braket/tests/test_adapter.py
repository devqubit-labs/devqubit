# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
End-to-end tests for the Braket adapter.

These tests verify the complete flow: circuit → tracked device → execution → results.
They use real LocalSimulator where possible for realistic validation.
"""

from __future__ import annotations

import json

from braket.circuits import Circuit
from devqubit_braket.adapter import BraketAdapter, TrackedDevice, TrackedTaskBatch
from devqubit_engine.tracking.run import track


# =============================================================================
# Test Helpers
# =============================================================================


def _artifact_kinds(run_loaded) -> list[str]:
    """Get list of artifact kinds from a loaded run."""
    return [a.kind for a in run_loaded.artifacts]


def _artifacts_of_kind(run_loaded, kind: str):
    """Get artifacts of a specific kind."""
    return [a for a in run_loaded.artifacts if a.kind == kind]


def _read_artifact_json(store, artifact) -> dict:
    """Read and parse JSON artifact."""
    payload = store.get_bytes(artifact.digest)
    return json.loads(payload.decode("utf-8"))


# =============================================================================
# UEC Contract Tests
# =============================================================================


class TestUECContract:
    """
    Tests for UEC (Uniform Execution Contract) compliance.

    These verify that the adapter produces correct envelope structure
    with all required artifacts and metadata.
    """

    def test_full_execution_flow(self, store, registry, local_simulator):
        """
        Complete execution flow produces all required artifacts.

        Verifies:
        - Program artifacts (JAQCD, OpenQASM, diagram)
        - Result artifacts (raw result, counts)
        - Envelope with correct structure and references
        """
        adapter = BraketAdapter()
        circuit = Circuit().h(0).cnot(0, 1).measure([0, 1])
        shots = 50

        with track(project="uec_contract", store=store, registry=registry) as run:
            device = adapter.wrap_executor(local_simulator, run)
            task = device.run(circuit, shots=shots)
            result = task.result()

            # Verify real execution
            counts = dict(result.measurement_counts)
            assert sum(counts.values()) == shots

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"

        # Verify required artifacts
        kinds = _artifact_kinds(loaded)
        assert "braket.ir.jaqcd" in kinds
        assert "braket.ir.openqasm" in kinds
        assert "braket.circuits.diagram" in kinds
        assert "result.braket.raw.json" in kinds
        assert "result.counts.json" in kinds
        assert "devqubit.envelope.json" in kinds

        # Verify envelope structure
        envelope_art = _artifacts_of_kind(loaded, "devqubit.envelope.json")[0]
        envelope = _read_artifact_json(store, envelope_art)

        assert envelope["schema"] == "devqubit.envelope/1.0"
        assert envelope["producer"]["adapter"] == "devqubit-braket"
        assert envelope["producer"]["sdk"] == "braket"
        assert envelope["device"]["provider"] == "local"
        assert envelope["execution"]["shots"] == shots
        assert envelope["execution"]["transpilation"]["mode"] == "managed"

        # Verify program artifacts are referenced in envelope
        logical = envelope["program"]["logical"]
        assert len(logical) == 3  # JAQCD + OpenQASM + diagram
        formats = {art["format"] for art in logical}
        assert formats == {"jaqcd", "openqasm3", "diagram"}

    def test_tags_and_params_logged(self, store, registry, local_simulator):
        """Execution logs correct tags and parameters."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).measure(0)
        shots = 25

        with track(project="tags_params", store=store, registry=registry) as run:
            device = adapter.wrap_executor(local_simulator, run)
            device.run(circuit, shots=shots).result()

        loaded = registry.load(run.run_id)

        assert loaded.record["data"]["tags"]["provider"] == "aws_braket"
        assert loaded.record["data"]["tags"]["adapter"] == "devqubit-braket"
        assert loaded.record["data"]["params"]["shots"] == shots
        assert loaded.record["data"]["params"]["num_circuits"] == 1


# =============================================================================
# Logging Frequency Tests
# =============================================================================


class TestLoggingFrequency:
    """Tests for configurable logging frequency."""

    def test_default_logs_first_only(self, store, registry, local_simulator):
        """Default behavior (log_every_n=0) logs first execution only."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).measure(0)

        with track(project="first_only", store=store, registry=registry) as run:
            device = adapter.wrap_executor(local_simulator, run)
            for _ in range(3):
                device.run(circuit, shots=10).result()

        loaded = registry.load(run.run_id)

        # Only one envelope logged
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 1

    def test_log_every_n_samples_correctly(self, store, registry, local_simulator):
        """log_every_n=2 logs executions 1, 2, 4 (3 total for 5 executions)."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).measure(0)

        with track(project="every_n", store=store, registry=registry) as run:
            device = adapter.wrap_executor(local_simulator, run, log_every_n=2)
            for _ in range(5):
                device.run(circuit, shots=5).result()

        loaded = registry.load(run.run_id)
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 3

    def test_log_new_circuits_detects_structure_changes(
        self, store, registry, local_simulator
    ):
        """log_new_circuits=True logs when circuit structure changes."""
        adapter = BraketAdapter()
        c1 = Circuit().h(0).measure(0)
        c2 = Circuit().x(0).measure(0)  # Different structure

        with track(project="new_circuits", store=store, registry=registry) as run:
            device = adapter.wrap_executor(
                local_simulator, run, log_every_n=0, log_new_circuits=True
            )
            device.run(c1, shots=5).result()  # First - logged
            device.run(c1, shots=5).result()  # Same - not logged
            device.run(c2, shots=5).result()  # New structure - logged

        loaded = registry.load(run.run_id)
        assert len(_artifacts_of_kind(loaded, "devqubit.envelope.json")) == 2


# =============================================================================
# Batch Execution Tests
# =============================================================================


class TestBatchExecution:
    """Tests for batch execution support."""

    def test_run_batch_returns_tracked_batch(self, store, registry, device_factory):
        """run_batch returns TrackedTaskBatch wrapper."""
        mock_device = device_factory(name="batch_test", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="batch", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuits = [Circuit().h(0).measure(0), Circuit().x(0).measure(0)]

            batch = device.run_batch(circuits, shots=100)

            assert isinstance(batch, TrackedTaskBatch)

    def test_run_batch_delegates_correctly(self, store, registry, device_factory):
        """run_batch calls device.run_batch, not device.run."""
        mock_device = device_factory(name="batch_delegate", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="batch_delegate", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            circuits = [Circuit().h(0), Circuit().x(0)]

            device.run_batch(circuits, shots=100)

        assert len(mock_device._run_batch_calls) == 1
        assert len(mock_device._run_calls) == 0
        assert mock_device._run_batch_calls[0]["kwargs"].get("shots") == 100


# =============================================================================
# ProgramSet Handling Tests
# =============================================================================


class TestProgramSetHandling:
    """Tests for ProgramSet task specification handling."""

    def test_program_set_sent_as_is(
        self, store, registry, device_factory, mock_program_set
    ):
        """
        ProgramSet is sent to device.run() unchanged.

        Critical: ProgramSet must not be converted to list - Braket
        handles it specially.
        """
        mock_device = device_factory(name="program_set", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="program_set", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            device.run(mock_program_set, shots=100)

        call = mock_device._run_calls[0]
        task_spec = call["task_spec"]

        # Should be original ProgramSet, verified by marker
        assert hasattr(task_spec, "marker")
        assert task_spec.marker == "test_program_set"


# =============================================================================
# Shots Handling Tests
# =============================================================================


class TestShotsHandling:
    """Tests for shots parameter handling."""

    def test_shots_none_uses_device_default(self, store, registry, device_factory):
        """shots=None lets device use its default."""
        mock_device = device_factory(name="shots_none", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="shots_none", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            device.run(Circuit().h(0))

        call = mock_device._run_calls[0]
        assert "shots" not in call["kwargs"]

    def test_shots_explicit_passed_through(self, store, registry, device_factory):
        """Explicit shots value is passed to device."""
        mock_device = device_factory(name="shots_explicit", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="shots_explicit", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            device.run(Circuit().h(0), shots=500)

        call = mock_device._run_calls[0]
        assert call["kwargs"]["shots"] == 500


# =============================================================================
# Artifact Determinism Tests
# =============================================================================


class TestArtifactDeterminism:
    """Tests for artifact digest determinism (critical for deduplication)."""

    def test_same_circuit_same_digest(self, store, registry, local_simulator):
        """Same circuit produces identical JAQCD digest across runs."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).cnot(0, 1).measure([0, 1])

        digests = []
        for i in range(2):
            with track(
                project=f"deterministic_{i}", store=store, registry=registry
            ) as run:
                device = adapter.wrap_executor(local_simulator, run)
                device.run(circuit, shots=10).result()

            loaded = registry.load(run.run_id)
            jaqcd = _artifacts_of_kind(loaded, "braket.ir.jaqcd")[0]
            digests.append(jaqcd.digest)

        assert digests[0] == digests[1]

    def test_different_circuits_different_digests(
        self, store, registry, local_simulator
    ):
        """Different circuits produce different artifact digests."""
        adapter = BraketAdapter()
        c1 = Circuit().h(0).measure(0)
        c2 = Circuit().x(0).measure(0)

        digests = []
        for i, circuit in enumerate([c1, c2]):
            with track(project=f"diff_{i}", store=store, registry=registry) as run:
                device = adapter.wrap_executor(local_simulator, run)
                device.run(circuit, shots=10).result()

            loaded = registry.load(run.run_id)
            jaqcd = _artifacts_of_kind(loaded, "braket.ir.jaqcd")[0]
            digests.append(jaqcd.digest)

        assert digests[0] != digests[1]


# =============================================================================
# Adapter Interface Tests
# =============================================================================


class TestAdapterInterface:
    """Tests for BraketAdapter interface."""

    def test_adapter_properties(self, local_simulator):
        """Adapter has correct name and supports LocalSimulator."""
        adapter = BraketAdapter()

        assert adapter.name == "braket"
        assert adapter.supports_executor(local_simulator) is True
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("not a device") is False

    def test_wrap_executor_returns_tracked_device(
        self, store, registry, local_simulator
    ):
        """wrap_executor returns TrackedDevice wrapper."""
        adapter = BraketAdapter()

        with track(project="wrap", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(local_simulator, run)

            assert isinstance(wrapped, TrackedDevice)
            assert wrapped.device is local_simulator

    def test_describe_executor(self, local_simulator):
        """describe_executor returns device info."""
        adapter = BraketAdapter()
        desc = adapter.describe_executor(local_simulator)

        assert "name" in desc
        assert desc["provider"] == "aws_braket"
