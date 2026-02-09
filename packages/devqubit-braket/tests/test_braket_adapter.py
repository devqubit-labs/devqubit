# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
End-to-end tests for the Braket adapter.

These tests verify the complete flow: circuit → tracked device → execution → results.
They use real LocalSimulator where possible for realistic validation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from braket.circuits import Circuit
from devqubit_braket.adapter import (
    BraketAdapter,
    TrackedDevice,
    TrackedTaskBatch,
    _extract_circuits_from_program_set,
    _get_program_set_metadata,
    _is_program_set,
    _materialize_task_spec,
)
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

        assert loaded.record["data"]["tags"]["provider"] == "local"
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
        assert desc["provider"] == "local"  # LocalSimulator is local, not aws_braket


# =============================================================================
# Provider Detection Tests
# =============================================================================


class TestProviderDetection:
    """Tests for accurate provider detection in tags and envelope."""

    def test_local_simulator_has_local_provider(self, store, registry, local_simulator):
        """LocalSimulator is tagged with provider='local', not 'aws_braket'."""
        adapter = BraketAdapter()
        circuit = Circuit().h(0).measure(0)

        with track(project="local_provider", store=store, registry=registry) as run:
            device = adapter.wrap_executor(local_simulator, run)
            device.run(circuit, shots=10).result()

        loaded = registry.load(run.run_id)

        # Tags should reflect actual provider
        assert loaded.record["data"]["tags"]["provider"] == "local"
        assert loaded.record["backend"]["provider"] == "local"

        # Envelope should have correct provider
        envelope_art = _artifacts_of_kind(loaded, "devqubit.envelope.json")[0]
        envelope = _read_artifact_json(store, envelope_art)
        assert envelope["device"]["provider"] == "local"

    def test_mock_aws_device_has_aws_provider(self, store, registry, device_factory):
        """Mock AWS device is tagged with provider='aws_braket'."""
        mock_device = device_factory(name="aws_qpu", qubit_count=2)
        adapter = BraketAdapter()

        with track(project="aws_provider", store=store, registry=registry) as run:
            device = adapter.wrap_executor(mock_device, run)
            device.run(Circuit().h(0), shots=10).result()

        loaded = registry.load(run.run_id)
        assert loaded.record["data"]["tags"]["provider"] == "aws_braket"


# ============================================================================
# Braket ProgramSet Tests
# ============================================================================


class TestIsProgramSet:
    """Tests for ProgramSet detection."""

    def test_none(self):
        assert _is_program_set(None) is False

    def test_regular_circuit(self):
        c = Circuit().h(0)
        assert _is_program_set(c) is False

    def test_with_entries_and_to_ir(self):
        obj = SimpleNamespace(entries=[], to_ir=lambda: None)
        assert _is_program_set(obj) is True

    def test_with_entries_and_total_executables(self):
        obj = SimpleNamespace(entries=[], total_executables=2)
        assert _is_program_set(obj) is True

    def test_by_class_name(self):
        """Detects ProgramSet by class name as fallback."""

        class MockProgramSetV2:
            pass

        assert _is_program_set(MockProgramSetV2()) is True

    def test_entries_alone_not_enough(self):
        """entries without to_ir/total_executables and non-ProgramSet name."""
        obj = SimpleNamespace(entries=[])
        assert _is_program_set(obj) is False


# ============================================================================
# ProgramSet Circuit Extraction Tests
# ============================================================================


class TestExtractCircuitsFromProgramSet:
    """Tests for circuit extraction from ProgramSets."""

    def test_extracts_circuit_attr(self):
        c = Circuit().h(0)
        entry = SimpleNamespace(circuit=c)
        ps = SimpleNamespace(entries=[entry])
        assert _extract_circuits_from_program_set(ps) == [c]

    def test_extracts_program_attr(self):
        c = Circuit().x(0)
        entry = SimpleNamespace(program=c)
        ps = SimpleNamespace(entries=[entry])
        assert _extract_circuits_from_program_set(ps) == [c]

    def test_no_entries(self):
        ps = SimpleNamespace(entries=None)
        assert _extract_circuits_from_program_set(ps) == []

    def test_entry_is_circuit(self):
        """Entry itself is a circuit (no circuit/program attr)."""
        c = Circuit().h(0)
        ps = SimpleNamespace(entries=[c])
        assert _extract_circuits_from_program_set(ps) == [c]


# ============================================================================
# ProgramSet Metadata Tests
# ============================================================================


class TestGetProgramSetMetadata:
    """Tests for ProgramSet metadata extraction."""

    def test_extracts_all_fields(self):
        ps = SimpleNamespace(
            total_executables=5,
            shots_per_executable=100,
            total_shots=500,
        )
        meta = _get_program_set_metadata(ps)
        assert meta["is_program_set"] is True
        assert meta["total_executables"] == 5
        assert meta["shots_per_executable"] == 100
        assert meta["total_shots"] == 500

    def test_missing_fields(self):
        ps = SimpleNamespace()
        meta = _get_program_set_metadata(ps)
        assert meta == {"is_program_set": True}


# ============================================================================
# ProgramSet Task Tests
# ============================================================================


class TestMaterializeTaskSpec:
    """Tests for task specification materialization."""

    def test_none(self):
        payload, circuits, single, meta = _materialize_task_spec(None)
        assert payload is None
        assert circuits == []
        assert single is False
        assert meta is None

    def test_single_circuit(self):
        c = Circuit().h(0)
        payload, circuits, single, meta = _materialize_task_spec(c)
        assert payload is c
        assert circuits == [c]
        assert single is True
        assert meta is None

    def test_list_passthrough(self):
        """List input is returned without copy."""
        lst = [Circuit().h(0), Circuit().x(0)]
        payload, circuits, single, meta = _materialize_task_spec(lst)
        assert payload is lst  # Same object — no copy
        assert circuits is lst
        assert single is False

    def test_tuple_converted_to_list(self):
        """Tuple input is converted to list."""
        t = (Circuit().h(0), Circuit().x(0))
        payload, circuits, single, meta = _materialize_task_spec(t)
        assert isinstance(payload, list)
        assert len(payload) == 2
        assert single is False

    def test_program_set(self):
        """ProgramSet is passed through as run_payload."""
        c = Circuit().h(0)
        entry = SimpleNamespace(circuit=c)
        ps = SimpleNamespace(
            entries=[entry],
            to_ir=lambda: None,
            total_executables=1,
        )
        payload, circuits, single, meta = _materialize_task_spec(ps)
        assert payload is ps
        assert circuits == [c]
        assert single is False
        assert meta is not None
        assert meta["is_program_set"] is True

    def test_generator_materialized(self):
        """Generator is materialized to list."""

        def gen():
            yield Circuit().h(0)
            yield Circuit().x(0)

        payload, _, single, _ = _materialize_task_spec(gen())
        assert isinstance(payload, list)
        assert len(payload) == 2
        assert single is False

    def test_non_iterable_wrapped(self):
        """Non-iterable, non-circuit treated as single."""
        payload, circuits, single, _ = _materialize_task_spec(42)
        assert payload == 42
        assert circuits == [42]
        assert single is True


# ============================================================================
# Logging Tests
# ============================================================================


class TestShouldLog:
    """Tests for logging frequency logic."""

    @pytest.fixture
    def _device(self):
        """Minimal TrackedDevice for testing _should_log."""
        tracker = SimpleNamespace(
            run_id="test",
            set_tag=lambda *a: None,
            log_param=lambda *a: None,
            record={},
        )
        # Create with minimal mock device
        mock_dev = SimpleNamespace(
            __module__="braket.devices",
            name="test",
            run=lambda *a, **kw: None,
        )
        return TrackedDevice(
            device=mock_dev,
            tracker=tracker,
            log_every_n=0,
            log_new_circuits=True,
        )

    def test_first_execution_always_logged(self, _device):
        assert _device._should_log(1, "hash1", True) is True

    def test_second_execution_same_hash_not_logged(self, _device):
        _device.log_every_n = 0
        assert _device._should_log(2, "hash1", False) is False

    def test_log_every_n_minus_1_always(self, _device):
        _device.log_every_n = -1
        assert _device._should_log(999, "hash1", False) is True

    def test_log_every_n_positive(self, _device):
        _device.log_every_n = 3
        assert _device._should_log(3, "hash1", False) is True
        assert _device._should_log(4, "hash1", False) is False
        assert _device._should_log(6, "hash1", False) is True

    def test_log_new_circuits(self, _device):
        _device.log_every_n = 0
        _device.log_new_circuits = True
        assert _device._should_log(5, "new_hash", True) is True

    def test_log_new_circuits_disabled(self, _device):
        _device.log_every_n = 0
        _device.log_new_circuits = False
        assert _device._should_log(5, "new_hash", True) is False
