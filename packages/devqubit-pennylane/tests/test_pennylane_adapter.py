# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for PennyLane adapter."""

import pennylane as qml
from devqubit_engine.core.run import track
from devqubit_pennylane.adapter import (
    PennyLaneAdapter,
    _compute_circuit_hash,
    patch_device,
)


def _count_kind(loaded, kind: str) -> int:
    """Count artifacts of a specific kind."""
    return sum(1 for a in loaded.artifacts if a.kind == kind)


class TestPennyLaneAdapter:
    """Tests for adapter registration and device detection."""

    def test_adapter_name(self):
        """Adapter has correct identifier."""
        assert PennyLaneAdapter().name == "pennylane"

    def test_supports_default_qubit(self, default_qubit):
        """Adapter supports default.qubit device."""
        adapter = PennyLaneAdapter()
        assert adapter.supports_executor(default_qubit) is True

    def test_rejects_non_devices(self):
        """Adapter rejects non-device objects."""
        adapter = PennyLaneAdapter()
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("not a device") is False
        assert adapter.supports_executor([]) is False

    def test_describe_executor(self, default_qubit):
        """Adapter correctly describes device."""
        desc = PennyLaneAdapter().describe_executor(default_qubit)

        assert desc["name"] == "default.qubit"
        assert desc["provider"] == "pennylane"
        assert desc["num_wires"] == 2

    def test_describe_executor_shows_analytic_mode(self, default_qubit):
        """Adapter description shows analytic mode for device without shots."""
        desc = PennyLaneAdapter().describe_executor(default_qubit)

        assert "shots_info" in desc
        assert desc["shots_info"]["analytic"] is True
        assert desc["shots_info"]["total_shots"] is None


class TestPatchDevice:
    """Tests for device patching."""

    def test_patch_sets_flag_and_preserves_original(self, default_qubit):
        """Patching sets flag and preserves original execute."""
        patch_device(default_qubit)

        assert default_qubit._devqubit_patched is True
        assert hasattr(default_qubit, "_devqubit_original_execute")
        assert default_qubit._devqubit_tracker is None

    def test_patch_is_idempotent_but_updates_config(self, default_qubit):
        """Second patch doesn't re-wrap but updates config."""
        patch_device(
            default_qubit,
            log_every_n=0,
            log_new_circuits=True,
            stats_update_interval=10,
        )
        execute_wrapped = default_qubit.execute

        patch_device(
            default_qubit,
            log_every_n=5,
            log_new_circuits=False,
            stats_update_interval=7,
        )

        # Not re-wrapped
        assert default_qubit.execute is execute_wrapped

        # Config updated
        assert default_qubit._devqubit_log_every_n == 5
        assert default_qubit._devqubit_log_new_circuits is False

    def test_patched_device_without_tracker_passes_through(self, default_qubit):
        """Patched device without tracker executes normally."""
        patch_device(default_qubit)
        default_qubit._devqubit_tracker = None

        @qml.qnode(default_qubit, shots=10)
        def circuit():
            qml.Hadamard(0)
            return qml.counts(wires=[0])

        result = circuit()
        assert result is not None


class TestWrapDevice:
    """Tests for device wrapping behavior."""

    def test_wrap_returns_same_device_patched(self, store, registry, default_qubit):
        """wrap() patches device in-place rather than returning wrapper."""
        with track(project="pl", store=store, registry=registry) as run:
            wrapped = run.wrap(default_qubit)
            assert wrapped is default_qubit
            assert default_qubit._devqubit_patched is True
            assert default_qubit._devqubit_tracker is run

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"

    def test_qnode_built_before_wrap_is_tracked(self, store, registry, default_qubit):
        """QNodes created before run context are still tracked when wrapped."""

        @qml.qnode(default_qubit, shots=25)
        def bell_counts():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=[0, 1])

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, stats_update_interval=1)
            _ = bell_counts()

        loaded = registry.load(run.run_id)
        kinds = {a.kind for a in loaded.artifacts}

        assert loaded.status == "FINISHED"
        assert "devqubit.envelope.json" in kinds
        assert "pennylane.tapes.json" in kinds
        assert "result.pennylane.output.json" in kinds
        assert loaded.record["device_snapshot"]["sdk"] == "pennylane"


class TestTrackedExecution:
    """Tests for tracked device execution."""

    def test_basic_qnode_tracking(self, store, registry, default_qubit):
        """Basic QNode execution is tracked with correct tags."""

        @qml.qnode(default_qubit, shots=1000)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            _ = circuit()

        loaded = registry.load(run.run_id)

        assert loaded.status == "FINISHED"
        assert loaded.record["data"]["tags"]["provider"] == "pennylane"
        assert loaded.record["backend"]["name"] == "default.qubit"
        assert loaded.record["backend"]["provider"] == "pennylane"

    def test_execution_count_incremented(self, store, registry, default_qubit):
        """Execution count is incremented correctly."""

        @qml.qnode(default_qubit, shots=1000)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            circuit()
            circuit()
            circuit()

        assert default_qubit._devqubit_execution_count == 3

    def test_tapes_and_results_logged(self, store, registry, default_qubit):
        """Tapes and results are logged as artifacts."""

        @qml.qnode(default_qubit, shots=1000)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            circuit()

        loaded = registry.load(run.run_id)

        artifact_kinds = {a.kind for a in loaded.artifacts}
        assert "pennylane.tapes.json" in artifact_kinds
        assert "pennylane.tapes.txt" in artifact_kinds

        assert "results" in loaded.record
        assert "completed_at" in loaded.record["results"]


class TestBatchExecution:
    """Tests for batch execution path (multiple tapes at once)."""

    def test_batch_execute_with_qml_execute(self, store, registry, default_qubit):
        """Batch execution via qml.execute is tracked correctly."""
        # Create multiple tapes (analytic mode - no shots needed for expval)
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

        with track(project="batch", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            # qml.execute is the canonical API for batch execution
            results = qml.execute([tape1, tape2], default_qubit)

        loaded = registry.load(run.run_id)

        assert loaded.status == "FINISHED"
        assert len(results) == 2

        # Check that batch was logged
        artifact_kinds = {a.kind for a in loaded.artifacts}
        assert "pennylane.tapes.json" in artifact_kinds

    def test_direct_device_execute_batch(self, store, registry, default_qubit):
        """Direct device.execute with list of tapes works."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.PauliX(wires=0)
            qml.expval(qml.PauliZ(0))

        with track(project="batch", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            # Direct device execution
            _ = default_qubit.execute([tape1, tape2])

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"


class TestExpectationValues:
    """Tests for expectation value execution."""

    def test_expval_tracking(self, store, registry, default_qubit):
        """Expectation value execution is tracked."""

        @qml.qnode(default_qubit)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            _ = circuit(0.5)

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"


class TestSamplingBehavior:
    """Tests for execution sampling to prevent logging explosion."""

    def test_default_logging_deduplicates_same_circuit(
        self, store, registry, default_qubit
    ):
        """Default policy logs first execution only for identical circuits."""

        @qml.qnode(default_qubit, shots=20)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=[0, 1])

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, stats_update_interval=1)
            circuit()
            circuit()
            circuit()

        loaded = registry.load(run.run_id)

        # De-duplication: structure and results logged once
        assert _count_kind(loaded, "pennylane.tapes.json") == 1
        assert _count_kind(loaded, "result.pennylane.output.json") == 1
        assert _count_kind(loaded, "devqubit.envelope.json") == 1

        # Stats reflect all executions
        stats = loaded.record.get("execution_stats", {})
        assert stats.get("total_executions") == 3
        assert stats.get("unique_circuits") == 1
        assert stats.get("logged_executions") == 1

    def test_log_every_n_logs_results_each_time(self, store, registry, default_qubit):
        """log_every_n=1 logs results each time but tapes only once."""

        @qml.qnode(default_qubit, shots=20)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=[0, 1])

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, log_every_n=1, stats_update_interval=1)
            circuit()
            circuit()

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "pennylane.tapes.json") == 1  # structure once
        assert _count_kind(loaded, "result.pennylane.output.json") == 2  # results twice
        assert _count_kind(loaded, "devqubit.envelope.json") == 2  # envelope twice

        assert loaded.record["execute"]["execution_count"] == 2

    def test_parameter_changes_do_not_create_new_structures(
        self, store, registry, default_qubit
    ):
        """Different parameter values don't create new circuit structures."""

        @qml.qnode(default_qubit)
        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, stats_update_interval=1)
            _ = circuit(0.1)
            _ = circuit(0.2)

        loaded = registry.load(run.run_id)

        # Same structure -> single unique circuit
        assert _count_kind(loaded, "pennylane.tapes.json") == 1
        assert _count_kind(loaded, "result.pennylane.output.json") == 1

        stats = loaded.record.get("execution_stats", {})
        assert stats.get("unique_circuits") == 1
        assert stats.get("total_executions") == 2


class TestCircuitHash:
    """Tests for circuit structure hashing."""

    def test_hash_is_structure_only(self):
        """Hash depends on structure, not parameter values."""
        with qml.tape.QuantumTape() as t1:
            qml.RX(0.1, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as t2:
            qml.RX(2.3, wires=0)  # Different value
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as t3:
            qml.RY(2.3, wires=0)  # Different gate
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        assert _compute_circuit_hash(t1) == _compute_circuit_hash(t2)
        assert _compute_circuit_hash(t1) != _compute_circuit_hash(t3)

    def test_different_wires_different_hash(self):
        """Different wire indices produce different hashes."""
        with qml.tape.QuantumTape() as t1:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as t2:
            qml.Hadamard(wires=1)
            qml.expval(qml.PauliZ(1))

        assert _compute_circuit_hash(t1) != _compute_circuit_hash(t2)

    def test_different_measurements_different_hash(self):
        """Different measurement types produce different hashes."""
        with qml.queuing.AnnotatedQueue() as q1:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))
        tape_expval = qml.tape.QuantumScript.from_queue(q1)

        with qml.queuing.AnnotatedQueue() as q2:
            qml.Hadamard(wires=0)
            qml.probs(wires=0)
        tape_probs = qml.tape.QuantumScript.from_queue(q2)

        assert _compute_circuit_hash(tape_expval) != _compute_circuit_hash(tape_probs)

    def test_single_tape_and_list_consistent(self):
        """Single tape and list of one tape produce same hash."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))
        tape = qml.tape.QuantumScript.from_queue(q)

        h_single = _compute_circuit_hash(tape)
        h_list = _compute_circuit_hash([tape])

        assert h_single == h_list


class TestBackendTypeCompliance:
    """Tests that device snapshots have schema-compliant backend_type."""

    # Strict schema compliance - only canonical values
    VALID_BACKEND_TYPES = {"simulator", "hardware"}

    def test_device_snapshot_backend_type(self, store, registry, default_qubit):
        """Device snapshot has valid backend_type."""
        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)

        loaded = registry.load(run.run_id)

        backend_type = loaded.record["device_snapshot"].get("backend_type")
        assert backend_type in self.VALID_BACKEND_TYPES

    def test_default_qubit_is_exactly_simulator(self, store, registry, default_qubit):
        """default.qubit backend_type is exactly 'simulator'."""
        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)

        loaded = registry.load(run.run_id)
        assert loaded.record["device_snapshot"]["backend_type"] == "simulator"


class TestUECCompliance:
    """Tests for UEC compliance in tracked records."""

    def test_device_snapshot_required_fields(self, store, registry, default_qubit):
        """Device snapshot record has required fields."""
        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)

        loaded = registry.load(run.run_id)
        snapshot = loaded.record["device_snapshot"]

        assert "backend_name" in snapshot
        assert "backend_type" in snapshot
        assert snapshot["sdk"] == "pennylane"

    def test_result_type_captured(self, store, registry, default_qubit):
        """Result type is captured in results record."""

        @qml.qnode(default_qubit, shots=100)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            circuit()

        loaded = registry.load(run.run_id)

        assert "results" in loaded.record
        assert "result_type" in loaded.record["results"]
        assert "completed_at" in loaded.record["results"]

    def test_tape_artifacts_captured(self, store, registry, default_qubit):
        """Tape artifacts are captured with expected kinds."""

        @qml.qnode(default_qubit, shots=100)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            circuit()

        loaded = registry.load(run.run_id)

        artifact_kinds = {a.kind for a in loaded.artifacts}
        assert "pennylane.tapes.json" in artifact_kinds
        assert "pennylane.tapes.txt" in artifact_kinds

    def test_raw_properties_artifact_created(self, store, registry, default_qubit):
        """Device raw_properties are logged as separate artifact."""
        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)

        loaded = registry.load(run.run_id)
        artifact_kinds = {a.kind for a in loaded.artifacts}

        assert "device.pennylane.raw_properties.json" in artifact_kinds
