# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""End-to-end tests for PennyLane adapter."""

import pennylane as qml
from devqubit_engine.tracking.run import track
from devqubit_pennylane.adapter import (
    PennyLaneAdapter,
    is_device_patched,
    patch_device,
    unpatch_device,
)


def _count_kind(loaded, kind: str) -> int:
    """Count artifacts of a specific kind."""
    return sum(1 for a in loaded.artifacts if a.kind == kind)


class TestAdapterRegistration:
    """Tests for adapter registration and device detection."""

    def test_adapter_name(self):
        """Adapter has correct identifier."""
        assert PennyLaneAdapter().name == "pennylane"

    def test_supports_default_qubit(self, default_qubit):
        """Adapter supports default.qubit device."""
        assert PennyLaneAdapter().supports_executor(default_qubit) is True

    def test_rejects_non_devices(self):
        """Adapter rejects non-device objects."""
        adapter = PennyLaneAdapter()
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("not a device") is False

    def test_describe_executor(self, default_qubit):
        """Adapter correctly describes device."""
        desc = PennyLaneAdapter().describe_executor(default_qubit)

        assert desc["name"] == "default.qubit"
        assert desc["provider"] == "local"
        assert desc["sdk"] == "pennylane"
        assert desc["num_wires"] == 2
        assert desc["shots_info"]["analytic"] is True


class TestDevicePatching:
    """Tests for device patching behavior."""

    def test_patch_sets_flag_and_preserves_original(self, default_qubit):
        """Patching sets flag and preserves original execute."""
        patch_device(default_qubit)

        assert default_qubit._devqubit_patched is True
        assert hasattr(default_qubit, "_devqubit_original_execute")
        assert default_qubit._devqubit_tracker is None

    def test_patch_is_idempotent(self, default_qubit):
        """Second patch doesn't re-wrap but updates config."""
        patch_device(default_qubit, log_every_n=0, log_new_circuits=True)
        execute_wrapped = default_qubit.execute

        patch_device(default_qubit, log_every_n=5, log_new_circuits=False)

        assert default_qubit.execute is execute_wrapped  # Not re-wrapped
        assert default_qubit._devqubit_log_every_n == 5  # Config updated

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

    def test_unpatch_restores_original_methods(self, default_qubit):
        """unpatch_device restores original execute method."""

        original_execute = default_qubit.execute
        patch_device(default_qubit)

        assert is_device_patched(default_qubit) is True
        assert default_qubit.execute is not original_execute

        result = unpatch_device(default_qubit)

        assert result is True
        assert is_device_patched(default_qubit) is False
        assert not hasattr(default_qubit, "_devqubit_tracker")

    def test_unpatch_unpatched_device_returns_false(self, default_qubit):
        """unpatch_device on unpatched device returns False."""

        assert is_device_patched(default_qubit) is False
        assert unpatch_device(default_qubit) is False


class TestDeviceWrapping:
    """Tests for device wrapping via run.wrap()."""

    def test_wrap_returns_same_device(self, store, registry, default_qubit):
        """wrap() patches device in-place rather than returning wrapper."""

        @qml.qnode(default_qubit, shots=10)
        def circuit():
            qml.Hadamard(0)
            return qml.counts(wires=[0])

        with track(project="pl", store=store, registry=registry) as run:
            wrapped = run.wrap(default_qubit)
            assert wrapped is default_qubit
            assert default_qubit._devqubit_patched is True
            assert default_qubit._devqubit_tracker is run
            _ = circuit()

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"

    def test_tracker_reassignment_resets_state(self, store, registry, default_qubit):
        """Reassigning device to new tracker resets execution counters."""

        @qml.qnode(default_qubit, shots=10)
        def circuit():
            qml.Hadamard(0)
            return qml.counts(wires=[0])

        # First run
        with track(project="pl", store=store, registry=registry) as run1:
            run1.wrap(default_qubit)
            circuit()
            circuit()
            assert default_qubit._devqubit_execution_count == 2

        # Second run with same device - counters should reset
        with track(project="pl", store=store, registry=registry) as run2:
            run2.wrap(default_qubit)
            assert default_qubit._devqubit_execution_count == 0
            assert default_qubit._devqubit_tracker is run2
            circuit()
            assert default_qubit._devqubit_execution_count == 1

    def test_qnode_built_before_wrap_is_tracked(self, store, registry, default_qubit):
        """QNodes created before run context are still tracked."""

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
        assert loaded.record["data"]["tags"]["provider"] == "local"
        assert loaded.record["data"]["tags"]["sdk"] == "pennylane"
        assert loaded.record["backend"]["name"] == "default.qubit"

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


class TestBatchExecution:
    """Tests for batch execution path."""

    def test_batch_execute_with_qml_execute(self, store, registry, default_qubit):
        """Batch execution via qml.execute is tracked correctly."""
        with qml.tape.QuantumTape() as tape1:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

        with track(project="batch", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            results = qml.execute([tape1, tape2], default_qubit)

        loaded = registry.load(run.run_id)

        assert loaded.status == "FINISHED"
        assert len(results) == 2
        assert "pennylane.tapes.json" in {a.kind for a in loaded.artifacts}


class TestDeduplication:
    """Tests for execution deduplication and sampling."""

    def test_default_deduplicates_same_circuit(self, store, registry, default_qubit):
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

        stats = loaded.record.get("execution_stats", {})
        assert stats.get("total_executions") == 3
        assert stats.get("unique_circuits") == 1

    def test_log_every_n_logs_results_each_time(self, store, registry, default_qubit):
        """log_every_n=1 logs results each time but tapes only once."""

        @qml.qnode(default_qubit, shots=20)
        def circuit():
            qml.Hadamard(0)
            return qml.counts(wires=[0, 1])

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, log_every_n=1, stats_update_interval=1)
            circuit()
            circuit()

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "pennylane.tapes.json") == 1  # structure once
        assert _count_kind(loaded, "result.pennylane.output.json") == 2  # results twice

    def test_parameter_changes_dont_create_new_structures(
        self, store, registry, default_qubit
    ):
        """Different parameter values don't create new circuit structures."""

        @qml.qnode(default_qubit)
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        with track(project="pl", store=store, registry=registry) as run:
            run.wrap(default_qubit, stats_update_interval=1)
            _ = circuit(0.1)
            _ = circuit(0.2)

        loaded = registry.load(run.run_id)

        assert _count_kind(loaded, "pennylane.tapes.json") == 1
        stats = loaded.record.get("execution_stats", {})
        assert stats.get("unique_circuits") == 1
        assert stats.get("total_executions") == 2


# =============================================================================
# UEC COMPLIANCE
# =============================================================================


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
        assert snapshot["backend_type"] in {"simulator", "hardware"}
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

    def test_raw_properties_artifact_created(self, store, registry, default_qubit):
        """Device raw_properties are logged as separate artifact."""
        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)

        loaded = registry.load(run.run_id)
        artifact_kinds = {a.kind for a in loaded.artifacts}

        assert "device.pennylane.raw_properties.json" in artifact_kinds

    def test_envelope_created(self, store, registry, default_qubit):
        """ExecutionEnvelope is created and logged."""

        @qml.qnode(default_qubit, shots=100)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts()

        with track(project="test", store=store, registry=registry) as run:
            run.wrap(default_qubit)
            circuit()

        loaded = registry.load(run.run_id)
        artifact_kinds = {a.kind for a in loaded.artifacts}

        assert "devqubit.envelope.json" in artifact_kinds
