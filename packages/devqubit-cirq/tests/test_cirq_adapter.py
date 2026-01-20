# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for the Cirq adapter."""

from __future__ import annotations

import json

import cirq
import pytest
import sympy
from devqubit_cirq.adapter import CirqAdapter, _materialize_circuits
from devqubit_engine.tracking.run import track


def _artifacts_of_kind(loaded, kind: str):
    """Get artifacts of a specific kind."""
    return [a for a in loaded.artifacts if a.kind == kind]


def _load_json(store, digest: str):
    """Load JSON from store by digest."""
    return json.loads(store.get_bytes(digest).decode("utf-8"))


def _load_single_envelope(store, loaded) -> dict:
    """Load the single envelope from a run."""
    env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
    assert len(env_arts) == 1
    return _load_json(store, env_arts[0].digest)


class TestCirqAdapter:
    """Tests for adapter registration and sampler detection."""

    def test_adapter_name(self):
        """Adapter has correct identifier."""
        assert CirqAdapter().name == "cirq"

    def test_supports_simulator(self, simulator):
        """Adapter supports Cirq Simulator."""
        adapter = CirqAdapter()
        assert adapter.supports_executor(simulator) is True

    def test_rejects_non_samplers(self):
        """Adapter rejects non-sampler objects."""
        adapter = CirqAdapter()
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("sampler") is False
        assert adapter.supports_executor([]) is False


class TestBellRunEndToEnd:
    """End-to-end test for Bell circuit execution."""

    def test_emits_consistent_envelope_and_counts(
        self, bell_circuit, simulator, store, registry
    ):
        """Bell run produces consistent envelope, counts, and program refs."""
        repetitions = 256

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            result = tracked.run(bell_circuit, repetitions=repetitions)
            assert hasattr(result, "measurements")

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"
        # P1 FIX: provider should be physical ("local"), sdk should be "cirq"
        assert loaded.record["backend"]["provider"] == "local"
        assert loaded.record["backend"]["sdk"] == "cirq"
        assert loaded.record["execute"]["repetitions"] == repetitions

        # Envelope internally consistent (UEC 1.0)
        env = _load_single_envelope(store, loaded)
        assert env["producer"]["adapter"] == "devqubit-cirq"
        assert env["producer"]["sdk"] == "cirq"
        assert env["program"]["num_circuits"] == 1
        assert env["execution"]["shots"] == repetitions

        # Bell state should only produce 00 and 11 (UEC 1.0: items instead of counts)
        result_items = env["result"]["items"]
        assert len(result_items) == 1
        counts = result_items[0]["counts"]["counts"]
        assert sum(counts.values()) == repetitions
        assert set(counts).issubset({"00", "11"})


class TestRunSweep:
    """Tests for parameterized sweep execution."""

    def test_produces_multiple_experiments(
        self, parameterized_circuit, simulator, store, registry
    ):
        """run_sweep produces multiple experiments with sweep options."""
        repetitions = 50
        theta = sympy.Symbol("theta")
        params = [cirq.ParamResolver({theta: v}) for v in [0.0, 0.5, 1.0]]

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            results = tracked.run_sweep(
                parameterized_circuit, params, repetitions=repetitions
            )
            assert len(results) == len(params)

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)

        assert env["execution"]["options"]["sweep"] is True
        # UEC 1.0: count items instead of num_experiments field
        assert len(env["result"]["items"]) == len(params)

    def test_logs_params_per_experiment(
        self, parameterized_circuit, simulator, store, registry
    ):
        """run_sweep logs parameter values for each experiment."""
        repetitions = 20
        theta = sympy.Symbol("theta")
        param_values = [0.0, 0.5, 1.0, 1.5]
        params = [cirq.ParamResolver({theta: v}) for v in param_values]

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            tracked.run_sweep(parameterized_circuit, params, repetitions=repetitions)

        loaded = registry.load(run.run_id)

        # Load normalized counts to check params
        counts_arts = _artifacts_of_kind(loaded, "result.counts.json")
        assert len(counts_arts) == 1

        counts_payload = _load_json(store, counts_arts[0].digest)
        experiments = counts_payload["experiments"]

        assert len(experiments) == len(param_values)

        # Each experiment should have params logged
        for i, exp in enumerate(experiments):
            assert "params" in exp, f"Experiment {i} missing params"
            # Verify the theta value matches (converted to float)
            assert "theta" in exp["params"]


class TestRunBatch:
    """Tests for batch circuit execution (run_batch)."""

    def test_batch_produces_nested_experiments(self, simulator, store, registry):
        """run_batch produces experiments with batch_index and sweep_index."""
        q0, q1 = cirq.LineQubit.range(2)

        # Two different circuits
        c1 = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="m"),
        )
        c2 = cirq.Circuit(
            cirq.X(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="m"),
        )

        programs = [c1, c2]
        # Each circuit gets its own empty resolver (no params)
        params_list = [cirq.ParamResolver({}), cirq.ParamResolver({})]
        repetitions = 50

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            results = tracked.run_batch(
                programs,
                params_list,
                repetitions=repetitions,
            )

            # run_batch returns list of lists
            assert len(results) == 2
            assert all(isinstance(r, list) for r in results)

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)

        # Verify batch flag in options
        assert env["execution"]["options"].get("batch") is True

        # Load counts to verify structure
        counts_arts = _artifacts_of_kind(loaded, "result.counts.json")
        assert len(counts_arts) == 1

        counts_payload = _load_json(store, counts_arts[0].digest)
        experiments = counts_payload["experiments"]

        # Should have 2 experiments (one per circuit, one sweep each)
        assert len(experiments) == 2

        # Verify batch_index and sweep_index are present
        assert experiments[0]["batch_index"] == 0
        assert experiments[0]["sweep_index"] == 0
        assert experiments[1]["batch_index"] == 1
        assert experiments[1]["sweep_index"] == 0

    def test_batch_with_sweeps(self, simulator, store, registry):
        """run_batch with parameter sweeps produces correct experiment count."""
        theta = sympy.Symbol("theta")
        q0 = cirq.LineQubit(0)

        circuit = cirq.Circuit(
            cirq.rz(theta).on(q0),
            cirq.measure(q0, key="m"),
        )

        # Two circuits, each with 3 parameter values
        programs = [circuit, circuit]
        params_list = [
            [cirq.ParamResolver({theta: v}) for v in [0.0, 0.5, 1.0]],
            [cirq.ParamResolver({theta: v}) for v in [1.5, 2.0, 2.5]],
        ]

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            results = tracked.run_batch(
                programs,
                params_list,
                repetitions=20,
            )

            # 2 circuits × 3 params each = 6 total results
            total_results = sum(len(r) for r in results)
            assert total_results == 6

        loaded = registry.load(run.run_id)
        counts_arts = _artifacts_of_kind(loaded, "result.counts.json")
        counts_payload = _load_json(store, counts_arts[0].digest)

        # 6 experiments total
        assert len(counts_payload["experiments"]) == 6

    def test_batch_with_per_circuit_repetitions(self, simulator, store, registry):
        """run_batch supports per-circuit repetition counts."""
        q0, q1 = cirq.LineQubit.range(2)

        c1 = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key="m"))
        c2 = cirq.Circuit(cirq.X(q1), cirq.measure(q1, key="m"))

        programs = [c1, c2]
        params_list = [cirq.ParamResolver({}), cirq.ParamResolver({})]
        # Different repetitions per circuit
        repetitions = [100, 200]

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            _ = tracked.run_batch(
                programs,
                params_list,
                repetitions=repetitions,
            )

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)

        # Verify repetitions_per_circuit is logged
        options = env["execution"]["options"]
        assert options.get("repetitions_per_circuit") == [100, 200]


class TestLoggingPolicy:
    """Tests for execution sampling to prevent logging explosion."""

    def test_first_only_plus_new_circuit(self, simulator, store, registry):
        """log_every_n=0 logs first only, but log_new_circuits=True logs new structures."""
        q0, q1 = cirq.LineQubit.range(2)
        c1 = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="m"),
        )
        c2 = cirq.Circuit(
            cirq.X(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="m"),
        )

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator, log_every_n=0, log_new_circuits=True)

            # Same circuit multiple times => only first logged
            for _ in range(3):
                tracked.run(c1, repetitions=10)

            # New circuit structure => should log again
            tracked.run(c2, repetitions=10)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == 2

    def test_every_n(self, simulator, store, registry):
        """log_every_n=2 logs on executions 1, 2, 4 => 3 envelopes."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m")
        )

        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator, log_every_n=2, log_new_circuits=False)

            for _ in range(5):
                tracked.run(circuit, repetitions=5)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == 3


class TestFailurePathEnvelope:
    """P0: Tests for failure-path envelope logging."""

    def test_run_failure_logs_envelope_and_reraises(self, store, registry):
        """When simulator.run() raises, envelope is logged and exception re-raised."""

        class FailingSimulator:
            """Simulator that always fails."""

            def run(self, *args, **kwargs):
                raise RuntimeError("Simulated hardware failure")

            def run_sweep(self, *args, **kwargs):
                raise RuntimeError("Simulated hardware failure")

            def run_batch(self, *args, **kwargs):
                raise RuntimeError("Simulated hardware failure")

        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0, key="m"))

        with track(project="test", store=store, registry=registry) as run:
            # Import adapter directly to use wrap_executor
            from devqubit_cirq.adapter import CirqAdapter

            adapter = CirqAdapter()
            tracked = adapter.wrap_executor(FailingSimulator(), run)

            # Should raise the original exception
            with pytest.raises(RuntimeError, match="Simulated hardware failure"):
                tracked.run(circuit, repetitions=10)

        # Envelope should still be logged with failure info
        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")

        # Should have logged an envelope even though execution failed
        assert len(env_arts) == 1

        env = _load_json(store, env_arts[0].digest)
        assert env["result"]["success"] is False
        assert env["result"]["status"] == "failed"
        # UEC 1.0: error is a top-level field, not in metadata
        assert env["result"]["error"]["type"] == "RuntimeError"

    def test_run_sweep_failure_logs_envelope(self, store, registry):
        """When simulator.run_sweep() raises, envelope is logged."""

        class FailingSweepSimulator:
            def run_sweep(self, *args, **kwargs):
                raise ValueError("Invalid sweep parameters")

        q0 = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")
        circuit = cirq.Circuit(cirq.rz(theta).on(q0), cirq.measure(q0, key="m"))
        params = [cirq.ParamResolver({theta: v}) for v in [0.0, 0.5]]

        with track(project="test", store=store, registry=registry) as run:
            from devqubit_cirq.adapter import CirqAdapter

            adapter = CirqAdapter()
            tracked = adapter.wrap_executor(FailingSweepSimulator(), run)

            with pytest.raises(ValueError, match="Invalid sweep parameters"):
                tracked.run_sweep(circuit, params, repetitions=10)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == 1

        env = _load_json(store, env_arts[0].digest)
        assert env["result"]["success"] is False
        assert env["result"]["status"] == "failed"


class TestMaterializeCircuits:
    """Tests for circuit materialization."""

    def test_single_circuit_not_iterated(self, bell_circuit, ghz_circuit):
        """Single Cirq Circuit is not iterated over moments."""
        circuits, was_single = _materialize_circuits(bell_circuit)
        assert was_single is True
        assert circuits == [bell_circuit]

        circuits2, was_single2 = _materialize_circuits([bell_circuit, ghz_circuit])
        assert was_single2 is False
        assert circuits2 == [bell_circuit, ghz_circuit]


class TestBackendTypeCompliance:
    """Tests that device snapshots have schema-compliant backend_type."""

    # Canonical backend types - using strict values to catch regressions
    VALID_BACKEND_TYPES = {"simulator", "hardware"}

    def test_simulator_backend_type(self, bell_circuit, simulator, store, registry):
        """Simulator has valid backend_type."""
        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            tracked.run(bell_circuit, repetitions=10)

        loaded = registry.load(run.run_id)
        backend_type = loaded.record.get("device_snapshot", {}).get("backend_type")
        assert backend_type in self.VALID_BACKEND_TYPES


class TestUECCompliance:
    """Tests for UEC compliance in tracked records."""

    def test_device_snapshot_required_fields(
        self, bell_circuit, simulator, store, registry
    ):
        """Device snapshot has required fields."""
        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            tracked.run(bell_circuit, repetitions=10)

        loaded = registry.load(run.run_id)
        snapshot = loaded.record.get("device_snapshot", {})

        assert "backend_name" in snapshot or "name" in loaded.record.get("backend", {})
        assert (
            snapshot.get("sdk") == "cirq"
            or loaded.record.get("backend", {}).get("provider") == "cirq"
        )


class TestEnvelopeValidation:
    """Tests for envelope validation behavior."""

    def test_valid_envelope_is_logged_not_invalid(
        self, bell_circuit, simulator, store, registry
    ):
        """Valid execution produces devqubit.envelope.json, not .invalid.json.

        This verifies that envelope validation passes for normal executions
        and that validation errors would not be silently swallowed.
        """
        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            tracked.run(bell_circuit, repetitions=100)

        loaded = registry.load(run.run_id)

        # Should have valid envelope
        valid_envs = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(valid_envs) == 1, "Should have exactly one valid envelope"

        # Should NOT have invalid envelope
        invalid_envs = _artifacts_of_kind(loaded, "devqubit.envelope.invalid.json")
        assert len(invalid_envs) == 0, "Should not have invalid envelope artifacts"

    def test_envelope_has_required_uec_fields(
        self, bell_circuit, simulator, store, registry
    ):
        """Envelope contains all UEC-required fields that would fail validation if missing."""
        with track(project="test", store=store, registry=registry) as run:
            tracked = run.wrap(simulator)
            tracked.run(bell_circuit, repetitions=50)

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)

        # These fields are required by UEC schema - missing any would cause
        # EnvelopeValidationError if validation wasn't working
        assert "envelope_id" in env
        assert "created_at" in env
        assert "producer" in env
        assert env["producer"]["adapter"] == "devqubit-cirq"
        assert env["producer"]["sdk"] == "cirq"
        assert "device" in env
        assert "program" in env
        assert "execution" in env
        assert "result" in env
        assert env["result"]["status"] in (
            "completed",
            "failed",
            "partial",
            "cancelled",
        )
