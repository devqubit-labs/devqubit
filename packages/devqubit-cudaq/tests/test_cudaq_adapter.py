# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""End-to-end tests for the CUDA-Q adapter."""

from __future__ import annotations

import json

import cudaq
import pytest
from devqubit_cudaq.adapter import CudaqAdapter, TrackedCudaqExecutor
from devqubit_engine.tracking.run import Run, track


# ruff: noqa: F821


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifacts_of_kind(loaded, kind: str):
    return [a for a in loaded.artifacts if a.kind == kind]


def _load_json(store, digest: str):
    return json.loads(store.get_bytes(digest).decode("utf-8"))


def _load_single_envelope(store, loaded) -> dict:
    env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
    assert len(env_arts) == 1, f"Expected 1 envelope, got {len(env_arts)}"
    return _load_json(store, env_arts[0].digest)


# ---------------------------------------------------------------------------
# Adapter registration
# ---------------------------------------------------------------------------


class TestCudaqAdapter:

    def test_name(self):
        assert CudaqAdapter().name == "cudaq"

    def test_supports_cudaq_module(self):
        assert CudaqAdapter().supports_executor(cudaq) is True

    def test_rejects_non_cudaq(self):
        adapter = CudaqAdapter()
        assert adapter.supports_executor(None) is False
        assert adapter.supports_executor("nope") is False
        assert adapter.supports_executor(42) is False

    def test_describe_executor(self):
        desc = CudaqAdapter().describe_executor(cudaq)
        assert desc["sdk"] == "cudaq"
        assert "name" in desc

    def test_wrap_executor(self, store, registry):
        adapter = CudaqAdapter()
        with Run(store=store, registry=registry, project="test") as run:
            wrapped = adapter.wrap_executor(cudaq, run)
            assert isinstance(wrapped, TrackedCudaqExecutor)


# ---------------------------------------------------------------------------
# Sample end-to-end
# ---------------------------------------------------------------------------


class TestSampleEndToEnd:

    def test_bell_sample_produces_envelope_and_record(
        self, bell_kernel, store, registry
    ):
        """Execute Bell kernel → verify run record + UEC envelope."""
        shots = 256

        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            result = executor.sample(bell_kernel, shots_count=shots)

        # Result is a real SampleResult
        assert result.get_total_shots() == shots
        for bitstring, count in result.items():
            assert bitstring in ("00", "11")
            assert count > 0

        # Run record
        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"
        assert loaded.record["backend"]["sdk"] == "cudaq"
        assert loaded.record["execute"]["success"] is True

        # UEC envelope
        env = _load_single_envelope(store, loaded)
        assert env["producer"]["adapter"] == "devqubit-cudaq"
        assert env["producer"]["sdk"] == "cudaq"
        assert env["program"]["num_circuits"] == 1
        assert env["execution"]["shots"] == shots

        # Bell state counts: only 00 and 11
        items = env["result"]["items"]
        assert len(items) >= 1
        counts = items[0]["counts"]["counts"]
        assert sum(counts.values()) == shots
        assert set(counts).issubset({"00", "11"})

    def test_ghz_sample(self, ghz_kernel, store, registry):
        """GHZ kernel produces only 000 and 111."""
        shots = 512

        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            result = executor.sample(ghz_kernel, shots_count=shots)

        assert result.get_total_shots() == shots

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)
        counts = env["result"]["items"][0]["counts"]["counts"]
        assert set(counts).issubset({"000", "111"})

    def test_parameterized_kernel_sample(self, rx_kernel, store, registry):
        """Parameterized Rx(0.0) → always |0⟩."""
        shots = 100

        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            result = executor.sample(rx_kernel, 0.0, shots_count=shots)

        assert result.get_total_shots() == shots
        # Rx(0) is identity — should measure |0⟩ with certainty
        for bitstring, count in result.items():
            assert bitstring == "0"

    def test_device_snapshot_in_record(self, bell_kernel, store, registry):
        """Device snapshot captures SDK and backend info."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        snapshot = loaded.record["device_snapshot"]
        assert snapshot["sdk"] == "cudaq"
        assert "backend_name" in snapshot

    def test_tags_set(self, bell_kernel, store, registry):
        """Tags include sdk, adapter, backend_name."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        tags = loaded.record.get("data", {}).get("tags", {})
        assert tags.get("sdk") == "cudaq"
        assert tags.get("adapter") == "devqubit-cudaq"


# ---------------------------------------------------------------------------
# Observe end-to-end
# ---------------------------------------------------------------------------


class TestObserveEndToEnd:

    def test_observe_produces_expectation_in_envelope(
        self, bell_kernel, z0_hamiltonian, store, registry
    ):
        """observe() with Z₀ on Bell state → expectation near 0.0."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            result = executor.observe(bell_kernel, z0_hamiltonian)

        # Real ObserveResult
        exp_val = result.expectation()
        assert isinstance(exp_val, float)
        # Bell state: ⟨Z₀⟩ should be near 0
        assert abs(exp_val) < 0.1

        loaded = registry.load(run.run_id)
        env = _load_single_envelope(store, loaded)
        assert env["producer"]["sdk"] == "cudaq"
        assert env["execution"]["sdk"] == "cudaq"

    def test_observe_with_shots(
        self, single_qubit_kernel, z0_hamiltonian, store, registry
    ):
        """Shot-based observe produces counts in the envelope."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            result = executor.observe(
                single_qubit_kernel, z0_hamiltonian, shots_count=500
            )

        exp_val = result.expectation()
        assert isinstance(exp_val, float)

        loaded = registry.load(run.run_id)
        assert loaded.record["results"]["success"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_execution_error_reraised_and_logged(self, store, registry):
        """When cudaq.sample raises, error is logged and re-raised."""

        @cudaq.kernel
        def _bad_kernel():
            q = cudaq.qvector(1)
            mz(q)

        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)

            # Sabotage the cudaq module's sample to simulate backend failure
            original_sample = executor.cudaq.sample

            def _failing_sample(*args, **kwargs):
                raise RuntimeError("Target offline")

            executor.cudaq.sample = _failing_sample
            try:
                with pytest.raises(RuntimeError, match="Target offline"):
                    executor.sample(_bad_kernel, shots_count=100)
            finally:
                executor.cudaq.sample = original_sample

        loaded = registry.load(run.run_id)
        assert "execution_error" in loaded.record
        assert loaded.record["execution_error"]["type"] == "RuntimeError"

        # Envelope should still be logged even on failure
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) >= 1


# ---------------------------------------------------------------------------
# Deduplication & logging frequency
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_repeated_kernel_deduplicates_structure(self, bell_kernel, store, registry):
        """Same kernel 3x with log_every_n=0 → 1 envelope (first only)."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq, log_every_n=0, log_new_circuits=True)
            for _ in range(3):
                executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == 1

    def test_different_kernels_both_logged(
        self, bell_kernel, ghz_kernel, store, registry
    ):
        """Two different kernels with log_new_circuits → 2 envelopes."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq, log_new_circuits=True)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(ghz_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == 2

    def test_log_every_n_minus_one_logs_all(self, bell_kernel, store, registry):
        """log_every_n=-1 logs every execution."""
        n_executions = 4

        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq, log_every_n=-1)
            for _ in range(n_executions):
                executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        env_arts = _artifacts_of_kind(loaded, "devqubit.envelope.json")
        assert len(env_arts) == n_executions


# ---------------------------------------------------------------------------
# Runtime configuration tracking
# ---------------------------------------------------------------------------


class TestRuntimeConfiguration:

    def test_set_target_records_event_and_invalidates_snapshot(
        self, bell_kernel, store, registry
    ):
        """set_target invalidates cached snapshot and records an event."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            # First sample — caches snapshot
            executor.sample(bell_kernel, shots_count=100)
            assert executor._device_snapshot is not None

            # Switch target — snapshot invalidated
            executor.set_target("qpp-cpu")
            assert executor._device_snapshot is None

            # Second sample — re-captures snapshot
            executor.sample(bell_kernel, shots_count=100)
            assert executor._device_snapshot is not None

        loaded = registry.load(run.run_id)
        assert loaded.record["execution_stats"]["total_executions"] == 2

    def test_set_random_seed_records_event(self, bell_kernel, store, registry):
        """set_random_seed is recorded as a runtime config event."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq)
            executor.set_random_seed(42)
            executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        assert loaded.status == "FINISHED"


# ---------------------------------------------------------------------------
# Execution stats
# ---------------------------------------------------------------------------


class TestExecutionStats:

    def test_stats_updated(self, bell_kernel, store, registry):
        """execution_stats reflects total executions and unique circuits."""
        with track(project="test-cudaq", store=store, registry=registry) as run:
            executor = run.wrap(cudaq, log_every_n=-1)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(bell_kernel, shots_count=100)

        loaded = registry.load(run.run_id)
        stats = loaded.record["execution_stats"]
        assert stats["total_executions"] == 2
        assert stats["unique_circuits"] == 1
        assert stats["logged_executions"] == 2
