# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for the CUDA-Q adapter (adapter.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from devqubit_engine.tracking.run import Run


# =============================================================================
# CudaqAdapter registration
# =============================================================================


class TestCudaqAdapter:

    def test_name(self):
        from devqubit_cudaq.adapter import CudaqAdapter

        assert CudaqAdapter().name == "cudaq"

    def test_rejects_non_cudaq(self):
        from devqubit_cudaq.adapter import CudaqAdapter

        adapter = CudaqAdapter()
        assert adapter.supports_executor("nope") is False
        assert adapter.supports_executor(42) is False
        assert adapter.supports_executor(None) is False


# =============================================================================
# TrackedCudaqExecutor construction
# =============================================================================


class TestTrackedCudaqExecutorInit:

    def test_initial_counters(self, store, registry, make_executor):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            assert executor._execution_count == 0
            assert executor._logged_execution_count == 0
            assert len(executor._seen_circuit_hashes) == 0

    def test_cudaq_module_accessible(self, store, registry, make_executor):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            assert hasattr(executor.cudaq, "sample")
            assert hasattr(executor.cudaq, "observe")

    def test_getattr_passthrough(self, store, registry, make_executor):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.set_target("qpp-cpu")
            executor.cudaq.set_target.assert_called_once_with("qpp-cpu")


# =============================================================================
# Tracked sample() execution
# =============================================================================


class TestSampleExecution:

    def test_returns_result(
        self, store, registry, make_executor, make_sample_result, bell_kernel
    ):
        expected = make_sample_result({"00": 500, "11": 500})
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, sample_result=expected)
            result = executor.sample(bell_kernel, shots_count=1000)
        assert result is expected

    def test_increments_counter(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(bell_kernel, shots_count=100)
            assert executor._execution_count == 3

    def test_calls_cudaq_sample(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.sample(bell_kernel, shots_count=1000)
        executor.cudaq.sample.assert_called_once()
        assert executor.cudaq.sample.call_args[0][0] is bell_kernel


# =============================================================================
# Tracked observe() execution
# =============================================================================


class TestObserveExecution:

    def test_returns_result(
        self, store, registry, make_executor, make_observe_result, bell_kernel
    ):
        expected = make_observe_result(expectation=-0.5)
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, observe_result=expected)
            result = executor.observe(bell_kernel, MagicMock())
        assert result is expected

    def test_calls_cudaq_observe(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.observe(bell_kernel, MagicMock(), 0.5)
        executor.cudaq.observe.assert_called_once()


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:

    def test_sample_error_reraised(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.cudaq.sample = MagicMock(
                side_effect=RuntimeError("Target offline"),
            )
            with pytest.raises(RuntimeError, match="Target offline"):
                executor.sample(bell_kernel, shots_count=100)

    def test_error_logged_to_record(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.cudaq.sample = MagicMock(
                side_effect=ValueError("Bad params"),
            )
            with pytest.raises(ValueError):
                executor.sample(bell_kernel, shots_count=100)
            assert "execution_error" in run.record


# =============================================================================
# Deduplication
# =============================================================================


class TestDeduplication:

    def test_same_kernel_deduplicates(
        self, store, registry, make_executor, bell_kernel
    ):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(
                run,
                log_every_n=0,
                log_new_circuits=True,
            )
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(bell_kernel, shots_count=100)
            assert executor._execution_count == 3
            assert len(executor._seen_circuit_hashes) == 1

    def test_different_kernels_both_logged(
        self, store, registry, make_executor, bell_kernel, ghz_kernel
    ):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, log_new_circuits=True)
            executor.sample(bell_kernel, shots_count=100)
            executor.sample(ghz_kernel, shots_count=100)
            assert len(executor._seen_circuit_hashes) == 2

    def test_log_every_n_minus_one_logs_all(
        self, store, registry, make_executor, bell_kernel
    ):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, log_every_n=-1)
            for _ in range(5):
                executor.sample(bell_kernel, shots_count=100)
            assert executor._execution_count == 5
            assert executor._logged_execution_count == 5


# =============================================================================
# set_target cache invalidation
# =============================================================================


class TestSetTargetInvalidation:

    def test_set_target_resets_snapshot_cache(
        self, store, registry, make_executor, bell_kernel
    ):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.sample(bell_kernel, shots_count=100)
            assert executor._device_snapshot is not None
            executor.set_target("nvidia")
            assert executor._device_snapshot is None


# =============================================================================
# UEC compliance
# =============================================================================


class TestUECCompliance:

    def test_device_snapshot_captured(
        self, store, registry, make_executor, make_target, bell_kernel
    ):
        target = make_target("nvidia", simulator="custatevec")
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, target=target)
            executor.sample(bell_kernel, shots_count=100)
            assert "device_snapshot" in run.record
            assert run.record["device_snapshot"]["sdk"] == "cudaq"

    def test_results_recorded(self, store, registry, make_executor, bell_kernel):
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run)
            executor.sample(bell_kernel, shots_count=1000)
            assert "results" in run.record
            assert run.record["results"]["success"] is True

    def test_tags_set(self, store, registry, make_executor, make_target, bell_kernel):
        target = make_target("ionq", _is_remote=True)
        with Run(store=store, registry=registry, project="test") as run:
            executor = make_executor(run, target=target)
            executor.sample(bell_kernel, shots_count=100)
            tags = run.record.get("data", {}).get("tags", {})
            assert tags.get("sdk") == "cudaq"
