# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for QiskitRuntimeAdapter registration and primitive detection."""

from __future__ import annotations

from devqubit_engine.tracking.run import track
from devqubit_qiskit_runtime.adapter import QiskitRuntimeAdapter


class TestAdapterRegistration:
    """Tests for adapter identification and primitive support detection."""

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

    def test_rejects_non_runtime_executor(self):
        """Adapter rejects non-Runtime executors."""
        adapter = QiskitRuntimeAdapter()

        class FakeExecutor:
            __module__ = "some.other.module"

            def run(self):
                pass

        assert adapter.supports_executor(FakeExecutor()) is False
        assert adapter.supports_executor(None) is False

    def test_rejects_executor_without_run(self):
        """Adapter rejects objects without run method."""
        adapter = QiskitRuntimeAdapter()

        class NoRunMethod:
            __module__ = "qiskit_ibm_runtime.sampler"

        assert adapter.supports_executor(NoRunMethod()) is False


class TestAdapterDescribe:
    """Tests for primitive description."""

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

        assert desc["provider"] == "fake"
        assert desc["primitive_type"] == "estimator"


class TestAdapterWrap:
    """Tests for wrapping primitives with tracking."""

    def test_wrap_returns_tracked_primitive(self, store, registry, fake_sampler):
        """Wrapping returns a TrackedRuntimePrimitive with correct attributes."""
        adapter = QiskitRuntimeAdapter()

        with track(project="test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)

            assert hasattr(wrapped, "run")
            assert wrapped.primitive is fake_sampler
            assert wrapped.primitive_type == "sampler"
            assert wrapped.tracker is run

    def test_wrap_with_custom_log_every_n(self, store, registry, fake_sampler):
        """Wrapping respects log_every_n parameter."""
        adapter = QiskitRuntimeAdapter()

        with track(project="test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run, log_every_n=5)
            assert wrapped.log_every_n == 5

    def test_wrap_with_log_new_circuits_disabled(self, store, registry, fake_sampler):
        """Wrapping respects log_new_circuits parameter."""
        adapter = QiskitRuntimeAdapter()

        with track(project="test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run, log_new_circuits=False)
            assert wrapped.log_new_circuits is False

    def test_wrapped_primitive_delegates_attributes(
        self, store, registry, fake_sampler
    ):
        """Wrapped primitive delegates attribute access to original."""
        adapter = QiskitRuntimeAdapter()

        with track(project="test", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)

            # Should delegate to original primitive
            assert wrapped.mode is fake_sampler.mode
            assert wrapped.options is fake_sampler.options
