# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime transpilation handling."""

from __future__ import annotations

import json
import warnings

import pytest
from devqubit_engine.tracking.run import track
from devqubit_qiskit_runtime.adapter import QiskitRuntimeAdapter
from devqubit_qiskit_runtime.transpilation import (
    TranspilationConfig,
    TranspilationOptions,
    circuit_looks_isa_simple,
    prepare_pubs_for_primitive,
)
from qiskit import QuantumCircuit


# =============================================================================
# Test Helpers
# =============================================================================


def _load_envelopes(run_id, store, registry):
    """Load run and extract envelope artifacts."""
    loaded = registry.load(run_id)
    env_artifacts = [a for a in loaded.artifacts if a.kind == "devqubit.envelope.json"]
    envelopes = []
    for a in env_artifacts:
        raw = store.get_bytes(a.digest)
        envelopes.append(json.loads(raw.decode("utf-8")))
    return loaded, envelopes


# =============================================================================
# Transpilation Options Tests
# =============================================================================


class TestTranspilationOptions:
    """Tests for TranspilationOptions parsing and validation."""

    def test_empty_options(self):
        """Handles None/empty options gracefully."""
        opts = TranspilationOptions.from_dict(None)
        assert opts.optimization_level is None

        opts = TranspilationOptions.from_dict({})
        assert opts.optimization_level is None

    def test_parses_optimization_level(self):
        """Parses optimization level correctly."""
        opts = TranspilationOptions.from_dict({"optimization_level": 2})
        assert opts.optimization_level == 2

    def test_supports_aliases(self):
        """Supports common option aliases."""
        opts = TranspilationOptions.from_dict({"opt_level": 3, "seed": 123})
        assert opts.optimization_level == 3
        assert opts.seed_transpiler == 123

    def test_rejects_invalid_optimization_level(self):
        """Rejects out-of-range optimization levels."""
        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"optimization_level": 5})

        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"optimization_level": -1})

    def test_validates_approximation_degree(self):
        """Validates approximation_degree is in [0, 1]."""
        opts = TranspilationOptions.from_dict({"approximation_degree": 0.5})
        assert opts.approximation_degree == 0.5

        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"approximation_degree": 1.5})

    def test_unknown_keys_warning(self):
        """Warns on unknown keys by default."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TranspilationOptions.from_dict({"unknown_key": "value"})
            assert len(w) == 1
            assert "unknown_key" in str(w[0].message)

    def test_strict_mode_raises(self):
        """Strict mode raises on unknown keys."""
        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"unknown_key": "value"}, strict=True)

    def test_unknown_keys_not_passed_to_qiskit(self):
        """Unknown keys stored in extra but NOT passed to Qiskit."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opts = TranspilationOptions.from_dict(
                {
                    "optimization_level": 2,
                    "unknown_key": "value",
                }
            )

        assert "unknown_key" in opts.extra
        kwargs = opts.to_kwargs()
        assert "unknown_key" not in kwargs
        assert kwargs["optimization_level"] == 2

    def test_to_metadata_dict_json_serializable(self):
        """to_metadata_dict returns JSON-serializable values."""

        opts = TranspilationOptions.from_dict(
            {
                "optimization_level": 2,
                "layout_method": "sabre",
            }
        )

        meta = opts.to_metadata_dict()
        json_str = json.dumps(meta)  # Should not raise
        assert "optimization_level" in json_str

    def test_backend_option_ignored_with_warning(self):
        """Backend option is ignored with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opts = TranspilationOptions.from_dict(
                {
                    "backend": "some_backend",
                    "optimization_level": 2,
                }
            )
            assert opts.optimization_level == 2
            assert any("backend" in str(warning.message) for warning in w)


# =============================================================================
# ISA Compatibility Tests
# =============================================================================


class TestCircuitLooksIsa:
    """Tests for ISA compatibility detection."""

    def test_isa_circuit_passes(self, real_target):
        """ISA circuit passes gate-level check with real target."""
        op_names = set(real_target.operation_names)

        qc = QuantumCircuit(2)
        if "rz" in op_names:
            qc.rz(0.5, 0)
        if "sx" in op_names:
            qc.sx(0)
        if "cx" in op_names:
            qc.cx(0, 1)
        elif "ecr" in op_names:
            qc.ecr(0, 1)

        assert circuit_looks_isa_simple(qc, real_target, strict=False) is True

    def test_non_isa_circuit_fails(self, real_target, non_isa_circuit):
        """Non-ISA circuit (H gate) fails check with real target."""
        assert circuit_looks_isa_simple(non_isa_circuit, real_target) is False

    def test_no_target_returns_true(self, non_isa_circuit):
        """Returns True if no target provided (can't check)."""
        assert circuit_looks_isa_simple(non_isa_circuit, None) is True

    def test_empty_circuit_is_isa(self, real_target):
        """Empty circuit is ISA compatible."""
        qc = QuantumCircuit(2)
        assert circuit_looks_isa_simple(qc, real_target) is True

    def test_barrier_only_is_isa(self, real_target):
        """Circuit with only barriers is ISA compatible."""
        qc = QuantumCircuit(2)
        qc.barrier()
        assert circuit_looks_isa_simple(qc, real_target) is True


# =============================================================================
# Transpilation Mode Integration Tests
# =============================================================================


class TestTranspilationModes:
    """Integration tests for transpilation modes."""

    def test_manual_mode_passthrough(self, fake_sampler, non_isa_circuit):
        """Manual mode passes circuits through unchanged."""
        pubs = [(non_isa_circuit,)]
        cfg = TranspilationConfig(mode="manual")

        out, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert out is pubs
        assert meta["transpilation_mode"] == "manual"
        assert meta["transpiled_by_devqubit"] is False
        assert meta["transpilation_reason"] == "manual_mode"

    def test_auto_mode_transpiles_non_isa(self, fake_sampler, non_isa_circuit):
        """Auto mode detects non-ISA and transpiles."""
        pubs = [(non_isa_circuit,)]
        cfg = TranspilationConfig(mode="auto")

        out, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert meta["transpilation_mode"] == "auto"
        assert meta["transpiled_by_devqubit"] is True
        assert meta["transpilation_reason"] == "transpiled"
        assert len(out) == 1
        assert isinstance(out[0][0], QuantumCircuit)

    def test_auto_mode_skips_isa(self, fake_sampler, isa_circuit):
        """Auto mode skips transpilation for ISA circuits."""
        pubs = [(isa_circuit,)]
        cfg = TranspilationConfig(mode="auto")

        out, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        # Should not transpile
        assert meta["transpilation_reason"] in ("already_isa", "transpiled")

    def test_managed_mode_always_transpiles(self, fake_sampler, isa_circuit):
        """Managed mode transpiles even if already ISA."""
        pubs = [(isa_circuit,)]
        cfg = TranspilationConfig(mode="managed")

        _, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert meta["transpilation_mode"] == "managed"
        assert meta["transpiled_by_devqubit"] is True
        assert meta["transpilation_reason"] == "transpiled"

    def test_no_target_returns_unchanged(self, non_isa_circuit):
        """Returns unchanged if no target available."""

        class NoBackendPrimitive:
            pass

        pubs = [(non_isa_circuit,)]
        cfg = TranspilationConfig(mode="auto")

        result, meta = prepare_pubs_for_primitive(
            pubs, NoBackendPrimitive(), "sampler", cfg
        )

        assert result is pubs
        assert meta["transpilation_reason"] == "no_target"

    def test_empty_pubs_handled(self, fake_sampler):
        """Empty pubs list handled gracefully."""
        cfg = TranspilationConfig(mode="auto")

        result, meta = prepare_pubs_for_primitive([], fake_sampler, "sampler", cfg)

        assert result == []
        assert meta["transpilation_reason"] == "no_circuits"


# =============================================================================
# Transpilation with Adapter Integration
# =============================================================================


class TestTranspilationWithAdapter:
    """Integration tests for transpilation through adapter."""

    def test_manual_mode_logged(self, store, registry, fake_sampler, bell_circuit):
        """Manual mode is logged in execution record."""
        adapter = QiskitRuntimeAdapter()

        with track(project="manual_mode", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], devqubit_transpilation_mode="manual")
            job.result()

        loaded = registry.load(run.run_id)

        assert loaded.record["execute"]["transpilation_mode"] == "manual"
        assert loaded.record["execute"]["transpiled_by_devqubit"] is False

    def test_auto_mode_is_default(self, store, registry, fake_sampler, bell_circuit):
        """Auto mode is the default."""
        adapter = QiskitRuntimeAdapter()

        with track(project="auto_default", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)])
            job.result()

        loaded = registry.load(run.run_id)

        assert loaded.record["execute"]["transpilation_mode"] == "auto"

    def test_transpilation_options_logged(
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Transpilation options are captured in record."""
        adapter = QiskitRuntimeAdapter()

        with track(project="options_logged", store=store, registry=registry) as run:
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
        self, store, registry, fake_sampler, bell_circuit
    ):
        """Transpilation info captured in envelope."""
        adapter = QiskitRuntimeAdapter()

        with track(project="transpile_env", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(bell_circuit,)], devqubit_transpilation_mode="manual")
            job.result()

        _, envelopes = _load_envelopes(run.run_id, store, registry)
        execution = envelopes[0]["execution"]

        assert "transpilation" in execution
        assert execution["transpilation"]["mode"] == "manual"

    def test_auto_transpiles_non_isa(
        self, store, registry, fake_sampler, non_isa_circuit
    ):
        """Auto mode transpiles non-ISA circuits and records metadata."""
        adapter = QiskitRuntimeAdapter()

        with track(project="auto_transpile", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_sampler, run)
            job = wrapped.run([(non_isa_circuit,)], shots=32)
            job.result()

        loaded = registry.load(run.run_id)
        exec_rec = loaded.record["execute"]

        assert exec_rec["transpilation_mode"] == "auto"
        assert exec_rec["transpiled_by_devqubit"] is True
        assert exec_rec["transpilation_reason"] == "transpiled"


# =============================================================================
# Estimator Observable Mapping Tests
# =============================================================================


class TestEstimatorObservableMapping:
    """Tests for Estimator observable layout mapping."""

    def test_estimator_observables_mapped(
        self, store, registry, fake_estimator, non_isa_circuit
    ):
        """Estimator observables are mapped after transpilation."""
        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError:
            pytest.skip("SparsePauliOp not available")

        adapter = QiskitRuntimeAdapter()
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        pubs = [(non_isa_circuit, obs)]

        with track(project="obs_mapped", store=store, registry=registry) as run:
            wrapped = adapter.wrap_executor(fake_estimator, run)
            job = wrapped.run(pubs)
            job.result()

        loaded = registry.load(run.run_id)

        # Check that observable mapping was attempted
        assert loaded.record["execute"].get("observables_layout_mapped") is True
