# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime transpilation utilities."""

from __future__ import annotations

import warnings

import pytest
from devqubit_qiskit_runtime.transpilation import (
    TranspilationConfig,
    TranspilationOptions,
    apply_layout_to_observables,
    circuit_looks_isa,
    prepare_pubs_for_primitive,
)
from qiskit import QuantumCircuit


class TestTranspilationOptions:
    """Tests for TranspilationOptions parsing and validation."""

    def test_empty_options(self):
        """Handles None/empty options."""
        opts = TranspilationOptions.from_dict(None)
        assert opts.optimization_level is None

        opts = TranspilationOptions.from_dict({})
        assert opts.optimization_level is None

    def test_optimization_level(self):
        """Parses optimization level."""
        opts = TranspilationOptions.from_dict({"optimization_level": 2})
        assert opts.optimization_level == 2

    def test_aliases(self):
        """Supports common aliases."""
        opts = TranspilationOptions.from_dict({"opt_level": 3, "seed": 123})
        assert opts.optimization_level == 3
        assert opts.seed_transpiler == 123

    def test_invalid_optimization_level(self):
        """Rejects invalid optimization levels."""
        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"optimization_level": 5})

        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"optimization_level": -1})

    def test_approximation_degree_validation(self):
        """Validates approximation_degree range [0, 1]."""
        opts = TranspilationOptions.from_dict({"approximation_degree": 0.5})
        assert opts.approximation_degree == 0.5

        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"approximation_degree": 1.5})

        with pytest.raises(ValueError):
            TranspilationOptions.from_dict({"approximation_degree": -0.1})

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

    def test_unknown_keys_not_in_to_kwargs(self):
        """Unknown keys are stored in extra but NOT passed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opts = TranspilationOptions.from_dict(
                {
                    "optimization_level": 2,
                    "unknown_key": "some_value",
                    "another_unknown": 123,
                }
            )

        assert "unknown_key" in opts.extra
        assert "another_unknown" in opts.extra

        kwargs = opts.to_kwargs()
        assert "unknown_key" not in kwargs
        assert "another_unknown" not in kwargs
        assert kwargs["optimization_level"] == 2

    def test_unknown_keys_available_with_include_extra(self):
        """Unknown keys can be included via include_extra for metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opts = TranspilationOptions.from_dict(
                {
                    "optimization_level": 1,
                    "custom_option": "test",
                }
            )

        assert "custom_option" not in opts.to_kwargs()
        kwargs_with_extra = opts.to_kwargs(include_extra=True)
        assert "custom_option" in kwargs_with_extra

    def test_to_kwargs(self):
        """Converts to kwargs dict, excluding None values."""
        opts = TranspilationOptions.from_dict(
            {
                "optimization_level": 2,
                "seed_transpiler": 42,
                "layout_method": "sabre",
            }
        )
        kwargs = opts.to_kwargs()

        assert kwargs["optimization_level"] == 2
        assert kwargs["seed_transpiler"] == 42
        assert kwargs["layout_method"] == "sabre"
        assert "routing_method" not in kwargs

    def test_to_metadata_dict_json_safe(self):
        """to_metadata_dict returns JSON-serializable values."""
        import json

        opts = TranspilationOptions.from_dict(
            {
                "optimization_level": 2,
                "layout_method": "sabre",
            }
        )

        meta = opts.to_metadata_dict()

        json_str = json.dumps(meta)
        assert "optimization_level" in json_str
        assert "layout_method" in json_str

    def test_backend_option_ignored_with_warning(self):
        """Backend option is ignored with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opts = TranspilationOptions.from_dict(
                {"backend": "some_backend", "optimization_level": 2}
            )
            assert opts.optimization_level == 2
            assert len(w) == 1
            assert "backend" in str(w[0].message)


class TestCircuitLooksIsa:
    """Tests for ISA compatibility detection."""

    def test_isa_circuit_with_real_target(self, real_target):
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

        assert circuit_looks_isa(qc, real_target, strict=False) is True

    def test_non_isa_circuit_with_real_target(self, real_target, non_isa_circuit):
        """Non-ISA circuit (H gate) fails check with real target."""
        assert circuit_looks_isa(non_isa_circuit, real_target) is False

    def test_no_target_returns_true(self, non_isa_circuit):
        """Returns True if no target provided (can't check)."""
        assert circuit_looks_isa(non_isa_circuit, None) is True

    def test_barrier_always_allowed(self, real_target):
        """Barriers are always allowed."""
        qc = QuantumCircuit(2)
        qc.barrier()
        assert circuit_looks_isa(qc, real_target) is True

    def test_empty_circuit(self, real_target):
        """Empty circuit is ISA compatible."""
        qc = QuantumCircuit(2)
        assert circuit_looks_isa(qc, real_target) is True

    def test_strict_mode_respects_ibm_rejection(self, real_target, non_isa_circuit):
        """Strict mode (default) respects IBM's ISA rejection."""
        result_strict = circuit_looks_isa(non_isa_circuit, real_target, strict=True)
        assert result_strict is False

    def test_non_strict_mode_fallback(self, real_target, non_isa_circuit):
        """Non-strict mode falls back to gate-name check."""
        result = circuit_looks_isa(non_isa_circuit, real_target, strict=False)
        assert result is False


class TestApplyLayoutToObservables:
    """Tests for observable layout mapping."""

    def test_none_input(self):
        """Returns None for None input."""
        assert apply_layout_to_observables(None, "layout") is None

    def test_none_layout(self):
        """Returns observable unchanged if no layout."""

        class Obs:
            pass

        obs = Obs()
        assert apply_layout_to_observables(obs, None) is obs

    def test_calls_apply_layout(self):
        """Calls apply_layout method if available."""

        class Obs:
            def __init__(self):
                self.called_with = None

            def apply_layout(self, layout):
                self.called_with = layout
                return "mapped"

        obs = Obs()
        result = apply_layout_to_observables(obs, "my_layout")

        assert result == "mapped"
        assert obs.called_with == "my_layout"

    def test_handles_list(self):
        """Maps list of observables."""

        class Obs:
            def apply_layout(self, layout):
                return f"mapped_{layout}"

        obs_list = [Obs(), Obs()]
        result = apply_layout_to_observables(obs_list, "L")

        assert result == ["mapped_L", "mapped_L"]

    def test_handles_tuple(self):
        """Maps tuple of observables, returns tuple."""

        class Obs:
            def apply_layout(self, layout):
                return "mapped"

        obs_tuple = (Obs(), Obs())
        result = apply_layout_to_observables(obs_tuple, "L")

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPreparePubsForPrimitive:
    """Tests for PUB preparation with different transpilation modes."""

    def test_manual_mode_passthrough(self, fake_sampler, non_isa_circuit):
        """Manual mode passes through unchanged, reports transpilation_needed."""
        pubs = [(non_isa_circuit,)]
        cfg = TranspilationConfig(mode="manual")

        out, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert out is pubs
        assert meta["transpilation_mode"] == "manual"
        assert meta["transpiled_by_devqubit"] is False
        assert meta["transpilation_reason"] == "manual_mode"

    def test_auto_mode_detects_and_transpiles_non_isa(
        self, fake_sampler, non_isa_circuit
    ):
        """Auto mode detects non-ISA and transpiles."""
        pubs = [(non_isa_circuit,)]
        cfg = TranspilationConfig(mode="auto")

        out, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert meta["transpilation_mode"] == "auto"
        assert meta["transpiled_by_devqubit"] is True
        assert meta["transpilation_reason"] == "transpiled"
        assert len(out) == 1
        assert isinstance(out[0], tuple) and isinstance(out[0][0], QuantumCircuit)

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

    def test_empty_pubs(self, fake_sampler):
        """Handles empty pubs list."""
        cfg = TranspilationConfig(mode="auto")

        result, meta = prepare_pubs_for_primitive([], fake_sampler, "sampler", cfg)

        assert result == []
        assert meta["transpilation_reason"] == "no_circuits"

    def test_meta_includes_options(self, fake_sampler, non_isa_circuit):
        """Metadata includes transpilation options when provided."""
        pubs = [(non_isa_circuit,)]
        opts = TranspilationOptions(optimization_level=2)
        cfg = TranspilationConfig(mode="manual", options=opts)

        _, meta = prepare_pubs_for_primitive(pubs, fake_sampler, "sampler", cfg)

        assert meta["transpilation_options"]["optimization_level"] == 2

    def test_estimator_observable_mapping_integration(
        self, fake_estimator, non_isa_circuit
    ):
        """Integration test: Estimator PUB observables are mapped after transpilation."""
        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError:
            pytest.skip("SparsePauliOp not available")

        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        pubs = [(non_isa_circuit, obs)]
        cfg = TranspilationConfig(mode="auto", map_observables=True)

        out, meta = prepare_pubs_for_primitive(pubs, fake_estimator, "estimator", cfg)

        assert meta["transpiled_by_devqubit"] is True
        assert meta["observables_layout_mapped"] is True
        assert len(out) == 1

        transpiled_circuit, mapped_obs = out[0][0], out[0][1]
        assert isinstance(transpiled_circuit, QuantumCircuit)
        assert mapped_obs is not None


class TestTranspilationConfig:
    """Tests for TranspilationConfig dataclass."""

    def test_defaults(self):
        """Default values are correct."""
        cfg = TranspilationConfig()

        assert cfg.mode == "auto"
        assert cfg.pass_manager is None
        assert cfg.map_observables is True
        assert cfg.strict_isa_check is True

    def test_valid_modes(self):
        """All valid modes are accepted."""
        valid_modes = {"auto", "manual", "managed", "force"}
        for mode in valid_modes:
            cfg = TranspilationConfig(mode=mode)
            assert cfg.mode == mode

    def test_strict_isa_check_configurable(self):
        """strict_isa_check can be configured."""
        cfg_strict = TranspilationConfig(strict_isa_check=True)
        cfg_lenient = TranspilationConfig(strict_isa_check=False)

        assert cfg_strict.strict_isa_check is True
        assert cfg_lenient.strict_isa_check is False


class TestTranspilationWithRealPassManager:
    """Tests that actually transpile circuits."""

    def test_transpile_non_isa_to_isa(self, fake_backend, non_isa_circuit):
        """Transpile a non-ISA circuit to ISA using real pass manager."""
        try:
            from qiskit.transpiler import generate_preset_pass_manager
        except ImportError:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )

        pm = generate_preset_pass_manager(optimization_level=1, backend=fake_backend)
        isa_circuit = pm.run(non_isa_circuit)

        assert circuit_looks_isa(isa_circuit, fake_backend.target, strict=False) is True

    def test_transpiled_circuit_has_layout(self, fake_backend, non_isa_circuit):
        """Transpiled circuit should have layout information."""
        try:
            from qiskit.transpiler import generate_preset_pass_manager
        except ImportError:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )

        pm = generate_preset_pass_manager(optimization_level=1, backend=fake_backend)
        isa_circuit = pm.run(non_isa_circuit)

        assert hasattr(isa_circuit, "layout")
        assert isa_circuit.layout is not None
