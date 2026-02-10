# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit hashing and canonicalization."""

import json

from devqubit_cudaq.circuits import (
    _native_json_to_op_stream,
    canonicalize_call_args,
    canonicalize_kernel_json,
    compute_circuit_hashes,
)


# ============================================================================
# canonicalize_kernel_json
# ============================================================================


class TestCanonicalizeKernelJson:

    def test_strips_debug_keys(self):
        raw = json.dumps(
            {
                "name": "bell",
                "instructions": [{"gate": "h", "qubits": [0]}],
                "__file__": "/tmp/foo.py",
                "__line__": 42,
                "debug": {"trace_id": "abc"},
            }
        )
        canonical = canonicalize_kernel_json(raw)
        parsed = json.loads(canonical)
        assert "__file__" not in parsed
        assert "__line__" not in parsed
        assert "debug" not in parsed
        assert parsed["name"] == "bell"

    def test_normalises_floats_to_hex(self):
        raw = json.dumps({"params": [0.5, 1.0]})
        canonical = canonicalize_kernel_json(raw)
        # Floats are now hex strings, not 0.5 / 1.0
        assert "0.5" not in canonical
        assert "1.0" not in canonical
        parsed = json.loads(canonical)
        assert len(parsed["params"]) == 2

    def test_sorted_keys_deterministic(self):
        a = canonicalize_kernel_json('{"b": 1, "a": 2}')
        b = canonicalize_kernel_json('{"a": 2, "b": 1}')
        assert a == b

    def test_negative_zero_normalised(self):
        a = canonicalize_kernel_json('{"v": 0.0}')
        b = canonicalize_kernel_json('{"v": -0.0}')
        assert a == b

    def test_nested_strip(self):
        raw = json.dumps({"instructions": [{"gate": "h", "location": "/src:10"}]})
        canonical = canonicalize_kernel_json(raw)
        parsed = json.loads(canonical)
        assert "location" not in parsed["instructions"][0]

    def test_invalid_json_raises(self):
        import pytest

        with pytest.raises(json.JSONDecodeError):
            canonicalize_kernel_json("not json")


# ============================================================================
# canonicalize_call_args
# ============================================================================


class TestCanonicalizeCallArgs:

    def test_empty_args_returns_empty(self):
        assert canonicalize_call_args(()) == ""

    def test_positional_args_deterministic(self):
        a = canonicalize_call_args((0.5, 1.0))
        b = canonicalize_call_args((0.5, 1.0))
        assert a == b
        assert a != ""

    def test_different_values_different_output(self):
        a = canonicalize_call_args((0.5,))
        b = canonicalize_call_args((1.0,))
        assert a != b

    def test_kwargs_sorted(self):
        a = canonicalize_call_args((), {"z": 1, "a": 2})
        b = canonicalize_call_args((), {"a": 2, "z": 1})
        assert a == b

    def test_mixed_args_and_kwargs(self):
        result = canonicalize_call_args((0.5,), {"theta": 1.0})
        parsed = json.loads(result)
        assert "a" in parsed
        assert "k" in parsed


# ============================================================================
# _native_json_to_op_stream
# ============================================================================


class TestNativeJsonToOpStream:

    def test_bell_kernel_json_schema(self, bell_kernel):
        """Real CUDAQ to_json() may use MLIR-based schema.

        If the SDK version emits a simple instruction list, we parse it;
        otherwise the function correctly returns None and the summarizer
        falls back to MLIR parsing.
        """
        raw = bell_kernel.to_json()
        parsed = _native_json_to_op_stream(raw)
        # Either parses successfully or returns None (both are valid)
        if parsed is not None:
            ops, nq = parsed
            assert nq == 2
            gate_names = [op["gate"] for op in ops]
            assert "h" in gate_names

    def test_parameterized_kernel_preserves_params(self, parameterized_kernel):
        """When JSON schema IS an instruction list, params are extracted."""
        raw = parameterized_kernel.to_json()
        parsed = _native_json_to_op_stream(raw)
        if parsed is not None:
            ops, _ = parsed
            param_ops = [op for op in ops if op.get("params")]
            assert len(param_ops) >= 1

    def test_unrecognised_schema_returns_none(self):
        assert _native_json_to_op_stream('{"weird": "schema"}') is None

    def test_invalid_json_returns_none(self):
        assert _native_json_to_op_stream("not json at all") is None

    def test_empty_instructions_returns_none(self):
        assert _native_json_to_op_stream('{"instructions": []}') is None

    def test_qubit_index_counting(self):
        j = json.dumps(
            {
                "instructions": [
                    {"gate": "h", "qubits": [0]},
                    {"gate": "cx", "qubits": [0, 4]},
                ]
            }
        )
        _, nq = _native_json_to_op_stream(j)
        assert nq == 5

    def test_controls_merged_into_qubits(self):
        j = json.dumps(
            {
                "instructions": [
                    {"gate": "x", "qubits": [2], "controls": [0, 1]},
                ]
            }
        )
        ops, nq = _native_json_to_op_stream(j)
        assert ops[0]["qubits"] == [0, 1, 2]
        assert nq == 3


# ============================================================================
# compute_circuit_hashes
# ============================================================================


class TestComputeCircuitHashes:

    def test_identical_kernels_same_hash(self, bell_kernel):
        h1 = compute_circuit_hashes(bell_kernel)
        h2 = compute_circuit_hashes(bell_kernel)
        assert h1 == h2

    def test_different_kernels_different_hash(self, bell_kernel, ghz_kernel):
        h_bell = compute_circuit_hashes(bell_kernel)
        h_ghz = compute_circuit_hashes(ghz_kernel)
        assert h_bell[0] != h_ghz[0]

    def test_hash_format_sha256(self, bell_kernel):
        s, p = compute_circuit_hashes(bell_kernel)
        assert s.startswith("sha256:")
        assert len(s) == 7 + 64

    def test_no_to_json_falls_back_to_name(self, bare_kernel):
        s, p = compute_circuit_hashes(bare_kernel)
        assert s is not None
        assert s == p  # No args → structural == parametric

    def test_non_kernel_does_not_crash(self):
        s, p = compute_circuit_hashes(42)
        assert isinstance(s, str)

    def test_different_args_different_parametric_hash(self, parameterized_kernel):
        _, p1 = compute_circuit_hashes(parameterized_kernel, (0.5,))
        _, p2 = compute_circuit_hashes(parameterized_kernel, (1.0,))
        assert p1 != p2

    def test_different_args_same_structural_hash(self, parameterized_kernel):
        """Structural hash is based on canonical JSON (no args), so
        two calls with different args share the same structural hash."""
        s1, _ = compute_circuit_hashes(parameterized_kernel, (0.5,))
        s2, _ = compute_circuit_hashes(parameterized_kernel, (1.0,))
        assert s1 == s2

    def test_no_args_structural_equals_parametric(self, bell_kernel):
        s, p = compute_circuit_hashes(bell_kernel)
        assert s == p

    def test_with_args_structural_differs_from_parametric(self, parameterized_kernel):
        s, p = compute_circuit_hashes(parameterized_kernel, (0.5,))
        assert s != p

    def test_kwargs_affect_parametric_hash(self, parameterized_kernel):
        _, p1 = compute_circuit_hashes(parameterized_kernel, (), {"theta": 0.5})
        _, p2 = compute_circuit_hashes(parameterized_kernel, (), {"theta": 1.0})
        assert p1 != p2
