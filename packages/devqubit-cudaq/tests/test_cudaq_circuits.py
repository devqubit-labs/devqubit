# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for circuit hashing (circuits.py)."""

import json

from devqubit_cudaq.circuits import (
    _native_json_to_op_stream,
    _parse_diagram_to_ops,
    compute_circuit_hashes,
)


class TestNativeJsonToOpStream:
    """Tests for parsing kernel.to_json() into op_stream."""

    def test_bell_kernel_ops(self, bell_kernel):
        parsed = _native_json_to_op_stream(bell_kernel.to_json())
        assert parsed is not None
        ops, nq = parsed
        gate_names = [op["gate"] for op in ops]
        assert "h" in gate_names
        assert "cx" in gate_names
        assert nq == 2

    def test_parameterized_kernel_preserves_params(self, parameterized_kernel):
        parsed = _native_json_to_op_stream(parameterized_kernel.to_json())
        assert parsed is not None
        ops, _ = parsed
        rx_ops = [op for op in ops if op["gate"] == "rx"]
        assert len(rx_ops) == 1
        assert rx_ops[0]["params"]["p0"] == 0.5

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


class TestComputeCircuitHashes:

    def test_identical_kernels_same_hash(self, bell_kernel):
        h1 = compute_circuit_hashes(bell_kernel)
        h2 = compute_circuit_hashes(bell_kernel)
        assert h1 == h2
        assert h1[0] is not None

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
        # No args → structural == parametric for name-only fallback
        assert s == p

    def test_non_kernel_does_not_crash(self):
        s, p = compute_circuit_hashes(42)
        assert isinstance(s, str)

    def test_different_args_different_parametric_hash(self, parameterized_kernel):
        _, p1 = compute_circuit_hashes(parameterized_kernel, (0.5,))
        _, p2 = compute_circuit_hashes(parameterized_kernel, (1.0,))
        assert p1 != p2

    def test_different_args_same_structural_hash(self, parameterized_kernel):
        s1, _ = compute_circuit_hashes(parameterized_kernel, (0.5,))
        s2, _ = compute_circuit_hashes(parameterized_kernel, (1.0,))
        # Structural hash should be the same (numbers normalized)
        assert s1 == s2

    def test_no_args_hashes_never_none(self, bell_kernel):
        s, p = compute_circuit_hashes(bell_kernel)
        assert s is not None
        assert p is not None


class TestDiagramParsing:
    """Diagram parsing is only for summarization, not hashing."""

    BELL_DIAGRAM = "q0 : ┤ h ├──●──┤ mz ├\n" "             │\n" "q1 : ────┤ x ├──┤ mz ├"

    def test_extracts_gates(self):
        ops, _ = _parse_diagram_to_ops(self.BELL_DIAGRAM)
        names = [op["gate"] for op in ops if not op["gate"].startswith("__")]
        assert "h" in names
        assert "x" in names
        assert "mz" in names

    def test_detects_qubit_count(self):
        _, nq = _parse_diagram_to_ops(self.BELL_DIAGRAM)
        assert nq == 2

    def test_empty_diagram(self):
        ops, nq = _parse_diagram_to_ops("")
        assert ops == []
        assert nq == 0

    def test_parameterized_gate(self):
        ops, _ = _parse_diagram_to_ops("q0 : ┤ rx(0.5) ├")
        rx = [op for op in ops if op["gate"] == "rx"]
        assert len(rx) == 1
        assert rx[0]["params"]["p0"] == 0.5
