# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for kernel serialization and summarization."""

import json

import pytest
from devqubit_cudaq.circuits import _CUDAQ_GATES
from devqubit_cudaq.serialization import (
    CudaqCircuitLoader,
    CudaqCircuitSerializer,
    capture_mlir,
    capture_qir,
    draw_kernel,
    kernel_to_text,
    serialize_kernel,
    serialize_kernel_native,
    summarize_cudaq_circuit,
)
from devqubit_engine.circuit.models import SDK, CircuitFormat
from devqubit_engine.circuit.registry import LoaderError, SerializerError


class TestSerializeKernelNative:

    def test_returns_valid_json(self, bell_kernel):
        result = serialize_kernel_native(bell_kernel)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == "bell"

    def test_no_to_json_returns_none(self, bare_kernel):
        assert serialize_kernel_native(bare_kernel) is None

    def test_non_kernel_returns_none(self):
        assert serialize_kernel_native("not a kernel") is None
        assert serialize_kernel_native(42) is None


class TestCaptureMlir:

    def test_real_kernel_returns_mlir(self, bell_kernel):
        mlir = capture_mlir(bell_kernel)
        assert mlir is not None
        assert "quake." in mlir or "func.func" in mlir

    def test_non_kernel_returns_none(self, bare_kernel):
        assert capture_mlir(bare_kernel) is None

    def test_none_input(self):
        assert capture_mlir(None) is None


class TestCaptureQir:

    def test_real_kernel_returns_qir(self, bell_kernel):
        qir = capture_qir(bell_kernel)
        assert qir is not None
        assert "ModuleID" in qir or "llvm" in qir.lower()

    def test_non_kernel_returns_none(self, bare_kernel):
        assert capture_qir(bare_kernel) is None


class TestDrawKernel:

    def test_real_kernel_returns_diagram(self, bell_kernel):
        diagram = draw_kernel(bell_kernel)
        assert diagram is not None
        assert "q0" in diagram or "q1" in diagram

    def test_non_kernel_returns_none(self, bare_kernel):
        assert draw_kernel(bare_kernel) is None


class TestSerializeKernel:

    def test_produces_circuit_data(self, bell_kernel):
        data = serialize_kernel(bell_kernel)
        assert data.format == CircuitFormat.CUDAQ_JSON
        assert data.sdk == SDK.CUDAQ
        assert data.name == "bell"
        parsed = json.loads(data.data)
        assert "instructions" in parsed

    def test_no_to_json_raises(self, bare_kernel):
        with pytest.raises(SerializerError):
            serialize_kernel(bare_kernel)

    def test_name_override(self, bell_kernel):
        data = serialize_kernel(bell_kernel, name="custom_bell")
        assert data.name == "custom_bell"


class TestCudaqCircuitSerializer:

    def test_name(self):
        assert CudaqCircuitSerializer().name == "cudaq"

    def test_sdk(self):
        assert CudaqCircuitSerializer().sdk == SDK.CUDAQ

    def test_can_serialize_mock(self, bell_kernel):
        result = CudaqCircuitSerializer().can_serialize(bell_kernel)
        assert isinstance(result, bool)

    def test_serialize_circuit(self, bell_kernel):
        data = CudaqCircuitSerializer().serialize_circuit(bell_kernel, name="bell")
        assert data.format == CircuitFormat.CUDAQ_JSON

    def test_supported_formats(self):
        assert CircuitFormat.CUDAQ_JSON in CudaqCircuitSerializer().supported_formats


class TestCudaqCircuitLoader:

    def test_can_load_native_json(self, bell_kernel):
        data = CudaqCircuitSerializer().serialize_circuit(bell_kernel)
        assert CudaqCircuitLoader().can_load(data) is True

    def test_rejects_wrong_sdk(self):
        from devqubit_engine.circuit.models import CircuitData

        data = CircuitData(
            data="{}",
            format=CircuitFormat.CUDAQ_JSON,
            sdk=SDK.PENNYLANE,
            name="x",
            index=0,
        )
        assert CudaqCircuitLoader().can_load(data) is False

    def test_rejects_wrong_format(self):
        from devqubit_engine.circuit.models import CircuitData

        data = CircuitData(
            data="{}",
            format=CircuitFormat.TAPE_JSON,
            sdk=SDK.CUDAQ,
            name="x",
            index=0,
        )
        assert CudaqCircuitLoader().can_load(data) is False

    def test_load_without_cudaq_raises(self, bell_kernel, monkeypatch):
        """When cudaq is not importable, loader raises LoaderError."""
        data = CudaqCircuitSerializer().serialize_circuit(bell_kernel)
        import builtins

        _real_import = builtins.__import__

        def _block_cudaq(name, *args, **kwargs):
            if name == "cudaq":
                raise ImportError("mocked: cudaq not installed")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_cudaq)
        with pytest.raises(LoaderError, match="cudaq not installed"):
            CudaqCircuitLoader().load(data)

    def test_load_with_cudaq_roundtrips(self, bell_kernel):
        """When cudaq IS available, load reconstructs a kernel."""
        data = CudaqCircuitSerializer().serialize_circuit(bell_kernel)
        loaded = CudaqCircuitLoader().load(data)
        assert loaded.name == "bell"
        assert loaded.sdk == SDK.CUDAQ


class TestKernelToText:

    def test_includes_name(self, bell_kernel):
        assert "bell" in kernel_to_text(bell_kernel)

    def test_includes_header(self, bell_kernel):
        assert "Kernel 0" in kernel_to_text(bell_kernel)


class TestGateClassification:

    def test_clifford_gates(self):
        for gate in ["h", "x", "y", "z", "s", "cnot", "cx", "cz", "swap"]:
            assert _CUDAQ_GATES[gate].is_clifford is True, f"{gate}"

    def test_non_clifford_gates(self):
        for gate in ["t", "rx", "ry", "rz", "toffoli"]:
            assert _CUDAQ_GATES[gate].is_clifford is False, f"{gate}"

    def test_measurement_gates(self):
        for gate in ["mz", "mx", "my", "measure"]:
            assert gate in _CUDAQ_GATES


# ============================================================================
# summarize_cudaq_circuit
# ============================================================================


class TestSummarizeCudaqCircuit:

    def test_bell_kernel_summary(self, bell_kernel):
        summary = summarize_cudaq_circuit(bell_kernel)
        assert summary.num_qubits == 2
        assert summary.sdk == SDK.CUDAQ
        assert summary.source_format == CircuitFormat.CUDAQ_JSON
        assert summary.gate_count_total > 0
        assert "h" in summary.gate_types

    def test_ghz_kernel_summary(self, ghz_kernel):
        summary = summarize_cudaq_circuit(ghz_kernel)
        assert summary.num_qubits == 3
        assert summary.gate_count_total > 0

    def test_parameterized_kernel_detects_params(self, parameterized_kernel):
        summary = summarize_cudaq_circuit(parameterized_kernel)
        assert summary.has_parameters is True
        assert summary.parameter_count >= 2  # rx + ry

    def test_bell_kernel_clifford_detection(self, bell_kernel):
        summary = summarize_cudaq_circuit(bell_kernel)
        if summary.gate_count_total > 0:
            assert summary.is_clifford is True

    def test_parameterized_kernel_not_clifford(self, parameterized_kernel):
        summary = summarize_cudaq_circuit(parameterized_kernel)
        if summary.gate_count_total > 0:
            assert summary.is_clifford is False

    def test_bare_kernel_returns_summary(self, bare_kernel):
        summary = summarize_cudaq_circuit(bare_kernel)
        assert summary.sdk == SDK.CUDAQ
        assert summary.gate_count_total == 0
