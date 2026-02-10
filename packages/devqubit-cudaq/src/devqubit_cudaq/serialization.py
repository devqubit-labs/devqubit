# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel serialization and summarization for the CUDA-Q adapter.

Serialization
    ``kernel.to_json()`` / ``PyKernelDecorator.from_json()``
Artifacts
    ``str(kernel)`` => MLIR, ``cudaq.translate()`` => QIR
Visual
    ``cudaq.draw()`` => diagram (logging only)
Summarization
    ``summarize_cudaq_circuit()`` — registered as the
    ``devqubit.circuit.summarizers`` entry-point for the ``cudaq`` SDK.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from devqubit_cudaq.circuits import (
    _CUDAQ_GATES,
    _get_native_json,
    _native_json_to_op_stream,
)
from devqubit_cudaq.utils import (
    get_kernel_name,
    get_kernel_num_qubits,
    is_cudaq_kernel,
)
from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    GateClassifier,
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_engine.circuit.summary import CircuitSummary


logger = logging.getLogger(__name__)

_classifier = GateClassifier(_CUDAQ_GATES)


# ============================================================================
# Capture functions
# ============================================================================


def serialize_kernel_native(kernel: Any) -> str | None:
    """
    Serialize a kernel via ``kernel.to_json()`` (lossless, SDK-native).

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel (``PyKernelDecorator`` or builder-style).

    Returns
    -------
    str or None
        JSON string on success, ``None`` if unavailable or invalid.
    """
    try:
        fn = getattr(kernel, "to_json", None)
        if fn is None or not callable(fn):
            return None
        out = fn()
        if not isinstance(out, str) or not out:
            return None
        json.loads(out)  # validate
        return out
    except json.JSONDecodeError as exc:
        logger.warning("kernel.to_json() returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.debug("kernel.to_json() failed: %s", exc)
        return None


def capture_mlir(kernel: Any) -> str | None:
    """
    Capture Quake MLIR text via ``str(kernel)``.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.

    Returns
    -------
    str or None
        Quake MLIR text, or ``None`` if unavailable.
    """
    try:
        mlir = str(kernel)
        if mlir and ("func.func" in mlir or "quake." in mlir or "module" in mlir):
            return mlir
    except Exception as exc:
        logger.debug("MLIR capture failed: %s", exc)
    return None


def capture_qir(kernel: Any, version: str = "1.0") -> str | None:
    """
    Capture QIR via ``cudaq.translate(kernel, format="qir:<version>")``.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.
    version : str
        QIR version string.  CUDA-Q documents ``0.1`` and ``1.0``.

    Returns
    -------
    str or None
        QIR (LLVM IR) text on success, else ``None``.
    """
    try:
        import cudaq  # local import

        out = cudaq.translate(kernel, format=f"qir:{version}")
        if isinstance(out, str) and out:
            return out
    except Exception as exc:
        logger.debug("QIR capture failed: %s", exc)
    return None


def draw_kernel(kernel: Any, args: tuple[Any, ...] = ()) -> str | None:
    """
    Draw circuit diagram via ``cudaq.draw(kernel, *args)``.

    Used for human-readable logging only — not for hashing.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.
    args : tuple, optional
        Kernel arguments for parameterized circuits.

    Returns
    -------
    str or None
        ASCII diagram on success, else ``None``.
    """
    try:
        import cudaq  # local import

        out = cudaq.draw(kernel, *args)
        if isinstance(out, str) and out:
            return out
    except Exception as exc:
        logger.debug("cudaq.draw() failed: %s", exc)
    return None


# ============================================================================
# Human-readable text
# ============================================================================


def kernel_to_text(
    kernel: Any,
    args: tuple[Any, ...] = (),
    index: int = 0,
) -> str:
    """
    Produce a human-readable kernel summary with circuit diagram.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.
    args : tuple, optional
        Kernel arguments for parameterized circuits.
    index : int, optional
        Kernel index for multi-circuit batches.

    Returns
    -------
    str
        Multi-line summary.
    """
    lines: list[str] = [f"=== Kernel {index} ==="]
    lines.append(f"Name: {get_kernel_name(kernel)}")

    num_qubits = get_kernel_num_qubits(kernel)
    if num_qubits is not None:
        lines.append(f"Qubits: {num_qubits}")

    diagram = draw_kernel(kernel, args)
    lines.append("")
    lines.append("Circuit:")
    lines.append(diagram if diagram else "<unavailable>")

    return "\n".join(lines)


# ============================================================================
# CircuitData serialization
# ============================================================================


def serialize_kernel(
    kernel: Any,
    *,
    name: str = "",
    index: int = 0,
) -> CircuitData:
    """
    Serialize a CUDA-Q kernel to ``CircuitData``.

    Uses ``kernel.to_json()`` as the data payload.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel (``PyKernelDecorator`` or builder-style).
    name : str, optional
        Override kernel name.
    index : int, optional
        Circuit index for multi-circuit batches.

    Returns
    -------
    CircuitData

    Raises
    ------
    SerializerError
        If ``to_json()`` is unavailable or returns invalid data.
    """
    kernel_name = name or get_kernel_name(kernel)
    native_json = serialize_kernel_native(kernel)

    if native_json is None:
        raise SerializerError(
            f"Cannot serialize kernel {kernel_name!r}: "
            f"to_json() unavailable on {type(kernel).__name__}"
        )

    return CircuitData(
        data=native_json,
        format=CircuitFormat.CUDAQ_JSON,
        sdk=SDK.CUDAQ,
        name=kernel_name,
        index=index,
    )


# ============================================================================
# Serializer / Loader
# ============================================================================


class CudaqCircuitSerializer:
    """
    CUDA-Q circuit serializer.

    Serializes kernels to native JSON via ``kernel.to_json()``.
    Registered as the ``devqubit.circuit.serializers`` entry-point.
    """

    name = "cudaq"
    default_format = CircuitFormat.CUDAQ_JSON

    @property
    def sdk(self) -> SDK:
        return SDK.CUDAQ

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        return [CircuitFormat.CUDAQ_JSON]

    def can_serialize(self, circuit: Any) -> bool:
        return is_cudaq_kernel(circuit)

    def serialize(
        self,
        circuit: Any,
        fmt: CircuitFormat | None = None,
        *,
        name: str = "",
        index: int = 0,
        args: tuple[Any, ...] = (),
    ) -> CircuitData:
        """
        Serialize a CUDA-Q kernel.

        Parameters
        ----------
        circuit : Any
            CUDA-Q kernel.
        fmt : CircuitFormat, optional
            Requested format (only ``CUDAQ_JSON`` supported).
        name : str, optional
            Override kernel name.
        index : int, optional
            Circuit index.
        args : tuple, optional
            Kernel arguments (accepted for interface compatibility;
            not currently used in serialization).

        Returns
        -------
        CircuitData
        """
        if fmt is not None and fmt not in self.supported_formats:
            raise SerializerError(f"Unsupported CUDA-Q format: {fmt}")
        return serialize_kernel(circuit, name=name, index=index)

    def serialize_circuit(
        self,
        circuit: Any,
        *,
        name: str = "",
        index: int = 0,
        args: tuple[Any, ...] = (),
    ) -> CircuitData:
        """Convenience alias for ``serialize``."""
        return self.serialize(circuit, name=name, index=index, args=args)


class CudaqCircuitLoader:
    """
    CUDA-Q circuit loader.

    Reconstructs callable kernels from native JSON via
    ``cudaq.PyKernelDecorator.from_json()``.
    """

    name = "cudaq"

    @property
    def sdk(self) -> SDK:
        return SDK.CUDAQ

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        return [CircuitFormat.CUDAQ_JSON]

    def can_load(self, data: CircuitData) -> bool:
        return data.sdk == SDK.CUDAQ and data.format == CircuitFormat.CUDAQ_JSON

    def load(self, data: CircuitData) -> LoadedCircuit:
        """
        Load kernel from native JSON via ``PyKernelDecorator.from_json()``.

        Returns a **callable** ``PyKernelDecorator``.
        """
        native_json = data.as_text()

        try:
            json.loads(native_json)
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Invalid kernel JSON: {exc}") from exc

        try:
            import cudaq  # local import

            kernel = cudaq.PyKernelDecorator.from_json(native_json)
        except ImportError as exc:
            raise LoaderError(
                "cudaq not installed — cannot reconstruct kernel"
            ) from exc
        except Exception as exc:
            raise LoaderError(f"from_json() failed: {exc}") from exc

        return LoadedCircuit(
            circuit=kernel,
            sdk=SDK.CUDAQ,
            source_format=CircuitFormat.CUDAQ_JSON,
            name=data.name,
            index=data.index,
        )


# ============================================================================
# MLIR-based gate extraction (fallback for summarization)
# ============================================================================

# Matches any Quake op name.
_QUAKE_GATE_RE = re.compile(r"quake\.([a-zA-Z_][\w]*)\b")

# Matches ``quake.alloca !quake.veq<N>``
_QUAKE_ALLOC_VEQ_RE = re.compile(r"quake\.alloca\s+!quake\.veq<(\d+)>")

# Matches ``quake.alloca !quake.ref``
_QUAKE_ALLOC_REF_RE = re.compile(r"quake\.alloca\s+!quake\.ref\b")

# Controlled gate pattern: ``quake.x [%c0] %t0``
_QUAKE_CTRL_RE = re.compile(r"quake\.(x|y|z)\s*\[([^\]]+)\]")

# Parameterised rotation ops.
_QUAKE_PARAM_GATES: frozenset[str] = frozenset(
    {"rx", "ry", "rz", "r1", "u3", "phaseshift", "crx", "cry", "crz", "cr1"}
)


def _summarize_from_mlir(
    mlir_text: str,
) -> tuple[int, Counter[str], bool, int] | None:
    """
    Extract gate counts and qubit count from Quake MLIR text.

    Handles controlled gates explicitly to avoid double-counting:
    ``quake.x [%c0] %t`` => ``cx`` (not ``x``).

    Parameters
    ----------
    mlir_text : str
        Raw Quake MLIR text from ``str(kernel)``.

    Returns
    -------
    tuple or None
        ``(num_qubits, gate_counts, has_params, param_count)``, or
        ``None`` if the text does not look like Quake IR.
    """
    if "quake." not in mlir_text:
        return None

    num_qubits = 0
    for m in _QUAKE_ALLOC_VEQ_RE.finditer(mlir_text):
        num_qubits += int(m.group(1))
    num_qubits += len(_QUAKE_ALLOC_REF_RE.findall(mlir_text))

    gate_counts: Counter[str] = Counter()
    has_params = False
    param_count = 0

    # Pass 1: count controlled X/Y/Z => cx/cy/cz/ccx
    for m in _QUAKE_CTRL_RE.finditer(mlir_text):
        base = m.group(1).lower()
        ctrl_list = m.group(2)
        ctrl_count = len([c for c in ctrl_list.split(",") if c.strip()])
        if ctrl_count == 1:
            gate_counts[f"c{base}"] += 1
        elif ctrl_count == 2 and base == "x":
            gate_counts["ccx"] += 1
        else:
            gate_counts[f"mc{base}"] += 1

    # Pass 2: count remaining gates (skip bare x/y/z to avoid double-count)
    for m in _QUAKE_GATE_RE.finditer(mlir_text):
        gate = m.group(1).lower()
        if gate in {"x", "y", "z"}:
            continue
        if gate in _CUDAQ_GATES:
            gate_counts[gate] += 1

    # Detect parameters
    for g in gate_counts:
        if g in _QUAKE_PARAM_GATES:
            has_params = True
            param_count += gate_counts[g]

    if not gate_counts and num_qubits == 0:
        return None

    return num_qubits, gate_counts, has_params, param_count


# ============================================================================
# Summarization — entry-point for devqubit.circuit.summarizers
# ============================================================================


def summarize_cudaq_circuit(kernel: Any) -> CircuitSummary:
    """
    Summarize a CUDA-Q kernel.

    Strategy (in order):
    1) ``kernel.to_json()`` => ``_native_json_to_op_stream()``
    2) ``str(kernel)`` => ``_summarize_from_mlir()``
    3) Minimal summary — never crashes.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.

    Returns
    -------
    CircuitSummary
    """
    gate_counts: Counter[str] = Counter()
    has_params = False
    param_count = 0
    num_qubits = get_kernel_num_qubits(kernel) or 0

    # --- Strategy 1: instruction-list JSON ---
    native_json = _get_native_json(kernel)
    if native_json is not None:
        parsed = _native_json_to_op_stream(native_json)
        if parsed is not None:
            ops, detected_qubits = parsed
            num_qubits = max(num_qubits, detected_qubits)

            for op in ops:
                g = str(op.get("gate", "unknown")).lower()
                gate_counts[g] += 1
                params = op.get("params")
                if params:
                    has_params = True
                    if isinstance(params, (dict, list, tuple)):
                        param_count += len(params)
                    else:
                        param_count += 1

    # --- Strategy 2: MLIR fallback ---
    if not gate_counts:
        mlir_text = capture_mlir(kernel)
        if mlir_text:
            mlir_result = _summarize_from_mlir(mlir_text)
            if mlir_result is not None:
                mlir_nq, mlir_gates, mlir_has_params, mlir_param_count = mlir_result
                num_qubits = max(num_qubits, mlir_nq)
                gate_counts = mlir_gates
                has_params = mlir_has_params
                param_count = mlir_param_count

    stats = _classifier.classify_counts(dict(gate_counts))

    return CircuitSummary(
        num_qubits=num_qubits,
        depth=0,  # CUDA-Q does not expose depth cheaply
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        has_parameters=has_params,
        parameter_count=param_count if has_params else 0,
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.CUDAQ_JSON,
        sdk=SDK.CUDAQ,
    )
