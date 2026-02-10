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
    _classifier,
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
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_engine.circuit.summary import CircuitSummary


logger = logging.getLogger(__name__)


# ============================================================================
# Capture functions
# ============================================================================


def serialize_kernel_native(kernel: Any) -> str | None:
    """
    Primary serialization via ``kernel.to_json()``.

    Lossless, round-trippable via ``PyKernelDecorator.from_json()``.
    """
    try:
        fn = getattr(kernel, "to_json", None)
        if fn is None or not callable(fn):
            return None
        result = fn()
        if not isinstance(result, str) or not result:
            return None
        json.loads(result)  # validate
        return result
    except json.JSONDecodeError as exc:
        logger.warning("kernel.to_json() returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.debug("kernel.to_json() failed: %s", exc)
        return None


def capture_mlir(kernel: Any) -> str | None:
    """Capture MLIR (Quake) text via ``str(kernel)``."""
    try:
        mlir = str(kernel)
        if mlir and ("func.func" in mlir or "quake." in mlir or "module" in mlir):
            return mlir
    except Exception as exc:
        logger.debug("MLIR capture failed: %s", exc)
    return None


def capture_qir(kernel: Any, version: str = "1.0") -> str | None:
    """Capture QIR via ``cudaq.translate(kernel, format="qir:<version>")``."""
    try:
        import cudaq

        result = cudaq.translate(kernel, format=f"qir:{version}")
        if isinstance(result, str) and result:
            return result
    except Exception as exc:
        logger.debug("QIR capture failed: %s", exc)
    return None


def draw_kernel(kernel: Any, args: tuple[Any, ...] = ()) -> str | None:
    """Draw circuit diagram via ``cudaq.draw()``.  Visual logging only."""
    try:
        import cudaq

        return cudaq.draw(kernel, *args)
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
    """Human-readable kernel summary with diagram."""
    lines: list[str] = [f"=== Kernel {index} ==="]
    lines.append(f"Name: {get_kernel_name(kernel)}")

    num_qubits = get_kernel_num_qubits(kernel)
    if num_qubits is not None:
        lines.append(f"Qubits: {num_qubits}")

    diagram = draw_kernel(kernel, args)
    if diagram:
        lines.append("")
        lines.append("Circuit:")
        lines.append(diagram)
    else:
        lines.append("Circuit: <unavailable>")

    return "\n".join(lines)


# ============================================================================
# CircuitData serialization
# ============================================================================


def serialize_kernel(
    kernel: Any,
    args: tuple[Any, ...] = (),
    *,
    name: str = "",
    index: int = 0,
) -> CircuitData:
    """
    Serialize a CUDA-Q kernel to ``CircuitData``.

    Uses ``kernel.to_json()`` as the data payload.

    Raises
    ------
    SerializerError
        If ``to_json()`` is unavailable.
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
    """CUDA-Q circuit serializer.  Native ``to_json()`` as primary format."""

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
        return serialize_kernel(circuit, args, name=name, index=index)

    def serialize_circuit(
        self,
        circuit: Any,
        *,
        name: str = "",
        index: int = 0,
        args: tuple[Any, ...] = (),
    ) -> CircuitData:
        return self.serialize(circuit, name=name, index=index, args=args)


class CudaqCircuitLoader:
    """
    CUDA-Q circuit loader.

    Reconstructs callable kernel via ``PyKernelDecorator.from_json()``.
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
            import cudaq

            kernel = cudaq.PyKernelDecorator.from_json(native_json)
        except ImportError:
            raise LoaderError("cudaq not installed — cannot reconstruct kernel")
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

# Quake gate operations that correspond to known quantum gates.
_QUAKE_GATE_RE = re.compile(
    r"quake\.(" + "|".join(sorted(_CUDAQ_GATES.keys())) + r")\b",
    re.IGNORECASE,
)

# Matches ``quake.alloca !quake.veq<N>``  (vector of qubits).
_QUAKE_ALLOC_VEQ_RE = re.compile(r"quake\.alloca\s+!quake\.veq<(\d+)>")

# Matches ``quake.alloca !quake.ref``  (single qubit).
_QUAKE_ALLOC_REF_RE = re.compile(r"quake\.alloca\s+!quake\.ref\b")

# Parameterised Quake rotation ops — presence implies parameters.
_QUAKE_PARAM_GATES: frozenset[str] = frozenset(
    {"rx", "ry", "rz", "r1", "u3", "phaseshift", "crx", "cry", "crz", "cr1"}
)


def _summarize_from_mlir(
    mlir_text: str,
) -> tuple[int, Counter[str], bool, int] | None:
    """
    Extract gate counts and qubit count from Quake MLIR text.

    Best-effort heuristic covering the common output of ``str(kernel)``
    for ``PyKernelDecorator`` kernels.  Does *not* attempt a full MLIR
    parse.

    Returns
    -------
    tuple or None
        ``(num_qubits, gate_counts, has_params, param_count)`` on
        success, or ``None`` if the text does not look like Quake IR.
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

    for m in _QUAKE_GATE_RE.finditer(mlir_text):
        gate_name = m.group(1).lower()
        gate_counts[gate_name] += 1
        if gate_name in _QUAKE_PARAM_GATES:
            has_params = True
            param_count += 1

    if not gate_counts and num_qubits == 0:
        return None

    return num_qubits, gate_counts, has_params, param_count


# ============================================================================
# Summarization — entry-point for devqubit.circuit.summarizers
# ============================================================================


def summarize_cudaq_circuit(kernel: Any) -> CircuitSummary:
    """
    Summarize a CUDA-Q kernel.

    Strategy (in order of preference):

    1. ``kernel.to_json()`` => ``_native_json_to_op_stream()`` — works
       when the SDK emits a simple instruction-list JSON schema.
    2. ``str(kernel)`` => ``_summarize_from_mlir()`` — works when the
       kernel prints Quake MLIR (the common case for
       ``PyKernelDecorator`` kernels in CUDA-Q ≥ 0.7).
    3. Minimal summary with zero counts — never crashes.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel (``PyKernelDecorator`` or builder-style).

    Returns
    -------
    CircuitSummary
        Circuit summary with gate counts and statistics.
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
            if detected_qubits > num_qubits:
                num_qubits = detected_qubits

            for op in ops:
                gate_name = op.get("gate", "unknown").lower()
                if gate_name.startswith("__"):
                    continue
                gate_counts[gate_name] += 1
                params = op.get("params")
                if params:
                    has_params = True
                    param_count += len(params) if isinstance(params, dict) else 1

    # --- Strategy 2: Quake MLIR fallback ---
    if not gate_counts:
        try:
            mlir_text = str(kernel)
            mlir_result = _summarize_from_mlir(mlir_text)
            if mlir_result is not None:
                mlir_nq, mlir_gates, mlir_has_params, mlir_param_count = mlir_result
                if mlir_nq > num_qubits:
                    num_qubits = mlir_nq
                gate_counts = mlir_gates
                has_params = mlir_has_params
                param_count = mlir_param_count
        except Exception as exc:
            logger.debug("MLIR summarization failed: %s", exc)

    stats = _classifier.classify_counts(dict(gate_counts))

    return CircuitSummary(
        num_qubits=num_qubits,
        depth=0,
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
