# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel serialization for the CUDA-Q adapter.

Primary:    ``kernel.to_json()`` / ``PyKernelDecorator.from_json()``
Artifacts:  ``str(kernel)`` → MLIR, ``cudaq.translate()`` → QIR
Visual:     ``cudaq.draw()`` → diagram (logging only)
"""

from __future__ import annotations

import json
import logging
from typing import Any

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
