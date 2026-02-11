# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel serialization and summarization for the CUDA-Q adapter.

Serialization
    Enriched JSON envelope with structured ``operations`` parsed from
    kernel source, plus the native ``to_json()`` blob for round-trip
    via ``PyKernelDecorator.from_json()``.
Capture helpers
    ``capture_mlir()``, ``capture_qir()``, ``draw_kernel()`` — used
    by the adapter to log separate MLIR, QIR, and diagram artifacts.
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

import numpy as np
from devqubit_cudaq.circuits import (
    _CUDAQ_GATES,
    _get_native_json,
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
    Render a CUDA-Q kernel as an ASCII circuit diagram.

    This helper calls ``cudaq.draw`` to obtain a human-readable circuit
    representation. It is intended for logging/debugging only (e.g., printing
    a GHZ circuit or a VQE ansatz). It must not be used for hashing or any
    logic that depends on a stable, deterministic textual representation.

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
        import cudaq

        # If user passed a single array/list/tuple as the only arg,
        # auto-expand it: args=((a,b,c),) -> args=(a,b,c)
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            args = tuple(args[0])

        # Convert numpy scalars to Python scalars
        norm_args = []
        for a in args:
            if isinstance(a, np.generic):  # np.float64, np.int64, ...
                a = a.item()
            norm_args.append(a)
        args = tuple(norm_args)

        # Try common call pattern
        out = cudaq.draw(kernel, *args)
        if isinstance(out, str) and out:
            return out

        # Fallback: explicit format first (some versions expose this signature)
        out = cudaq.draw("ascii", kernel, *args)
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
# funcSrc → structured operations
# ============================================================================

# Gate call: ``h(q[0])`` or ``rx(0.5, q[0])``
_GATE_CALL_RE = re.compile(r"\b([a-z]\w*)\s*\(([^)]*)\)")
# Controlled: ``x.ctrl(q[0], q[1])``
_CTRL_CALL_RE = re.compile(r"\b([a-z]\w*)\.ctrl\s*\(([^)]*)\)")
# Qubit reference: ``q[0]`` or ``q[i + 1]``
_QREF_RE = re.compile(r"\w+\[([^\]]+)\]")

# Gates recognized by the parser.
_KNOWN_GATES: frozenset[str] = frozenset(
    {
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "rx",
        "ry",
        "rz",
        "r1",
        "sdg",
        "tdg",
        "swap",
        "u3",
        "phaseshift",
        "mz",
        "mx",
        "my",
    }
)

# Qubit allocation: ``cudaq.qvector(N)`` or ``cudaq.qubit()``
_QVECTOR_RE = re.compile(r"cudaq\.qvector\s*\(\s*(\d+)\s*\)")
_QUBIT_RE = re.compile(r"cudaq\.qubit\s*\(\s*\)")


def _parse_num_qubits_from_funcSrc(func_src: str) -> int:
    """
    Extract qubit count from ``cudaq.qvector(N)`` / ``cudaq.qubit()`` calls.

    Parameters
    ----------
    func_src : str
        Python source from ``funcSrc``.

    Returns
    -------
    int
        Total qubits allocated (0 if none found).
    """
    total = 0
    for m in _QVECTOR_RE.finditer(func_src):
        total += int(m.group(1))
    total += len(_QUBIT_RE.findall(func_src))
    return total


def _parse_operations_from_funcSrc(func_src: str) -> list[dict[str, Any]]:
    """
    Extract gate operations from kernel Python source code.

    Parses common CUDA-Q gate-call patterns into structured dicts.
    Loops and variables are **not** unrolled — this captures the
    *template* structure, not the execution trace.

    Parameters
    ----------
    func_src : str
        Python source from the ``funcSrc`` field of ``kernel.to_json()``.

    Returns
    -------
    list of dict
        Ordered gate operations.  Each dict has at minimum a ``"gate"``
        key.  Qubit targets that resolve to integers get ``"targets"``
        / ``"controls"`` lists; unresolved expressions (loop variables)
        appear under ``"targets_expr"`` / ``"controls_expr"``.

    Examples
    --------
    >>> ops = _parse_operations_from_funcSrc("h(q[0])\\ncx.ctrl(q[0], q[1])")
    >>> ops[0]
    {'gate': 'h', 'targets': [0]}
    """
    ops: list[dict[str, Any]] = []

    for line in func_src.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@", "def ", "q =", "q=")):
            continue

        # --- Controlled gates: x.ctrl(q[c], q[t]) ---
        m = _CTRL_CALL_RE.search(stripped)
        if m:
            gate = m.group(1).lower()
            qrefs = _QREF_RE.findall(m.group(2))
            op: dict[str, Any] = {"gate": f"c{gate}"}
            if len(qrefs) >= 2:
                ctrls, tgt = qrefs[:-1], qrefs[-1]
                try:
                    op["controls"] = [int(c) for c in ctrls]
                except ValueError:
                    op["controls_expr"] = ctrls
                try:
                    op["targets"] = [int(tgt)]
                except ValueError:
                    op["targets_expr"] = [tgt]
            ops.append(op)
            continue

        # --- Regular gate calls ---
        m = _GATE_CALL_RE.search(stripped)
        if m:
            gate = m.group(1).lower()
            if gate not in _KNOWN_GATES:
                continue
            arg_str = m.group(2)
            qrefs = _QREF_RE.findall(arg_str)

            op = {"gate": gate}

            # Measurement on whole register: mz(q)
            if gate in {"mz", "mx", "my"} and not qrefs:
                op["targets"] = "all"
                ops.append(op)
                continue

            # Parameters: float literals or variable names before qubit refs
            params: list[Any] = []
            for part in (p.strip() for p in arg_str.split(",")):
                if not _QREF_RE.search(part) and part:
                    try:
                        params.append(float(part))
                    except ValueError:
                        params.append(part)
            if params:
                op["params"] = params

            if qrefs:
                try:
                    op["targets"] = [int(q) for q in qrefs]
                except ValueError:
                    op["targets_expr"] = qrefs

            ops.append(op)

    return ops


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
    Serialize a CUDA-Q kernel to enriched ``CircuitData``.

    Produces a JSON envelope with structured ``operations`` parsed from
    ``funcSrc``, plus the original ``to_json()`` blob preserved in a
    ``native`` field for lossless ``from_json()`` round-trip.

    MLIR, QIR, and diagram representations are **not** included here;
    the adapter logs those as separate artifacts.

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

    # Build enriched envelope
    try:
        native_parsed = json.loads(native_json)
    except json.JSONDecodeError:
        native_parsed = None

    envelope: dict[str, Any] = {
        "sdk": "cudaq",
        "format_version": 1,
        "name": kernel_name,
    }

    nq = get_kernel_num_qubits(kernel)

    # Parse operations from funcSrc
    if native_parsed is not None:
        func_src = native_parsed.get("funcSrc", "")
        if func_src:
            if nq is None:
                nq = _parse_num_qubits_from_funcSrc(func_src) or None
            ops = _parse_operations_from_funcSrc(func_src)
            if ops:
                envelope["operations"] = ops

    if nq is not None:
        envelope["num_qubits"] = nq

    # Preserve original for from_json() round-trip
    envelope["native"] = native_parsed if native_parsed is not None else native_json

    enriched = json.dumps(envelope, indent=2, default=str)

    return CircuitData(
        data=enriched,
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

    Produces enriched JSON with structured ``operations`` parsed from
    kernel source, plus the native ``to_json()`` blob for round-trip.
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
        Reconstruct a callable kernel from serialized JSON.

        Handles both the enriched envelope (extracts ``native`` field)
        and legacy raw ``funcSrc`` format from ``kernel.to_json()``.

        Parameters
        ----------
        data : CircuitData
            Serialized circuit data.

        Returns
        -------
        LoadedCircuit
            Container with a callable ``PyKernelDecorator``.

        Raises
        ------
        LoaderError
            If JSON is invalid or reconstruction fails.
        """
        raw_json = data.as_text()

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Invalid kernel JSON: {exc}") from exc

        # Enriched format: extract native blob for from_json()
        if isinstance(parsed, dict) and "native" in parsed:
            native_json = json.dumps(parsed["native"])
        elif isinstance(parsed, dict) and "funcSrc" in parsed:
            # Legacy: raw to_json() output
            native_json = raw_json
        else:
            raise LoaderError(
                "Kernel JSON missing 'native' or 'funcSrc' — "
                "cannot reconstruct kernel"
            )

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

    1. Parse ``funcSrc`` from ``kernel.to_json()`` via
       ``_parse_operations_from_funcSrc()`` — works for all
       ``PyKernelDecorator`` kernels.
    2. ``str(kernel)`` → ``_summarize_from_mlir()`` — fallback for
       builder-style kernels without ``funcSrc``.
    3. Minimal summary — never crashes.

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

    # --- Strategy 1: parse operations from funcSrc ---
    native_json = _get_native_json(kernel)
    if native_json is not None:
        try:
            parsed = json.loads(native_json)
            func_src = parsed.get("funcSrc", "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            func_src = ""

        if func_src:
            num_qubits = max(num_qubits, _parse_num_qubits_from_funcSrc(func_src))
            ops = _parse_operations_from_funcSrc(func_src)
            for op in ops:
                g = str(op.get("gate", "unknown")).lower()
                gate_counts[g] += 1
                params = op.get("params")
                if params:
                    has_params = True
                    if isinstance(params, (list, tuple)):
                        param_count += len(params)
                    else:
                        param_count += 1

    # --- Strategy 2: MLIR fallback (gates) or supplement (qubit count) ---
    if not gate_counts or num_qubits == 0:
        mlir_text = capture_mlir(kernel)
        if mlir_text:
            mlir_result = _summarize_from_mlir(mlir_text)
            if mlir_result is not None:
                mlir_nq, mlir_gates, mlir_has_params, mlir_param_count = mlir_result
                num_qubits = max(num_qubits, mlir_nq)
                if not gate_counts:
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
