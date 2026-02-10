# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel hashing and canonicalization for the CUDA-Q adapter.

Hash types (UEC contract)
-------------------------
- **structural_hash**: identifies the circuit template (gate sequence,
  wiring, parameter arity) while ignoring concrete parameter values.
- **parametric_hash**: identifies a *concrete execution* of the template
  by incorporating call-site argument values.  For non-parameterized
  circuits, ``parametric_hash == structural_hash``.

Primary representation for hashing
----------------------------------
1) ``kernel.to_json()`` => canonicalized JSON => SHA-256 (SDK-native; most
   common path because CUDA-Q returns ``funcSrc`` schema, not instruction
   lists).
2) Best-effort conversion to the devqubit-engine canonical *op_stream*
   (cross-SDK hashing) when an instruction-list schema is recognized.
3) Fallback: MLIR / diagram / kernel name.

We explicitly avoid using ``cudaq.draw()`` for hashing because it is a
presentational format and (per CUDA-Q docs) omits measurement operations.

Notes
-----
- ``kernel.to_json()`` / ``cudaq.PyKernelDecorator.from_json`` are the
  supported round-trip mechanism in CUDA-Q.
- Float normalization uses IEEE-754 binary64 big-endian hex to avoid
  platform-dependent string representations.
- Hash parts are length-prefixed before feeding SHA-256 to prevent
  ambiguous concatenation collisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import struct
from typing import Any

from devqubit_cudaq.utils import get_kernel_name
from devqubit_engine.circuit.hashing import hash_circuit_pair
from devqubit_engine.circuit.models import GateCategory, GateInfo


logger = logging.getLogger(__name__)

# ============================================================================
# Gate classification (shared between hashing & summarization)
# ============================================================================

_CUDAQ_GATES: dict[str, GateInfo] = {
    # Single-qubit Clifford
    "h": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "x": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "y": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "z": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "s": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sdag": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    # Single-qubit non-Clifford
    "t": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "tdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "tdag": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ry": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "r1": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u3": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "phaseshift": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    # Two-qubit Clifford
    "cnot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "swap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    # Two-qubit non-Clifford
    "crx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cry": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "crz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cr1": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cs": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "ct": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    # Multi-qubit
    "toffoli": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "ccx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "ccnot": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "cswap": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "fredkin": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    # Measurement
    "mz": GateInfo(GateCategory.MEASURE),
    "mx": GateInfo(GateCategory.MEASURE),
    "my": GateInfo(GateCategory.MEASURE),
    "measure": GateInfo(GateCategory.MEASURE),
}

# ============================================================================
# JSON canonicalization
# ============================================================================

# Keys that are unstable across SDK versions / environments.
_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "__file__",
        "__line__",
        "__source__",
        "docstring",
        "doc",
        "metadata",
        "debug",
        "source_location",
        "location",
        "id",
        "uuid",
    }
)


def _float_to_hex(value: float) -> str:
    """
    Convert float to deterministic IEEE-754 binary64 big-endian hex.

    Parameters
    ----------
    value : float

    Returns
    -------
    str
        ``'nan'``, ``'inf'``/``'-inf'``, or 16-char hex string.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        value = 0.0  # normalize -0.0
    return struct.pack(">d", value).hex()


def _canonicalize_obj(obj: Any) -> Any:
    """
    Canonicalize a JSON-like Python object for hashing.

    * Floats => IEEE-754 hex strings.
    * Dict keys in ``_STRIP_KEYS`` are removed.
    * Lists preserve order.

    Parameters
    ----------
    obj : Any

    Returns
    -------
    Any
        Canonicalized object of JSON-safe primitives.
    """
    if isinstance(obj, float):
        return _float_to_hex(obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {
            str(k): _canonicalize_obj(v)
            for k, v in obj.items()
            if str(k) not in _STRIP_KEYS
        }
    return obj


def canonicalize_kernel_json(native_json: str) -> str:
    """
    Canonicalize ``kernel.to_json()`` output for hashing.

    Parameters
    ----------
    native_json : str
        Raw JSON string from ``kernel.to_json()``.

    Returns
    -------
    str
        Deterministic JSON string suitable for SHA-256 hashing.

    Raises
    ------
    json.JSONDecodeError
        If *native_json* is not valid JSON.
    """
    parsed = json.loads(native_json)
    canonical = _canonicalize_obj(parsed)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


# ============================================================================
# Hashing primitives — length-prefixed SHA-256
# ============================================================================


def _sha256_tag_bytes(*parts: bytes) -> str:
    """
    SHA-256 over length-prefixed byte parts (collision-safe).

    Parameters
    ----------
    *parts : bytes

    Returns
    -------
    str
        ``sha256:<hex>``
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return f"sha256:{h.hexdigest()}"


def _sha256_tag_parts(*parts: str) -> str:
    """
    SHA-256 over length-prefixed UTF-8 string parts (collision-safe).

    Parameters
    ----------
    *parts : str

    Returns
    -------
    str
        ``sha256:<hex>``
    """
    h = hashlib.sha256()
    for part in parts:
        b = part.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return f"sha256:{h.hexdigest()}"


# ============================================================================
# Argument serialization
# ============================================================================


def _to_jsonable(value: Any) -> Any:
    """
    Best-effort conversion to a JSON-safe primitive.

    Must be stable across processes and must not embed memory addresses.

    Parameters
    ----------
    value : Any

    Returns
    -------
    Any
        JSON-safe primitive (or stable string representation).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {
            "__bytes__": True,
            "len": len(value),
            "sha256": _sha256_tag_bytes(value),
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    # numpy scalars / arrays (avoid importing numpy at module level)
    mod = getattr(type(value), "__module__", "") or ""
    if "numpy" in mod:
        type_name = type(value).__name__.lower()
        if "float" in type_name:
            return float(value)
        if "int" in type_name:
            return int(value)
        tobytes = getattr(value, "tobytes", None)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if callable(tobytes) and shape is not None and dtype is not None:
            try:
                return {
                    "__ndarray__": True,
                    "dtype": str(dtype),
                    "shape": list(shape),
                    "sha256": _sha256_tag_bytes(tobytes()),
                }
            except Exception:
                pass

    # CUDA-Q types: str() is more stable than repr()
    if "cudaq" in mod:
        try:
            return str(value)
        except Exception:
            pass

    # Last resort: repr, scrub hex addresses for cross-process stability
    s = repr(value)
    s = re.sub(r"0x[0-9a-fA-F]+", "0x…", s)
    return s


def canonicalize_call_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> str:
    """
    Deterministically serialize call-site kernel arguments.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the kernel.
    kwargs : dict, optional
        Keyword arguments passed to the kernel.

    Returns
    -------
    str
        Deterministic JSON string, or empty string if no arguments.
    """
    if not args and not kwargs:
        return ""
    payload: dict[str, Any] = {}
    if args:
        payload["a"] = [_canonicalize_obj(_to_jsonable(a)) for a in args]
    if kwargs:
        payload["k"] = {
            str(k): _canonicalize_obj(_to_jsonable(v))
            for k, v in sorted(kwargs.items())
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ============================================================================
# Hashing — public API
# ============================================================================


def _get_native_json(kernel: Any) -> str | None:
    """Get JSON string from ``kernel.to_json()``, or ``None``."""
    try:
        fn = getattr(kernel, "to_json", None)
        if fn is not None and callable(fn):
            out = fn()
            if isinstance(out, str) and out:
                return out
    except Exception as exc:
        logger.debug("kernel.to_json() failed: %s", exc)
    return None


def compute_circuit_hashes(
    kernel: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Compute structural and parametric hashes for a CUDA-Q kernel.

    Strategy
    --------
    1. **Canonical JSON** — ``kernel.to_json()`` => canonicalize => hash.
       Most common path (CUDA-Q ``funcSrc`` schema).
    2. **op_stream** — if JSON parses into an instruction list, use the
       engine's ``hash_circuit_pair()`` for cross-SDK hashing.
    3. **Raw JSON** — if canonicalization fails.
    4. **MLIR / diagram** — if ``to_json()`` is unavailable.
    5. **Kernel name** — last resort, never returns ``None``.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.
    args : tuple
        Concrete positional kernel arguments.
    kwargs : dict, optional
        Concrete keyword kernel arguments.

    Returns
    -------
    tuple[str, str]
        ``(structural_hash, parametric_hash)``, both ``sha256:<hex>``.
    """
    native_json = _get_native_json(kernel)
    args_canonical = canonicalize_call_args(args, kwargs)

    if native_json is not None:
        # --- Strategy 1: canonical JSON ---
        try:
            canonical = canonicalize_kernel_json(native_json)
            structural = _sha256_tag_parts(canonical)
            parametric = (
                _sha256_tag_parts(canonical, args_canonical)
                if args_canonical
                else structural
            )
            return structural, parametric
        except json.JSONDecodeError as exc:
            logger.debug("Canonical JSON failed (bad JSON): %s", exc)
        except Exception as exc:
            logger.debug("Canonical JSON hashing failed: %s", exc)

        # --- Strategy 2: op_stream via engine (cross-SDK) ---
        try:
            parsed = _native_json_to_op_stream(native_json)
            if parsed is not None:
                ops, nq = parsed
                structural, parametric = hash_circuit_pair(
                    ops, num_qubits=nq, num_clbits=0
                )
                if args_canonical:
                    parametric = _sha256_tag_parts(parametric, args_canonical)
                return structural, parametric
        except Exception as exc:
            logger.debug("op_stream hashing failed: %s", exc)

        # --- Strategy 3: raw JSON hash ---
        structural = _sha256_tag_parts(native_json)
        parametric = (
            _sha256_tag_parts(native_json, args_canonical)
            if args_canonical
            else structural
        )
        return structural, parametric

    # --- Strategy 4: MLIR / diagram ---
    fallback = _get_fallback_content(kernel)
    if fallback:
        structural = _sha256_tag_parts(fallback)
        parametric = (
            _sha256_tag_parts(fallback, args_canonical)
            if args_canonical
            else structural
        )
        return structural, parametric

    # --- Strategy 5: kernel name ---
    name = get_kernel_name(kernel)
    structural = _sha256_tag_parts(name)
    parametric = (
        _sha256_tag_parts(name, args_canonical) if args_canonical else structural
    )
    return structural, parametric


def _get_fallback_content(kernel: Any) -> str | None:
    """
    Best-effort content for hashing when ``to_json()`` is unavailable.

    Tries ``str(kernel)`` (Quake MLIR), then ``cudaq.draw()``.
    """
    try:
        mlir = str(kernel)
        if mlir and ("func.func" in mlir or "quake." in mlir or "module" in mlir):
            return mlir
    except Exception:
        pass

    try:
        import cudaq  # local import

        diagram = cudaq.draw(kernel)
        if isinstance(diagram, str) and diagram.strip():
            return diagram
    except Exception:
        pass

    return None


# ============================================================================
# Native JSON => op_stream (best-effort)
# ============================================================================


def _native_json_to_op_stream(
    native_json: str,
) -> tuple[list[dict[str, Any]], int] | None:
    """
    Parse CUDA-Q native JSON into devqubit-engine op_stream.

    Intentionally conservative: succeeds only for schemas that resemble
    an instruction list.  Returns ``None`` if unrecognized.

    Parameters
    ----------
    native_json : str
        String returned by ``kernel.to_json()``.

    Returns
    -------
    tuple[list[dict[str, Any]], int] or None
        ``(ops, num_qubits)`` or ``None``.
    """
    try:
        data = json.loads(native_json)
    except json.JSONDecodeError:
        return None

    instructions = (
        data.get("instructions") or data.get("operations") or data.get("gates")
    )
    if not isinstance(instructions, list) or not instructions:
        return None

    ops: list[dict[str, Any]] = []
    max_q = -1

    for instr in instructions:
        if not isinstance(instr, dict):
            continue

        gate = instr.get("gate") or instr.get("name") or instr.get("op")
        if gate is None:
            continue

        targets = instr.get("qubits", instr.get("targets", []))
        if isinstance(targets, int):
            targets = [targets]
        controls = instr.get("controls", [])
        if isinstance(controls, int):
            controls = [controls]

        qubits = [int(q) for q in list(controls) + list(targets)]
        if qubits:
            max_q = max(max_q, max(qubits))

        op: dict[str, Any] = {"gate": str(gate).lower(), "qubits": qubits}

        clbits = instr.get("clbits") or instr.get("classical_bits") or []
        if isinstance(clbits, int):
            clbits = [clbits]
        if isinstance(clbits, list) and clbits:
            op["clbits"] = [int(c) for c in clbits]

        raw_params = instr.get("params", instr.get("parameters"))
        if raw_params is not None:
            if isinstance(raw_params, dict):
                op["params"] = {str(k): _to_jsonable(v) for k, v in raw_params.items()}
            elif isinstance(raw_params, (list, tuple)):
                op["params"] = [_to_jsonable(v) for v in raw_params]
            else:
                op["params"] = {"p0": _to_jsonable(raw_params)}

        ops.append(op)

    if not ops:
        return None

    num_qubits = max_q + 1 if max_q >= 0 else 0
    return ops, num_qubits
