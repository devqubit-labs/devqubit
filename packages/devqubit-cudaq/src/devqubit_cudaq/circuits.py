# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel hashing and canonicalization for the CUDA-Q adapter.

Hashing
-------
Produces two distinct hashes per UEC contract:

- **structural_hash**: Identifies the circuit *template* (gate sequence,
  wiring, parameter arity) while ignoring concrete parameter values.
  Two executions of the same kernel with different angles share a
  structural hash.
- **parametric_hash**: Includes concrete parameter values *and* the
  call-site arguments.  Two executions of the same kernel with
  *different* angles produce different parametric hashes.

Primary path:  ``kernel.to_json()`` => canonicalise JSON => hash.
To build the *parametric* hash the canonicalised call arguments are
appended.  ``cudaq.draw()`` is **not** used for hashing (it is a
presentational format that omits measurements and may change between
SDK versions) — it is logged as a human-readable artifact only.

Gate classification
-------------------
``_CUDAQ_GATES`` and ``_classifier`` live here because they are
fundamental circuit-analysis knowledge shared by both hashing
(``compute_circuit_hashes``) and summarization (in ``serialization.py``).
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
from devqubit_engine.circuit.models import (
    GateCategory,
    GateClassifier,
    GateInfo,
)


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

_classifier = GateClassifier(_CUDAQ_GATES)


# ============================================================================
# JSON canonicalisation
# ============================================================================

# Top-level keys that are debug / unstable metadata and must be stripped
# before hashing so that changes between SDK versions or environments do
# not break hash stability.
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
    """IEEE-754 binary64 big-endian hex — deterministic across platforms."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        value = 0.0  # normalise -0.0
    return struct.pack(">d", value).hex()


def _canonicalise_value(value: Any) -> Any:
    """
    Make a single JSON value deterministic.

    * Floats => IEEE-754 hex string (avoids repr precision issues).
    * Ints / bools / None / strings => pass through.
    * Lists => recurse.
    * Dicts => recurse (keys sorted later by json.dumps).
    """
    if isinstance(value, float):
        return _float_to_hex(value)
    if isinstance(value, dict):
        return {
            k: _canonicalise_value(v) for k, v in value.items() if k not in _STRIP_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalise_value(v) for v in value]
    return value


def canonicalize_kernel_json(native_json: str) -> str:
    """
    Canonicalise ``kernel.to_json()`` output for hashing.

    Steps:

    1. Parse JSON.
    2. Strip unstable / debug keys (file paths, line numbers, UUIDs).
    3. Normalise floats to IEEE-754 hex (platform-stable).
    4. Re-serialise with sorted keys and compact separators.

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
    data = json.loads(native_json)
    canonical = _canonicalise_value(data)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def canonicalize_call_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> str:
    """
    Deterministically serialise call-site arguments.

    Produces a stable string that changes whenever the concrete argument
    values change, so that the *parametric* hash properly reflects the
    invocation parameters.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the kernel.
    kwargs : dict, optional
        Keyword arguments passed to the kernel.

    Returns
    -------
    str
        Deterministic JSON string of the arguments.
    """
    payload: dict[str, Any] = {}
    if args:
        payload["a"] = [_canonicalise_value(_to_jsonable(a)) for a in args]
    if kwargs:
        payload["k"] = {
            str(k): _canonicalise_value(_to_jsonable(v))
            for k, v in sorted(kwargs.items())
        }
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion to a JSON-safe primitive."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    # numpy scalar or similar
    type_name = type(value).__name__
    if "float" in type_name.lower():
        return float(value)
    if "int" in type_name.lower():
        return int(value)
    # For known CUDA-Q types prefer str() (more stable than repr)
    module = getattr(type(value), "__module__", "") or ""
    if "cudaq" in module:
        try:
            return str(value)
        except Exception:
            pass
    # Last resort — use repr, but strip memory addresses (0x...) for
    # cross-process hash stability.
    s = repr(value)
    s = re.sub(r"0x[0-9a-fA-F]+", "0x…", s)
    return s


# ============================================================================
# Hashing — public API
# ============================================================================


def _sha256_tag(payload: str) -> str:
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _get_native_json(kernel: Any) -> str | None:
    """Get JSON string from ``kernel.to_json()``, or ``None``."""
    try:
        fn = getattr(kernel, "to_json", None)
        if fn is not None and callable(fn):
            result = fn()
            if isinstance(result, str) and result:
                return result
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
    1. **Primary — canonical JSON**:
       ``kernel.to_json()`` => canonicalise => structural hash.
       Append canonicalised call arguments => parametric hash.
    2. **Secondary — op_stream** (if JSON parses into an instruction list):
       Feed the engine's ``hash_circuit_pair()`` for cross-SDK–compatible
       hashing, then mix in call arguments for the parametric hash.
    3. **Last resort**: hash the kernel name so that we never return
       ``None``.

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
    (structural_hash, parametric_hash)
        Both ``sha256:<hex>``.  Never returns ``None``.
    """
    native_json = _get_native_json(kernel)

    if native_json is not None:
        # --- Strategy 1: canonical JSON ---
        try:
            canonical = canonicalize_kernel_json(native_json)
            structural_hash = _sha256_tag(canonical)

            args_canonical = canonicalize_call_args(args, kwargs)
            if args_canonical:
                parametric_hash = _sha256_tag(canonical + "|" + args_canonical)
            else:
                parametric_hash = structural_hash

            return structural_hash, parametric_hash
        except json.JSONDecodeError as exc:
            logger.debug("Canonical JSON failed (bad JSON): %s", exc)
        except Exception as exc:
            logger.debug("Canonical JSON hashing failed: %s", exc)

        # --- Strategy 2: op_stream via engine ---
        try:
            parsed = _native_json_to_op_stream(native_json)
            if parsed is not None:
                ops, nq = parsed
                header = {
                    "gate": "__kernel__",
                    "qubits": [],
                    "meta": {"nq": nq, "nc": 0},
                }
                structural, parametric = hash_circuit_pair([header] + ops, nq, 0)
                if args or kwargs:
                    args_canonical = canonicalize_call_args(args, kwargs)
                    if args_canonical:
                        parametric = _sha256_tag(f"{parametric}|{args_canonical}")
                return structural, parametric
        except Exception as exc:
            logger.debug("op_stream hashing failed: %s", exc)

        # Fallback: raw JSON hash
        structural = _sha256_tag(native_json)
        if args or kwargs:
            args_canonical = canonicalize_call_args(args, kwargs)
            parametric = _sha256_tag(f"{native_json}|{args_canonical}")
        else:
            parametric = structural
        return structural, parametric

    # --- Strategy 3: MLIR / str(kernel) / diagram ---
    fallback_content = _get_fallback_content(kernel)
    if fallback_content is not None:
        structural = _sha256_tag(fallback_content)
        if args or kwargs:
            args_canonical = canonicalize_call_args(args, kwargs)
            parametric = _sha256_tag(f"{fallback_content}|{args_canonical}")
        else:
            parametric = structural
        return structural, parametric

    # --- Strategy 4: kernel name (last resort) ---
    name = get_kernel_name(kernel)
    structural = _sha256_tag(name)
    if args or kwargs:
        args_canonical = canonicalize_call_args(args, kwargs)
        parametric = _sha256_tag(f"{name}|{args_canonical}")
    else:
        parametric = structural
    return structural, parametric


def _get_fallback_content(kernel: Any) -> str | None:
    """
    Get best-effort content for hashing when ``to_json()`` is unavailable.

    Tries, in order: ``str(kernel)`` (MLIR), ``cudaq.draw()``.
    """
    # Try MLIR via str(kernel)
    try:
        mlir = str(kernel)
        if mlir and ("func.func" in mlir or "quake." in mlir or "module" in mlir):
            return mlir
    except Exception:
        pass

    # Try diagram
    try:
        import cudaq

        diagram = cudaq.draw(kernel)
        if isinstance(diagram, str) and diagram.strip():
            return diagram
    except Exception:
        pass

    return None


# ============================================================================
# Native JSON => op_stream
# ============================================================================


def _native_json_to_op_stream(
    native_json: str,
) -> tuple[list[dict[str, Any]], int] | None:
    """
    Parse CUDA-Q native JSON into canonical op_stream.

    Probes the JSON for a list of gate instructions under common
    schema keys (``instructions``, ``operations``, ``gates``).
    Returns ``None`` if the schema is not recognised.
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
    num_qubits = 0

    for instr in instructions:
        if not isinstance(instr, dict):
            continue

        gate = instr.get("gate") or instr.get("name") or instr.get("op") or "unknown"
        qubits = instr.get("qubits", instr.get("targets", []))
        if isinstance(qubits, int):
            qubits = [qubits]
        controls = instr.get("controls", [])
        if isinstance(controls, int):
            controls = [controls]
        all_qubits = list(controls) + list(qubits)

        op: dict[str, Any] = {
            "gate": str(gate).lower(),
            "qubits": all_qubits,
            "clbits": [],
        }

        raw_params = instr.get("params", instr.get("parameters"))
        if raw_params is not None:
            if isinstance(raw_params, (list, tuple)):
                op["params"] = {
                    f"p{i}": float(p) if isinstance(p, (int, float)) else None
                    for i, p in enumerate(raw_params)
                }
            elif isinstance(raw_params, (int, float)):
                op["params"] = {"p0": float(raw_params)}

        ops.append(op)
        if all_qubits:
            num_qubits = max(num_qubits, max(all_qubits) + 1)

    return (ops, num_qubits) if ops else None
