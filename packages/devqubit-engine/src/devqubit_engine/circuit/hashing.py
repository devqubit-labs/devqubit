# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Canonical circuit hashing for cross-SDK consistency.

All adapters must use these functions to ensure identical circuits
produce identical hashes regardless of SDK.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


def _float_to_hex(value: float) -> str:
    """IEEE-754 binary64 big-endian hex encoding."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        value = 0.0  # Normalize -0.0
    return struct.pack(">d", value).hex()


def _encode_value(value: Any) -> str:
    """Encode parameter value deterministically."""
    if value is None:
        return "__unbound__"
    if isinstance(value, float):
        return _float_to_hex(value)
    if isinstance(value, int):
        return _float_to_hex(float(value))
    return str(value)


def _normalize_op(op: dict[str, Any], with_values: bool) -> dict[str, Any]:
    """Normalize operation for hashing."""
    out: dict[str, Any] = {
        "g": str(op.get("gate", "?")).lower(),
        "q": [int(q) for q in op.get("qubits", [])],
    }

    if op.get("clbits"):
        out["c"] = [int(c) for c in op["clbits"]]

    params = op.get("params")
    if params:
        if with_values:
            if isinstance(params, dict):
                out["p"] = {str(k): _encode_value(v) for k, v in sorted(params.items())}
            else:
                out["p"] = [_encode_value(v) for v in params]
        else:
            # Structural: only arity
            out["pa"] = len(params) if isinstance(params, (dict, list, tuple)) else 1

    cond = op.get("condition")
    if cond:
        if isinstance(cond, dict):
            out["cond"] = {str(k): v for k, v in sorted(cond.items())}
        else:
            out["cond"] = str(cond)

    return out


def _compute_hash(
    ops: list[dict[str, Any]],
    with_values: bool,
    num_qubits: int,
    num_clbits: int,
) -> str:
    """Compute hash from normalized ops."""
    normalized = [_normalize_op(op, with_values) for op in ops]
    payload = {"nq": num_qubits, "nc": num_clbits, "ops": normalized}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def hash_structural(
    op_stream: list[dict[str, Any]],
    num_qubits: int,
    num_clbits: int,
) -> str:
    """
    Compute structural hash (ignores parameter values).

    Parameters
    ----------
    op_stream : list of dict
        Operations: {gate, qubits, clbits?, params?, condition?}
    num_qubits : int
        Total qubit count.
    num_clbits : int
        Total classical bit count.

    Returns
    -------
    str
        Hash as ``sha256:<hex>``
    """
    return _compute_hash(
        op_stream,
        with_values=False,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
    )


def hash_parametric(
    op_stream: list[dict[str, Any]],
    num_qubits: int,
    num_clbits: int,
) -> str:
    """
    Compute parametric hash (includes parameter values).

    Parameters
    ----------
    op_stream : list of dict
        Operations with bound parameter values.
    num_qubits : int
        Total qubit count.
    num_clbits : int
        Total classical bit count.

    Returns
    -------
    str
        Hash as ``sha256:<hex>``

    Notes
    -----
    If no parameters have values, returns same as structural hash.
    """
    # Check if any param has a value
    has_values = False
    for op in op_stream:
        params = op.get("params")
        if params:
            if isinstance(params, dict):
                if any(v is not None for v in params.values()):
                    has_values = True
                    break
            elif isinstance(params, (list, tuple)) and params:
                has_values = True
                break

    if not has_values:
        return hash_structural(op_stream, num_qubits, num_clbits)

    return _compute_hash(
        op_stream,
        with_values=True,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
    )


def hash_circuit_pair(
    op_stream: list[dict[str, Any]],
    num_qubits: int,
    num_clbits: int,
) -> tuple[str, str]:
    """
    Compute both hashes in one call.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)
    """
    structural = hash_structural(op_stream, num_qubits, num_clbits)
    parametric = hash_parametric(op_stream, num_qubits, num_clbits)
    return structural, parametric
