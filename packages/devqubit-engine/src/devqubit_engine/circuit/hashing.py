# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Canonical circuit hashing for cross-SDK consistency.

This module provides canonical hashing functions for quantum circuits.
All adapters must use these functions to ensure identical circuits
produce identical hashes regardless of SDK.

Hash Types
----------
- **structural_hash**: Hash of circuit structure (gates, qubits, conditions).
  Ignores parameter values. Same hash = same circuit template.

- **parametric_hash**: Hash of structure + bound parameter values.
  Same hash = identical circuit execution.

Contract
--------
For circuits without parameters: ``parametric_hash == structural_hash``

Encoding
--------
- Integers: "i:<decimal>" prefix preserves arbitrary precision
- Floats: IEEE-754 binary64 big-endian hex
- Booleans: "b:true" or "b:false"
- Negative zero: normalized to positive zero
- NaN: encoded as "nan"
- Infinity: encoded as "inf" or "-inf"
- Unbound params: encoded as "__unbound__"
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


def _float_to_hex(value: float) -> str:
    """
    Convert float to IEEE-754 binary64 big-endian hex representation.

    This encoding is deterministic across platforms and languages,
    ensuring consistent hashing regardless of runtime environment.

    Parameters
    ----------
    value : float
        Float value to encode.

    Returns
    -------
    str
        Deterministic string representation:
        - Normal floats: 16-character hex string
        - NaN: "nan"
        - +inf: "inf"
        - -inf: "-inf"
        - -0.0: same as 0.0 (normalized)
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        value = 0.0  # Normalize -0.0 to 0.0
    return struct.pack(">d", value).hex()


def encode_value(value: Any) -> str:
    """
    Encode parameter value deterministically.

    Parameters
    ----------
    value : Any
        Parameter value (float, int, str, or None).

    Returns
    -------
    str
        Deterministic string representation.
    """
    if value is None:
        return "__unbound__"
    if isinstance(value, bool):
        return f"b:{value}".lower()
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return _float_to_hex(value)
    # numpy scalar types (avoid numpy import)
    type_name = type(value).__name__
    if "float" in type_name.lower():
        return _float_to_hex(float(value))
    if "int" in type_name.lower():
        return f"i:{int(value)}"
    return str(value)


def _normalize_op(op: dict[str, Any], with_values: bool) -> dict[str, Any]:
    """
    Normalize an operation dict for hashing.

    Creates a minimal, deterministic representation of the operation
    suitable for JSON serialization and hashing.

    Parameters
    ----------
    op : dict
        Operation dictionary with keys:
        - gate : str - Gate name
        - qubits : list of int - Qubit indices (order preserved!)
        - clbits : list of int, optional - Classical bit indices
        - params : dict or list, optional - Parameters
        - condition : dict or str, optional - Classical condition

    with_values : bool
        If True, include parameter values (for parametric hash).
        If False, only include parameter arity (for structural hash).

    Returns
    -------
    dict
        Normalized operation with keys:
        - g : str - Gate name (lowercase)
        - q : list of int - Qubit indices
        - c : list of int, optional - Classical bit indices
        - p : dict or list, optional - Parameter values (if with_values)
        - pa : int, optional - Parameter arity (if not with_values)
        - cond : dict or str, optional - Condition
    """
    out: dict[str, Any] = {
        "g": str(op.get("gate", "?")).lower(),
        "q": [int(q) for q in op.get("qubits", [])],  # Order preserved!
    }

    # Classical bits - preserve order for measurement mapping
    if op.get("clbits"):
        out["c"] = [int(c) for c in op["clbits"]]

    # Parameters
    params = op.get("params")
    if params:
        if with_values:
            # Parametric hash: include actual values
            if isinstance(params, dict):
                out["p"] = {str(k): encode_value(v) for k, v in sorted(params.items())}
            else:
                out["p"] = [encode_value(v) for v in params]
        else:
            # Structural hash: only record arity
            out["pa"] = len(params) if isinstance(params, (dict, list, tuple)) else 1

    # Classical condition
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
    """
    Compute hash from normalized operations.

    Parameters
    ----------
    ops : list of dict
        Operation stream.
    with_values : bool
        Include parameter values.
    num_qubits : int
        Total qubit count.
    num_clbits : int
        Total classical bit count.

    Returns
    -------
    str
        Hash in format "sha256:<hex>".
    """
    normalized = [_normalize_op(op, with_values) for op in ops]
    payload = {
        "nq": num_qubits,
        "nc": num_clbits,
        "ops": normalized,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def hash_structural(
    op_stream: list[dict[str, Any]],
    num_qubits: int,
    num_clbits: int,
) -> str:
    """
    Compute structural hash of circuit operation stream.

    The structural hash captures the circuit template - gate types,
    qubit connectivity, parameter arity, and conditions - but NOT
    parameter values. Two circuits with the same structure but
    different parameter values will have the same structural hash.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries. Each operation should have:

        - gate : str
            Gate name (case insensitive).
        - qubits : list of int
            Qubit indices (order preserved for directional gates).
        - clbits : list of int, optional
            Classical bit indices.
        - params : dict or list, optional
            Parameter placeholders or values.
        - condition : dict or str, optional
            Classical condition.

    num_qubits : int
        Total number of qubits in circuit. Required to prevent
        collisions between circuits with idle qubits.
    num_clbits : int
        Total number of classical bits in circuit.

    Returns
    -------
    str
        SHA-256 hash in format "sha256:<hex>".

    See Also
    --------
    hash_parametric : Hash including parameter values.
    hash_circuit_pair : Compute both hashes at once.
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
    Compute parametric hash of circuit with bound parameters.

    The parametric hash captures both the circuit structure AND the
    bound parameter values. Two circuits with the same structure but
    different parameter values will have different parametric hashes.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries with bound parameter values.
    num_qubits : int
        Total number of qubits in circuit.
    num_clbits : int
        Total number of classical bits in circuit.

    Returns
    -------
    str
        SHA-256 hash in format "sha256:<hex>".

    See Also
    --------
    hash_structural : Hash ignoring parameter values.
    hash_circuit_pair : Compute both hashes at once.
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

    # If no values, return structural hash
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
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed,
    as adapters typically need both for UEC compliance.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries.
    num_qubits : int
        Total number of qubits in circuit.
    num_clbits : int
        Total number of classical bits in circuit.

    Returns
    -------
    structural_hash : str
        Structure-only hash (ignores parameter values).
    parametric_hash : str
        Hash including bound parameter values.

    See Also
    --------
    hash_structural : Compute only structural hash.
    hash_parametric : Compute only parametric hash.

    Examples
    --------
    >>> ops = [{"gate": "rx", "qubits": [0], "params": {"theta": 0.5}}]
    >>> structural, parametric = hash_circuit_pair(ops, num_qubits=1, num_clbits=0)
    >>> structural != parametric  # Different because has params
    True

    >>> ops = [{"gate": "h", "qubits": [0]}]
    >>> structural, parametric = hash_circuit_pair(ops, num_qubits=1, num_clbits=0)
    >>> structural == parametric  # Same because no params
    True
    """
    structural = hash_structural(op_stream, num_qubits, num_clbits)
    parametric = hash_parametric(op_stream, num_qubits, num_clbits)
    return structural, parametric
