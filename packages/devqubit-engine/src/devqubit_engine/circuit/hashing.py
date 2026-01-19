# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit hashing utilities.

This module provides canonical hashing functions for quantum circuits.
All adapters should use these functions to ensure consistent hash computation
across different SDKs.

Hash Types
----------
- **structural_hash**: Hash of circuit structure (gates, qubits, controls).
  Ignores parameter values. Same hash means same circuit template.

- **parametric_hash**: Hash of structure + bound parameter values.
  Same hash means identical circuit for this specific execution.

Usage
-----
Adapters map their SDK-specific circuit representations to a canonical
operation stream, then use these functions:

>>> from devqubit_engine.circuit.hashing import hash_structural, hash_parametric
>>>
>>> ops = [
...     {"gate": "h", "qubits": [0]},
...     {"gate": "cx", "qubits": [0, 1]},
...     {"gate": "rz", "qubits": [0], "params": ["theta"]},
... ]
>>> structural = hash_structural(ops)
>>>
>>> bound_params = {"theta": 1.5707963267948966}
>>> parametric = hash_parametric(ops, bound_params)
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


def _float_to_hex(value: float) -> str:
    """
    Convert float to deterministic IEEE-754 binary64 hex representation.

    Handles special cases:
    - -0.0 → 0.0 (normalized)
    - NaN → "nan"
    - ±inf → "inf" / "-inf"

    Parameters
    ----------
    value : float
        Float value to encode.

    Returns
    -------
    str
        Deterministic string representation.
    """
    # Handle special cases
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    # Normalize -0.0 to 0.0
    if value == 0.0:
        value = 0.0
    # IEEE-754 binary64 big-endian hex
    return struct.pack(">d", value).hex()


def _collect_param_names(op_stream: list[dict[str, Any]]) -> set[str]:
    """
    Collect all parameter names used in the operation stream.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries.

    Returns
    -------
    set of str
        Set of parameter names referenced in ops.
    """
    names: set[str] = set()
    for op in op_stream:
        params = op.get("params")
        if params is None:
            continue
        if isinstance(params, dict):
            for key, val in params.items():
                # If value is None or a string placeholder, it's a param name
                if val is None or isinstance(val, str):
                    names.add(str(key))
        elif isinstance(params, (list, tuple)):
            for val in params:
                if isinstance(val, str):
                    names.add(val)
    return names


def _normalize_param_value(value: Any) -> str:
    """
    Normalize a parameter value to deterministic string.

    Parameters
    ----------
    value : Any
        Parameter value (float, int, or str).

    Returns
    -------
    str
        Normalized string representation.
    """
    if isinstance(value, float):
        return _float_to_hex(value)
    if isinstance(value, int):
        # Convert int to float for consistent representation
        return _float_to_hex(float(value))
    return str(value)


def _normalize_operation(
    op: dict[str, Any],
    include_params: bool = False,
    resolved_params: dict[str, Any] | None = None,
) -> dict:
    """
    Normalize an operation dict for hashing.

    Parameters
    ----------
    op : dict
        Operation dictionary with keys like 'gate', 'qubits', 'params', 'controls'.
    include_params : bool, default=False
        If True, include parameter values. If False, only include param structure.
    resolved_params : dict, optional
        Pre-resolved parameter values for parametric hashing.

    Returns
    -------
    dict
        Normalized operation suitable for hashing.
    """
    normalized: dict[str, Any] = {
        "gate": str(op.get("gate", "unknown")).lower(),
        "qubits": [int(q) for q in op.get("qubits", [])],
    }

    # Controls can be sorted (control qubits are symmetric)
    if op.get("controls"):
        normalized["controls"] = sorted(int(c) for c in op["controls"])

    # Classical bits - preserve order for measurement mapping
    if op.get("clbits"):
        normalized["clbits"] = [int(c) for c in op["clbits"]]

    # Parameters
    params = op.get("params")
    if params:
        if include_params:
            # Parametric hash: include resolved values
            if isinstance(params, dict):
                norm_params = {}
                for k, v in sorted(params.items()):
                    # Use resolved value if available
                    if resolved_params and k in resolved_params:
                        norm_params[str(k)] = _normalize_param_value(resolved_params[k])
                    elif v is not None:
                        norm_params[str(k)] = _normalize_param_value(v)
                    else:
                        # Unresolved placeholder
                        norm_params[str(k)] = f"__unbound:{k}__"
                normalized["params"] = norm_params
            elif isinstance(params, (list, tuple)):
                normalized["params"] = [_normalize_param_value(v) for v in params]
        else:
            # Structural hash: include param structure, not values
            if isinstance(params, dict):
                normalized["param_names"] = sorted(str(k) for k in params.keys())
            elif isinstance(params, (list, tuple)):
                # Check if list contains string names
                if all(isinstance(v, str) for v in params):
                    # List of parameter names - include them for structural identity
                    normalized["param_names"] = list(params)
                else:
                    # List of values - only record count
                    normalized["param_count"] = len(params)

    # Condition (for conditional gates)
    if op.get("condition"):
        cond = op["condition"]
        # Normalize condition to deterministic form
        if isinstance(cond, dict):
            normalized["condition"] = {str(k): v for k, v in sorted(cond.items())}
        else:
            normalized["condition"] = str(cond)

    return normalized


def hash_structural(
    op_stream: list[dict[str, Any]],
    *,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
) -> str:
    """
    Compute structural hash of circuit operation stream.

    The structural hash captures the circuit template - gate types, qubit
    connectivity, and parameter placeholders - but NOT parameter values.
    Two circuits with the same structure but different parameter values
    will have the same structural hash.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries. Each operation should have:
        - gate: Gate name (str)
        - qubits: List of qubit indices (order preserved!)
        - params: Optional parameter names or values
        - controls: Optional control qubit indices
        - clbits: Optional classical bit indices
    num_qubits : int, optional
        Total number of qubits in circuit. If provided, circuits with
        different qubit counts will have different hashes even if the
        gate stream is identical (prevents idle qubit collisions).
    num_clbits : int, optional
        Total number of classical bits in circuit.

    Returns
    -------
    str
        SHA-256 hash of normalized circuit structure.

    Examples
    --------
    >>> ops = [
    ...     {"gate": "h", "qubits": [0]},
    ...     {"gate": "cx", "qubits": [0, 1]},
    ...     {"gate": "rz", "qubits": [0], "params": {"theta": 0.5}},
    ... ]
    >>> hash_structural(ops, num_qubits=2)
    'sha256:abc123...'

    Notes
    -----
    Adapters should convert their SDK-specific circuit representations
    to this canonical operation stream format before hashing.

    Important: qubit order is preserved (not sorted) because many gates
    are directional (e.g., CX(0,1) != CX(1,0)).
    """
    normalized_ops = [
        _normalize_operation(op, include_params=False) for op in op_stream
    ]

    # Build payload with optional circuit dimensions
    payload: dict[str, Any] = {"ops": normalized_ops}
    if num_qubits is not None:
        payload["nq"] = int(num_qubits)
    if num_clbits is not None:
        payload["nc"] = int(num_clbits)

    # Create deterministic JSON representation
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_parametric(
    op_stream: list[dict[str, Any]],
    bound_params: dict[str, Any] | None = None,
    *,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
) -> str:
    """
    Compute parametric hash of circuit with bound parameters.

    The parametric hash captures both the circuit structure AND the bound
    parameter values. Two circuits with the same structure but different
    parameter values will have different parametric hashes.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries (same format as hash_structural).
    bound_params : dict, optional
        Dictionary mapping parameter names to values. If provided, these
        values are merged with inline param values from op_stream.
        Only parameters actually used in op_stream are included in hash.
    num_qubits : int, optional
        Total number of qubits in circuit.
    num_clbits : int, optional
        Total number of classical bits in circuit.

    Returns
    -------
    str
        SHA-256 hash of normalized circuit with bound parameters.

    Examples
    --------
    >>> ops = [
    ...     {"gate": "h", "qubits": [0]},
    ...     {"gate": "rz", "qubits": [0], "params": {"theta": None}},
    ... ]
    >>> bound = {"theta": 1.5707963267948966}
    >>> hash_parametric(ops, bound, num_qubits=1)
    'sha256:def456...'

    Notes
    -----
    If circuit has no parameters, ``parametric_hash == structural_hash``.
    This is the expected behavior per UEC spec.

    Only parameters actually referenced in op_stream are included in the
    hash payload, preventing drift from unused bound_params.
    """
    bound_params = bound_params or {}

    # Collect parameter names used in op_stream
    used_param_names = _collect_param_names(op_stream)

    # Filter bound_params to only used ones
    filtered_bound = {k: v for k, v in bound_params.items() if k in used_param_names}

    # Resolve parameters in ops
    resolved_ops: list[dict[str, Any]] = []
    for op in op_stream:
        resolved_op = dict(op)
        if op.get("params"):
            params = op["params"]
            if isinstance(params, dict):
                # Merge bound params
                resolved_params = {}
                for k, v in params.items():
                    if v is None and k in filtered_bound:
                        resolved_params[k] = filtered_bound[k]
                    elif v is not None:
                        resolved_params[k] = v
                    else:
                        resolved_params[k] = None  # Unbound
                resolved_op["params"] = resolved_params
        resolved_ops.append(resolved_op)

    normalized_ops = [
        _normalize_operation(op, include_params=True, resolved_params=filtered_bound)
        for op in resolved_ops
    ]

    # Normalize filtered bound params for inclusion in payload
    normalized_bound = {
        str(k): _normalize_param_value(v) for k, v in sorted(filtered_bound.items())
    }

    # Build payload with optional circuit dimensions
    payload: dict[str, Any] = {
        "ops": normalized_ops,
        "bound": normalized_bound,
    }
    if num_qubits is not None:
        payload["nq"] = int(num_qubits)
    if num_clbits is not None:
        payload["nc"] = int(num_clbits)

    # Create deterministic JSON representation
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_circuit_pair(
    op_stream: list[dict[str, Any]],
    bound_params: dict[str, Any] | None = None,
    *,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
) -> tuple[str, str]:
    """
    Compute both structural and parametric hashes in one call.

    Convenience function for adapters that need both hashes.

    Parameters
    ----------
    op_stream : list of dict
        List of operation dictionaries.
    bound_params : dict, optional
        Bound parameter values.
    num_qubits : int, optional
        Total number of qubits in circuit.
    num_clbits : int, optional
        Total number of classical bits in circuit.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)

    Examples
    --------
    >>> ops = [{"gate": "rx", "qubits": [0], "params": {"theta": 0.5}}]
    >>> struct, param = hash_circuit_pair(ops, num_qubits=1)
    """
    structural = hash_structural(
        op_stream,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
    )
    parametric = hash_parametric(
        op_stream,
        bound_params,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
    )
    return structural, parametric
