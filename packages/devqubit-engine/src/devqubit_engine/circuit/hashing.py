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
from typing import Any

from devqubit_engine.utils.serialization import to_jsonable


def _normalize_float(value: float, precision: int = 10) -> str:
    """
    Normalize float to deterministic string representation.

    Parameters
    ----------
    value : float
        Float value to normalize.
    precision : int, default=10
        Number of significant digits.

    Returns
    -------
    str
        Normalized string representation.
    """
    if value == 0.0:
        return "0"
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.{precision}g}"


def _normalize_operation(
    op: dict[str, Any],
    include_params: bool = False,
) -> dict:
    """
    Normalize an operation dict for hashing.

    Parameters
    ----------
    op : dict
        Operation dictionary with keys like 'gate', 'qubits', 'params', 'controls'.
    include_params : bool, default=False
        If True, include parameter values. If False, only include param names.

    Returns
    -------
    dict
        Normalized operation suitable for hashing.
    """
    normalized: dict[str, Any] = {
        "gate": str(op.get("gate", "unknown")).lower(),
        "qubits": sorted(int(q) for q in op.get("qubits", [])),
    }

    # Controls (for controlled gates)
    if op.get("controls"):
        normalized["controls"] = sorted(int(c) for c in op["controls"])

    # Classical bits (for measurements)
    if op.get("clbits"):
        normalized["clbits"] = sorted(int(c) for c in op["clbits"])

    # Parameters - structural hash only includes names, parametric includes values
    if op.get("params"):
        params = op["params"]
        if include_params:
            # Include actual values (for parametric hash)
            if isinstance(params, dict):
                normalized["params"] = {
                    str(k): (
                        _normalize_float(float(v))
                        if isinstance(v, (int, float))
                        else str(v)
                    )
                    for k, v in sorted(params.items())
                }
            elif isinstance(params, (list, tuple)):
                normalized["params"] = [
                    (
                        _normalize_float(float(v))
                        if isinstance(v, (int, float))
                        else str(v)
                    )
                    for v in params
                ]
        else:
            # Only param names/positions (for structural hash)
            if isinstance(params, dict):
                normalized["param_names"] = sorted(str(k) for k in params.keys())
            elif isinstance(params, (list, tuple)):
                normalized["param_count"] = len(params)

    # Condition (for conditional gates)
    if op.get("condition"):
        normalized["condition"] = to_jsonable(op["condition"])

    return normalized


def hash_structural(op_stream: list[dict[str, Any]]) -> str:
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
        - qubits: List of qubit indices
        - params: Optional parameter names or values
        - controls: Optional control qubit indices
        - clbits: Optional classical bit indices

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
    >>> hash_structural(ops)
    'sha256:abc123...'

    Notes
    -----
    Adapters should convert their SDK-specific circuit representations
    to this canonical operation stream format before hashing.
    """
    normalized_ops = [
        _normalize_operation(op, include_params=False) for op in op_stream
    ]

    # Create deterministic JSON representation
    canonical = json.dumps(
        {"version": "1.0", "ops": normalized_ops},
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_parametric(
    op_stream: list[dict[str, Any]],
    bound_params: dict[str, Any] | None = None,
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
    >>> hash_parametric(ops, bound)
    'sha256:def456...'

    Notes
    -----
    If circuit has no parameters, ``parametric_hash == structural_hash``.
    This is the expected behavior per UEC spec.
    """
    bound_params = bound_params or {}

    # Apply bound params to operations
    resolved_ops: list[dict[str, Any]] = []
    for op in op_stream:
        resolved_op = dict(op)
        if op.get("params"):
            params = op["params"]
            if isinstance(params, dict):
                # Merge bound params
                resolved_params = {}
                for k, v in params.items():
                    if v is None and k in bound_params:
                        resolved_params[k] = bound_params[k]
                    else:
                        resolved_params[k] = v
                resolved_op["params"] = resolved_params
        resolved_ops.append(resolved_op)

    normalized_ops = [
        _normalize_operation(op, include_params=True) for op in resolved_ops
    ]

    # Create deterministic JSON representation
    canonical = json.dumps(
        {"version": "1.0", "ops": normalized_ops, "bound": to_jsonable(bound_params)},
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_circuit_pair(
    op_stream: list[dict[str, Any]],
    bound_params: dict[str, Any] | None = None,
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

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)

    Examples
    --------
    >>> ops = [{"gate": "rx", "qubits": [0], "params": {"theta": 0.5}}]
    >>> struct, param = hash_circuit_pair(ops)
    """
    structural = hash_structural(op_stream)
    parametric = hash_parametric(op_stream, bound_params)
    return structural, parametric
