# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit/tape handling utilities for PennyLane adapter.

This module provides functions for hashing and converting PennyLane
tape objects for logging purposes.

Hashing Contract
----------------
All hashing is delegated to ``devqubit_engine.circuit.hashing`` to ensure:

- Identical tapes produce identical hashes across SDKs
- IEEE-754 float encoding for determinism
- For tapes without parameters: ``parametric_hash == structural_hash``

Notes
-----
PennyLane uses "tapes" as the internal circuit representation.
A tape contains operations and measurements on wires.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.hashing import hash_circuit_pair


logger = logging.getLogger(__name__)


def _is_tape_like(obj: Any) -> bool:
    """Check if object has tape-like interface."""
    return hasattr(obj, "operations") and hasattr(obj, "measurements")


def _normalize_wire(wire: Any) -> int:
    """
    Normalize a wire to integer index.

    Parameters
    ----------
    wire : Any
        PennyLane wire (int, string, or Wires object).

    Returns
    -------
    int
        Integer wire index.
    """
    # Try direct int conversion
    try:
        return int(wire)
    except (TypeError, ValueError):
        pass

    # Try to get index from wire object
    idx = getattr(wire, "index", None)
    if idx is not None:
        try:
            return int(idx)
        except (TypeError, ValueError):
            pass

    # Last resort: hash the string representation
    # This ensures determinism even for non-integer wires
    return hash(str(wire)) % (2**31)


def _wires_to_list(wires: Any) -> list[int]:
    """
    Convert wires to list of integer indices.

    Parameters
    ----------
    wires : Any
        PennyLane wires (Wires object, list, tuple, or single wire).

    Returns
    -------
    list of int
        List of integer wire indices (order preserved).
    """
    try:
        wire_list = list(wires)
        return [_normalize_wire(w) for w in wire_list]
    except TypeError:
        # Single wire
        return [_normalize_wire(wires)]


def tape_to_op_stream(tape: Any) -> list[dict[str, Any]]:
    """
    Convert a PennyLane tape to canonical op_stream format.

    The op_stream format is SDK-agnostic and used for hashing.

    Parameters
    ----------
    tape : Any
        PennyLane tape object.

    Returns
    -------
    list of dict
        List of operation dictionaries with keys:
        - gate: lowercase operation name
        - qubits: list of wire indices (order preserved)
        - clbits: list of classical bit indices (empty for PennyLane)
        - params: dict with p0, p1, ... for parameter values/names
        - meta: additional metadata (e.g., observable for measurements)

    Notes
    -----
    Wire order is preserved for directional gates (e.g., CNOT).
    Parameters include both bound numeric values and symbolic parameter names.
    Measurements are included as special operations.
    """
    ops: list[dict[str, Any]] = []

    # Process operations
    operations = getattr(tape, "operations", [])
    for op in operations:
        op_dict = _convert_operation(op)
        ops.append(op_dict)

    # Process measurements (included in hash for completeness)
    measurements = getattr(tape, "measurements", [])
    for meas in measurements:
        meas_dict = _convert_measurement(meas)
        ops.append(meas_dict)

    return ops


def _convert_operation(op: Any) -> dict[str, Any]:
    """
    Convert a PennyLane operation to canonical format.

    Parameters
    ----------
    op : Any
        PennyLane operation.

    Returns
    -------
    dict
        Operation dictionary.
    """
    # Gate name (lowercase for consistency)
    name = getattr(op, "name", None)
    op_name = (
        name.lower() if isinstance(name, str) and name else type(op).__name__.lower()
    )

    # Wires as integer indices (order preserved!)
    wires = _wires_to_list(getattr(op, "wires", ()))

    # Parameter handling
    params: dict[str, Any] = {}
    param_values = _get_param_values(op)

    for i, p in enumerate(param_values):
        key = f"p{i}"
        if _is_trainable_param(p):
            # Trainable/symbolic parameter - store as unbound
            params[key] = None
            param_name = getattr(p, "name", None) or str(p)
            params[f"{key}_name"] = str(param_name)[:100]
        else:
            # Numeric value
            try:
                params[key] = float(p)
            except (TypeError, ValueError):
                params[key] = None
                params[f"{key}_expr"] = str(p)[:100]

    op_dict: dict[str, Any] = {
        "gate": op_name,
        "qubits": wires,
        "clbits": [],  # PennyLane doesn't have classical bits in operations
    }

    if params:
        op_dict["params"] = params

    return op_dict


def _convert_measurement(meas: Any) -> dict[str, Any]:
    """
    Convert a PennyLane measurement to canonical format.

    Parameters
    ----------
    meas : Any
        PennyLane measurement process.

    Returns
    -------
    dict
        Measurement dictionary.
    """
    # Measurement type
    mtype = type(meas).__name__.lower()

    # Wires
    wires = _wires_to_list(getattr(meas, "wires", ()))

    # Observable info
    obs = getattr(meas, "obs", None)
    obs_name = ""
    if obs is not None:
        obs_name_attr = getattr(obs, "name", None)
        obs_name = (
            obs_name_attr.lower()
            if isinstance(obs_name_attr, str) and obs_name_attr
            else type(obs).__name__.lower()
        )

    # Return type
    rtype = getattr(meas, "return_type", None)
    rtype_str = str(rtype).lower() if rtype is not None else ""

    # Include observable in gate name so it's always hashed
    if obs_name:
        gate_name = f"__meas_{mtype}_{obs_name}__"
    else:
        gate_name = f"__meas_{mtype}__"

    return {
        "gate": gate_name,
        "qubits": wires,
        "clbits": [],
        "meta": {
            "meas_type": mtype,
            "return_type": rtype_str,
            "observable": obs_name,
        },
    }


def _get_param_values(op: Any) -> list[Any]:
    """
    Get parameter values from a PennyLane operation.

    Parameters
    ----------
    op : Any
        PennyLane operation.

    Returns
    -------
    list
        List of parameter values.
    """
    # Try 'parameters' first (preferred)
    params = getattr(op, "parameters", None)
    if isinstance(params, (list, tuple)) and params:
        return list(params)

    # Try 'data' (older interface)
    data = getattr(op, "data", None)
    if isinstance(data, (list, tuple)) and data:
        return list(data)

    return []


def _is_trainable_param(p: Any) -> bool:
    """
    Check if parameter is a trainable/symbolic parameter.

    Parameters
    ----------
    p : Any
        Parameter value.

    Returns
    -------
    bool
        True if parameter is symbolic/trainable.
    """
    # Check for PennyLane's Variable or similar
    if hasattr(p, "requires_grad"):
        return True

    # Check for symbolic parameter markers
    if hasattr(p, "name") and not isinstance(p, (int, float, complex)):
        return True

    # Check type name
    type_name = type(p).__name__.lower()
    if any(marker in type_name for marker in ("variable", "tensor", "arraybox")):
        # Additional check: ArrayBox with numeric value is not symbolic
        try:
            float(p)
            return False
        except (TypeError, ValueError):
            return True

    return False


def _get_num_wires(tape: Any) -> int:
    """
    Get the number of wires in a tape.

    Parameters
    ----------
    tape : Any
        PennyLane tape object.

    Returns
    -------
    int
        Number of wires (max wire index + 1 or wire count).
    """
    # Try wires attribute
    wires = getattr(tape, "wires", None)
    if wires is not None:
        try:
            wire_list = list(wires)
            if wire_list:
                # Try to get max index
                indices = [_normalize_wire(w) for w in wire_list]
                return max(indices) + 1
            return len(wire_list)
        except (TypeError, ValueError):
            pass

    # Try num_wires attribute
    num_wires = getattr(tape, "num_wires", None)
    if num_wires is not None:
        try:
            return int(num_wires)
        except (TypeError, ValueError):
            pass

    # Fall back to scanning operations
    max_wire = -1
    for op in getattr(tape, "operations", []):
        op_wires = getattr(op, "wires", ())
        try:
            for w in op_wires:
                idx = _normalize_wire(w)
                max_wire = max(max_wire, idx)
        except (TypeError, ValueError):
            pass

    return max_wire + 1 if max_wire >= 0 else 0


def compute_circuit_hashes(
    circuits: Any,
) -> tuple[str | None, str | None]:
    """
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed,
    as it avoids redundant computation.

    Parameters
    ----------
    circuits : Any
        A tape-like object or list of tapes.

    Returns
    -------
    structural_hash : str or None
        Structure-only hash (ignores parameter values).
    parametric_hash : str or None
        Hash including bound parameter values.

    Notes
    -----
    Both hashes use IEEE-754 float encoding for determinism.
    For tapes without parameters, parametric_hash == structural_hash.
    """
    tapes = _get_tapes(circuits)
    if not tapes:
        return None, None
    return _compute_hashes(tapes)


def compute_structural_hash(circuits: Any) -> str | None:
    """
    Compute a structure-only hash for PennyLane tape objects.

    Captures tape structure (gates, wires, measurements) while ignoring
    parameter values for deduplication purposes.

    Parameters
    ----------
    circuits : Any
        A tape-like object or list of tapes.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if invalid.
    """
    tapes = _get_tapes(circuits)
    if not tapes:
        return None
    structural, _ = _compute_hashes(tapes)
    return structural


def compute_parametric_hash(circuits: Any) -> str | None:
    """
    Compute a parametric hash for PennyLane tapes.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : Any
        A tape-like object or list of tapes.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if invalid.

    Notes
    -----
    UEC Contract: For tapes without parameters, parametric_hash == structural_hash.

    IMPORTANT: trainable_params is NOT included in hash (it's training metadata,
    not circuit semantics).
    """
    tapes = _get_tapes(circuits)
    if not tapes:
        return None
    _, parametric = _compute_hashes(tapes)
    return parametric


def _get_tapes(circuits: Any) -> list[Any]:
    """
    Extract tape list from circuits.

    Parameters
    ----------
    circuits : Any
        A tape-like object, list of tapes, or None.

    Returns
    -------
    list
        List of tapes (empty if input is invalid).
    """
    if circuits is None:
        return []
    if _is_tape_like(circuits):
        return [circuits]
    if isinstance(circuits, (list, tuple)) and circuits:
        if all(_is_tape_like(t) for t in circuits):
            return list(circuits)
    return []


def _compute_hashes(
    tapes: list[Any],
) -> tuple[str, str]:
    """
    Internal hash computation using devqubit_engine canonical hashing.

    Converts all tapes to canonical op_stream format and delegates
    to devqubit_engine.circuit.hashing for actual hash computation.

    Parameters
    ----------
    tapes : list
        Non-empty list of PennyLane tape objects.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)
    """
    all_ops: list[dict[str, Any]] = []
    total_nq = 0
    total_nc = 0

    for tape in tapes:
        try:
            # Determine tape dimensions
            nq = _get_num_wires(tape)
            nc = 0  # PennyLane doesn't have classical bits at tape level
            total_nq += nq
            total_nc += nc

            # Add tape boundary marker for multi-tape batches
            all_ops.append(
                {
                    "gate": "__tape__",
                    "qubits": [],
                    "meta": {"nq": nq, "nc": nc},
                }
            )

            # Convert tape to op_stream
            ops = tape_to_op_stream(tape)
            all_ops.extend(ops)

        except Exception as e:
            logger.debug("Failed to convert tape to op_stream: %s", e)
            # Fallback: use string representation
            all_ops.append(
                {
                    "gate": "__fallback__",
                    "qubits": [],
                    "meta": {"repr": str(tape)[:200]},
                }
            )

    return hash_circuit_pair(all_ops, total_nq, total_nc)


def tapes_to_text(tapes: list[Any]) -> str:
    """
    Convert tapes to human-readable text diagrams.

    Parameters
    ----------
    tapes : list
        List of PennyLane tape objects.

    Returns
    -------
    str
        Combined text representation of all tapes.
    """
    parts: list[str] = []

    for i, tape in enumerate(tapes):
        if i > 0:
            parts.append("")

        parts.append(f"[Tape {i}]")

        try:
            # Try to use tape's string representation
            parts.append(str(tape))
        except Exception:
            # Fallback: manual formatting
            nq = _get_num_wires(tape)
            parts.append(f"<Tape with {nq} wires>")

            ops = getattr(tape, "operations", [])
            for op in ops:
                op_name = getattr(op, "name", type(op).__name__)
                op_wires = list(getattr(op, "wires", []))
                parts.append(f"  {op_name} @ {op_wires}")

            meas = getattr(tape, "measurements", [])
            for m in meas:
                mtype = type(m).__name__
                m_wires = list(getattr(m, "wires", []))
                parts.append(f"  {mtype} @ {m_wires}")

    return "\n".join(parts)
