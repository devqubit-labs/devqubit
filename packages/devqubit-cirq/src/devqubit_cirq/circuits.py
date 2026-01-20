# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Cirq adapter.

This module provides functions for hashing and converting Cirq Circuit
objects for logging purposes.

Hashing Contract
----------------
All hashing is delegated to ``devqubit_engine.circuit.hashing`` to ensure:

- Identical circuits produce identical hashes across SDKs
- IEEE-754 float encoding for determinism
- For circuits without parameters: ``parametric_hash == structural_hash``

Qubit Indexing
--------------
Qubit-to-index mapping is deterministic across Python process invocations:

- LineQubit: uses x coordinate
- GridQubit: uses row * 1000 + col
- NamedQubit: uses sorted string representation order

This avoids Python's per-process hash randomization which would otherwise
cause identical circuits to produce different hashes across runs.

Notes
-----
Cirq uses "moments" to organize operations that happen simultaneously.
Operations within a moment are sorted by qubit indices for deterministic hashing.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.hashing import hash_circuit_pair


logger = logging.getLogger(__name__)


def _qubit_to_index(qubit: Any, qubit_map: dict[str, int] | None = None) -> int:
    """
    Convert a Cirq qubit to an integer index.

    Parameters
    ----------
    qubit : Any
        Cirq qubit (LineQubit, GridQubit, NamedQubit, etc.).
    qubit_map : dict, optional
        Pre-computed deterministic mapping from qubit string representation
        to index. When provided, this takes precedence for NamedQubit and
        other non-standard qubit types to ensure deterministic hashing.

    Returns
    -------
    int
        Integer qubit index.

    Notes
    -----
    For LineQubit, uses the x coordinate.
    For GridQubit, uses row * 1000 + col for uniqueness.
    For NamedQubit or other types, uses the provided qubit_map for
    deterministic indexing (falls back to string-based offset if no map).
    """
    # LineQubit: has .x attribute
    if hasattr(qubit, "x"):
        try:
            return int(qubit.x)
        except (TypeError, ValueError):
            pass

    # GridQubit: has .row and .col attributes
    if hasattr(qubit, "row") and hasattr(qubit, "col"):
        try:
            return int(qubit.row) * 1000 + int(qubit.col)
        except (TypeError, ValueError):
            pass

    # NamedQubit or other: use deterministic mapping
    qubit_str = str(qubit)
    if qubit_map is not None and qubit_str in qubit_map:
        return qubit_map[qubit_str]

    # Fallback: deterministic string-based index (sum of character codes)
    # This is stable across process runs unlike hash()
    return sum(ord(c) for c in qubit_str) % (2**31)


def _qubits_to_list(
    qubits: Any,
    qubit_map: dict[str, int] | None = None,
) -> list[int]:
    """
    Convert qubits to list of integer indices.

    Parameters
    ----------
    qubits : Any
        Iterable of Cirq qubit objects.
    qubit_map : dict, optional
        Pre-computed deterministic mapping from qubit string representation
        to index for NamedQubit and other non-standard qubit types.

    Returns
    -------
    list of int
        List of integer qubit indices (order preserved).
    """
    try:
        return [_qubit_to_index(q, qubit_map) for q in qubits]
    except TypeError:
        return [_qubit_to_index(qubits, qubit_map)]


def _get_gate_params(gate: Any) -> list[tuple[str, Any]]:
    """
    Extract parameters from a Cirq gate.

    Only extracts parameters that are:
    1. Symbolic (sympy expressions)
    2. OR explicitly user-specified (non-default values)

    This ensures that gates with only default parameters (like H)
    satisfy the UEC contract: structural == parametric for no-param circuits.

    Parameters
    ----------
    gate : Any
        Cirq gate object.

    Returns
    -------
    list of tuple
        List of (param_name, param_value) tuples.
    """
    params: list[tuple[str, Any]] = []

    if gate is None:
        return params

    # Try Cirq's is_parameterized check first (most reliable)
    try:
        import cirq

        if hasattr(cirq, "is_parameterized") and cirq.is_parameterized(gate):
            # Gate has symbolic parameters - extract them
            for attr in (
                "_radians",
                "rads",
                "_exponent",
                "exponent",
                "theta",
                "phi",
                "gamma",
                "_phase_exponent",
                "phase_exponent",
            ):
                val = getattr(gate, attr, None)
                if val is not None and _is_symbolic(val):
                    params.append((attr, val))
            return params
    except (ImportError, TypeError, AttributeError):
        pass

    # For non-parameterized gates, only include non-default values
    # from gates that REQUIRE user-specified parameters
    gate_type = type(gate).__name__.lower()

    # Rotation gates - always have user-specified angle
    if gate_type in (
        "rx",
        "ry",
        "rz",
        "rxx",
        "ryy",
        "rzz",
        "phasedxpow",
        "phasedxzpow",
        "fsim",
        "iswappow",
        "cphase",
        "zz",
    ):
        for attr in (
            "_radians",
            "rads",
            "_exponent",
            "exponent",
            "theta",
            "phi",
            "gamma",
        ):
            val = getattr(gate, attr, None)
            if val is not None:
                try:
                    float_val = float(val)
                    params.append((attr, float_val))
                except (TypeError, ValueError):
                    if _is_symbolic(val):
                        params.append((attr, val))
        return params

    # For other gates (H, X, Y, Z, CNOT, etc.), don't extract default params
    # This ensures structural == parametric for these gates
    return params


def _is_symbolic(value: Any) -> bool:
    """Check if a value is a symbolic parameter."""
    # Check for sympy symbols
    if hasattr(value, "free_symbols"):
        try:
            if value.free_symbols:
                return True
        except (TypeError, AttributeError):
            pass

    # Check for Cirq Symbol
    type_name = type(value).__name__.lower()
    if "symbol" in type_name or "param" in type_name:
        return True

    # Check if it's not numeric
    try:
        float(value)
        return False
    except (TypeError, ValueError):
        return True


def _build_qubit_map(circuit: Any) -> dict[str, int]:
    """
    Build a deterministic qubit-to-index mapping for a circuit.

    Creates a stable mapping by sorting qubits by their string representation,
    ensuring identical circuits produce identical mappings across process runs.

    Parameters
    ----------
    circuit : Any
        Cirq Circuit object.

    Returns
    -------
    dict
        Mapping from qubit string representation to deterministic integer index.
        For LineQubit/GridQubit, the index matches their natural coordinate-based
        index. For NamedQubit and other types, indices are assigned based on
        sorted string representation order.

    Notes
    -----
    This function is critical for hash determinism. It avoids using Python's
    built-in hash() which is randomized per process since Python 3.3.
    """
    try:
        all_qubits = list(circuit.all_qubits())
    except Exception:
        return {}

    if not all_qubits:
        return {}

    # Sort qubits by string representation for deterministic ordering
    sorted_qubits = sorted(all_qubits, key=str)

    # Build mapping - use natural indices for standard qubit types,
    # sequential indices for others
    qubit_map: dict[str, int] = {}
    named_offset = 10000  # Offset for non-standard qubits to avoid collisions

    for i, qubit in enumerate(sorted_qubits):
        qubit_str = str(qubit)

        # LineQubit: use x coordinate
        if hasattr(qubit, "x"):
            try:
                qubit_map[qubit_str] = int(qubit.x)
                continue
            except (TypeError, ValueError):
                pass

        # GridQubit: use row * 1000 + col
        if hasattr(qubit, "row") and hasattr(qubit, "col"):
            try:
                qubit_map[qubit_str] = int(qubit.row) * 1000 + int(qubit.col)
                continue
            except (TypeError, ValueError):
                pass

        # NamedQubit or other: use deterministic sequential index
        qubit_map[qubit_str] = named_offset + i

    return qubit_map


def circuit_to_op_stream(
    circuit: Any,
    resolver: Any | None = None,
    qubit_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a Cirq Circuit to canonical op_stream format.

    The op_stream format is SDK-agnostic and used for hashing.

    Parameters
    ----------
    circuit : Any
        Cirq Circuit object.
    resolver : cirq.ParamResolver or None
        Parameter resolver for binding symbolic parameters.
    qubit_map : dict, optional
        Pre-computed deterministic qubit mapping. If not provided, a mapping
        will be built from the circuit.

    Returns
    -------
    list of dict
        List of operation dictionaries with keys:
        - gate: lowercase gate type name
        - qubits: list of qubit indices (order preserved)
        - clbits: list of classical bit indices (empty for Cirq)
        - params: dict with p0, p1, ... for parameter values/names
        - meta: additional metadata (e.g., measurement key)

    Notes
    -----
    Qubit order is preserved for directional gates (e.g., CNOT).
    Operations within moments are sorted by qubit indices for determinism.
    Qubit indices are deterministic across process runs.
    """
    # Build qubit map if not provided
    if qubit_map is None:
        qubit_map = _build_qubit_map(circuit)

    ops: list[dict[str, Any]] = []

    for moment in circuit:
        # Sort operations within moment by qubit indices for determinism
        moment_ops = list(moment)
        moment_ops.sort(
            key=lambda op: tuple(_qubits_to_list(getattr(op, "qubits", ()), qubit_map))
        )

        for op in moment_ops:
            op_dict = _convert_operation(op, resolver, qubit_map)
            ops.append(op_dict)

    return ops


def _convert_operation(
    op: Any,
    resolver: Any | None = None,
    qubit_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Convert a Cirq operation to canonical format.

    Parameters
    ----------
    op : Any
        Cirq operation.
    resolver : cirq.ParamResolver or None
        Parameter resolver for binding symbolic parameters.
    qubit_map : dict, optional
        Pre-computed deterministic qubit mapping for NamedQubit and
        other non-standard qubit types.

    Returns
    -------
    dict
        Operation dictionary.
    """
    gate = getattr(op, "gate", None)

    # Gate name (lowercase for consistency)
    if gate is not None:
        gate_name = type(gate).__name__.lower()
    else:
        gate_name = type(op).__name__.lower()

    # Include measurement key in gate name (so it affects hash)
    if gate is not None:
        meas_key = getattr(gate, "key", None)
        if meas_key is not None:
            # Handle MeasurementKey object
            key_str = getattr(meas_key, "name", None) or str(meas_key)
            gate_name = f"{gate_name}_key:{key_str}"

    # Qubits as integer indices (order preserved!)
    qubits = _qubits_to_list(getattr(op, "qubits", ()), qubit_map)

    # Parameter handling
    params: dict[str, Any] = {}
    gate_params = _get_gate_params(gate)

    for i, (param_name, param_value) in enumerate(gate_params):
        key = f"p{i}"

        if _is_symbolic(param_value):
            # Try to resolve symbolic parameter
            resolved_value = None
            if resolver is not None:
                try:
                    resolved_value = resolver.value_of(param_value)
                except Exception:
                    pass

            if resolved_value is not None:
                try:
                    params[key] = float(resolved_value)
                    # Also include the symbolic name for structural hash
                    param_str = getattr(param_value, "name", None) or str(param_value)
                    params[f"{key}_name"] = str(param_str)[:100]
                except (TypeError, ValueError):
                    params[key] = None
                    params[f"{key}_name"] = str(param_value)[:100]
            else:
                # Unresolved symbolic parameter
                params[key] = None
                param_str = getattr(param_value, "name", None) or str(param_value)
                params[f"{key}_name"] = str(param_str)[:100]
        else:
            # Numeric value
            try:
                params[key] = float(param_value)
            except (TypeError, ValueError):
                params[key] = None
                params[f"{key}_expr"] = str(param_value)[:100]

    # Build operation dict
    op_dict: dict[str, Any] = {
        "gate": gate_name,
        "qubits": qubits,
        "clbits": [],  # Cirq doesn't have classical bits in operations
    }

    if params:
        op_dict["params"] = params

    return op_dict


def _get_num_qubits(
    circuit: Any,
    qubit_map: dict[str, int] | None = None,
) -> int:
    """
    Get the number of qubits in a circuit.

    Parameters
    ----------
    circuit : Any
        Cirq Circuit object.
    qubit_map : dict, optional
        Pre-computed deterministic qubit mapping.

    Returns
    -------
    int
        Number of qubits (based on all qubits used in circuit).
    """
    try:
        all_qubits = circuit.all_qubits()
        if all_qubits:
            indices = [_qubit_to_index(q, qubit_map) for q in all_qubits]
            return max(indices) + 1
        return 0
    except Exception:
        return 0


def compute_circuit_hashes(
    circuits: list[Any],
    resolver: Any | None = None,
) -> tuple[str | None, str | None]:
    """
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed,
    as it avoids redundant computation.

    Parameters
    ----------
    circuits : list
        List of Cirq Circuit objects.
    resolver : cirq.ParamResolver or None
        Parameter resolver with bound values.

    Returns
    -------
    structural_hash : str or None
        Structure-only hash (ignores parameter values).
    parametric_hash : str or None
        Hash including bound parameter values.

    Notes
    -----
    Both hashes use IEEE-754 float encoding for determinism.
    For circuits without parameters, parametric_hash == structural_hash.
    """
    if not circuits:
        return None, None
    return _compute_hashes(circuits, resolver)


def compute_structural_hash(circuits: list[Any]) -> str | None:
    """
    Compute a structure-only hash for Cirq circuits.

    Captures circuit structure (gates, qubits) while ignoring
    parameter values for deduplication purposes.

    Parameters
    ----------
    circuits : list
        List of Cirq Circuit objects.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.
    """
    if not circuits:
        return None
    structural, _ = _compute_hashes(circuits, resolver=None)
    return structural


def compute_parametric_hash(
    circuits: list[Any],
    resolver: Any | None = None,
) -> str | None:
    """
    Compute a parametric hash for Cirq circuits.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : list
        List of Cirq Circuit objects.
    resolver : cirq.ParamResolver or None
        Parameter resolver with bound values.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.

    Notes
    -----
    UEC Contract: For circuits without parameters, parametric_hash == structural_hash.
    """
    if not circuits:
        return None
    _, parametric = _compute_hashes(circuits, resolver)
    return parametric


def _compute_hashes(
    circuits: list[Any],
    resolver: Any | None = None,
) -> tuple[str, str]:
    """
    Internal hash computation using devqubit_engine canonical hashing.

    Converts all circuits to canonical op_stream format and delegates
    to devqubit_engine.circuit.hashing for actual hash computation.

    Parameters
    ----------
    circuits : list
        Non-empty list of Cirq Circuit objects.
    resolver : cirq.ParamResolver or None
        Parameter resolver with bound values.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)

    Notes
    -----
    Uses deterministic qubit mapping to ensure hash stability across
    different Python process invocations.
    """
    all_ops: list[dict[str, Any]] = []
    total_nq = 0
    total_nc = 0

    for circuit in circuits:
        try:
            # Build deterministic qubit mapping for this circuit
            qubit_map = _build_qubit_map(circuit)

            # Determine circuit dimensions
            nq = _get_num_qubits(circuit, qubit_map)
            nc = 0  # Cirq doesn't have classical bits at circuit level
            total_nq += nq
            total_nc += nc

            # Add circuit boundary marker for multi-circuit batches
            all_ops.append(
                {
                    "gate": "__circuit__",
                    "qubits": [],
                    "meta": {"nq": nq, "nc": nc},
                }
            )

            # Convert circuit to op_stream (with resolver for parametric)
            ops = circuit_to_op_stream(circuit, resolver, qubit_map)
            all_ops.extend(ops)

        except Exception as e:
            logger.debug("Failed to convert circuit to op_stream: %s", e)
            # Fallback: use string representation
            all_ops.append(
                {
                    "gate": "__fallback__",
                    "qubits": [],
                    "meta": {"repr": str(circuit)[:200]},
                }
            )

    return hash_circuit_pair(all_ops, total_nq, total_nc)
