# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit hashing utilities for Braket adapter.

This module provides functions for hashing Braket Circuit objects for
deduplication and tracking purposes.

Hashing Contract
----------------
All hashing is delegated to ``devqubit_engine.circuit.hashing`` to ensure:

- Identical circuits produce identical hashes across SDKs
- IEEE-754 float encoding for determinism
- For circuits without parameters: ``parametric_hash == structural_hash``
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.hashing import hash_circuit_pair


logger = logging.getLogger(__name__)


def circuit_to_op_stream(circuit: Any) -> list[dict[str, Any]]:
    """
    Convert a Braket Circuit to canonical op_stream format.

    The op_stream format is SDK-agnostic and used for hashing.

    Parameters
    ----------
    circuit : Any
        Braket Circuit object.

    Returns
    -------
    list of dict
        List of operation dictionaries with keys:
        - gate: lowercase operation name
        - qubits: list of qubit indices (order preserved)
        - clbits: list of classical bit indices (empty for Braket)
        - params: dict with p0, p1, ... for parameter values/names

    Notes
    -----
    Qubit order is preserved for directional gates (e.g., CNOT).
    Parameters include both bound numeric values and FreeParameter names.
    """
    ops: list[dict[str, Any]] = []

    instrs = getattr(circuit, "instructions", None)
    if instrs is None:
        return ops

    for instr in instrs:
        op = getattr(instr, "operator", None)

        # Gate name (lowercase for consistency)
        if op is not None:
            op_name = getattr(op, "name", None)
            op_name = (
                op_name.lower()
                if isinstance(op_name, str) and op_name
                else type(op).__name__.lower()
            )
        else:
            op_name = type(instr).__name__.lower()

        # Target qubits as integer indices (order preserved!)
        qubits = _extract_qubits(instr)

        # Parameter handling
        params = _extract_params(op)

        op_dict: dict[str, Any] = {
            "gate": op_name,
            "qubits": qubits,
            "clbits": [],  # Braket doesn't have classical bits in instructions
        }

        if params:
            op_dict["params"] = params

        ops.append(op_dict)

    return ops


def _extract_qubits(instr: Any) -> list[int]:
    """
    Extract qubit indices from an instruction.

    Parameters
    ----------
    instr : Any
        Braket instruction object.

    Returns
    -------
    list of int
        Qubit indices (order preserved for directional gates).
    """
    qubits: list[int] = []
    tgt = getattr(instr, "target", None)
    if tgt is None:
        return qubits

    try:
        for q in tgt:
            idx = getattr(q, "index", None)
            if idx is not None:
                qubits.append(int(idx))
            else:
                qubits.append(int(q))
    except (TypeError, ValueError):
        # Single target or non-iterable
        idx = getattr(tgt, "index", None)
        if idx is not None:
            qubits.append(int(idx))

    return qubits


def _extract_params(op: Any) -> dict[str, Any]:
    """
    Extract parameters from an operator.

    Parameters
    ----------
    op : Any
        Braket operator object.

    Returns
    -------
    dict
        Parameter dictionary with p0, p1, ... keys and optional _name/_expr suffixes.
    """
    params: dict[str, Any] = {}
    if op is None:
        return params

    param_values: list[Any] = []
    for attr in ("parameters", "params", "angles", "angle"):
        val = getattr(op, attr, None)
        if val is not None:
            if isinstance(val, (list, tuple)):
                param_values = list(val)
            else:
                param_values = [val]
            break

    for i, p in enumerate(param_values):
        key = f"p{i}"
        if hasattr(p, "name"):
            # FreeParameter - store as unbound
            params[key] = None
            params[f"{key}_name"] = str(p.name)
        else:
            try:
                params[key] = float(p)
            except (TypeError, ValueError):
                params[key] = None
                params[f"{key}_expr"] = str(p)[:100]

    return params


def compute_circuit_hashes(
    circuits: list[Any],
    inputs: dict[str, float] | None = None,
) -> tuple[str | None, str | None]:
    """
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed,
    as it avoids redundant computation.

    Parameters
    ----------
    circuits : list
        List of Braket Circuit objects.
    inputs : dict, optional
        Parameter bindings for FreeParameters (name -> value).

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
    return _compute_hashes(circuits, inputs)


def compute_structural_hash(circuits: list[Any]) -> str | None:
    """
    Compute a structure-only hash for Braket Circuit objects.

    Captures circuit structure (gates, qubits) while ignoring
    parameter values for deduplication purposes.

    Parameters
    ----------
    circuits : list[Any]
        List of Braket Circuit objects.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.
    """
    if not circuits:
        return None
    structural, _ = _compute_hashes(circuits, None)
    return structural


def compute_parametric_hash(
    circuits: list[Any],
    inputs: dict[str, float] | None = None,
) -> str | None:
    """
    Compute a parametric hash for Braket circuits.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : list[Any]
        List of Braket Circuit objects.
    inputs : dict[str, float] or None
        Parameter bindings for FreeParameters.

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
    _, parametric = _compute_hashes(circuits, inputs)
    return parametric


def _compute_hashes(
    circuits: list[Any],
    inputs: dict[str, float] | None,
) -> tuple[str, str]:
    """
    Internal hash computation using devqubit_engine canonical hashing.

    Converts all circuits to canonical op_stream format and delegates
    to devqubit_engine.circuit.hashing for actual hash computation.

    Parameters
    ----------
    circuits : list
        Non-empty list of Braket Circuit objects.
    inputs : dict or None
        Parameter bindings for FreeParameters.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)
    """
    all_ops: list[dict[str, Any]] = []
    total_nq = 0
    total_nc = 0

    for circuit in circuits:
        try:
            nq = _get_num_qubits(circuit)
            nc = 0  # Braket doesn't have classical bits at circuit level
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

            ops = circuit_to_op_stream(circuit)

            if inputs:
                ops = _apply_inputs_to_ops(ops, inputs)

            all_ops.extend(ops)

        except Exception as e:
            logger.debug("Failed to convert circuit to op_stream: %s", e)
            all_ops.append(
                {
                    "gate": "__fallback__",
                    "qubits": [],
                    "meta": {"repr": str(circuit)[:200]},
                }
            )

    return hash_circuit_pair(all_ops, total_nq, total_nc)


def _get_num_qubits(circuit: Any) -> int:
    """
    Get the number of qubits in a Braket circuit.

    Parameters
    ----------
    circuit : Any
        Braket Circuit object.

    Returns
    -------
    int
        Number of qubits (max index + 1 from instructions).
    """
    qc = getattr(circuit, "qubit_count", None)
    if qc is not None:
        try:
            return int(qc)
        except (TypeError, ValueError):
            pass

    max_idx = -1
    instrs = getattr(circuit, "instructions", None)
    if instrs is not None:
        for instr in instrs:
            tgt = getattr(instr, "target", None)
            if tgt is not None:
                try:
                    for q in tgt:
                        idx = getattr(q, "index", None)
                        if idx is not None:
                            max_idx = max(max_idx, int(idx))
                        else:
                            max_idx = max(max_idx, int(q))
                except (TypeError, ValueError):
                    pass

    return max_idx + 1 if max_idx >= 0 else 0


def _apply_inputs_to_ops(
    ops: list[dict[str, Any]],
    inputs: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Apply FreeParameter bindings to ops.

    Parameters
    ----------
    ops : list of dict
        Operations from circuit_to_op_stream.
    inputs : dict
        Parameter name -> value bindings.

    Returns
    -------
    list of dict
        Ops with parameter values substituted.
    """
    result = []
    for op in ops:
        if "params" not in op:
            result.append(op)
            continue

        new_op = dict(op)
        new_params = dict(op["params"])

        for key in list(new_params.keys()):
            if key.endswith("_name"):
                base_key = key[:-5]
                param_name = new_params[key]
                if param_name in inputs:
                    new_params[base_key] = float(inputs[param_name])

        new_op["params"] = new_params
        result.append(new_op)

    return result
