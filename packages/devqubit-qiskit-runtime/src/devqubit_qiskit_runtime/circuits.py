# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit Runtime adapter.

This module provides functions for hashing and converting Qiskit
QuantumCircuit objects for logging purposes.
"""

from __future__ import annotations

import hashlib
from typing import Any


def compute_structural_hash(circuits: list[Any]) -> str | None:
    """
    Compute a structure-only hash for Qiskit QuantumCircuit objects.

    Captures circuit structure (gates, qubits, classical bits) while
    ignoring parameter values for deduplication purposes.

    Parameters
    ----------
    circuits : list[Any]
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.

    Notes
    -----
    The hash captures:
    - Operation names (e.g., 'rx', 'cx', 'measure')
    - Ordered qubit indices
    - Ordered clbit indices (measurement wiring)
    - Parameter arity (count only, not values)
    - Classical condition presence

    This is the STRUCTURAL hash - use compute_parametric_hash for full identity.
    """
    if not circuits:
        return None

    circuit_signatures: list[str] = []

    for circuit in circuits:
        try:
            qubit_index = {
                q: i for i, q in enumerate(getattr(circuit, "qubits", ()) or ())
            }
            clbit_index = {
                c: i for i, c in enumerate(getattr(circuit, "clbits", ()) or ())
            }

            op_sigs: list[str] = []
            for instr in getattr(circuit, "data", []) or []:
                op = getattr(instr, "operation", None)
                name = getattr(op, "name", None)
                op_name = name if isinstance(name, str) and name else type(op).__name__

                qs: list[int] = []
                for q in getattr(instr, "qubits", ()) or ():
                    if q in qubit_index:
                        qs.append(qubit_index[q])
                    else:
                        qs.append(getattr(circuit.find_bit(q), "index", -1))
                cs: list[int] = []
                for c in getattr(instr, "clbits", ()) or ():
                    if c in clbit_index:
                        cs.append(clbit_index[c])
                    else:
                        cs.append(getattr(circuit.find_bit(c), "index", -1))

                params = getattr(op, "params", None)
                parity = len(params) if isinstance(params, (list, tuple)) else 0
                cond = getattr(op, "condition", None)
                has_cond = 1 if cond is not None else 0

                op_sigs.append(
                    f"{op_name}|p{parity}|q{tuple(qs)}|c{tuple(cs)}|if{has_cond}"
                )

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            circuit_signatures.append(str(circuit)[:500])

    payload = "\n".join(circuit_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def compute_parametric_hash(circuits: list[Any]) -> str | None:
    """
    Compute a parametric hash for Qiskit QuantumCircuit objects.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : list[Any]
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.

    Notes
    -----
    Includes:
    - All structural information (gates, qubits, classical bits)
    - Bound parameter values (rounded to 10 decimal places for stability)
    - Unbound parameter names
    """
    if not circuits:
        return None

    circuit_signatures: list[str] = []

    for circuit in circuits:
        try:
            qubit_index = {
                q: i for i, q in enumerate(getattr(circuit, "qubits", ()) or ())
            }
            clbit_index = {
                c: i for i, c in enumerate(getattr(circuit, "clbits", ()) or ())
            }

            op_sigs: list[str] = []
            for instr in getattr(circuit, "data", []) or []:
                op = getattr(instr, "operation", None)
                name = getattr(op, "name", None)
                op_name = name if isinstance(name, str) and name else type(op).__name__

                qs: list[int] = []
                for q in getattr(instr, "qubits", ()) or ():
                    if q in qubit_index:
                        qs.append(qubit_index[q])
                    else:
                        qs.append(getattr(circuit.find_bit(q), "index", -1))
                cs: list[int] = []
                for c in getattr(instr, "clbits", ()) or ():
                    if c in clbit_index:
                        cs.append(clbit_index[c])
                    else:
                        cs.append(getattr(circuit.find_bit(c), "index", -1))

                # Include actual parameter values
                params = getattr(op, "params", None) or []
                param_strs: list[str] = []
                for p in params:
                    try:
                        # Check if it's a Parameter (unbound)
                        if hasattr(p, "name") and hasattr(p, "_symbol_expr"):
                            param_strs.append(f"<param:{p.name}>")
                        else:
                            # Numeric value - round for stability
                            val = float(p)
                            param_strs.append(f"{val:.10f}")
                    except (TypeError, ValueError):
                        param_strs.append(str(p)[:50])

                cond = getattr(op, "condition", None)
                has_cond = 1 if cond is not None else 0

                op_sigs.append(
                    f"{op_name}|params=[{','.join(param_strs)}]|q{tuple(qs)}|c{tuple(cs)}|if{has_cond}"
                )

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            circuit_signatures.append(str(circuit)[:500])

    payload = "\n".join(circuit_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def circuits_to_text(circuits: list[Any]) -> str:
    """
    Convert circuits to human-readable text diagrams.

    Parameters
    ----------
    circuits : list
        List of QuantumCircuit objects.

    Returns
    -------
    str
        Combined text diagram of all circuits.
    """
    parts: list[str] = []

    for i, circuit in enumerate(circuits):
        if i > 0:
            parts.append("")

        name = getattr(circuit, "name", None) or f"circuit_{i}"
        parts.append(f"[{i}] {name}")

        try:
            diagram = circuit.draw(output="text", fold=80)
            if hasattr(diagram, "single_string"):
                parts.append(diagram.single_string())
            else:
                parts.append(str(diagram))
        except Exception:
            parts.append(str(circuit))

    return "\n".join(parts)
