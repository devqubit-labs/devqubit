# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit Runtime adapter.

This module provides functions for hashing and converting Qiskit
QuantumCircuit objects for logging purposes.

Hashing Contract
----------------
All hashing is delegated to ``devqubit_engine.circuit.hashing`` to ensure:

- Identical circuits produce identical hashes across SDKs
- IEEE-754 float encoding for determinism
- For circuits without parameters: ``parametric_hash == structural_hash``

This module reuses ``devqubit_qiskit.utils.circuit_to_op_stream`` for
consistent op_stream conversion across both Qiskit adapters.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.hashing import hash_circuit_pair
from devqubit_qiskit.circuits import circuit_to_op_stream


logger = logging.getLogger(__name__)


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
    - Ordered qubit indices (preserves gate directionality)
    - Ordered clbit indices (measurement wiring)
    - Parameter arity (count only, not values)
    - Classical condition details (not just presence)
    - Circuit dimensions (nq, nc)

    This uses the canonical devqubit_engine hashing for cross-SDK consistency.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> compute_structural_hash([qc])
    'sha256:abc123...'
    """
    if not circuits:
        return None
    structural, _ = _compute_hashes(circuits)
    return structural


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
    - Bound parameter values (IEEE-754 binary64 encoding)
    - Unbound parameter names

    UEC Contract: For circuits without parameters, parametric_hash == structural_hash.
    This is enforced by the engine's hash_circuit_pair function.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> from qiskit.circuit import Parameter
    >>> theta = Parameter('θ')
    >>> qc = QuantumCircuit(1)
    >>> qc.rx(theta, 0)
    >>> bound = qc.assign_parameters({theta: 0.5})
    >>> compute_parametric_hash([bound])
    'sha256:def456...'
    """
    if not circuits:
        return None
    _, parametric = _compute_hashes(circuits)
    return parametric


def compute_circuit_hashes(circuits: list[Any]) -> tuple[str | None, str | None]:
    """
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed,
    as it avoids redundant computation.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.

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

    Examples
    --------
    >>> structural, parametric = compute_circuit_hashes([qc])
    >>> structural == parametric  # True for non-parameterized circuits
    True
    """
    if not circuits:
        return None, None
    return _compute_hashes(circuits)


def _compute_hashes(circuits: list[Any]) -> tuple[str, str]:
    """
    Internal hash computation using devqubit_engine canonical hashing.

    Converts all circuits to canonical op_stream format and delegates
    to devqubit_engine.circuit.hashing for actual hash computation.

    Parameters
    ----------
    circuits : list
        Non-empty list of QuantumCircuit objects.

    Returns
    -------
    tuple of (str, str)
        (structural_hash, parametric_hash)

    Notes
    -----
    UEC Contract: For circuits without parameter values in ops,
    parametric_hash == structural_hash. This is handled by the
    engine's hash_circuit_pair function.
    """
    all_ops: list[dict[str, Any]] = []
    total_nq = 0
    total_nc = 0

    for circuit in circuits:
        try:
            nq = getattr(circuit, "num_qubits", 0) or 0
            nc = getattr(circuit, "num_clbits", 0) or 0
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

            # Add circuit operations using shared op_stream converter
            ops = circuit_to_op_stream(circuit)
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
        Combined text diagram of all circuits, each prefixed with
        its index and name.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2, name="bell")
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> print(circuits_to_text([qc]))
    [0] bell
         ┌───┐
    q_0: ┤ H ├──■──
         └───┘┌─┴─┐
    q_1: ─────┤ X ├
              └───┘
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
