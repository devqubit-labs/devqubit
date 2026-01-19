# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit adapter.

This module provides functions for materializing, hashing, serializing,
and logging Qiskit QuantumCircuit objects. Uses canonical devqubit_engine
hashing for cross-SDK consistency.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.hashing import hash_circuit_pair
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.program import ProgramArtifact, ProgramRole
from devqubit_qiskit.serialization import QiskitCircuitSerializer
from devqubit_qiskit.utils import qiskit_version
from qiskit import QuantumCircuit


logger = logging.getLogger(__name__)

# Module-level serializer instance
_serializer = QiskitCircuitSerializer()


def materialize_circuits(circuits: Any) -> tuple[list[Any], bool]:
    """
    Materialize circuit inputs exactly once.

    Prevents consumption bugs when the user provides generators or iterators.
    QuantumCircuit is iterable over instructions, so it must be checked
    explicitly before attempting iteration.

    Parameters
    ----------
    circuits : Any
        A QuantumCircuit, or an iterable of QuantumCircuit objects.

    Returns
    -------
    circuit_list : list
        List of circuit-like objects.
    was_single : bool
        True if the input was a single circuit-like object.
    """
    if circuits is None:
        return [], False

    # QuantumCircuit is iterable over instructions, so check explicitly
    if isinstance(circuits, QuantumCircuit):
        return [circuits], True

    if isinstance(circuits, (list, tuple)):
        return list(circuits), False

    # Generic iterables (generator, iterator, etc.)
    try:
        return list(circuits), False
    except TypeError:
        # Not iterable -> treat as a single circuit-like payload
        return [circuits], True


def circuit_to_op_stream(circuit: QuantumCircuit) -> list[dict[str, Any]]:
    """
    Convert a Qiskit QuantumCircuit to canonical operation stream.

    The operation stream format is SDK-agnostic and used by the
    devqubit hashing functions for cross-SDK consistency.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qiskit circuit to convert.

    Returns
    -------
    list of dict
        Canonical operation stream where each dict contains:

        - gate : str
            Operation name (lowercase).
        - qubits : list of int
            Ordered qubit indices. Order is preserved (not sorted)
            because gate semantics depend on operand order.
        - clbits : list of int, optional
            Ordered classical bit indices for measurements.
        - params : dict, optional
            Parameter dict with keys like "p0", "p1", etc.
        - condition : dict, optional
            Classical condition info with type, target, and value.

    Notes
    -----
    Qubit order is preserved (not sorted) because many gates are
    directional. For example, CX(0,1) has control=0, target=1,
    while CX(1,0) has control=1, target=0. Sorting would lose
    this distinction.
    """
    # Build index maps for fast lookup
    qubit_idx = {q: i for i, q in enumerate(circuit.qubits)}
    clbit_idx = {c: i for i, c in enumerate(circuit.clbits)}

    ops: list[dict[str, Any]] = []

    for instr in circuit.data:
        operation = instr.operation
        name = getattr(operation, "name", None)
        if not isinstance(name, str) or not name:
            name = type(operation).__name__

        # Extract qubit indices - preserve order for directional gates!
        qubits: list[int] = []
        for q in instr.qubits:
            if q in qubit_idx:
                qubits.append(qubit_idx[q])
            else:
                # Fallback for unusual bit containers
                bit_info = circuit.find_bit(q)
                qubits.append(getattr(bit_info, "index", -1))

        # Build operation dict
        op_dict: dict[str, Any] = {
            "gate": name.lower(),
            "qubits": qubits,
        }

        # Classical bits - preserve order for measurement mapping
        if instr.clbits:
            clbits: list[int] = []
            for c in instr.clbits:
                if c in clbit_idx:
                    clbits.append(clbit_idx[c])
                else:
                    bit_info = circuit.find_bit(c)
                    clbits.append(getattr(bit_info, "index", -1))
            op_dict["clbits"] = clbits

        # Extract parameters
        raw_params = getattr(operation, "params", None)
        if raw_params:
            params = _extract_params(raw_params)
            if params:
                op_dict["params"] = params

        # Extract classical condition
        condition = _extract_condition(operation, clbit_idx)
        if condition:
            op_dict["condition"] = condition

        ops.append(op_dict)

    return ops


def _extract_params(raw_params: Any) -> dict[str, Any] | None:
    """
    Extract parameters from Qiskit gate params.

    Handles three cases:
    - Unbound Parameter: stores None with parameter name
    - ParameterExpression: stores None with expression string
    - Numeric value: stores float value

    Parameters
    ----------
    raw_params : Any
        Gate parameters (list of Parameter, ParameterExpression, or numeric).

    Returns
    -------
    dict or None
        Parameter dict with keys like "p0", "p1", etc., or None if empty.
    """
    if not isinstance(raw_params, (list, tuple)) or not raw_params:
        return None

    params: dict[str, Any] = {}

    for i, p in enumerate(raw_params):
        key = f"p{i}"

        # Qiskit Parameter (unbound)
        if hasattr(p, "name") and hasattr(p, "_symbol_expr"):
            params[key] = None
            params[f"{key}_name"] = str(p.name)
        # Qiskit ParameterExpression
        elif hasattr(p, "parameters") and hasattr(p, "_symbol_expr"):
            params[key] = None
            params[f"{key}_expr"] = str(p)
        else:
            # Numeric value
            try:
                params[key] = float(p)
            except (TypeError, ValueError):
                params[key] = str(p)[:50]

    return params if params else None


def _extract_condition(
    operation: Any,
    clbit_idx: dict[Any, int],
) -> dict[str, Any] | None:
    """
    Extract classical condition from operation.

    Conditions can be on a ClassicalRegister or a single classical bit.
    This function extracts the full condition details including the
    target and comparison value.

    Parameters
    ----------
    operation : Any
        Qiskit operation with potential condition.
    clbit_idx : dict
        Classical bit to index mapping.

    Returns
    -------
    dict or None
        Condition dict with keys:

        - type : str
            One of "register", "clbit", "unknown", or "present".
        - register : str, optional
            Register name (for register conditions).
        - index : int, optional
            Classical bit index (for clbit conditions).
        - value : int
            Comparison value.
    """
    cond = getattr(operation, "condition", None)
    if cond is None:
        return None

    try:
        target, value = cond

        # Determine if condition is on register or single bit
        if hasattr(target, "name"):
            # ClassicalRegister
            return {
                "type": "register",
                "register": str(target.name),
                "value": int(value),
            }
        elif target in clbit_idx:
            # Single classical bit
            return {
                "type": "clbit",
                "index": clbit_idx[target],
                "value": int(value),
            }
        else:
            # Fallback for unknown condition types
            return {
                "type": "unknown",
                "target": str(target),
                "value": int(value),
            }
    except Exception:
        # Condition exists but couldn't be parsed
        return {"type": "present"}


def compute_structural_hash(circuits: list[Any]) -> str | None:
    """
    Compute structural hash for Qiskit circuits.

    The structural hash captures the circuit template - gate types,
    qubit connectivity, and parameter arity - but NOT parameter values.
    Two circuits with the same structure but different parameter values
    will have the same structural hash.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        SHA-256 hash in format "sha256:<hex>", or None if empty list.

    See Also
    --------
    compute_parametric_hash : Hash including parameter values.
    compute_circuit_hashes : Compute both hashes at once.

    Examples
    --------
    >>> qc1 = QuantumCircuit(2)
    >>> qc1.h(0)
    >>> qc1.cx(0, 1)
    >>> h1 = compute_structural_hash([qc1])
    >>> h1.startswith("sha256:")
    True
    """
    if not circuits:
        return None
    structural, _ = _compute_hashes(circuits)
    return structural


def compute_parametric_hash(circuits: list[Any]) -> str | None:
    """
    Compute parametric hash for Qiskit circuits.

    The parametric hash captures both the circuit structure AND the
    bound parameter values. Two circuits with the same structure but
    different parameter values will have different parametric hashes.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        SHA-256 hash in format "sha256:<hex>", or None if empty list.

    Notes
    -----
    If circuits have no parameters, parametric_hash == structural_hash.
    This is required by the UEC specification.

    See Also
    --------
    compute_structural_hash : Hash ignoring parameter values.
    compute_circuit_hashes : Compute both hashes at once.
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
    """
    if not circuits:
        return None, None
    return _compute_hashes(circuits)


def _compute_hashes(circuits: list[Any]) -> tuple[str, str]:
    """
    Internal hash computation.

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
    engine's hash_parametric function which checks for actual
    parameter values in the op_stream.
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

            # Add circuit operations
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
        Combined text diagram of all circuits, separated by blank lines.
    """
    parts: list[str] = []

    for i, circuit in enumerate(circuits):
        if i > 0:
            parts.append("")  # Blank line between circuits

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


def serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    backend_name: str,
    structural_hash: str | None,
) -> list[ProgramArtifact]:
    """
    Serialize and log circuits in multiple formats.

    Creates ProgramArtifact references for each circuit in each format,
    properly handling multi-circuit batches.

    Formats logged:
    - QPY: Binary format, batch, lossless (Qiskit-specific)
    - OpenQASM3: Text format, per-circuit, portable
    - Diagram: Human-readable text representation

    Parameters
    ----------
    tracker : Run
        Tracker instance for logging artifacts.
    circuits : list
        List of QuantumCircuit objects.
    backend_name : str
        Backend name for metadata.
    structural_hash : str or None
        Structural hash of circuits.

    Returns
    -------
    list of ProgramArtifact
        References to logged program artifacts.

    Notes
    -----
    QPY format is logged with a security note indicating it contains
    opaque bytes that should not be executed untrusted.
    """
    artifacts: list[ProgramArtifact] = []
    meta = {
        "backend_name": backend_name,
        "qiskit_version": qiskit_version(),
        "structural_hash": structural_hash,
        "num_circuits": len(circuits),
    }

    # Log circuits in QPY format (batch, lossless)
    try:
        qpy_data = _serializer.serialize(circuits, CircuitFormat.QPY)
        ref = tracker.log_bytes(
            kind="qiskit.qpy.circuits",
            data=qpy_data.as_bytes(),
            media_type="application/vnd.qiskit.qpy",
            role="program",
            meta={**meta, "security_note": "opaque_bytes_only"},
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="qpy",
                name="circuits_batch",
                index=0,
            )
        )
    except Exception as e:
        logger.debug("Failed to serialize circuits to QPY: %s", e)

    # Log circuits in QASM3 format (per circuit, portable)
    oq3_items: list[dict[str, Any]] = []
    for i, c in enumerate(circuits):
        try:
            qasm_data = _serializer.serialize(c, CircuitFormat.OPENQASM3, index=i)
            qc_name = getattr(c, "name", None) or f"circuit_{i}"
            oq3_items.append(
                {
                    "source": qasm_data.as_text(),
                    "name": f"circuit_{i}:{qc_name}",
                    "index": i,
                }
            )
        except Exception:
            continue

    if oq3_items:
        oq3_result = tracker.log_openqasm3(oq3_items, name="circuits", meta=meta)
        # Generate ProgramArtifact per circuit
        for item in oq3_result.get("items", []):
            ref = item.get("raw_ref")
            if ref:
                item_index = item.get("index", 0)
                item_name = item.get("name", f"circuit_{item_index}")
                artifacts.append(
                    ProgramArtifact(
                        ref=ref,
                        role=ProgramRole.LOGICAL,
                        format="openqasm3",
                        name=item_name,
                        index=item_index,
                    )
                )

    # Log circuit diagrams (human-readable text)
    try:
        diagram_text = circuits_to_text(circuits)
        ref = tracker.log_bytes(
            kind="qiskit.circuits.diagram",
            data=diagram_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
            meta={"num_circuits": len(circuits)},
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="diagram",
                name="circuits",
                index=0,
            )
        )
    except Exception:
        pass  # Diagram logging is best-effort

    return artifacts
