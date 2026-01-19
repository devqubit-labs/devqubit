# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit adapter.

This module provides functions for materializing, hashing, serializing,
and logging Qiskit QuantumCircuit objects.
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

    Prevents consumption bugs when the user provides generators/iterators.

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
    devqubit hashing functions.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qiskit circuit to convert.

    Returns
    -------
    list of dict
        Canonical operation stream where each dict contains:
        - gate: Operation name (str)
        - qubits: Ordered qubit indices (list[int]) - order preserved!
        - clbits: Ordered classical bit indices (list[int])
        - params: Parameter dict or list
        - condition: Classical condition info or None

    Notes
    -----
    Qubit order is preserved (not sorted) because gate semantics
    depend on operand order (e.g., CX control vs target).
    """
    # Build index maps for fast lookup
    qubit_index = {q: i for i, q in enumerate(circuit.qubits)}
    clbit_index = {c: i for i, c in enumerate(circuit.clbits)}

    ops: list[dict[str, Any]] = []

    for instr in circuit.data:
        op = instr.operation
        op_name = getattr(op, "name", None)
        if not isinstance(op_name, str) or not op_name:
            op_name = type(op).__name__

        # Extract qubit indices - preserve order!
        qubits: list[int] = []
        for q in instr.qubits:
            if q in qubit_index:
                qubits.append(qubit_index[q])
            else:
                # Fallback for unusual bit containers
                bit_info = circuit.find_bit(q)
                qubits.append(getattr(bit_info, "index", -1))

        # Extract classical bit indices - preserve order for measurement mapping
        clbits: list[int] = []
        for c in instr.clbits:
            if c in clbit_index:
                clbits.append(clbit_index[c])
            else:
                bit_info = circuit.find_bit(c)
                clbits.append(getattr(bit_info, "index", -1))

        # Build operation dict
        op_dict: dict[str, Any] = {
            "gate": op_name.lower(),
            "qubits": qubits,
        }

        if clbits:
            op_dict["clbits"] = clbits

        # Extract parameters
        raw_params = getattr(op, "params", None)
        if raw_params:
            params = _extract_params(raw_params)
            if params:
                op_dict["params"] = params

        # Extract classical condition
        condition = _extract_condition(op, circuit, clbit_index)
        if condition is not None:
            op_dict["condition"] = condition

        ops.append(op_dict)

    return ops


def _extract_params(raw_params: Any) -> dict[str, Any] | None:
    """
    Extract parameters from Qiskit gate params.

    Parameters
    ----------
    raw_params : Any
        Gate parameters (list of Parameter, ParameterExpression, or numeric).

    Returns
    -------
    dict or None
        Parameter dict with names/values, or None if no params.
    """
    if not isinstance(raw_params, (list, tuple)) or not raw_params:
        return None

    params: dict[str, Any] = {}

    for i, p in enumerate(raw_params):
        param_key = f"p{i}"

        # Check for unbound Parameter
        if hasattr(p, "name") and hasattr(p, "_symbol_expr"):
            # Qiskit Parameter - use name as placeholder
            params[param_key] = None
            params[f"{param_key}_name"] = str(p.name)
        elif hasattr(p, "parameters") and hasattr(p, "_symbol_expr"):
            # Qiskit ParameterExpression - store expression string
            params[param_key] = None
            params[f"{param_key}_expr"] = str(p)
        else:
            # Numeric value
            try:
                params[param_key] = float(p)
            except (TypeError, ValueError):
                params[param_key] = str(p)[:50]

    return params if params else None


def _extract_condition(
    op: Any,
    circuit: QuantumCircuit,
    clbit_index: dict[Any, int],
) -> dict[str, Any] | None:
    """
    Extract classical condition from operation.

    Parameters
    ----------
    op : Any
        Qiskit operation with potential condition.
    circuit : QuantumCircuit
        Parent circuit for register lookup.
    clbit_index : dict
        Classical bit to index mapping.

    Returns
    -------
    dict or None
        Condition dict with target and value, or None if no condition.
    """
    cond = getattr(op, "condition", None)
    if cond is None:
        return None

    try:
        cond_target, cond_value = cond

        # Determine if condition is on register or single bit
        if hasattr(cond_target, "name"):
            # ClassicalRegister
            return {
                "type": "register",
                "register": str(cond_target.name),
                "value": int(cond_value),
            }
        elif cond_target in clbit_index:
            # Single classical bit
            return {
                "type": "clbit",
                "index": clbit_index[cond_target],
                "value": int(cond_value),
            }
        else:
            # Fallback
            return {
                "type": "unknown",
                "target": str(cond_target),
                "value": int(cond_value),
            }
    except Exception:
        return {"type": "present"}


def compute_circuit_hashes(
    circuits: list[Any],
) -> tuple[str | None, str | None]:
    """
    Compute structural and parametric hashes for Qiskit circuits.

    Uses the canonical devqubit_engine hashing for cross-SDK consistency.

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

    # Build combined op stream and track total dimensions
    all_ops: list[dict[str, Any]] = []
    total_qubits = 0
    total_clbits = 0

    for circuit in circuits:
        try:
            nq = getattr(circuit, "num_qubits", 0) or 0
            nc = getattr(circuit, "num_clbits", 0) or 0
            total_qubits += nq
            total_clbits += nc

            # Add circuit boundary marker (for multi-circuit batches)
            all_ops.append(
                {
                    "gate": "__circuit__",
                    "qubits": [],
                    "params": {"nq": nq, "nc": nc},
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
                    "params": {"repr": str(circuit)[:200]},
                }
            )

    try:
        structural, parametric = hash_circuit_pair(
            all_ops,
            num_qubits=total_qubits,
            num_clbits=total_clbits,
        )
        return structural, parametric
    except Exception as e:
        logger.warning("Failed to compute circuit hashes: %s", e)
        return None, None


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

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    circuits : list
        List of QuantumCircuit objects.
    backend_name : str
        Backend name for metadata.
    structural_hash : str or None
        Structural hash of circuits (UEC v1.0).

    Returns
    -------
    list of ProgramArtifact
        References to logged program artifacts, one per format per circuit.
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
        # QPY is a batch format - single artifact for all circuits
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
        # Generate ProgramArtifact per circuit, not just the first one
        items = oq3_result.get("items", [])
        for item in items:
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
