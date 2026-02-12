# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit adapter.

Provides functions for materializing, hashing, serializing, and logging
Qiskit QuantumCircuit objects. Uses canonical devqubit_engine hashing
for cross-SDK consistency.

Hashing Contract
----------------
All hashing is delegated to ``devqubit_engine.circuit.hashing`` to ensure:
- Identical circuits produce identical hashes across SDKs
- IEEE-754 float encoding for determinism
- For circuits without parameters: ``parametric_hash == structural_hash``
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from devqubit_engine.circuit.hashing import encode_value, hash_circuit_pair
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.program import ProgramArtifact, ProgramRole
from devqubit_qiskit.serialization import QiskitCircuitSerializer
from devqubit_qiskit.utils import qiskit_version
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression


logger = logging.getLogger(__name__)

_serializer = QiskitCircuitSerializer()


# =============================================================================
# Circuit Materialization
# =============================================================================


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

    if isinstance(circuits, QuantumCircuit):
        return [circuits], True

    if isinstance(circuits, (list, tuple)):
        return list(circuits), False

    try:
        return list(circuits), False
    except TypeError as e:
        logger.debug("Suppressed error: %s", e)
        return [circuits], True


# =============================================================================
# Circuit Hashing
# =============================================================================


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
        - gate : str - Operation name (lowercase)
        - qubits : list of int - Ordered qubit indices
        - clbits : list of int, optional - Classical bit indices
        - params : dict, optional - Parameter dict with encoded values
        - condition : dict, optional - Classical condition

    Notes
    -----
    Qubit order is preserved (not sorted) because many gates are
    directional. For example, CX(0,1) has control=0, target=1.
    """
    qubit_idx = {q: i for i, q in enumerate(circuit.qubits)}
    clbit_idx = {c: i for i, c in enumerate(circuit.clbits)}

    ops: list[dict[str, Any]] = []

    for instr in circuit.data:
        operation = instr.operation
        name = getattr(operation, "name", None)
        if not isinstance(name, str) or not name:
            name = type(operation).__name__

        qubits = _extract_qubit_indices(instr.qubits, qubit_idx, circuit)

        op_dict: dict[str, Any] = {
            "gate": name.lower(),
            "qubits": qubits,
        }

        if instr.clbits:
            clbits = _extract_clbit_indices(instr.clbits, clbit_idx, circuit)
            op_dict["clbits"] = clbits

        raw_params = getattr(operation, "params", None)
        if raw_params:
            params = _extract_params(raw_params)
            if params:
                op_dict["params"] = params

        condition = _extract_condition(operation, clbit_idx)
        if condition:
            op_dict["condition"] = condition

        ops.append(op_dict)

    return ops


def _extract_qubit_indices(
    qubits: Any,
    qubit_idx: dict[Any, int],
    circuit: QuantumCircuit,
) -> list[int]:
    """Extract qubit indices from instruction qubits."""
    indices: list[int] = []
    for q in qubits:
        if q in qubit_idx:
            indices.append(qubit_idx[q])
        else:
            bit_info = circuit.find_bit(q)
            indices.append(getattr(bit_info, "index", -1))
    return indices


def _extract_clbit_indices(
    clbits: Any,
    clbit_idx: dict[Any, int],
    circuit: QuantumCircuit,
) -> list[int]:
    """Extract classical bit indices from instruction clbits."""
    indices: list[int] = []
    for c in clbits:
        if c in clbit_idx:
            indices.append(clbit_idx[c])
        else:
            bit_info = circuit.find_bit(c)
            indices.append(getattr(bit_info, "index", -1))
    return indices


def _extract_params(raw_params: Any) -> dict[str, Any] | None:
    """
    Extract parameters from Qiskit gate params.

    Handles three cases:
    - Unbound Parameter: stores None with parameter name
    - ParameterExpression: stores None with expression string
    - Numeric value: stores float value
    """
    if not isinstance(raw_params, (list, tuple)) or not raw_params:
        return None

    params: dict[str, Any] = {}

    for i, p in enumerate(raw_params):
        key = f"p{i}"

        if isinstance(p, Parameter):
            # Unbound parameter (no value yet)
            params[key] = None
            params[f"{key}_name"] = str(p.name)
        elif isinstance(p, ParameterExpression):
            # Bound or compound expression
            params[key] = None
            params[f"{key}_expr"] = str(p)
        else:
            try:
                params[key] = float(p)
            except (TypeError, ValueError):
                params[key] = str(p)[:50]

    return params if params else None


def _extract_condition(
    operation: Any,
    clbit_idx: dict[Any, int],
) -> dict[str, Any] | None:
    """Extract classical condition from operation."""
    cond = getattr(operation, "condition", None)
    if cond is None:
        return None

    try:
        target, value = cond

        if hasattr(target, "name"):
            return {
                "type": "register",
                "register": str(target.name),
                "value": int(value),
            }
        elif target in clbit_idx:
            return {
                "type": "clbit",
                "index": clbit_idx[target],
                "value": int(value),
            }
        else:
            return {
                "type": "unknown",
                "target": str(target),
                "value": int(value),
            }
    except (TypeError, ValueError) as e:
        logger.debug("Failed to serialize condition: %s", e)
        return {"type": "present"}


def compute_structural_hash(circuits: list[Any]) -> str | None:
    """
    Compute structural hash for Qiskit circuits.

    The structural hash captures the circuit template - gate types,
    qubit connectivity, and parameter arity - but NOT parameter values.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        SHA-256 hash in format "sha256:<hex>", or None if empty list.
    """
    if not circuits:
        return None
    structural, _ = _compute_hashes(circuits)
    return structural


def compute_parametric_hash(circuits: list[Any]) -> str | None:
    """
    Compute parametric hash for Qiskit circuits.

    The parametric hash captures both the circuit structure AND the
    bound parameter values.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        SHA-256 hash in format "sha256:<hex>", or None if empty list.
    """
    if not circuits:
        return None
    _, parametric = _compute_hashes(circuits)
    return parametric


def compute_circuit_hashes(
    circuits: list[Any],
    parameter_binds: list[dict[Any, float]] | None = None,
) -> tuple[str | None, str | None]:
    """
    Compute both structural and parametric hashes in one call.

    This is the preferred method when both hashes are needed.

    Parameters
    ----------
    circuits : list
        List of Qiskit QuantumCircuit objects.
    parameter_binds : list of dict, optional
        Parameter bindings passed to backend.run(). If provided,
        these values are incorporated into the parametric hash
        to distinguish runs with different parameter values.

    Returns
    -------
    structural_hash : str or None
        Structure-only hash (ignores parameter values).
    parametric_hash : str or None
        Hash including bound parameter values. When parameter_binds
        is provided, includes a digest of the bind values.
    """
    if not circuits:
        return None, None
    return _compute_hashes(circuits, parameter_binds)


def _compute_hashes(
    circuits: list[Any],
    parameter_binds: list[dict[Any, float]] | None = None,
) -> tuple[str, str]:
    """Internal hash computation with optional parameter binds."""
    all_ops: list[dict[str, Any]] = []
    total_nq = 0
    total_nc = 0

    for circuit in circuits:
        try:
            nq = getattr(circuit, "num_qubits", 0) or 0
            nc = getattr(circuit, "num_clbits", 0) or 0
            total_nq += nq
            total_nc += nc

            all_ops.append(
                {
                    "gate": "__circuit__",
                    "qubits": [],
                    "meta": {"nq": nq, "nc": nc},
                }
            )

            ops = circuit_to_op_stream(circuit)
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

    structural, parametric = hash_circuit_pair(all_ops, total_nq, total_nc)

    # If parameter_binds provided, incorporate into parametric hash
    if parameter_binds:
        binds_digest = _compute_binds_digest(parameter_binds)
        if binds_digest:
            parametric = _combine_hash_with_binds(parametric, binds_digest)

    return structural, parametric


def _compute_binds_digest(
    parameter_binds: list[dict[Any, float]],
) -> str | None:
    """
    Compute a deterministic digest of parameter bindings.

    Creates a canonical representation of parameter binds that can
    be combined with the circuit hash to distinguish runs with
    different parameter values.

    Parameters
    ----------
    parameter_binds : list of dict
        Parameter bindings as passed to backend.run().
        Each dict maps Parameter objects to float values.

    Returns
    -------
    str or None
        SHA-256 digest of canonicalized binds, or None if empty.
    """
    if not parameter_binds:
        return None

    # Canonicalize binds: sort by parameter name, encode values deterministically
    canonical_binds: list[list[tuple[str, str]]] = []

    for bind_dict in parameter_binds:
        if not bind_dict:
            canonical_binds.append([])
            continue

        sorted_items: list[tuple[str, str]] = []
        for param, value in bind_dict.items():
            # Extract parameter name
            param_name = str(getattr(param, "name", param))

            # Encode value deterministically using engine function
            encoded = encode_value(value)

            sorted_items.append((param_name, encoded))

        # Sort by parameter name for determinism
        sorted_items.sort(key=lambda x: x[0])
        canonical_binds.append(sorted_items)

    # Serialize to JSON and hash
    canonical_json = json.dumps(canonical_binds, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


def _combine_hash_with_binds(base_hash: str, binds_digest: str) -> str:
    """
    Combine base parametric hash with binds digest.

    Parameters
    ----------
    base_hash : str
        Base parametric hash in format "sha256:<hex>".
    binds_digest : str
        Digest of parameter binds.

    Returns
    -------
    str
        Combined hash in format "sha256:<hex>".
    """
    # Extract hex from base hash
    if base_hash.startswith("sha256:"):
        base_hex = base_hash[7:]
    else:
        base_hex = base_hash

    combined = f"{base_hex}:{binds_digest}"
    new_hash = hashlib.sha256(combined.encode()).hexdigest()
    return f"sha256:{new_hash}"


# =============================================================================
# Circuit Serialization and Logging
# =============================================================================


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
        except (TypeError, ValueError) as e:
            logger.debug("Failed to draw circuit diagram: %s", e)
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
    """
    artifacts: list[ProgramArtifact] = []
    meta = {
        "backend_name": backend_name,
        "qiskit_version": qiskit_version(),
        "structural_hash": structural_hash,
        "num_circuits": len(circuits),
    }

    # QPY serialization (lossless)
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

    # OpenQASM3 serialization (portable)
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
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed to serialize circuit QASM: %s", e)
            continue

    if oq3_items:
        oq3_result = tracker.log_openqasm3(oq3_items, name="circuits", meta=meta)
        for item in oq3_result.get("items", []):
            ref = item.get("ref")
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

    # Diagram (human-readable)
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
    except (TypeError, ValueError) as e:
        logger.debug("Failed to create diagram artifact: %s", e)

    return artifacts
