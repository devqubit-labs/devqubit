# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handling utilities for Qiskit adapter.

This module provides functions for materializing, hashing, serializing,
and logging Qiskit QuantumCircuit objects.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.core.run import Run
from devqubit_engine.uec.program import ProgramArtifact
from devqubit_engine.uec.types import ProgramRole
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


def compute_circuit_hash(circuits: list[Any]) -> str | None:
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
    """
    if not circuits:
        return None

    circuit_signatures: list[str] = []

    for circuit in circuits:
        try:
            # Precompute indices for speed and stability
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

                # Qubits / clbits in order (control-target order matters)
                qs: list[int] = []
                for q in getattr(instr, "qubits", ()) or ():
                    if q in qubit_index:
                        qs.append(qubit_index[q])
                    else:
                        # Fallback if circuit has unusual bit containers
                        qs.append(getattr(circuit.find_bit(q), "index", -1))
                cs: list[int] = []
                for c in getattr(instr, "clbits", ()) or ():
                    if c in clbit_index:
                        cs.append(clbit_index[c])
                    else:
                        cs.append(getattr(circuit.find_bit(c), "index", -1))

                # Parameter arity (count only)
                params = getattr(op, "params", None)
                parity = len(params) if isinstance(params, (list, tuple)) else 0

                # Classical condition presence
                cond = getattr(op, "condition", None)
                has_cond = 1 if cond is not None else 0

                op_sigs.append(
                    f"{op_name}|p{parity}|q{tuple(qs)}|c{tuple(cs)}|if{has_cond}"
                )

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            # Conservative fallback: avoid breaking tracking
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
    circuit_hash: str | None,
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
    circuit_hash : str or None
        Circuit structure hash.

    Returns
    -------
    list of ProgramArtifact
        References to logged program artifacts, one per format per circuit.
    """
    artifacts: list[ProgramArtifact] = []
    meta = {
        "backend_name": backend_name,
        "qiskit_version": qiskit_version(),
        "circuit_hash": circuit_hash,
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
