# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit serialization for Braket adapter.

Provides JAQCD and OpenQASM serialization, loading, and summarization
for Amazon Braket circuits.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    GateCategory,
    GateClassifier,
    GateInfo,
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_engine.circuit.summary import CircuitSummary


# Gate classification table for Braket gates
_BRAKET_GATES: dict[str, GateInfo] = {
    # Single-qubit Clifford gates
    "h": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "x": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "y": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "z": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "s": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "si": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "v": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "vi": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "i": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    # Single-qubit non-Clifford gates
    "t": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ti": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ry": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "phaseshift": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "gpi": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "gpi2": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    # Two-qubit Clifford gates
    "cnot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "swap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "iswap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "ecr": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    # Two-qubit non-Clifford gates
    "xx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "yy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "zz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "xy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cphaseshift": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cphaseshift00": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cphaseshift01": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cphaseshift10": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "pswap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "ms": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    # Multi-qubit gates
    "ccnot": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "cswap": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    # Measurement and control
    "measure": GateInfo(GateCategory.MEASURE),
    "reset": GateInfo(GateCategory.MEASURE),
    "barrier": GateInfo(GateCategory.BARRIER),
}

_classifier = GateClassifier(_BRAKET_GATES)


def is_braket_circuit(obj: Any) -> bool:
    """
    Check if object is a Braket Circuit.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object is a Braket Circuit.
    """
    try:
        from braket.circuits import Circuit

        return isinstance(obj, Circuit)
    except ImportError:
        return False


# =============================================================================
# Serialization Functions
# =============================================================================


def serialize_jaqcd(circuit: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize circuit to JAQCD format.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit to serialize.
    name : str, optional
        Circuit name for metadata.
    index : int, optional
        Circuit index in batch.

    Returns
    -------
    CircuitData
        Serialized circuit data.

    Raises
    ------
    SerializerError
        If serialization fails.

    Notes
    -----
    JAQCD format does not support measure instructions. If the circuit
    contains measurements, a gate-only copy is serialized instead.
    """
    from braket.circuits.serialization import IRType

    try:
        prog = circuit.to_ir(ir_type=IRType.JAQCD)
        data = prog.json() if hasattr(prog, "json") else str(prog)
        return CircuitData(
            data=data,
            format=CircuitFormat.JAQCD,
            sdk=SDK.BRAKET,
            name=name or f"circuit_{index}",
            index=index,
        )
    except NotImplementedError:
        # JAQCD doesn't support measure instructions - create gate-only copy
        try:
            circuit_copy = _copy_circuit_without_measurements(circuit)
            prog = circuit_copy.to_ir(ir_type=IRType.JAQCD)
            data = prog.json() if hasattr(prog, "json") else str(prog)
            return CircuitData(
                data=data,
                format=CircuitFormat.JAQCD,
                sdk=SDK.BRAKET,
                name=name or f"circuit_{index}",
                index=index,
            )
        except Exception as e:
            raise SerializerError(
                f"JAQCD serialization failed (even without measurements): {e}"
            ) from e
    except Exception as e:
        raise SerializerError(f"JAQCD serialization failed: {e}") from e


def serialize_openqasm(circuit: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize circuit to OpenQASM format.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit to serialize.
    name : str, optional
        Circuit name for metadata.
    index : int, optional
        Circuit index in batch.

    Returns
    -------
    CircuitData
        Serialized circuit data.

    Raises
    ------
    SerializerError
        If serialization fails.
    """
    from braket.circuits.serialization import IRType

    try:
        prog = circuit.to_ir(ir_type=IRType.OPENQASM)
        src = getattr(prog, "source", None)
        if isinstance(src, str) and src.strip():
            return CircuitData(
                data=src,
                format=CircuitFormat.OPENQASM3,
                sdk=SDK.BRAKET,
                name=name or f"circuit_{index}",
                index=index,
            )
        raise SerializerError("OpenQASM source not available")
    except Exception as e:
        raise SerializerError(f"OpenQASM serialization failed: {e}") from e


def _copy_circuit_without_measurements(circuit: Any) -> Any:
    """
    Create a copy of a circuit without measurement operations.

    JAQCD format does not support measure instructions, so this function
    creates a gate-only copy of the circuit for serialization.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit potentially containing measurements.

    Returns
    -------
    braket.circuits.Circuit
        New circuit with only gate operations (no measurements).
    """
    from braket.circuits import Circuit

    circuit_copy = Circuit()

    for instr in circuit.instructions:
        op = instr.operator
        op_name = getattr(op, "name", type(op).__name__).lower()

        if op_name in ("measure", "measurement"):
            continue

        targets = [int(q) for q in instr.target]
        angle = getattr(op, "angle", None)

        gate_fn = getattr(circuit_copy, op_name, None)
        if gate_fn is not None:
            if angle is not None:
                gate_fn(*targets, angle)
            else:
                gate_fn(*targets)

    return circuit_copy


# =============================================================================
# Text Conversion
# =============================================================================


def circuit_to_text(circuit: Any, index: int = 0) -> str:
    """
    Convert circuit to human-readable text format.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit.
    index : int, optional
        Circuit index for labeling.

    Returns
    -------
    str
        Human-readable representation.
    """
    try:
        return f"[{index}]\n{circuit}"
    except Exception:
        return f"[{index}]\n(circuit diagram unavailable)"


def circuits_to_text(circuits: list[Any]) -> str:
    """
    Convert multiple circuits to human-readable text format.

    Parameters
    ----------
    circuits : list
        List of Braket circuits.

    Returns
    -------
    str
        Combined human-readable representation.
    """
    return "\n\n".join(circuit_to_text(c, i) for i, c in enumerate(circuits))


# =============================================================================
# Circuit Summary
# =============================================================================


def summarize_circuit(circuit: Any) -> CircuitSummary:
    """
    Generate a summary of a Braket circuit.

    Extracts gate counts, depth, qubit count, and classification
    information from the circuit.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit to summarize.

    Returns
    -------
    CircuitSummary
        Circuit summary with statistics and gate counts.
    """
    gate_counts: Counter[str] = Counter()

    for instr in circuit.instructions:
        op = instr.operator
        gate_name = getattr(op, "name", type(op).__name__).lower()
        gate_counts[gate_name] += 1

    stats = _classifier.classify_counts(dict(gate_counts))

    return CircuitSummary(
        num_qubits=circuit.qubit_count,
        depth=circuit.depth,
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.JAQCD,
        sdk=SDK.BRAKET,
    )


# =============================================================================
# Loader and Serializer Classes
# =============================================================================


class BraketCircuitLoader:
    """
    Loader for Braket circuits from serialized formats.

    Supports loading from JAQCD and OpenQASM formats.

    Attributes
    ----------
    name : str
        Loader identifier.
    sdk : SDK
        Target SDK (BRAKET).
    supported_formats : list of CircuitFormat
        Formats this loader can handle.
    """

    name = "braket"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this loader handles."""
        return SDK.BRAKET

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [CircuitFormat.JAQCD, CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2]

    def load(self, data: CircuitData) -> LoadedCircuit:
        """
        Load circuit from serialized data.

        Parameters
        ----------
        data : CircuitData
            Serialized circuit data.

        Returns
        -------
        LoadedCircuit
            Container with loaded circuit object.

        Raises
        ------
        LoaderError
            If format is unsupported or loading fails.
        """
        if data.format == CircuitFormat.JAQCD:
            return self._load_jaqcd(data)
        if data.format in (CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2):
            return self._load_openqasm(data)
        raise LoaderError(f"Unsupported format: {data.format}")

    def _load_jaqcd(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from JAQCD format."""
        from braket.circuits import Circuit
        from braket.ir.jaqcd import Program

        try:
            prog = Program.parse_raw(data.as_text())
            circuit = Circuit()

            for instr in prog.instructions:
                gate_name = instr.type.lower()
                targets = getattr(instr, "targets", None) or getattr(
                    instr, "target", []
                )
                if not isinstance(targets, list):
                    targets = [targets]
                control = getattr(instr, "control", None)
                angle = getattr(instr, "angle", None)

                gate_fn = getattr(circuit, gate_name, None)
                if gate_fn is not None:
                    args = [control] if control is not None else []
                    args.extend(targets)
                    if angle is not None:
                        gate_fn(*args, angle)
                    else:
                        gate_fn(*args)

            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.BRAKET,
                source_format=CircuitFormat.JAQCD,
                name=data.name,
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"JAQCD load failed: {e}") from e

    def _load_openqasm(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from OpenQASM format."""
        from braket.circuits import Circuit

        try:
            circuit = Circuit.from_ir(data.as_text())
            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.BRAKET,
                source_format=data.format,
                name=data.name,
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"OpenQASM load failed: {e}") from e


class BraketCircuitSerializer:
    """
    Serializer for Braket circuits.

    Supports serialization to JAQCD and OpenQASM formats.

    Attributes
    ----------
    name : str
        Serializer identifier.
    sdk : SDK
        Source SDK (BRAKET).
    supported_formats : list of CircuitFormat
        Formats this serializer can produce.
    """

    name = "braket"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this serializer handles."""
        return SDK.BRAKET

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported output formats."""
        return [CircuitFormat.JAQCD, CircuitFormat.OPENQASM3]

    def can_serialize(self, circuit: Any) -> bool:
        """
        Check if this serializer can handle a circuit.

        Parameters
        ----------
        circuit : Any
            Circuit object to check.

        Returns
        -------
        bool
            True if circuit is a Braket Circuit.
        """
        return is_braket_circuit(circuit)

    def serialize(
        self,
        circuit: Any,
        fmt: CircuitFormat,
        *,
        name: str = "",
        index: int = 0,
    ) -> CircuitData:
        """
        Serialize circuit to specified format.

        Parameters
        ----------
        circuit : braket.circuits.Circuit
            Braket circuit to serialize.
        fmt : CircuitFormat
            Target format.
        name : str, optional
            Circuit name for metadata.
        index : int, optional
            Circuit index in batch.

        Returns
        -------
        CircuitData
            Serialized circuit data.

        Raises
        ------
        SerializerError
            If format is unsupported or serialization fails.
        """
        if fmt == CircuitFormat.JAQCD:
            return serialize_jaqcd(circuit, name=name, index=index)
        if fmt == CircuitFormat.OPENQASM3:
            return serialize_openqasm(circuit, name=name, index=index)
        raise SerializerError(f"Unsupported format: {fmt}")


def summarize_braket_circuit(circuit: Any) -> CircuitSummary:
    """
    Generate a summary of a Braket circuit.

    Extracts gate counts, depth, qubit count, and classification
    information from the circuit.

    Parameters
    ----------
    circuit : braket.circuits.Circuit
        Braket circuit to summarize.

    Returns
    -------
    CircuitSummary
        Circuit summary with statistics and gate counts.
    """
    gate_counts: Counter[str] = Counter()

    for instr in circuit.instructions:
        op = instr.operator
        gate_name = getattr(op, "name", type(op).__name__).lower()
        gate_counts[gate_name] += 1

    # Classify gates using the classifier
    stats = _classifier.classify_counts(dict(gate_counts))

    return CircuitSummary(
        num_qubits=circuit.qubit_count,
        depth=circuit.depth,
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.JAQCD,
        sdk=SDK.BRAKET,
    )
