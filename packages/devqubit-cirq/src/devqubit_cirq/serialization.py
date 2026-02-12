# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit serialization for Cirq adapter.

Provides Cirq JSON and OpenQASM serialization, loading, and summarization
for Google Cirq circuits.
"""

from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


# Gate classification table for Cirq gates
_CIRQ_GATES: dict[str, GateInfo] = {
    # Single-qubit Clifford gates
    "h": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "x": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "y": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "z": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "s": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sxdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    # Single-qubit non-Clifford gates
    "t": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "tdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ry": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    # Two-qubit Clifford gates
    "cnot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "swap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "iswap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    # Two-qubit non-Clifford gates
    "fsim": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "xx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "yy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "zz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cp": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cphase": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    # Multi-qubit gates
    "ccz": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "ccx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "cswap": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    # Measurement and reset
    "measure": GateInfo(GateCategory.MEASURE),
    "reset": GateInfo(GateCategory.MEASURE),
}

_classifier = GateClassifier(_CIRQ_GATES)


def _normalize_cirq_gate(gate: Any) -> str:
    """
    Normalize Cirq gate to canonical name.

    Cirq uses class names like HPowGate, CNotPowGate, ZPowGate.
    This maps them to standard gate names.

    Parameters
    ----------
    gate : cirq.Gate
        Cirq gate object.

    Returns
    -------
    str
        Normalized lowercase gate name.
    """
    import cirq

    # Handle power gates with specific exponents
    if isinstance(gate, cirq.ZPowGate):
        exp = getattr(gate, "exponent", 1)
        mapping = {1: "z", 0.5: "s", -0.5: "sdg", 0.25: "t", -0.25: "tdg"}
        return mapping.get(exp, "rz")

    if isinstance(gate, cirq.XPowGate):
        return "x" if getattr(gate, "exponent", 1) == 1 else "rx"

    if isinstance(gate, cirq.YPowGate):
        return "y" if getattr(gate, "exponent", 1) == 1 else "ry"

    if isinstance(gate, cirq.HPowGate):
        return "h"

    if isinstance(gate, cirq.CNotPowGate):
        return "cnot"

    if isinstance(gate, cirq.CZPowGate):
        return "cz"

    if isinstance(gate, cirq.SwapPowGate):
        return "swap"

    if isinstance(gate, cirq.ISwapPowGate):
        return "iswap"

    if isinstance(gate, cirq.MeasurementGate):
        return "measure"

    # Fall back to class name
    return type(gate).__name__.lower().replace("powgate", "").replace("gate", "")


def is_cirq_circuit(obj: Any) -> bool:
    """
    Check if object is a Cirq circuit.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object is a Cirq Circuit or FrozenCircuit.
    """
    try:
        import cirq

        # Check against known circuit types
        circuit_types = [
            getattr(cirq, name, None)
            for name in ("AbstractCircuit", "Circuit", "FrozenCircuit")
        ]
        circuit_types = [cls for cls in circuit_types if isinstance(cls, type)]

        if circuit_types and isinstance(obj, tuple(circuit_types)):
            return True

        # Duck-type check for circuit-like objects
        return hasattr(obj, "all_operations") and hasattr(obj, "moments")
    except ImportError:
        return False


def serialize_json(circuit: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize circuit to Cirq JSON format.

    Parameters
    ----------
    circuit : cirq.Circuit
        Cirq circuit to serialize.
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
    import cirq

    try:
        json_str = cirq.to_json(circuit, file_or_fn=None, indent=2)
        return CircuitData(
            data=json_str,
            format=CircuitFormat.CIRQ_JSON,
            sdk=SDK.CIRQ,
            name=name or f"circuit_{index}",
            index=index,
        )
    except Exception as e:
        raise SerializerError(f"Cirq JSON serialization failed: {e}") from e


def serialize_openqasm3(circuit: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize circuit to OpenQASM format (3.0 preferred, 2.0 fallback).

    Attempts QASM 3.0 first, falls back to 2.0 if unsupported.
    The returned CircuitData.format reflects the actual version used.

    Parameters
    ----------
    circuit : cirq.Circuit
        Cirq circuit to serialize.
    name : str, optional
        Circuit name for metadata.
    index : int, optional
        Circuit index in batch.

    Returns
    -------
    CircuitData
        Serialized circuit data with appropriate format indicator.

    Raises
    ------
    SerializerError
        If circuit doesn't support QASM export or serialization fails.

    Notes
    -----
    Cirq's QASM import/export is experimental with limited compatibility
    for both QASM 2.0 and 3.0. Some circuits may not round-trip correctly.
    """
    to_qasm = getattr(circuit, "to_qasm", None)
    if not callable(to_qasm):
        raise SerializerError("Circuit does not support to_qasm()")

    # Try QASM 3.0 first with different argument names (Cirq API varies)
    src: str | None = None
    qasm_version = CircuitFormat.OPENQASM3

    for kwarg in ("version", "output_version"):
        try:
            src = to_qasm(**{kwarg: "3.0"})
            if isinstance(src, str) and src.strip():
                break
            src = None
        except TypeError as e:
            logger.debug("Suppressed error: %s", e)
            continue
        except Exception as e:
            raise SerializerError(f"OpenQASM 3.0 serialization failed: {e}") from e

    # Fallback to default (QASM 2.0)
    if src is None:
        try:
            src = to_qasm()
            qasm_version = CircuitFormat.OPENQASM2
        except Exception as e:
            raise SerializerError(f"OpenQASM serialization failed: {e}") from e

    if not isinstance(src, str) or not src.strip():
        raise SerializerError("Empty OpenQASM output")

    return CircuitData(
        data=src,
        format=qasm_version,
        sdk=SDK.CIRQ,
        name=name or f"circuit_{index}",
        index=index,
    )


def circuit_to_text(circuit: Any, index: int = 0) -> str:
    """
    Convert a Cirq circuit to human-readable text diagram.

    Parameters
    ----------
    circuit : cirq.Circuit
        Cirq circuit.
    index : int, optional
        Circuit index for labeling.

    Returns
    -------
    str
        Human-readable circuit diagram with index header.
    """
    header = f"[{index}]"
    try:
        to_text = getattr(circuit, "to_text_diagram", None)
        if callable(to_text):
            return f"{header}\n{to_text()}"
    except (TypeError, ValueError) as e:
        logger.debug("Failed to render circuit diagram: %s", e)
    return f"{header}\n{repr(circuit)}"


def circuits_to_text(circuits: Any) -> str:
    """
    Convert multiple Cirq circuits to human-readable text format.

    Parameters
    ----------
    circuits : Circuit or list of Circuit
        Single circuit or list of circuits.

    Returns
    -------
    str
        Combined circuit diagrams separated by blank lines.
    """
    if is_cirq_circuit(circuits):
        return circuit_to_text(circuits, 0)

    # Convert to list
    if isinstance(circuits, (list, tuple)):
        circuit_list = list(circuits)
    else:
        try:
            circuit_list = list(circuits)
        except TypeError:
            circuit_list = [circuits]

    return "\n\n".join(circuit_to_text(c, i) for i, c in enumerate(circuit_list))


class CirqCircuitLoader:
    """
    Loader for Cirq circuits from serialized formats.

    Supports loading from Cirq JSON and OpenQASM formats.

    Attributes
    ----------
    name : str
        Loader identifier.
    sdk : SDK
        Target SDK (CIRQ).
    supported_formats : list of CircuitFormat
        Formats this loader can handle.
    """

    name = "cirq"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this loader handles."""
        return SDK.CIRQ

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [
            CircuitFormat.CIRQ_JSON,
            CircuitFormat.OPENQASM3,
            CircuitFormat.OPENQASM2,
        ]

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
        if data.format == CircuitFormat.CIRQ_JSON:
            return self._load_json(data)
        if data.format in (CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2):
            return self._load_openqasm(data)
        raise LoaderError(f"Unsupported format: {data.format}")

    def _load_json(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from Cirq JSON format."""
        import cirq

        try:
            obj = cirq.read_json(json_text=data.as_text())

            # Handle wrapped circuit lists
            if isinstance(obj, dict) and obj.get("type") == "list":
                circuits = obj.get("circuits", [])
                circuit = circuits[0] if circuits else cirq.Circuit()
            else:
                circuit = obj

            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.CIRQ,
                source_format=CircuitFormat.CIRQ_JSON,
                name=data.name,
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"Cirq JSON load failed: {e}") from e

    def _load_openqasm(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from OpenQASM format."""
        try:
            from cirq.contrib.qasm_import import circuit_from_qasm

            circuit = circuit_from_qasm(data.as_text())
            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.CIRQ,
                source_format=data.format,
                name=data.name,
                index=data.index,
            )
        except ImportError as e:
            raise LoaderError(
                "Cirq QASM import requires cirq.contrib.qasm_import"
            ) from e
        except Exception as e:
            raise LoaderError(f"OpenQASM load failed: {e}") from e


class CirqCircuitSerializer:
    """
    Serializer for Cirq circuits.

    Supports serialization to Cirq JSON and OpenQASM formats.

    Attributes
    ----------
    name : str
        Serializer identifier.
    sdk : SDK
        Source SDK (CIRQ).
    supported_formats : list of CircuitFormat
        Formats this serializer can produce.
    """

    name = "cirq"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this serializer handles."""
        return SDK.CIRQ

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported output formats."""
        return [
            CircuitFormat.CIRQ_JSON,
            CircuitFormat.OPENQASM3,
            CircuitFormat.OPENQASM2,
        ]

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
            True if circuit is a Cirq circuit.
        """
        return is_cirq_circuit(circuit)

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
        circuit : cirq.Circuit
            Cirq circuit to serialize.
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
        if fmt == CircuitFormat.CIRQ_JSON:
            return serialize_json(circuit, name=name, index=index)
        if fmt in (CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2):
            return serialize_openqasm3(circuit, name=name, index=index)
        raise SerializerError(f"Unsupported format: {fmt}")


def summarize_cirq_circuit(circuit: Any) -> CircuitSummary:
    """
    Generate a summary of a Cirq circuit.

    Extracts gate counts, depth, qubit count, and classification
    information from the circuit.

    Parameters
    ----------
    circuit : cirq.Circuit
        Cirq circuit to summarize.

    Returns
    -------
    CircuitSummary
        Circuit summary with statistics and gate counts.
    """
    import cirq

    all_qubits = sorted(circuit.all_qubits(), key=str)
    gate_counts: Counter[str] = Counter()
    has_params = False

    for op in circuit.all_operations():
        gate = op.gate
        if gate is None:
            continue

        gate_counts[_normalize_cirq_gate(gate)] += 1

        try:
            if cirq.is_parameterized(gate):
                has_params = True
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to check parameterized gate: %s", e)

    stats = _classifier.classify_counts(dict(gate_counts))

    try:
        depth = len(circuit)
    except (TypeError, AttributeError) as e:
        logger.debug("Failed to get circuit depth: %s", e)
        depth = 0

    return CircuitSummary(
        num_qubits=len(all_qubits),
        depth=depth,
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        has_parameters=has_params,
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.CIRQ_JSON,
        sdk=SDK.CIRQ,
    )
