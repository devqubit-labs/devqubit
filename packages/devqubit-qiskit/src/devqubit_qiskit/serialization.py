# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit serialization for Qiskit adapter.

Provides QPY and OpenQASM 3.0 serialization using devqubit_engine.circuit types.
Supports lossless (QPY) and portable (OpenQASM 3) representations following
the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from dataclasses import dataclass
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
from qiskit import qpy
from qiskit.qasm3 import dumps as qasm3_dumps


logger = logging.getLogger(__name__)


_QISKIT_GATES: dict[str, GateInfo] = {
    # Single-qubit Clifford gates
    "h": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "x": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "y": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "z": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "s": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sxdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "id": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "i": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    # Single-qubit non-Clifford gates
    "t": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "tdg": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ry": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "p": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u1": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u2": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u3": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    # Two-qubit Clifford gates
    "cx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cnot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "swap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "iswap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "ecr": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    # Two-qubit non-Clifford gates
    "rzz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "rxx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "ryy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "rzx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cp": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "crx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cry": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "crz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cu": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    # Multi-qubit gates
    "ccx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "ccz": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "cswap": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "mcx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    # Measurement and control
    "measure": GateInfo(GateCategory.MEASURE),
    "reset": GateInfo(GateCategory.MEASURE),
    "barrier": GateInfo(GateCategory.BARRIER),
}

_classifier = GateClassifier(_QISKIT_GATES)


def serialize_qpy(circuits: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize circuits to QPY format (lossless, SDK-native).

    QPY is Qiskit's native binary format that preserves all circuit
    information including parameters, metadata, and calibrations.

    Parameters
    ----------
    circuits : QuantumCircuit or list
        Circuit(s) to serialize.
    name : str, optional
        Circuit name for metadata.
    index : int, optional
        Circuit index for metadata.

    Returns
    -------
    CircuitData
        Serialized circuit data.

    Notes
    -----
    QPY is the recommended format for lossless circuit storage.
    Use OpenQASM 3 for cross-SDK portability.
    """
    buffer = io.BytesIO()
    qpy.dump(circuits, buffer)
    return CircuitData(
        data=buffer.getvalue(),
        format=CircuitFormat.QPY,
        sdk=SDK.QISKIT,
        name=name,
        index=index,
    )


def serialize_qasm3(circuits: Any) -> list[CircuitData]:
    """
    Serialize circuits to OpenQASM 3.0 format (portable).

    OpenQASM 3 is a portable text format that can be read by
    multiple quantum SDKs, making it ideal for cross-platform
    comparison and archival.

    Parameters
    ----------
    circuits : Any
        Single circuit or list of circuits.

    Returns
    -------
    list[CircuitData]
        List of serialized circuits (skips failures silently).

    Notes
    -----
    Not all Qiskit circuits can be serialized to OpenQASM 3,
    particularly those with custom gates or certain pulse-level
    operations.
    """
    if not isinstance(circuits, (list, tuple)):
        circuits = [circuits]

    results: list[CircuitData] = []
    for i, c in enumerate(circuits):
        qc_name = getattr(c, "name", None) or f"circuit_{i}"
        try:
            qasm = qasm3_dumps(c)
            if qasm:
                results.append(
                    CircuitData(
                        data=qasm,
                        format=CircuitFormat.OPENQASM3,
                        sdk=SDK.QISKIT,
                        name=f"circuit_{i}:{qc_name}",
                        index=i,
                        metadata={"circuit_name": qc_name},
                    )
                )
        except Exception:
            continue

    return results


@dataclass
class LoadedCircuitBatch:
    """
    Container for a batch of loaded circuits from a single QPY file.

    QPY format supports multiple circuits in a single file. This class
    provides access to all circuits while maintaining compatibility
    with single-circuit workflows.

    Parameters
    ----------
    circuits : list
        List of loaded QuantumCircuit objects.
    sdk : SDK
        Source SDK (QISKIT).
    source_format : CircuitFormat
        Original format (QPY).
    name : str, optional
        Batch name/identifier.

    Attributes
    ----------
    circuits : list
        All loaded circuits.
    count : int
        Number of circuits in the batch.
    """

    circuits: list[Any]
    sdk: SDK
    source_format: CircuitFormat
    name: str = ""

    @property
    def count(self) -> int:
        """Get the number of circuits in this batch."""
        return len(self.circuits)

    def __len__(self) -> int:
        """Return the number of circuits."""
        return len(self.circuits)

    def __iter__(self):
        """Iterate over circuits."""
        return iter(self.circuits)

    def __getitem__(self, index: int) -> Any:
        """Get circuit by index."""
        return self.circuits[index]

    def to_loaded_circuits(self) -> list[LoadedCircuit]:
        """
        Convert batch to list of LoadedCircuit instances.

        Returns
        -------
        list of LoadedCircuit
            Individual circuit containers with proper indexing.
        """
        return [
            LoadedCircuit(
                circuit=circuit,
                sdk=self.sdk,
                source_format=self.source_format,
                name=getattr(circuit, "name", "") or f"circuit_{i}",
                index=i,
            )
            for i, circuit in enumerate(self.circuits)
        ]


class QiskitCircuitLoader:
    """
    Qiskit circuit loader for multiple formats.

    Loads circuits from QPY, OpenQASM 2, and OpenQASM 3 formats
    back into Qiskit QuantumCircuit objects.

    Attributes
    ----------
    name : str
        Loader identifier ("qiskit").
    sdk : SDK
        Target SDK (QISKIT).
    supported_formats : list of CircuitFormat
        Formats this loader can handle.
    """

    name = "qiskit"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this loader handles."""
        return SDK.QISKIT

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [CircuitFormat.QPY, CircuitFormat.OPENQASM3, CircuitFormat.OPENQASM2]

    def load(self, data: CircuitData) -> LoadedCircuit:
        """
        Load circuit from CircuitData.

        For single-circuit data, returns a LoadedCircuit. For QPY data
        that may contain multiple circuits, use load_batch() instead.

        Parameters
        ----------
        data : CircuitData
            Serialized circuit data.

        Returns
        -------
        LoadedCircuit
            Loaded circuit container.

        Raises
        ------
        LoaderError
            If format is unsupported or loading fails.

        See Also
        --------
        load_batch : Load all circuits from QPY batch data.
        """
        if data.format == CircuitFormat.QPY:
            return self._load_qpy(data)
        elif data.format == CircuitFormat.OPENQASM3:
            return self._load_qasm3(data)
        elif data.format == CircuitFormat.OPENQASM2:
            return self._load_qasm2(data)
        else:
            raise LoaderError(f"Unsupported format: {data.format}")

    def load_batch(self, data: CircuitData) -> LoadedCircuitBatch:
        """
        Load all circuits from QPY batch data.

        QPY format supports multiple circuits in a single file. This method
        returns all circuits as a batch, enabling proper restoration of
        multi-circuit submissions.

        Parameters
        ----------
        data : CircuitData
            Serialized QPY data containing one or more circuits.

        Returns
        -------
        LoadedCircuitBatch
            Container with all circuits from the QPY file.

        Raises
        ------
        LoaderError
            If format is not QPY or loading fails.
        """
        if data.format != CircuitFormat.QPY:
            raise LoaderError(f"load_batch only supports QPY format, got {data.format}")

        try:
            circuits = qpy.load(io.BytesIO(data.as_bytes()))
            # qpy.load always returns a list
            if not isinstance(circuits, list):
                circuits = [circuits]

            return LoadedCircuitBatch(
                circuits=circuits,
                sdk=SDK.QISKIT,
                source_format=CircuitFormat.QPY,
                name=data.name,
            )
        except Exception as e:
            raise LoaderError(f"QPY batch load failed: {e}") from e

    def _load_qpy(self, data: CircuitData) -> LoadedCircuit:
        """
        Load single circuit from QPY format.

        For backward compatibility, returns the first circuit when QPY
        contains multiple circuits. Use load_batch() to get all circuits.
        """
        try:
            circuits = qpy.load(io.BytesIO(data.as_bytes()))
            # qpy.load always returns a list
            if isinstance(circuits, list):
                if len(circuits) > 1:
                    logger.warning(
                        "QPY file contains %d circuits, returning first only. "
                        "Use load_batch() to get all circuits.",
                        len(circuits),
                    )
                circuit = circuits[0]
            else:
                circuit = circuits

            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.QISKIT,
                source_format=CircuitFormat.QPY,
                name=data.name or getattr(circuit, "name", ""),
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"QPY load failed: {e}") from e

    def _load_qasm3(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from OpenQASM 3 format."""
        # Try qiskit.qasm3.loads (Qiskit 1.0+)
        try:
            from qiskit.qasm3 import loads
        except ImportError:
            loads = None

        if loads is not None:
            try:
                circuit = loads(data.as_text())
                return LoadedCircuit(
                    circuit=circuit,
                    sdk=SDK.QISKIT,
                    source_format=CircuitFormat.OPENQASM3,
                    name=data.name,
                    index=data.index,
                )
            except Exception as e:
                raise LoaderError(f"QASM3 load failed: {e}") from e

        # Fallback to qiskit-qasm3-import
        try:
            from qiskit_qasm3_import import parse

            circuit = parse(data.as_text())
            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.QISKIT,
                source_format=CircuitFormat.OPENQASM3,
                name=data.name,
                index=data.index,
            )
        except ImportError:
            raise LoaderError("QASM3 requires Qiskit 1.0+ or qiskit-qasm3-import")
        except Exception as e:
            raise LoaderError(f"QASM3 load failed: {e}") from e

    def _load_qasm2(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from OpenQASM 2 format."""
        from qiskit import QuantumCircuit

        try:
            circuit = QuantumCircuit.from_qasm_str(data.as_text())
            return LoadedCircuit(
                circuit=circuit,
                sdk=SDK.QISKIT,
                source_format=CircuitFormat.OPENQASM2,
                name=data.name,
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"QASM2 load failed: {e}") from e


class QiskitCircuitSerializer:
    """
    Qiskit circuit serializer for multiple formats.

    Serializes circuits to QPY (lossless) and OpenQASM 3 (portable)
    formats following the UEC two-layer storage principle.

    Attributes
    ----------
    name : str
        Serializer identifier ("qiskit").
    sdk : SDK
        Source SDK (QISKIT).
    supported_formats : list of CircuitFormat
        Formats this serializer can produce.
    """

    name = "qiskit"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this serializer handles."""
        return SDK.QISKIT

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [CircuitFormat.QPY, CircuitFormat.OPENQASM3]

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
            True if circuit is a Qiskit QuantumCircuit.
        """
        try:
            from qiskit import QuantumCircuit

            return isinstance(circuit, QuantumCircuit)
        except ImportError:
            return False

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
        circuit : QuantumCircuit
            Qiskit circuit.
        fmt : CircuitFormat
            Target format (QPY or OPENQASM3).
        name : str, optional
            Circuit name for metadata.
        index : int, optional
            Circuit index for batches.

        Returns
        -------
        CircuitData
            Serialized circuit data.

        Raises
        ------
        SerializerError
            If format is unsupported or serialization fails.
        """
        circuit_name = name or getattr(circuit, "name", "") or f"circuit_{index}"

        if fmt == CircuitFormat.QPY:
            return serialize_qpy(circuit, name=circuit_name, index=index)

        elif fmt == CircuitFormat.OPENQASM3:
            try:
                source = qasm3_dumps(circuit)
                return CircuitData(
                    data=source,
                    format=CircuitFormat.OPENQASM3,
                    sdk=SDK.QISKIT,
                    name=circuit_name,
                    index=index,
                )
            except Exception as e:
                raise SerializerError(f"QASM3 serialize failed: {e}") from e

        else:
            raise SerializerError(f"Unsupported format: {fmt}")


def summarize_qiskit_circuit(circuit: Any) -> CircuitSummary:
    """
    Summarize a Qiskit QuantumCircuit.

    Extracts gate counts, depth, and other structural information
    for quick circuit characterization.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to summarize.

    Returns
    -------
    CircuitSummary
        Circuit summary with gate counts and statistics.
    """
    gate_counts: Counter[str] = Counter()

    for instruction in circuit.data:
        gate = instruction.operation
        gate_name = gate.name.lower()
        gate_counts[gate_name] += 1

    # Classify gates
    stats = _classifier.classify_counts(dict(gate_counts))

    return CircuitSummary(
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        depth=circuit.depth(),
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        has_parameters=circuit.num_parameters > 0,
        parameter_count=circuit.num_parameters,
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.QPY,
        sdk=SDK.QISKIT,
    )
