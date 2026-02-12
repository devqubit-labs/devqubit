# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit serialization for PennyLane adapter.

Provides tape serialization, loading, and summarization for PennyLane.
"""

from __future__ import annotations

import json
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
from devqubit_pennylane.utils import extract_shots_info


logger = logging.getLogger(__name__)


# Gate classification table for PennyLane gates
_PENNYLANE_GATES: dict[str, GateInfo] = {
    # Single-qubit Clifford gates
    "hadamard": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "paulix": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "pauliy": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "pauliz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "s": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    "sx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=True),
    # Single-qubit non-Clifford gates
    "t": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rx": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "ry": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rz": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "rot": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "u3": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    "phaseshift": GateInfo(GateCategory.SINGLE_QUBIT, is_clifford=False),
    # Two-qubit Clifford gates
    "cnot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "cy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "swap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    "iswap": GateInfo(GateCategory.TWO_QUBIT, is_clifford=True),
    # Two-qubit non-Clifford gates
    "crx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "cry": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "crz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "crot": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "controlledphaseshift": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "isingxx": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "isingyy": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    "isingzz": GateInfo(GateCategory.TWO_QUBIT, is_clifford=False),
    # Multi-qubit gates
    "toffoli": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "ccx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "cswap": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    "multicontrolledx": GateInfo(GateCategory.MULTI_QUBIT, is_clifford=False),
    # Measurement - include both base names and MP suffixed names
    "measure": GateInfo(GateCategory.MEASURE),
    "sample": GateInfo(GateCategory.MEASURE),
    "samplemp": GateInfo(GateCategory.MEASURE),
    "counts": GateInfo(GateCategory.MEASURE),
    "countsmp": GateInfo(GateCategory.MEASURE),
    "expval": GateInfo(GateCategory.MEASURE),
    "expectationmp": GateInfo(GateCategory.MEASURE),
    "var": GateInfo(GateCategory.MEASURE),
    "variancemp": GateInfo(GateCategory.MEASURE),
    "probs": GateInfo(GateCategory.MEASURE),
    "probabilitymp": GateInfo(GateCategory.MEASURE),
    "state": GateInfo(GateCategory.MEASURE),
    "statemp": GateInfo(GateCategory.MEASURE),
}

_classifier = GateClassifier(_PENNYLANE_GATES)


# Mapping from serialized return_type names to PennyLane measurement functions
_MEASUREMENT_MAP = {
    "expectationmp": "expval",
    "expval": "expval",
    "samplemp": "sample",
    "sample": "sample",
    "countsmp": "counts",
    "counts": "counts",
    "probabilitymp": "probs",
    "probs": "probs",
    "variancemp": "var",
    "var": "var",
    "statemp": "state",
    "state": "state",
}


def is_pennylane_tape(obj: Any) -> bool:
    """
    Check if object is a PennyLane tape.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object is a PennyLane tape.
    """
    try:
        import pennylane as qml

        if isinstance(obj, qml.tape.QuantumTape):
            return True
        if hasattr(qml.tape, "QuantumScript") and isinstance(
            obj, qml.tape.QuantumScript
        ):
            return True
        return False
    except ImportError as e:
        logger.debug("Suppressed error: %s", e)
        return False


def _param_to_jsonable(p: Any) -> Any:
    """
    Convert a single parameter to a JSON-safe value.

    Handles scalars, 0-d arrays, Autograd ArrayBox, and batched
    (multi-element) arrays.  Batched arrays are summarised instead of
    being dumped verbatim.

    Returns
    -------
    float | dict
        A float for scalar values, or a summary dict for batched params.
    """
    # 1. Plain Python scalar
    if isinstance(p, (int, float)):
        return float(p)

    # 2. Try .item() — works for 0-d numpy/torch/jax arrays
    if hasattr(p, "item"):
        try:
            return float(p.item())
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("Float extraction via .item() failed: %s", e)

    # 3. Autograd ArrayBox — extract wrapped _value
    if hasattr(p, "_value"):
        try:
            val = p._value
            if hasattr(val, "item"):
                return float(val.item())
            return float(val)
        except (TypeError, ValueError) as e:
            logger.debug("Float conversion failed: %s", e)

    # 4. Direct float() — catches remaining scalar-like objects
    try:
        return float(p)
    except (TypeError, ValueError) as e:
        logger.debug("Float conversion failed: %s", e)

    # 5. Multi-element array (batched parameters) — summarise
    shape = getattr(p, "shape", None)
    if shape is not None:
        dtype = str(getattr(p, "dtype", "unknown"))
        return {"batched": True, "shape": list(shape), "dtype": dtype}

    # 6. Last resort — truncated string
    return str(p)[:120]


def _serialize_operation(op: Any) -> dict[str, Any]:
    """Serialize a single operation to dict."""
    op_dict: dict[str, Any] = {
        "name": op.name,
        "wires": list(op.wires),
    }
    if op.parameters:
        op_dict["parameters"] = [_param_to_jsonable(p) for p in op.parameters]
    return op_dict


def _serialize_measurement(m: Any) -> dict[str, Any]:
    """Serialize a single measurement to dict."""
    m_dict: dict[str, Any] = {
        "return_type": str(type(m).__name__),
        "wires": list(m.wires) if m.wires else None,
    }

    # Store observable info for reconstruction
    if hasattr(m, "obs") and m.obs is not None:
        obs = m.obs
        obs_dict: dict[str, Any] = {
            "name": getattr(obs, "name", type(obs).__name__),
        }

        # Store observable wires (may differ from measurement wires)
        if hasattr(obs, "wires"):
            obs_dict["wires"] = list(obs.wires)

        # For tensor products or composed observables
        if hasattr(obs, "operands"):
            obs_dict["operands"] = [
                {"name": getattr(o, "name", type(o).__name__), "wires": list(o.wires)}
                for o in obs.operands
            ]

        m_dict["observable"] = obs_dict

    return m_dict


def _format_param_text(p: Any) -> str:
    """Format a single parameter for text display."""
    if isinstance(p, (int, float)):
        return f"{p:.4f}" if isinstance(p, float) else str(p)

    # 0-d array / ArrayBox => scalar
    if hasattr(p, "item"):
        try:
            return f"{float(p.item()):.4f}"
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("Float extraction via .item() failed: %s", e)
    if hasattr(p, "_value"):
        try:
            val = p._value
            v = float(val.item()) if hasattr(val, "item") else float(val)
            return f"{v:.4f}"
        except (TypeError, ValueError) as e:
            logger.debug("Float extraction via .item() failed: %s", e)
    try:
        return f"{float(p):.4f}"
    except (TypeError, ValueError) as e:
        logger.debug("Float conversion failed: %s", e)

    # Batched array => summary
    shape = getattr(p, "shape", None)
    if shape is not None:
        return f"<batch {list(shape)}>"
    return str(p)[:60]


def _strip_param_definitions(diagram: str) -> str:
    """
    Remove parameter definition blocks (M0 = ...) from tape_text output.

    Keeps only the wire diagram lines.
    """
    lines = diagram.splitlines()
    result: list[str] = []
    in_param_block = False
    for line in lines:
        # Detect start of parameter definition block (e.g. "M0 = ")
        stripped = line.strip()
        if (
            stripped
            and len(stripped) >= 3
            and stripped[0] == "M"
            and "=" in stripped[:6]
        ):
            # Check pattern: M<digits> = or M<digits> =
            prefix = stripped.split("=", 1)[0].strip()
            if prefix[0] == "M" and prefix[1:].isdigit():
                in_param_block = True
                continue
        if in_param_block:
            # Parameter blocks contain numpy-style arrays: lines starting
            # with [ or containing floats in scientific notation
            if (
                stripped.startswith("[")
                or stripped.startswith("-")
                or stripped.startswith("0")
                or (stripped and stripped[0].isdigit())
            ):
                continue
            # Continuation lines with just numbers
            if stripped.endswith("]"):
                in_param_block = False
                continue
            # Empty line ends param block
            if not stripped:
                in_param_block = False
                # Don't add extra blank line
                continue
            # Non-array line — we're past the param block
            in_param_block = False
        result.append(line)
    # Strip trailing blank lines
    while result and not result[-1].strip():
        result.pop()
    return "\n".join(result)


def _tape_to_diagram(tape: Any) -> str | None:
    """
    Render a tape as an ASCII wire diagram via PennyLane's drawer.

    Returns None if PennyLane is unavailable or drawing fails.
    Parameter definition blocks (M0 = ...) are stripped — only the
    wire diagram is kept.
    """
    try:
        import pennylane as qml

        # tape_text is the correct API for tapes (qml.draw is for QNodes)
        tape_text_fn = getattr(getattr(qml, "drawer", None), "tape_text", None)
        if tape_text_fn is not None:
            text = tape_text_fn(tape)
            if text and text.strip():
                return _strip_param_definitions(text)

        # Fallback: older PennyLane versions may have it at top level
        tape_text_fn = getattr(qml, "tape_text", None)
        if tape_text_fn is not None:
            text = tape_text_fn(tape)
            if text and text.strip():
                return _strip_param_definitions(text)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to render tape diagram: %s", e)
    return None


def tape_to_text(tape: Any, index: int = 0) -> str:
    """
    Convert a PennyLane tape to human-readable diagram.

    Uses ``qml.draw`` for a proper ASCII wire diagram when available,
    falling back to a compact operation listing.

    Parameters
    ----------
    tape : QuantumTape
        PennyLane tape.
    index : int, optional
        Tape index for labeling.

    Returns
    -------
    str
        Human-readable tape diagram.
    """
    header_lines = [f"=== Tape {index} ==="]

    # Wires info
    header_lines.append(f"Wires: {list(tape.wires)}")

    # Shots info using shared utility
    shots_info = extract_shots_info(tape)
    if shots_info.analytic:
        header_lines.append("Shots: None (analytic)")
    else:
        header_lines.append(f"Shots: {shots_info.total_shots}")
        if shots_info.shot_vector:
            header_lines.append(f"Shot vector: {shots_info.shot_vector}")

    # Try native qml.draw first
    diagram = _tape_to_diagram(tape)
    if diagram:
        header_lines.append("")
        header_lines.append(diagram)
        return "\n".join(header_lines)

    # Fallback: compact operation listing with safe param formatting
    header_lines.append("")
    header_lines.append("Operations:")
    for op in tape.operations:
        params_str = ""
        if op.parameters:
            parts = [_format_param_text(p) for p in op.parameters]
            params_str = f"({', '.join(parts)})"
        header_lines.append(f"  {op.name}{params_str} @ wires={list(op.wires)}")

    header_lines.append("")
    header_lines.append("Measurements:")
    for m in tape.measurements:
        m_type = type(m).__name__
        obs_str = f" of {m.obs}" if hasattr(m, "obs") and m.obs else ""
        wires_str = f" @ wires={list(m.wires)}" if m.wires else ""
        header_lines.append(f"  {m_type}{obs_str}{wires_str}")

    return "\n".join(header_lines)


def tapes_to_text(tapes: list[Any]) -> str:
    """
    Convert multiple PennyLane tapes to human-readable diagrams.

    Parameters
    ----------
    tapes : list
        List of PennyLane tapes.

    Returns
    -------
    str
        Combined human-readable tape diagrams.
    """
    parts = [tape_to_text(t, i) for i, t in enumerate(tapes)]
    return "\n\n".join(parts)


def _serialize_tape_dict(tape: Any) -> dict[str, Any]:
    """
    Serialize a single tape to a dictionary.

    Parameters
    ----------
    tape : QuantumTape
        PennyLane tape.

    Returns
    -------
    dict
        Serialized tape dictionary (without index field).
    """
    ops = [_serialize_operation(op) for op in tape.operations]
    measurements = [_serialize_measurement(m) for m in tape.measurements]

    tape_dict: dict[str, Any] = {
        "num_wires": len(tape.wires),
        "wires": list(tape.wires),
        "num_operations": len(ops),
        "operations": ops,
        "num_measurements": len(measurements),
        "measurements": measurements,
    }

    # Handle Shots using shared utility
    shots_info = extract_shots_info(tape)
    if not shots_info.analytic:
        tape_dict["shots"] = shots_info.total_shots
        if shots_info.shot_vector:
            tape_dict["shot_vector"] = shots_info.shot_vector

    return tape_dict


def serialize_tape(tape: Any, *, name: str = "", index: int = 0) -> CircuitData:
    """
    Serialize a single tape to JSON format.

    Parameters
    ----------
    tape : QuantumTape
        PennyLane tape.
    name : str, optional
        Tape name.
    index : int, optional
        Tape index.

    Returns
    -------
    CircuitData
        Serialized tape.

    Raises
    ------
    SerializerError
        If serialization fails.
    """
    try:
        tape_dict = _serialize_tape_dict(tape)
        data = json.dumps(tape_dict, indent=2)
        return CircuitData(
            data=data,
            format=CircuitFormat.TAPE_JSON,
            sdk=SDK.PENNYLANE,
            name=name or f"tape_{index}",
            index=index,
        )
    except Exception as e:
        raise SerializerError(f"Tape serialize failed: {e}") from e


def serialize_tapes(tapes: list[Any]) -> CircuitData:
    """
    Serialize multiple tapes to JSON format.

    Parameters
    ----------
    tapes : list
        List of PennyLane tapes.

    Returns
    -------
    CircuitData
        Serialized tapes.

    Raises
    ------
    SerializerError
        If serialization fails.
    """
    try:
        tape_dicts = []
        for i, tape in enumerate(tapes):
            tape_dict = _serialize_tape_dict(tape)
            tape_dict["index"] = i  # Add index for batch format
            tape_dicts.append(tape_dict)
        data = json.dumps({"num_tapes": len(tapes), "tapes": tape_dicts}, indent=2)
        return CircuitData(
            data=data,
            format=CircuitFormat.TAPE_JSON,
            sdk=SDK.PENNYLANE,
            name="tapes",
            index=0,
        )
    except Exception as e:
        raise SerializerError(f"Tapes serialize failed: {e}") from e


def _reconstruct_observable(obs_dict: dict[str, Any], qml: Any) -> Any:
    """
    Reconstruct a PennyLane observable from serialized dict.

    Parameters
    ----------
    obs_dict : dict
        Serialized observable dictionary.
    qml : module
        PennyLane module.

    Returns
    -------
    Observable
        Reconstructed PennyLane observable.
    """
    obs_name = obs_dict.get("name", "")
    obs_wires = obs_dict.get("wires", [0])

    # Handle tensor products
    if "operands" in obs_dict:
        operands = []
        for op_dict in obs_dict["operands"]:
            op_obs = _reconstruct_observable(op_dict, qml)
            if op_obs is not None:
                operands.append(op_obs)
        if len(operands) >= 2:
            result = operands[0]
            for op in operands[1:]:
                result = result @ op
            return result
        elif operands:
            return operands[0]

    # Map common observable names to PennyLane observables
    obs_map = {
        "PauliX": qml.PauliX,
        "PauliY": qml.PauliY,
        "PauliZ": qml.PauliZ,
        "Hadamard": qml.Hadamard,
        "Identity": qml.Identity,
    }

    obs_class = obs_map.get(obs_name)
    if obs_class:
        return obs_class(wires=obs_wires[0] if len(obs_wires) == 1 else obs_wires)

    # Fallback to PauliZ
    logger.debug("Unknown observable %s, falling back to PauliZ", obs_name)
    return qml.PauliZ(wires=obs_wires[0] if obs_wires else 0)


def _reconstruct_measurement(m_dict: dict[str, Any], qml: Any) -> Any | None:
    """
    Reconstruct a PennyLane measurement from serialized dict.

    Parameters
    ----------
    m_dict : dict
        Serialized measurement dictionary.
    qml : module
        PennyLane module.

    Returns
    -------
    MeasurementProcess or None
        Reconstructed measurement, or None if reconstruction fails.
    """
    return_type = m_dict.get("return_type", "").lower()
    wires = m_dict.get("wires")

    # Map to PennyLane measurement function
    meas_name = _MEASUREMENT_MAP.get(return_type)
    if not meas_name:
        logger.debug("Unknown measurement type: %s", return_type)
        return None

    meas_fn = getattr(qml, meas_name, None)
    if meas_fn is None:
        logger.debug("Measurement function not found: %s", meas_name)
        return None

    # Reconstruct observable if present
    obs_dict = m_dict.get("observable")

    try:
        if meas_name in ("expval", "var", "sample", "counts"):
            # These measurements require an observable
            if obs_dict:
                obs = _reconstruct_observable(obs_dict, qml)
                return meas_fn(obs)
            elif wires is not None:
                # Fallback: use PauliZ on measurement wires
                if len(wires) == 1:
                    return meas_fn(qml.PauliZ(wires=wires[0]))
                else:
                    # Tensor product for multiple wires
                    obs = qml.PauliZ(wires=wires[0])
                    for w in wires[1:]:
                        obs = obs @ qml.PauliZ(wires=w)
                    return meas_fn(obs)
            else:
                # No wires specified, use wire 0
                return meas_fn(qml.PauliZ(wires=0))

        elif meas_name in ("probs", "state"):
            # These can work with or without wires
            if wires is not None:
                return meas_fn(wires=wires)
            else:
                return meas_fn()

        else:
            # Unknown measurement
            return None

    except Exception as e:
        logger.debug("Failed to reconstruct measurement: %s", e)
        return None


class PennyLaneCircuitLoader:
    """
    PennyLane circuit loader.

    Loads circuits from PennyLane tape JSON format, reconstructing
    operations, measurements, and shots configuration.

    Attributes
    ----------
    name : str
        Loader identifier.
    sdk : SDK
        Target SDK (PENNYLANE).
    supported_formats : list of CircuitFormat
        Formats this loader can handle.
    """

    name = "pennylane"

    @property
    def sdk(self) -> SDK:
        """Get the SDK this loader handles."""
        return SDK.PENNYLANE

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [CircuitFormat.TAPE_JSON]

    def load(self, data: CircuitData) -> LoadedCircuit:
        """
        Load circuit from CircuitData.

        Fully reconstructs the tape including operations, measurements,
        and shots configuration.

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
        """
        if data.format != CircuitFormat.TAPE_JSON:
            raise LoaderError(f"Unsupported format: {data.format}")

        try:
            import pennylane as qml

            tape_data = json.loads(data.as_text())

            # Handle both single tape and multi-tape format
            if "tapes" in tape_data:
                tapes_json = tape_data.get("tapes", [])
                if not tapes_json:
                    raise LoaderError("No tapes found in data")
                first = tapes_json[0]
            else:
                first = tape_data

            # Reconstruct tape with operations and measurements
            with qml.tape.QuantumTape() as tape:
                # Reconstruct operations
                for op in first.get("operations", []):
                    gate_fn = getattr(qml, op["name"], None)
                    if gate_fn:
                        params = op.get("parameters", [])
                        wires = op["wires"]
                        if params:
                            gate_fn(*params, wires=wires)
                        else:
                            gate_fn(wires=wires)
                    else:
                        logger.debug("Unknown gate: %s", op["name"])

                # Reconstruct measurements
                for m_dict in first.get("measurements", []):
                    _reconstruct_measurement(m_dict, qml)

            # Set shots on the tape if available
            shots = first.get("shots")
            shot_vector = first.get("shot_vector")

            if shots is not None or shot_vector is not None:
                # Create Shots object if available
                try:
                    from pennylane.measurements import Shots

                    if shot_vector:
                        # Convert shot vector to format Shots expects
                        tape._shots = Shots(shot_vector)
                    elif shots:
                        tape._shots = Shots(shots)
                except (ImportError, AttributeError):
                    # Older PennyLane version, just set integer shots
                    if shots:
                        tape._shots = shots

            return LoadedCircuit(
                circuit=tape,
                sdk=SDK.PENNYLANE,
                source_format=CircuitFormat.TAPE_JSON,
                name=data.name,
                index=data.index,
            )
        except Exception as e:
            raise LoaderError(f"Tape load failed: {e}") from e

    def load_batch(self, data: CircuitData) -> list[LoadedCircuit]:
        """
        Load multiple circuits from CircuitData.

        Parameters
        ----------
        data : CircuitData
            Serialized circuit data containing multiple tapes.

        Returns
        -------
        list of LoadedCircuit
            List of loaded circuit containers.

        Raises
        ------
        LoaderError
            If format is unsupported or loading fails.
        """
        if data.format != CircuitFormat.TAPE_JSON:
            raise LoaderError(f"Unsupported format: {data.format}")

        try:
            tape_data = json.loads(data.as_text())

            # Handle multi-tape format
            if "tapes" not in tape_data:
                # Single tape, wrap in list
                return [self.load(data)]

            tapes_json = tape_data.get("tapes", [])
            if not tapes_json:
                raise LoaderError("No tapes found in data")

            loaded_circuits = []
            for i, tape_dict in enumerate(tapes_json):
                # Create CircuitData for single tape
                single_data = CircuitData(
                    data=json.dumps(tape_dict),
                    format=CircuitFormat.TAPE_JSON,
                    sdk=SDK.PENNYLANE,
                    name=f"{data.name}_{i}",
                    index=i,
                )
                loaded_circuits.append(self.load(single_data))

            return loaded_circuits

        except Exception as e:
            raise LoaderError(f"Batch tape load failed: {e}") from e


class PennyLaneCircuitSerializer:
    """
    PennyLane circuit serializer.

    Serializes circuits to PennyLane tape JSON format.

    Attributes
    ----------
    name : str
        Serializer identifier.
    sdk : SDK
        Source SDK (PENNYLANE).
    supported_formats : list of CircuitFormat
        Formats this serializer can produce.
    default_format : CircuitFormat
        Default serialization format (TAPE_JSON).
    """

    name = "pennylane"
    default_format = CircuitFormat.TAPE_JSON

    @property
    def sdk(self) -> SDK:
        """Get the SDK this serializer handles."""
        return SDK.PENNYLANE

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        return [CircuitFormat.TAPE_JSON]

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
            True if circuit is a PennyLane tape.
        """
        return is_pennylane_tape(circuit)

    def serialize(
        self,
        circuit: Any,
        fmt: CircuitFormat | None = None,
        *,
        name: str = "",
        index: int = 0,
    ) -> CircuitData:
        """
        Serialize circuit to specified format.

        Parameters
        ----------
        circuit : QuantumTape or list of QuantumTape
            PennyLane tape(s).
        fmt : CircuitFormat, optional
            Target format. Defaults to TAPE_JSON.
        name : str, optional
            Circuit name.
        index : int, optional
            Circuit index.

        Returns
        -------
        CircuitData
            Serialized circuit data.

        Raises
        ------
        SerializerError
            If format is unsupported or serialization fails.
        """
        if fmt is None:
            fmt = self.default_format

        if fmt != CircuitFormat.TAPE_JSON:
            raise SerializerError(f"Unsupported format: {fmt}")

        if is_pennylane_tape(circuit):
            return serialize_tape(circuit, name=name, index=index)

        # List of tapes
        tapes = list(circuit)
        return serialize_tapes(tapes)

    def serialize_circuit(
        self,
        circuit: Any,
        *,
        name: str = "",
        index: int = 0,
    ) -> CircuitData:
        """
        Serialize a single circuit/tape using default format.

        Parameters
        ----------
        circuit : QuantumTape
            PennyLane tape.
        name : str, optional
            Circuit name.
        index : int, optional
            Circuit index.

        Returns
        -------
        CircuitData
            Serialized circuit data.
        """
        return self.serialize(
            circuit,
            self.default_format,
            name=name,
            index=index,
        )

    def serialize_batch(self, tapes: list[Any]) -> CircuitData:
        """
        Serialize multiple tapes using default format.

        Parameters
        ----------
        tapes : list of QuantumTape
            PennyLane tapes.

        Returns
        -------
        CircuitData
            Serialized tapes data.
        """
        return serialize_tapes(tapes)


def summarize_pennylane_tape(tape: Any) -> CircuitSummary:
    """
    Summarize a PennyLane tape.

    Parameters
    ----------
    tape : QuantumTape
        PennyLane tape to summarize.

    Returns
    -------
    CircuitSummary
        Circuit summary with gate counts and statistics.
    """
    gate_counts: Counter[str] = Counter()
    has_params = False
    param_count = 0

    for op in tape.operations:
        gate_name = op.name.lower()
        gate_counts[gate_name] += 1
        if op.parameters:
            has_params = True
            param_count += len(op.parameters)

    # Count measurements - use class name lowercased
    for m in tape.measurements:
        m_name = type(m).__name__.lower()
        gate_counts[m_name] += 1

    stats = _classifier.classify_counts(dict(gate_counts))

    # Get depth if available
    depth = 0
    try:
        if hasattr(tape, "graph"):
            depth = tape.graph.get_depth()
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to get depth: %s", e)

    return CircuitSummary(
        num_qubits=len(tape.wires),
        depth=depth,
        gate_count_1q=stats["gate_count_1q"],
        gate_count_2q=stats["gate_count_2q"],
        gate_count_multi=stats["gate_count_multi"],
        gate_count_measure=stats["gate_count_measure"],
        gate_count_total=sum(gate_counts.values()),
        gate_types=dict(gate_counts),
        has_parameters=has_params,
        parameter_count=param_count if has_params else 0,
        is_clifford=stats["is_clifford"],
        source_format=CircuitFormat.TAPE_JSON,
        sdk=SDK.PENNYLANE,
    )
