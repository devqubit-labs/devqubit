# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device handling for Braket adapter.

Creates structured DeviceSnapshot objects from Braket devices, capturing
identity, topology, native gates, and calibration data.

Notes
-----
Amazon Braket exposes device topology and native gate set through device properties.
In particular:
- device.topology_graph (networkx DiGraph) is constructed from properties.paradigm.connectivity
- device.properties.paradigm.nativeGateSet contains the native gate set (when available)
"""

from __future__ import annotations

import logging
from statistics import median
from typing import TYPE_CHECKING, Any

from devqubit_braket.utils import (
    braket_version,
    get_backend_name,
    get_nested,
    obj_to_dict,
    to_float,
)
from devqubit_engine.uec.models.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.utils.common import utc_now_iso


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)


# =============================================================================
# Calibration Extraction
# =============================================================================


def _parse_qubit_key(qubits_key: str) -> list[int]:
    """
    Parse a qubit key string to a list of qubit indices.

    Handles formats like "0-1", "(0,1)", "[0,1]", "0,1".

    Parameters
    ----------
    qubits_key : str
        Qubit key string.

    Returns
    -------
    list of int
        Parsed qubit indices.
    """
    try:
        cleaned = qubits_key.translate(str.maketrans("()-[]", ",,,,,", " "))
        return [int(t) for t in cleaned.split(",") if t]
    except (ValueError, TypeError):
        return []


def _extract_fidelity_entries(
    container: Any,
    candidate_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """
    Extract a list of fidelity entries from a container.

    Parameters
    ----------
    container : Any
        Dict-like or object-like container.
    candidate_keys : tuple of str
        Candidate keys that may store a list of fidelities.

    Returns
    -------
    list of dict
        List entries, each expected to contain gateName and fidelity.
    """
    d = obj_to_dict(container) or {}
    if not isinstance(d, dict):
        return []

    for k in candidate_keys:
        v = d.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    return []


def _gate_name_and_fidelity(entry: dict[str, Any]) -> tuple[str | None, float | None]:
    """Extract (gate_name, fidelity) from a fidelity entry dict."""
    gname = entry.get("gateName") or entry.get("gate_name") or entry.get("name")
    fidelity = to_float(entry.get("fidelity"))
    return (str(gname) if gname else None), fidelity


def extract_calibration(device: Any) -> DeviceCalibration | None:
    """
    Extract DeviceCalibration from Braket device properties.

    Parameters
    ----------
    device : Any
        Braket device instance (AwsDevice, LocalSimulator, etc.)

    Returns
    -------
    DeviceCalibration or None
        Calibration bundle when standardized calibration metrics are found.

    Notes
    -----
    Prefers standardized properties (gate fidelities). For each fidelity entry,
    this maps to a devqubit GateCalibration with:

        error = 1 - fidelity

    This keeps your schema consistent with "lower is better" error-style metrics.
    """
    try:
        props_obj = getattr(device, "properties", None)
    except (AttributeError, TypeError):
        return None

    props = obj_to_dict(props_obj)
    if not isinstance(props, dict) or not props:
        return None

    cal_time = get_nested(props, ("service", "updatedAt"))
    cal_time = str(cal_time) if cal_time else utc_now_iso()

    std = obj_to_dict(get_nested(props, ("standardized",)))
    if not isinstance(std, dict) or not std:
        return None

    oneq_props = std.get("oneQubitProperties")
    twoq_props = std.get("twoQubitProperties")

    gates: list[GateCalibration] = []
    qubit_errors: dict[int, list[float]] = {}

    # 1Q fidelities -> GateCalibration + per-qubit aggregation
    if isinstance(oneq_props, dict):
        for q_key, q_entry in oneq_props.items():
            try:
                q = int(q_key)
            except (ValueError, TypeError):
                continue

            entries = _extract_fidelity_entries(
                q_entry,
                candidate_keys=("oneQubitGateFidelity", "one_qubit_gate_fidelity"),
            )
            for e in entries:
                gname, fidelity = _gate_name_and_fidelity(e)
                if not gname or fidelity is None:
                    continue
                err = max(0.0, 1.0 - float(fidelity))
                gates.append(GateCalibration(gate=gname, qubits=(q,), error=err))
                qubit_errors.setdefault(q, []).append(err)

    # 2Q fidelities -> GateCalibration
    if isinstance(twoq_props, dict):
        for edge_key, edge_entry in twoq_props.items():
            if not isinstance(edge_key, str):
                continue
            qubits = _parse_qubit_key(edge_key)
            if len(qubits) != 2:
                continue
            qpair = tuple(qubits)

            entries = _extract_fidelity_entries(
                edge_entry,
                candidate_keys=("twoQubitGateFidelity", "two_qubit_gate_fidelity"),
            )
            for e in entries:
                gname, fidelity = _gate_name_and_fidelity(e)
                if not gname or fidelity is None:
                    continue
                err = max(0.0, 1.0 - float(fidelity))
                gates.append(GateCalibration(gate=gname, qubits=qpair, error=err))

    if not gates:
        return None

    # Build QubitCalibration records with derived 1Q error medians
    qubits_out: list[QubitCalibration] = []
    for q, errs in sorted(qubit_errors.items()):
        if errs:
            try:
                qubits_out.append(
                    QubitCalibration(qubit=q, gate_error_1q=float(median(errs)))
                )
            except (ValueError, TypeError):
                qubits_out.append(QubitCalibration(qubit=q))

    cal = DeviceCalibration(
        calibration_time=cal_time,
        qubits=qubits_out,
        gates=gates,
    )
    cal.compute_medians()
    return cal


# =============================================================================
# Device Snapshot Creation
# =============================================================================


def _detect_physical_provider(device: Any) -> str:
    """
    Detect physical provider from Braket device.

    UEC requires provider to be the physical backend provider,
    not the SDK name. SDK goes in producer.frontends[].

    Parameters
    ----------
    device : Any
        Braket device instance.

    Returns
    -------
    str
        Physical provider: "aws_braket" for AWS devices, "local" for local simulators.
    """
    class_name = device.__class__.__name__.lower()
    module_name = getattr(device, "__module__", "").lower()

    if "local" in class_name or "localsimulator" in class_name:
        return "local"

    if "local" in module_name:
        return "local"

    try:
        arn = getattr(device, "arn", None)
        if arn:
            arn_str = str(arn() if callable(arn) else arn).lower()
            if "aws" in arn_str or "braket" in arn_str:
                return "aws_braket"
    except (AttributeError, TypeError):
        pass

    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            device_type_str = str(device_type).lower()
            if "simulator" in device_type_str and "aws" not in str(module_name):
                return "local"
    except (AttributeError, TypeError):
        pass

    return "aws_braket"


def _resolve_backend_type(device: Any) -> str:
    """
    Resolve backend_type to a schema-valid value.

    Parameters
    ----------
    device : Any
        Braket device instance.

    Returns
    -------
    str
        One of: "simulator", "hardware".
    """
    class_name = device.__class__.__name__.lower()

    if any(s in class_name for s in ("simulator", "sim", "local")):
        return "simulator"

    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            device_type_str = str(device_type).lower()
            if "simulator" in device_type_str:
                return "simulator"
            if "qpu" in device_type_str:
                return "hardware"
    except (AttributeError, TypeError):
        pass

    try:
        arn = getattr(device, "arn", None)
        if arn and "simulator" in str(arn).lower():
            return "simulator"
    except (AttributeError, TypeError):
        pass

    return "hardware" if "awsdevice" in class_name else "simulator"


def _extract_native_gates(
    device: Any,
    props_dict: dict[str, Any] | None,
) -> list[str] | None:
    """
    Extract native gates supported by the device (best-effort).

    Preference order:
    1) device.properties.paradigm.nativeGateSet
    2) properties dict: ["paradigm"]["nativeGateSet"]
    3) properties dict: ["action"]["braket.ir.openqasm.program"]["supportedOperations"]

    Parameters
    ----------
    device : Any
        Braket device instance.
    props_dict : dict or None
        JSONable device properties dict (if available).

    Returns
    -------
    list of str or None
        Native gate names (or supported operations fallback), if found.
    """
    try:
        props_obj = getattr(device, "properties", None)
        ng = get_nested(props_obj, ("paradigm", "nativeGateSet"))
        if isinstance(ng, list) and ng:
            return [str(x) for x in ng]
    except (AttributeError, TypeError):
        pass

    if isinstance(props_dict, dict):
        ng = get_nested(props_dict, ("paradigm", "nativeGateSet"))
        if isinstance(ng, list) and ng:
            return [str(x) for x in ng]

        ops = get_nested(
            props_dict,
            ("action", "braket.ir.openqasm.program", "supportedOperations"),
        )
        if isinstance(ops, list) and ops:
            return [str(x) for x in ops]

    return None


def _extract_topology(
    device: Any,
    props_dict: dict[str, Any] | None,
) -> tuple[int | None, list[tuple[int, int]] | None]:
    """
    Extract qubit topology from device.

    Preference order:
    1) device.topology_graph (networkx DiGraph)
    2) properties.paradigm.connectivity.connectivityGraph (+ fullyConnected)

    Parameters
    ----------
    device : Any
        Braket device instance.
    props_dict : dict or None
        JSONable properties dict if available.

    Returns
    -------
    tuple
        (num_qubits, connectivity) - either may be None.
    """
    num_qubits: int | None = None
    connectivity: list[tuple[int, int]] | None = None

    # 1) Preferred: topology_graph if implemented
    try:
        tg = getattr(device, "topology_graph", None)
        if tg is not None:
            nodes = list(getattr(tg, "nodes", []))
            if nodes:
                num_qubits = len(nodes)

            edges = list(getattr(tg, "edges", []))
            if edges:
                conn = [(int(u), int(v)) for u, v in edges]
                if conn:
                    connectivity = conn

            if num_qubits is not None or connectivity is not None:
                return num_qubits, connectivity
    except (AttributeError, TypeError, ValueError):
        pass

    # 2) Fallback: parse properties
    try:
        props_obj = getattr(device, "properties", None)
    except (AttributeError, TypeError):
        props_obj = None

    qc = get_nested(props_obj, ("paradigm", "qubitCount"))
    if qc is None and isinstance(props_dict, dict):
        qc = get_nested(props_dict, ("paradigm", "qubitCount"))
    if qc is not None:
        try:
            num_qubits = int(qc)
        except (ValueError, TypeError):
            pass

    # fullyConnected flag - if true, don't expand edges
    fully = get_nested(props_obj, ("paradigm", "connectivity", "fullyConnected"))
    if fully is None and isinstance(props_dict, dict):
        fully = get_nested(props_dict, ("paradigm", "connectivity", "fullyConnected"))
    if fully:
        return num_qubits, None

    # connectivityGraph dict-of-lists
    cg = get_nested(props_obj, ("paradigm", "connectivity", "connectivityGraph"))
    if cg is None and isinstance(props_dict, dict):
        cg = get_nested(props_dict, ("paradigm", "connectivity", "connectivityGraph"))

    if isinstance(cg, dict) and cg:
        edge_set: set[tuple[int, int]] = set()
        for u, nbrs in cg.items():
            try:
                ui = int(u)
            except (ValueError, TypeError):
                continue
            if isinstance(nbrs, list):
                for v in nbrs:
                    try:
                        edge_set.add((ui, int(v)))
                    except (ValueError, TypeError):
                        continue
        if edge_set:
            connectivity = sorted(edge_set)

    return num_qubits, connectivity


def _extract_backend_id(device: Any) -> str | None:
    """
    Extract stable backend identifier (ARN for AWS devices).

    Parameters
    ----------
    device : Any
        Braket device instance.

    Returns
    -------
    str or None
        ARN or other stable identifier.
    """
    try:
        arn = getattr(device, "arn", None)
        if arn:
            return str(arn() if callable(arn) else arn)
    except (AttributeError, TypeError):
        pass
    return None


def _build_raw_properties(
    device: Any,
    props_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build raw_properties dictionary for artifact logging.

    Parameters
    ----------
    device : Any
        Braket device instance.
    props_dict : dict or None
        JSONable device properties dict.

    Returns
    -------
    dict
        Raw properties for lossless capture.
    """
    raw_properties: dict[str, Any] = {
        "device_class": device.__class__.__name__,
        "device_module": getattr(device, "__module__", ""),
    }

    arn = _extract_backend_id(device)
    if arn:
        raw_properties["arn"] = arn

    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            raw_properties["device_type"] = str(device_type)
    except (AttributeError, TypeError):
        pass

    try:
        if arn and ":" in arn:
            parts = arn.split("/")
            if len(parts) >= 2:
                raw_properties["provider_name"] = parts[1]
    except (AttributeError, TypeError, IndexError):
        pass

    try:
        status = getattr(device, "status", None)
        if status is not None:
            raw_properties["status"] = str(status)
    except (AttributeError, TypeError):
        pass

    if props_dict:
        raw_properties["properties"] = props_dict

    return raw_properties


def create_device_snapshot(
    device: Any,
    *,
    tracker: Run | None = None,
) -> DeviceSnapshot:
    """
    Create a DeviceSnapshot from a Braket device.

    Captures device properties, topology, native gates, and calibration data.

    Parameters
    ----------
    device : Any
        Braket device instance (LocalSimulator or AwsDevice).
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.
        If provided, raw properties are logged and referenced via ``raw_properties_ref``.

    Returns
    -------
    DeviceSnapshot
        Structured device snapshot with calibration data.

    Raises
    ------
    ValueError
        If device is None.

    Notes
    -----
    When ``tracker`` is provided, raw device properties are logged as a separate
    artifact for lossless capture. This includes the full device properties dict,
    ARN, status, and other provider-specific metadata.
    """
    if device is None:
        raise ValueError("Cannot create device snapshot from None device")

    captured_at = utc_now_iso()
    backend_name = get_backend_name(device)
    backend_type = _resolve_backend_type(device)
    backend_id = _extract_backend_id(device)
    sdk_version = braket_version()

    try:
        props_obj = getattr(device, "properties", None)
        props_dict = obj_to_dict(props_obj) if props_obj else None
    except Exception as e:
        logger.debug("Failed to get device properties: %s", e)
        props_dict = None

    try:
        num_qubits, connectivity = _extract_topology(device, props_dict)
    except Exception as e:
        logger.debug("Failed to extract topology: %s", e)
        num_qubits, connectivity = None, None

    try:
        native_gates = _extract_native_gates(device, props_dict)
    except Exception as e:
        logger.debug("Failed to extract native gates: %s", e)
        native_gates = None

    try:
        calibration = extract_calibration(device)
    except Exception as e:
        logger.debug("Failed to extract calibration: %s", e)
        calibration = None

    raw_properties_ref = None
    if tracker is not None:
        raw_properties = _build_raw_properties(device, props_dict)
        try:
            raw_properties_ref = tracker.log_json(
                name="device_raw_properties",
                obj=raw_properties,
                role="device_raw",
                kind="device.braket.raw_properties.json",
            )
            logger.debug("Logged raw Braket device properties artifact")
        except Exception as e:
            logger.warning("Failed to log raw_properties artifact: %s", e)

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=_detect_physical_provider(device),
        backend_id=backend_id,
        num_qubits=num_qubits,
        connectivity=connectivity,
        native_gates=native_gates,
        calibration=calibration,
        sdk_versions={"braket": sdk_version},
        raw_properties_ref=raw_properties_ref,
    )
