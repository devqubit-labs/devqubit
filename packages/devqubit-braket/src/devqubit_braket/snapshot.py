# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot creation for Braket devices.

Creates structured DeviceSnapshot objects from Braket devices, capturing:
- identity and raw properties
- topology (num_qubits + connectivity)
- native gates (best-effort)
- calibration (via devqubit_braket.calibration)

Notes
-----
Amazon Braket exposes device topology and native gate set through device properties.
In particular:
- device.topology_graph (networkx DiGraph) is constructed from properties.paradigm.connectivity
- device.properties.paradigm.nativeGateSet contains the native gate set (when available)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_braket.calibration import extract_calibration_from_device
from devqubit_braket.utils import (
    braket_version,
    get_backend_name,
    get_nested,
    obj_to_dict,
)
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.utils.common import utc_now_iso


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)


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

    # Local simulator
    if "local" in class_name or "localsimulator" in class_name:
        return "local"

    # Check module for local
    if "local" in module_name:
        return "local"

    # Check ARN for AWS device
    try:
        arn = getattr(device, "arn", None)
        if arn:
            arn_str = str(arn() if callable(arn) else arn).lower()
            if "aws" in arn_str or "braket" in arn_str:
                return "aws_braket"
    except Exception:
        pass

    # Check device type
    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            device_type_str = str(device_type).lower()
            if "simulator" in device_type_str and "aws" not in str(module_name):
                return "local"
    except Exception:
        pass

    # Default: AWS Braket (since most non-local devices are AWS)
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

    # Check class name for simulator indicators
    if any(s in class_name for s in ("simulator", "sim", "local")):
        return "simulator"

    # Check device type attribute if available
    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            device_type_str = str(device_type).lower()
            if "simulator" in device_type_str:
                return "simulator"
            if "qpu" in device_type_str:
                return "hardware"
    except Exception:
        pass

    # Check ARN for simulator pattern
    try:
        arn = getattr(device, "arn", None)
        if arn and "simulator" in str(arn).lower():
            return "simulator"
    except Exception:
        pass

    # Default to hardware for AwsDevice, simulator otherwise
    return "hardware" if "awsdevice" in class_name else "simulator"


def _extract_native_gates(
    device: Any, props_dict: dict[str, Any] | None
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
    # Try attribute path first
    try:
        props_obj = getattr(device, "properties", None)
        ng = get_nested(props_obj, ("paradigm", "nativeGateSet"))
        if isinstance(ng, list) and ng:
            return [str(x) for x in ng]
    except Exception:
        pass

    # Try dict paths
    if isinstance(props_dict, dict):
        ng = get_nested(props_dict, ("paradigm", "nativeGateSet"))
        if isinstance(ng, list) and ng:
            return [str(x) for x in ng]

        # Fallback: supported operations
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
    except Exception:
        pass

    # 2) Fallback: parse properties
    try:
        props_obj = getattr(device, "properties", None)
    except Exception:
        props_obj = None

    # num_qubits from qubitCount
    qc = get_nested(props_obj, ("paradigm", "qubitCount"))
    if qc is None and isinstance(props_dict, dict):
        qc = get_nested(props_dict, ("paradigm", "qubitCount"))
    if qc is not None:
        try:
            num_qubits = int(qc)
        except Exception:
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
            except Exception:
                continue
            if isinstance(nbrs, list):
                for v in nbrs:
                    try:
                        edge_set.add((ui, int(v)))
                    except Exception:
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
    except Exception:
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

    # ARN
    arn = _extract_backend_id(device)
    if arn:
        raw_properties["arn"] = arn

    # Device type
    try:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            raw_properties["device_type"] = str(device_type)
    except Exception:
        pass

    # Provider name (from ARN or properties)
    try:
        if arn and ":" in arn:
            # ARN format: arn:aws:braket:<region>::device/<provider>/<device_name>
            parts = arn.split("/")
            if len(parts) >= 2:
                raw_properties["provider_name"] = parts[1]
    except Exception:
        pass

    # Status (for AwsDevice)
    try:
        status = getattr(device, "status", None)
        if status is not None:
            raw_properties["status"] = str(status)
    except Exception:
        pass

    # Include full properties dict if available
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

    # Get properties dict for fallback lookups
    try:
        props_obj = getattr(device, "properties", None)
        props_dict = obj_to_dict(props_obj) if props_obj else None
    except Exception as e:
        logger.debug("Failed to get device properties: %s", e)
        props_dict = None

    # Topology
    try:
        num_qubits, connectivity = _extract_topology(device, props_dict)
    except Exception as e:
        logger.debug("Failed to extract topology: %s", e)
        num_qubits, connectivity = None, None

    # Native gates (best-effort)
    try:
        native_gates = _extract_native_gates(device, props_dict)
    except Exception as e:
        logger.debug("Failed to extract native gates: %s", e)
        native_gates = None

    # Calibration (from standardized fidelities, etc.)
    try:
        calibration = extract_calibration_from_device(device)
    except Exception as e:
        logger.debug("Failed to extract calibration: %s", e)
        calibration = None

    # Log raw_properties as artifact if tracker is provided
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
