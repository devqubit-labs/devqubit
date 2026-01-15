# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot creation for Qiskit BackendV2-compatible backends.

Provides Qiskit-specific extraction functions for creating DeviceSnapshot
instances from Qiskit backends (BackendV2, AerSimulator).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.models.calibration import DeviceCalibration
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.calibration import extract_calibration_from_properties
from devqubit_qiskit.utils import get_backend_name, qiskit_version


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)


def _extract_connectivity_from_coupling_map(
    coupling_map: Any,
) -> list[tuple[int, int]] | None:
    """
    Extract connectivity from a CouplingMap-like object.

    Parameters
    ----------
    coupling_map : Any
        Object with ``get_edges()`` method (e.g., Qiskit CouplingMap).

    Returns
    -------
    list[tuple[int, int]] | None
        List of directed (qubit_i, qubit_j) edges, or None if extraction fails.

    Notes
    -----
    Qiskit CouplingMap is directed: edge direction usually corresponds to
    control->target for entangling ops (e.g., CX).
    """
    if coupling_map is None or not hasattr(coupling_map, "get_edges"):
        return None

    try:
        edges = coupling_map.get_edges()
        return [(int(e[0]), int(e[1])) for e in edges]
    except Exception:
        return None


def _extract_from_target(
    target: Any,
) -> tuple[int | None, list[tuple[int, int]] | None, list[str] | None, dict[str, Any]]:
    """
    Extract device information from a Qiskit Target object.

    Parameters
    ----------
    target : Any
        Qiskit Target instance (typically from BackendV2.target).

    Returns
    -------
    num_qubits : int | None
        Number of qubits.
    connectivity : list[tuple[int, int]] | None
        Directed edge list of connected qubit pairs.
    native_gates : list[str] | None
        Sorted list of native operation names (excluding measure/reset/delay).
    raw : dict[str, Any]
        Raw target information for artifact storage.
    """
    if target is None:
        return None, None, None, {}

    raw: dict[str, Any] = {}
    num_qubits: int | None = None
    connectivity: list[tuple[int, int]] | None = None
    native_gates: list[str] | None = None

    # num_qubits (Target has this in modern Qiskit)
    try:
        if hasattr(target, "num_qubits") and target.num_qubits is not None:
            num_qubits = int(target.num_qubits)
            raw["num_qubits"] = num_qubits
    except Exception as e:
        logger.debug("Failed to extract num_qubits from target: %s", e)

    # Connectivity via Target.build_coupling_map()
    try:
        if hasattr(target, "build_coupling_map") and callable(
            target.build_coupling_map
        ):
            coupling_map = target.build_coupling_map()
            connectivity = _extract_connectivity_from_coupling_map(coupling_map)
            if connectivity:
                raw["connectivity"] = connectivity
    except Exception as e:
        logger.debug("Failed to build coupling map from target: %s", e)

    # Native operation names
    try:
        if hasattr(target, "operation_names"):
            ops = list(target.operation_names)
            raw["operation_names"] = ops
            native_gates = sorted(
                op for op in ops if op not in ("measure", "reset", "delay")
            )
    except Exception as e:
        logger.debug("Failed to extract operation_names from target: %s", e)

    return num_qubits, connectivity, native_gates, raw


def _call_backend_properties(
    backend: Any,
    *,
    refresh: bool,
) -> Any | None:
    """
    Call backend.properties() in a signature-tolerant way.

    Some providers expose:
      - properties()
      - properties(refresh=False)
      - properties(refresh=False, datetime=None)

    Parameters
    ----------
    backend : Any
        BackendV2-compatible backend instance.
    refresh : bool
        Whether to request a refresh (when supported).

    Returns
    -------
    Any | None
        BackendProperties-like object, or None if unavailable.
    """
    if not hasattr(backend, "properties") or not callable(backend.properties):
        return None

    # Try refresh-only signature (fake providers, some simulators).
    try:
        return backend.properties(refresh=refresh)
    except TypeError:
        pass

    # Last resort: no-arg call.
    try:
        return backend.properties()
    except Exception:
        return None


def _extract_calibration(
    backend: Any,
    *,
    refresh_properties: bool,
) -> tuple[DeviceCalibration | None, dict[str, Any]]:
    """
    Extract calibration data from backend.properties().

    Parameters
    ----------
    backend : Any
        BackendV2-compatible backend instance.
    refresh_properties : bool
        If True, re-query provider for latest properties (when supported).

    Returns
    -------
    calibration : DeviceCalibration | None
        Extracted calibration data.
    raw : dict[str, Any]
        Raw properties payload and selected metadata for artifact storage.
    """
    props = _call_backend_properties(
        backend,
        refresh=refresh_properties,
    )
    if props is None:
        return None, {}

    try:
        props_dict = props.to_dict() if hasattr(props, "to_dict") else {}
    except Exception:
        props_dict = {}

    # Capture last_update_date explicitly (useful even if to_dict drops/changes format).
    last_update_date = getattr(props, "last_update_date", None)

    calibration: DeviceCalibration | None = None
    if extract_calibration_from_properties is not None:
        try:
            calibration = extract_calibration_from_properties(
                props_dict,
                source="provider",
            )
        except Exception as e:
            logger.debug("Failed to parse calibration from properties dict: %s", e)
    else:
        logger.debug(
            "Calibration extraction not available: "
            "extract_calibration_from_properties not imported"
        )

    raw: dict[str, Any] = {}
    if props_dict:
        raw["properties"] = props_dict
    if last_update_date is not None:
        # Store as-is; downstream can normalize to ISO if needed.
        raw["properties_last_update_date"] = last_update_date

    return calibration, raw


def _detect_backend_type(backend: Any) -> str:
    """
    Detect whether a backend is a simulator or hardware.

    Parameters
    ----------
    backend : Any
        BackendV2-compatible backend instance.

    Returns
    -------
    str
        Backend type: "simulator" or "hardware".
    """
    backend_name = get_backend_name(backend).lower()
    class_name = type(backend).__name__.lower()
    module_name = type(backend).__module__.lower()

    # Prefer explicit "options.simulator" flag when present (BackendV2-style).
    try:
        opts = getattr(backend, "options", None)
        if opts is not None and hasattr(opts, "simulator") and bool(opts.simulator):
            return "simulator"
    except Exception:
        pass

    # Name-based heuristics
    simulator_indicators = (
        "sim",
        "simulator",
        "fake",
        "aer",
        "statevector",
        "unitary",
        "qasm",
        "density_matrix",
        "stabilizer",
    )

    if any(ind in backend_name for ind in simulator_indicators):
        return "simulator"
    if any(ind in class_name for ind in simulator_indicators):
        return "simulator"
    if any(ind in module_name for ind in ("aer", "fake", "simulator")):
        return "simulator"

    # Hardware indicators
    if any(ind in backend_name for ind in ("ibm_", "ibmq_", "ionq", "rigetti", "aqt")):
        return "hardware"

    # Conservative default: unknown backends are treated as simulator unless identified.
    return "simulator"


def _detect_provider(backend: Any) -> str:
    """
    Detect the physical provider for a Qiskit backend.

    This returns the physical backend provider (ibm_quantum, aer, etc.),
    not the SDK. The SDK (qiskit) goes in producer.frontends[].

    Parameters
    ----------
    backend : Any
        BackendV2-compatible backend instance.

    Returns
    -------
    str
        Provider identifier: "ibm_quantum", "aer", "fake", or "local".
    """
    module_name = type(backend).__module__.lower()
    backend_name = get_backend_name(backend).lower()

    if "ibm" in module_name or "ibm_" in backend_name:
        return "ibm_quantum"
    if "qiskit_aer" in module_name or "aer" in module_name:
        return "aer"
    if "fake" in module_name:
        return "fake"

    return "local"


def _get_sdk_versions() -> dict[str, str]:
    """
    Collect SDK version information.

    Returns
    -------
    dict[str, str]
        SDK name to version mapping.
    """
    sdk_versions: dict[str, str] = {"qiskit": qiskit_version()}

    try:
        import qiskit_aer

        sdk_versions["qiskit_aer"] = getattr(qiskit_aer, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import qiskit_ibm_runtime

        sdk_versions["qiskit_ibm_runtime"] = getattr(
            qiskit_ibm_runtime, "__version__", "unknown"
        )
    except ImportError:
        pass

    return sdk_versions


def create_device_snapshot(
    backend: Any,
    *,
    refresh_properties: bool = False,
    tracker: Run | None = None,
) -> DeviceSnapshot:
    """
    Create a DeviceSnapshot from a Qiskit BackendV2-compatible backend.

    Parameters
    ----------
    backend : Any
        Qiskit BackendV2-compatible backend instance.
    refresh_properties : bool, optional
        If True, re-query the provider for backend properties instead of using cache
        (when supported). Default is False.
    tracker : Run, optional
        Tracker instance for logging raw properties as artifact.
        If provided, raw properties are logged and referenced via ``raw_properties_ref``.

    Returns
    -------
    DeviceSnapshot
        Complete device snapshot with optional raw_properties_ref.

    Raises
    ------
    ValueError
        If backend is None.
    """
    if backend is None:
        raise ValueError("Cannot create device snapshot from None backend")

    captured_at = utc_now_iso()
    backend_name = get_backend_name(backend)
    backend_type = _detect_backend_type(backend)
    provider = _detect_provider(backend)

    # Extract from Target (BackendV2 canonical source)
    target = getattr(backend, "target", None)
    num_qubits, connectivity, native_gates, target_raw = _extract_from_target(target)

    # BackendV2 fallback for num_qubits if Target didn't provide it
    if num_qubits is None:
        try:
            if hasattr(backend, "num_qubits") and backend.num_qubits is not None:
                num_qubits = int(backend.num_qubits)
        except Exception as e:
            logger.debug("Failed to get num_qubits from backend: %s", e)

    # BackendV2 fallback for native gates if Target didn't provide it
    if native_gates is None:
        try:
            if hasattr(backend, "operation_names"):
                ops = list(backend.operation_names)
                native_gates = sorted(
                    op for op in ops if op not in ("measure", "reset", "delay")
                )
        except Exception as e:
            logger.debug("Failed to get operation_names from backend: %s", e)

    # Extract calibration from backend.properties()
    calibration, calibration_raw = _extract_calibration(
        backend,
        refresh_properties=refresh_properties,
    )

    # SDK versions
    sdk_versions = _get_sdk_versions()

    # Build raw properties for artifact logging (lossless capture)
    raw_properties: dict[str, Any] = {
        "backend_class": type(backend).__name__,
        "backend_module": type(backend).__module__,
    }

    if target_raw:
        raw_properties["target"] = target_raw
    if calibration_raw:
        raw_properties.update(calibration_raw)

    # Log raw_properties as artifact if tracker is provided
    raw_properties_ref = None
    if tracker is not None and len(raw_properties) > 2:
        try:
            raw_properties_ref = tracker.log_json(
                name="backend_raw_properties",
                obj=to_jsonable(raw_properties),
                role="device_raw",
                kind="device.qiskit.raw_properties.json",
            )
            logger.debug("Logged raw backend properties artifact")
        except Exception as e:
            logger.warning("Failed to log raw properties artifact: %s", e)

    # Create snapshot
    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        num_qubits=num_qubits,
        connectivity=connectivity,
        native_gates=native_gates,
        calibration=calibration,
        sdk_versions=sdk_versions,
        raw_properties_ref=raw_properties_ref,
    )
