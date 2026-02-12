# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot and calibration for Qiskit backends.

Provides Qiskit-specific extraction for creating DeviceSnapshot instances
with calibration data from BackendV2-compatible backends.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from statistics import median
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.models.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.utils import (
    as_int_tuple,
    convert_duration_to_ns,
    convert_freq_to_ghz,
    convert_time_to_us,
    get_backend_name,
    qiskit_version,
    to_float,
)


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run


logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


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
        If True, re-query the provider for backend properties instead of
        using cache (when supported). Default is False.
    tracker : Run, optional
        Tracker instance for logging raw properties as artifact.

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
    provider = detect_provider(backend)

    # Extract from Target (BackendV2 canonical source)
    target = getattr(backend, "target", None)
    num_qubits, connectivity, native_gates, target_raw = _extract_from_target(target)

    # Fallbacks for num_qubits and native_gates
    if num_qubits is None:
        num_qubits = _get_num_qubits_fallback(backend)
    if native_gates is None:
        native_gates = _get_native_gates_fallback(backend)

    # Extract calibration from backend.properties()
    calibration, calibration_raw = _extract_calibration(
        backend, refresh_properties=refresh_properties
    )

    # Build raw properties for artifact logging
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
        except Exception as e:
            logger.warning("Failed to log raw properties artifact: %s", e)

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        num_qubits=num_qubits,
        connectivity=connectivity,
        native_gates=native_gates,
        calibration=calibration,
        sdk_versions=_get_sdk_versions(),
        raw_properties_ref=raw_properties_ref,
    )


def detect_provider(backend: Any) -> str:
    """
    Detect the physical provider for a Qiskit backend.

    This returns the physical backend provider (ibm_quantum, aer, etc.),
    not the SDK. The SDK (qiskit) goes in producer.frontends[].

    Parameters
    ----------
    backend : Any
        Qiskit backend instance.

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


def extract_calibration(backend: Any) -> DeviceCalibration | None:
    """
    Extract calibration data from a Qiskit backend.

    Tries backend.properties() first, then Target if available.

    Parameters
    ----------
    backend : Any
        Qiskit backend instance.

    Returns
    -------
    DeviceCalibration or None
        Extracted calibration data, or None if unavailable.
    """
    calibration, _ = _extract_calibration(backend, refresh_properties=False)
    if calibration is not None:
        return calibration

    # Try Target as fallback
    target = getattr(backend, "target", None)
    if target is not None:
        return _extract_calibration_from_target(target)

    return None


def extract_calibration_from_properties(
    props: dict[str, Any],
    *,
    source: str = "provider",
) -> DeviceCalibration | None:
    """
    Extract DeviceCalibration from a Qiskit BackendProperties dict.

    Parameters
    ----------
    props : dict
        BackendProperties.to_dict() output or similar structure.
    source : str, optional
        Data source indicator for UEC compliance. Default is "provider".

    Returns
    -------
    DeviceCalibration or None
        Extracted calibration data, or None if no useful data exists.
    """
    if not isinstance(props, dict) or not props:
        return None

    # Extract calibration timestamp
    cal_time = _parse_calibration_timestamp(props)

    # Extract qubit and gate properties
    qubits_out = _extract_qubit_calibrations(props.get("qubits", []))
    gates_out = _extract_gate_calibrations(props.get("gates", []))

    # Check if we have any calibration data
    has_qubit_data = any(
        q.t1_us is not None
        or q.t2_us is not None
        or q.readout_error is not None
        or q.frequency_ghz is not None
        for q in qubits_out
    )
    has_gate_data = any(
        g.error is not None or g.duration_ns is not None for g in gates_out
    )

    if not has_qubit_data and not has_gate_data:
        return None

    # Derive per-qubit 1Q gate errors
    if qubits_out and gates_out:
        qubits_out = _derive_1q_gate_errors(qubits_out, gates_out)

    calibration = DeviceCalibration(
        calibration_time=cal_time,
        qubits=qubits_out,
        gates=gates_out,
        source=source,
    )
    calibration.compute_medians()

    return calibration


# =============================================================================
# Internal: Device Info Extraction
# =============================================================================


def _detect_backend_type(backend: Any) -> str:
    """Detect whether a backend is a simulator or hardware."""
    backend_name = get_backend_name(backend).lower()
    class_name = type(backend).__name__.lower()
    module_name = type(backend).__module__.lower()

    # Check explicit simulator flag
    try:
        opts = getattr(backend, "options", None)
        if opts is not None and hasattr(opts, "simulator") and bool(opts.simulator):
            return "simulator"
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to check simulator flag: %s", e)

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

    return "simulator"


def _extract_from_target(
    target: Any,
) -> tuple[int | None, list[tuple[int, int]] | None, list[str] | None, dict[str, Any]]:
    """Extract device info from a Qiskit Target object."""
    if target is None:
        return None, None, None, {}

    raw: dict[str, Any] = {}
    num_qubits: int | None = None
    connectivity: list[tuple[int, int]] | None = None
    native_gates: list[str] | None = None

    # num_qubits
    try:
        if hasattr(target, "num_qubits") and target.num_qubits is not None:
            num_qubits = int(target.num_qubits)
            raw["num_qubits"] = num_qubits
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to get num_qubits: %s", e)

    # Connectivity via Target.build_coupling_map()
    try:
        if hasattr(target, "build_coupling_map") and callable(
            target.build_coupling_map
        ):
            coupling_map = target.build_coupling_map()
            connectivity = _extract_connectivity_from_coupling_map(coupling_map)
            if connectivity:
                raw["connectivity"] = connectivity
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to build coupling map: %s", e)

    # Native operation names
    try:
        if hasattr(target, "operation_names"):
            ops = list(target.operation_names)
            raw["operation_names"] = ops
            native_gates = sorted(
                op for op in ops if op not in ("measure", "reset", "delay")
            )
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to get operation_names: %s", e)

    return num_qubits, connectivity, native_gates, raw


def _extract_connectivity_from_coupling_map(
    coupling_map: Any,
) -> list[tuple[int, int]] | None:
    """Extract connectivity from a CouplingMap-like object."""
    if coupling_map is None or not hasattr(coupling_map, "get_edges"):
        return None
    try:
        edges = coupling_map.get_edges()
        return [(int(e[0]), int(e[1])) for e in edges]
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to extract connectivity: %s", e)
        return None


def _get_num_qubits_fallback(backend: Any) -> int | None:
    """Get num_qubits from backend directly."""
    try:
        if hasattr(backend, "num_qubits") and backend.num_qubits is not None:
            return int(backend.num_qubits)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to get backend num_qubits: %s", e)
    return None


def _get_native_gates_fallback(backend: Any) -> list[str] | None:
    """Get native gates from backend directly."""
    try:
        if hasattr(backend, "operation_names"):
            ops = list(backend.operation_names)
            return sorted(op for op in ops if op not in ("measure", "reset", "delay"))
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to get operation_names: %s", e)
    return None


def _get_sdk_versions() -> dict[str, str]:
    """Collect SDK version information."""
    sdk_versions: dict[str, str] = {"qiskit": qiskit_version()}

    try:
        import qiskit_aer

        sdk_versions["qiskit_aer"] = getattr(qiskit_aer, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    try:
        import qiskit_ibm_runtime

        sdk_versions["qiskit_ibm_runtime"] = getattr(
            qiskit_ibm_runtime, "__version__", "unknown"
        )
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    return sdk_versions


# =============================================================================
# Internal: Calibration Extraction
# =============================================================================


def _extract_calibration(
    backend: Any,
    *,
    refresh_properties: bool,
) -> tuple[DeviceCalibration | None, dict[str, Any]]:
    """Extract calibration data from backend.properties()."""
    props = _call_backend_properties(backend, refresh=refresh_properties)
    if props is None:
        return None, {}

    try:
        props_dict = props.to_dict() if hasattr(props, "to_dict") else {}
    except (TypeError, ValueError, AttributeError) as e:
        logger.debug("Failed to convert properties to dict: %s", e)
        props_dict = {}

    last_update_date = getattr(props, "last_update_date", None)

    calibration: DeviceCalibration | None = None
    if props_dict:
        try:
            calibration = extract_calibration_from_properties(
                props_dict, source="provider"
            )
        except Exception as e:
            logger.debug("Failed to parse calibration from properties: %s", e)

    raw: dict[str, Any] = {}
    if props_dict:
        raw["properties"] = props_dict
    if last_update_date is not None:
        raw["properties_last_update_date"] = last_update_date

    return calibration, raw


def _call_backend_properties(backend: Any, *, refresh: bool) -> Any | None:
    """Call backend.properties() in a signature-tolerant way."""
    if not hasattr(backend, "properties") or not callable(backend.properties):
        return None

    try:
        return backend.properties(refresh=refresh)
    except TypeError as e:
        logger.debug("Suppressed error: %s", e)

    try:
        return backend.properties()
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to get backend properties: %s", e)
        return None


def _extract_calibration_from_target(
    target: Any,
    *,
    source: str = "derived",
) -> DeviceCalibration | None:
    """Extract calibration data from a Qiskit Target object."""
    if target is None:
        return None

    gates_out: list[GateCalibration] = []

    try:
        for op_name in target.operation_names:
            for qargs in target.qargs_for_operation_name(op_name):
                try:
                    inst_props = target[op_name][qargs]
                    if inst_props is None:
                        continue

                    error = getattr(inst_props, "error", None)
                    duration = getattr(inst_props, "duration", None)

                    if error is None and duration is None:
                        continue

                    duration_ns = (
                        float(duration) * 1e9 if duration is not None else None
                    )

                    gates_out.append(
                        GateCalibration(
                            gate=op_name,
                            qubits=tuple(qargs),
                            error=float(error) if error is not None else None,
                            duration_ns=duration_ns,
                        )
                    )
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to extract gate metric: %s", e)
                    continue
    except (TypeError, ValueError) as e:
        logger.debug("Failed to extract gate metric: %s", e)
        return None

    if not gates_out:
        return None

    calibration = DeviceCalibration(
        calibration_time=utc_now_iso(),
        gates=gates_out,
        source=source,
    )
    calibration.compute_medians()

    return calibration


def _parse_calibration_timestamp(props: dict[str, Any]) -> str:
    """Parse calibration timestamp from properties dict."""
    for key in ("last_update_date", "last_update_datetime", "last_update"):
        val = props.get(key)
        if val is not None:
            if isinstance(val, datetime):
                try:
                    return val.isoformat()
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to format datetime: %s", e)
                    return str(val)
            s = str(val).strip()
            if s:
                return s
    return utc_now_iso()


def _extract_qubit_calibrations(qubits: list[Any]) -> list[QubitCalibration]:
    """Extract per-qubit calibration data from properties."""
    qubits_out: list[QubitCalibration] = []

    for q_idx, qprops in enumerate(qubits):
        if not isinstance(qprops, list):
            continue

        t1_us = None
        t2_us = None
        readout_error = None
        p01 = None
        p10 = None
        freq_ghz = None
        anharm_ghz = None

        for p in qprops:
            if not isinstance(p, dict):
                continue

            name = str(p.get("name") or p.get("parameter") or "").strip().lower()
            val = to_float(p.get("value"))
            unit = p.get("unit")

            if not name or val is None:
                continue

            if name == "t1":
                t1_us = convert_time_to_us(val, str(unit) if unit else None)
            elif name == "t2":
                t2_us = convert_time_to_us(val, str(unit) if unit else None)
            elif name == "readout_error":
                readout_error = val
            elif name == "prob_meas0_prep1":
                p01 = val
            elif name == "prob_meas1_prep0":
                p10 = val
            elif name == "frequency":
                freq_ghz = convert_freq_to_ghz(val, str(unit) if unit else None)
            elif name == "anharmonicity":
                anharm_ghz = convert_freq_to_ghz(val, str(unit) if unit else None)

        # Approximate readout_error from assignment probabilities if missing
        if readout_error is None:
            vals = [x for x in (p01, p10) if x is not None]
            if vals:
                readout_error = sum(vals) / float(len(vals))

        qubits_out.append(
            QubitCalibration(
                qubit=int(q_idx),
                t1_us=t1_us,
                t2_us=t2_us,
                readout_error=readout_error,
                gate_error_1q=None,
                frequency_ghz=freq_ghz,
                anharmonicity_ghz=anharm_ghz,
            )
        )

    return qubits_out


def _extract_gate_calibrations(gates: list[Any]) -> list[GateCalibration]:
    """Extract per-gate calibration data from properties."""
    gates_out: list[GateCalibration] = []

    for g in gates:
        if not isinstance(g, dict):
            continue

        gname = str(g.get("gate") or g.get("name") or "").strip()
        if not gname:
            continue

        gqubits = as_int_tuple(g.get("qubits"))
        if gqubits is None:
            continue

        error = None
        duration_ns = None

        params = g.get("parameters", [])
        if isinstance(params, list):
            for p in params:
                if not isinstance(p, dict):
                    continue
                pname = str(p.get("name") or p.get("parameter") or "").strip().lower()
                val = to_float(p.get("value"))
                unit = p.get("unit")

                if val is None or not pname:
                    continue

                if pname in ("gate_error", "error"):
                    error = val
                elif pname in ("gate_length", "duration"):
                    duration_ns = convert_duration_to_ns(
                        val, str(unit) if unit else None
                    )

        if error is None and duration_ns is None:
            continue

        gates_out.append(
            GateCalibration(
                gate=gname,
                qubits=gqubits,
                error=error,
                duration_ns=duration_ns,
            )
        )

    return gates_out


def _derive_1q_gate_errors(
    qubits: list[QubitCalibration],
    gates: list[GateCalibration],
) -> list[QubitCalibration]:
    """Derive per-qubit single-qubit gate errors from gate calibrations."""
    oneq_errors: dict[int, list[float]] = {}
    for g in gates:
        if g.error is None or len(g.qubits) != 1:
            continue
        q = int(g.qubits[0])
        oneq_errors.setdefault(q, []).append(float(g.error))

    if not oneq_errors:
        return qubits

    updated: list[QubitCalibration] = []
    for q in qubits:
        if q.gate_error_1q is None and q.qubit in oneq_errors and oneq_errors[q.qubit]:
            try:
                ge = float(median(oneq_errors[q.qubit]))
                updated.append(replace(q, gate_error_1q=ge))
            except (TypeError, ValueError) as e:
                logger.debug("Failed to compute median gate error: %s", e)
                updated.append(q)
        else:
            updated.append(q)

    return updated
