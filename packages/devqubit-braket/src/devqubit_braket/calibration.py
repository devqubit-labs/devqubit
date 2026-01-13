# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Calibration extraction for Braket devices.

Extracts calibration-like metrics from Braket device properties.
Prefer standardized device properties (gate fidelities) and map them
to the devqubit calibration schema.

Notes
-----
In Amazon Braket, numerical gate-quality metrics are typically available via:

- device.properties.standardized.oneQubitProperties[...].oneQubitGateFidelity
- device.properties.standardized.twoQubitProperties[...].twoQubitGateFidelity

Pulse "gate calibrations" (AwsDevice.gate_calibrations / refresh_gate_calibrations)
are pulse-sequence definitions and are not reliably "error/duration" metrics.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from devqubit_braket.utils import get_nested, obj_to_dict, to_float
from devqubit_engine.uec.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.utils.time_utils import utc_now_iso


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
        # Remove brackets/parens and normalize separators
        cleaned = qubits_key.translate(str.maketrans("()-[]", ",,,,,", " "))
        return [int(t) for t in cleaned.split(",") if t]
    except Exception:
        return []


def _extract_fidelity_entries(
    container: Any, candidate_keys: tuple[str, ...]
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


def extract_calibration_from_device(device: Any) -> DeviceCalibration | None:
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
    except Exception:
        return None

    props = obj_to_dict(props_obj)
    if not isinstance(props, dict) or not props:
        return None

    # Timestamp: documented as "properties last updated"
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
            except Exception:
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
            except Exception:
                qubits_out.append(QubitCalibration(qubit=q))

    cal = DeviceCalibration(
        calibration_time=cal_time,
        qubits=qubits_out,
        gates=gates,
    )
    cal.compute_medians()
    return cal
