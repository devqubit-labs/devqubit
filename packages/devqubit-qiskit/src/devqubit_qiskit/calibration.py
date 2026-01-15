# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Calibration extraction for Qiskit backends.

Extracts device calibration data from Qiskit BackendProperties
and converts it to the devqubit calibration schema, following
the Uniform Execution Contract (UEC) with explicit source tracking.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from statistics import median
from typing import Any

from devqubit_engine.uec.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.utils.common import utc_now_iso
from devqubit_qiskit.utils import (
    as_int_tuple,
    convert_duration_to_ns,
    convert_freq_to_ghz,
    convert_time_to_us,
    to_float,
)


logger = logging.getLogger(__name__)


def _parse_timestamp(x: Any) -> str:
    """
    Parse timestamp to ISO format string.

    Parameters
    ----------
    x : Any
        Timestamp value (datetime, string, or other).

    Returns
    -------
    str
        ISO format timestamp string.
    """
    if x is None:
        return utc_now_iso()
    if isinstance(x, datetime):
        try:
            return x.isoformat()
        except Exception:
            return str(x)
    s = str(x).strip()
    return s if s else utc_now_iso()


def _extract_qubit_calibrations(
    qubits: list[Any],
) -> list[QubitCalibration]:
    """
    Extract per-qubit calibration data from properties.

    Parameters
    ----------
    qubits : list
        Qubit properties list from BackendProperties.

    Returns
    -------
    list of QubitCalibration
        Extracted qubit calibration records.

    Notes
    -----
    Handles standard Qiskit property names:
    - t1, t2: Coherence times
    - readout_error: Direct readout error
    - prob_meas0_prep1, prob_meas1_prep0: Assignment errors
    - frequency, anharmonicity: Qubit frequencies
    """
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

        qc = QubitCalibration(
            qubit=int(q_idx),
            t1_us=t1_us,
            t2_us=t2_us,
            readout_error=readout_error,
            gate_error_1q=None,  # Filled later from gate table
            frequency_ghz=freq_ghz,
            anharmonicity_ghz=anharm_ghz,
        )
        qubits_out.append(qc)

    return qubits_out


def _extract_gate_calibrations(gates: list[Any]) -> list[GateCalibration]:
    """
    Extract per-gate calibration data from properties.

    Parameters
    ----------
    gates : list
        Gate properties list from BackendProperties.

    Returns
    -------
    list of GateCalibration
        Extracted gate calibration records.

    Notes
    -----
    Handles standard Qiskit gate property names:
    - gate_error, error: Gate error rate
    - gate_length, duration: Gate duration
    """
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
    """
    Derive per-qubit single-qubit gate errors from gate calibrations.

    For each qubit, compute the median error of all single-qubit gates
    acting on that qubit.

    Parameters
    ----------
    qubits : list of QubitCalibration
        Qubit calibrations to update.
    gates : list of GateCalibration
        Gate calibrations to extract 1Q errors from.

    Returns
    -------
    list of QubitCalibration
        Updated qubit calibrations with gate_error_1q filled in.
    """
    # Collect 1Q gate errors per qubit
    oneq_errors: dict[int, list[float]] = {}
    for g in gates:
        if g.error is None:
            continue
        if len(g.qubits) == 1:
            q = int(g.qubits[0])
            oneq_errors.setdefault(q, []).append(float(g.error))

    if not oneq_errors:
        return qubits

    # Update qubits with median 1Q gate error
    updated: list[QubitCalibration] = []
    for q in qubits:
        if q.gate_error_1q is None and q.qubit in oneq_errors and oneq_errors[q.qubit]:
            try:
                ge = float(median(oneq_errors[q.qubit]))
                updated.append(replace(q, gate_error_1q=ge))
            except Exception:
                updated.append(q)
        else:
            updated.append(q)

    return updated


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
        Data source indicator for UEC compliance. Default is "provider"
        indicating data comes directly from backend properties.

    Returns
    -------
    DeviceCalibration or None
        Extracted calibration data, or None if no useful data exists.

    Notes
    -----
    This function handles the standard Qiskit BackendProperties format
    with 'qubits' and 'gates' lists, as well as various timestamp fields.

    The source parameter follows UEC conventions:
    - "provider": Direct from backend properties API
    - "derived": Computed from other available data
    - "manual": User-provided values
    """
    if not isinstance(props, dict) or not props:
        return None

    # Extract calibration timestamp
    cal_time = None
    for k in ("last_update_date", "last_update_datetime", "last_update"):
        if props.get(k):
            cal_time = _parse_timestamp(props.get(k))
            break
    if not cal_time:
        cal_time = utc_now_iso()

    # Extract qubit properties
    qubits_out: list[QubitCalibration] = []
    qubits = props.get("qubits")
    if isinstance(qubits, list):
        qubits_out = _extract_qubit_calibrations(qubits)

    # Extract gate properties
    gates_out: list[GateCalibration] = []
    gates = props.get("gates")
    if isinstance(gates, list):
        gates_out = _extract_gate_calibrations(gates)

    # Check if we have any calibration data
    has_any_qubit_metric = any(
        (
            q.t1_us is not None
            or q.t2_us is not None
            or q.readout_error is not None
            or q.frequency_ghz is not None
            or q.anharmonicity_ghz is not None
        )
        for q in qubits_out
    )
    has_any_gate_metric = any(
        (g.error is not None or g.duration_ns is not None) for g in gates_out
    )

    if not has_any_qubit_metric and not has_any_gate_metric:
        logger.debug("No calibration data found in properties")
        return None

    # Derive per-qubit 1Q gate errors
    if qubits_out and gates_out:
        qubits_out = _derive_1q_gate_errors(qubits_out, gates_out)

    # Build calibration object with source tracking
    calibration = DeviceCalibration(
        calibration_time=cal_time,
        qubits=qubits_out,
        gates=gates_out,
        source=source,
    )
    calibration.compute_medians()

    logger.debug(
        "Extracted calibration: %d qubits, %d gates, source=%s",
        len(qubits_out),
        len(gates_out),
        source,
    )

    return calibration


def extract_calibration_from_target(
    target: Any,
    *,
    source: str = "derived",
) -> DeviceCalibration | None:
    """
    Extract calibration data from a Qiskit Target object.

    BackendV2 backends may not have traditional properties() but expose
    calibration through the Target object.

    Parameters
    ----------
    target : Any
        Qiskit Target instance.
    source : str, optional
        Data source indicator. Default is "derived" since Target
        calibration may be computed/interpolated.

    Returns
    -------
    DeviceCalibration or None
        Extracted calibration data, or None if no useful data exists.

    Notes
    -----
    Target-based calibration is typically less detailed than
    BackendProperties but may be the only option for some backends.
    """
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

                    # Convert duration to ns (Target uses seconds)
                    duration_ns = None
                    if duration is not None:
                        duration_ns = float(duration) * 1e9

                    gates_out.append(
                        GateCalibration(
                            gate=op_name,
                            qubits=tuple(qargs),
                            error=float(error) if error is not None else None,
                            duration_ns=duration_ns,
                        )
                    )
                except Exception:
                    continue
    except Exception as e:
        logger.debug("Failed to extract calibration from Target: %s", e)
        return None

    if not gates_out:
        return None

    # Build calibration object
    calibration = DeviceCalibration(
        calibration_time=utc_now_iso(),
        gates=gates_out,
        source=source,
    )
    calibration.compute_medians()

    return calibration
