# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Calibration data structures for quantum devices.

This module defines per-qubit and per-gate calibration records,
along with the aggregated DeviceCalibration bundle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


logger = logging.getLogger(__name__)


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_GATE_CLASS_SUFFIXES = ("powgate", "gate")


def _norm_gate(name: str) -> str:
    """Normalize gate names for consistent matching across SDKs."""
    s = _NON_ALNUM_RE.sub("", str(name).lower())
    for suffix in _GATE_CLASS_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


# Common two-qubit gate names for median error calculation
TWO_QUBIT_GATES = frozenset(
    _norm_gate(x)
    for x in {
        # Common controlled / entangling gates
        "cx",
        "cnot",
        "cz",
        "cy",
        "ecr",
        "rzx",
        # Swap family
        "swap",
        "iswap",
        "pswap",
        "phased_iswap",
        # Interaction / rotation families
        "rxx",
        "ryy",
        "rzz",
        "xx",
        "yy",
        "zz",
        "xy",
        # Ion-trap style
        "ms",
        # FSim family (Cirq/Google)
        "fsim",
        "phased_fsim",
        # Google-specific named aliases
        "sycamore",
        "syc",
        "willow",
        # Braket controlled-phase (2-qubit)
        "cphaseshift",
        "cphaseshift00",
        "cphaseshift01",
        "cphaseshift10",
        "cphaseshift11",
    }
)


def _median_val(values: Sequence[float | None]) -> float | None:
    """Compute median of values, ignoring None entries."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.median(valid))


@dataclass
class QubitCalibration:
    """
    Per-qubit calibration record.

    Parameters
    ----------
    qubit : int
        Qubit index (0-based).
    t1_us : float, optional
        Energy relaxation time (T1) in microseconds.
    t2_us : float, optional
        Dephasing time (T2) in microseconds.
    readout_error : float, optional
        Assignment/readout error probability (0.0 to 1.0).
    gate_error_1q : float, optional
        Representative single-qubit gate error probability.
    frequency_ghz : float, optional
        Qubit frequency in GHz.
    anharmonicity_ghz : float, optional
        Qubit anharmonicity in GHz.
    """

    qubit: int
    t1_us: float | None = None
    t2_us: float | None = None
    readout_error: float | None = None
    gate_error_1q: float | None = None
    frequency_ghz: float | None = None
    anharmonicity_ghz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"qubit": int(self.qubit)}
        for key in (
            "t1_us",
            "t2_us",
            "readout_error",
            "gate_error_1q",
            "frequency_ghz",
            "anharmonicity_ghz",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = float(value)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QubitCalibration:
        return cls(
            qubit=int(d["qubit"]),
            t1_us=d.get("t1_us"),
            t2_us=d.get("t2_us"),
            readout_error=d.get("readout_error"),
            gate_error_1q=d.get("gate_error_1q"),
            frequency_ghz=d.get("frequency_ghz"),
            anharmonicity_ghz=d.get("anharmonicity_ghz"),
        )


@dataclass
class GateCalibration:
    """
    Calibration record for a gate applied on specific qubits.

    Parameters
    ----------
    gate : str
        Gate name (e.g., "cx", "cz", "rx").
    qubits : tuple of int
        Tuple of qubit indices the gate acts on.
    error : float, optional
        Gate error probability (0.0 to 1.0).
    duration_ns : float, optional
        Gate duration in nanoseconds.
    """

    gate: str
    qubits: tuple[int, ...]
    error: float | None = None
    duration_ns: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "gate": str(self.gate),
            "qubits": list(self.qubits),
        }
        if self.error is not None:
            d["error"] = float(self.error)
        if self.duration_ns is not None:
            d["duration_ns"] = float(self.duration_ns)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateCalibration:
        return cls(
            gate=str(d["gate"]),
            qubits=tuple(int(x) for x in d.get("qubits", [])),
            error=d.get("error"),
            duration_ns=d.get("duration_ns"),
        )

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) >= 2


@dataclass
class DeviceCalibration:
    """
    Device-level calibration bundle with derived summary metrics.

    Parameters
    ----------
    calibration_time : str, optional
        Provider/SDK calibration timestamp (ISO 8601).
    qubits : list of QubitCalibration
        Per-qubit calibration records.
    gates : list of GateCalibration
        Per-gate calibration records.
    median_t1_us : float, optional
        Median T1 across all qubits.
    median_t2_us : float, optional
        Median T2 across all qubits.
    median_readout_error : float, optional
        Median readout error across all qubits.
    median_2q_error : float, optional
        Median two-qubit gate error.
    source : str, optional
        Data source: "provider", "derived", or "manual".
    """

    calibration_time: str | None = None
    qubits: list[QubitCalibration] = field(default_factory=list)
    gates: list[GateCalibration] = field(default_factory=list)

    median_t1_us: float | None = None
    median_t2_us: float | None = None
    median_readout_error: float | None = None
    median_2q_error: float | None = None

    source: str | None = None
    schema_version: str = "devqubit.calibration/0.1"

    def compute_medians(self) -> None:
        """Compute derived median summary metrics in-place."""
        if self.median_t1_us is None:
            self.median_t1_us = _median_val([q.t1_us for q in self.qubits])

        if self.median_t2_us is None:
            self.median_t2_us = _median_val([q.t2_us for q in self.qubits])

        if self.median_readout_error is None:
            self.median_readout_error = _median_val(
                [q.readout_error for q in self.qubits]
            )

        if self.median_2q_error is None:
            gate_errors = [
                g.error
                for g in self.gates
                if g.error is not None and _norm_gate(g.gate) in TWO_QUBIT_GATES
            ]
            self.median_2q_error = _median_val(gate_errors)

        logger.debug(
            "Computed calibration medians: T1=%.2f µs, T2=%.2f µs, "
            "readout_err=%.4f, 2q_err=%.4f",
            self.median_t1_us or 0,
            self.median_t2_us or 0,
            self.median_readout_error or 0,
            self.median_2q_error or 0,
        )

    def to_dict(self) -> dict[str, Any]:
        has_missing = any(
            x is None
            for x in (
                self.median_t1_us,
                self.median_t2_us,
                self.median_readout_error,
                self.median_2q_error,
            )
        )
        if has_missing:
            self.compute_medians()

        d: dict[str, Any] = {
            "schema": self.schema_version,
            "calibration_time": self.calibration_time,
            "qubits": [q.to_dict() for q in self.qubits],
            "gates": [g.to_dict() for g in self.gates],
        }

        if self.source:
            d["source"] = self.source

        for key in (
            "median_t1_us",
            "median_t2_us",
            "median_readout_error",
            "median_2q_error",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = float(value)

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceCalibration:
        qubits = [
            QubitCalibration.from_dict(x)
            for x in d.get("qubits", [])
            if isinstance(x, dict)
        ]
        gates = [
            GateCalibration.from_dict(x)
            for x in d.get("gates", [])
            if isinstance(x, dict)
        ]

        return cls(
            calibration_time=d.get("calibration_time"),
            qubits=qubits,
            gates=gates,
            median_t1_us=d.get("median_t1_us"),
            median_t2_us=d.get("median_t2_us"),
            median_readout_error=d.get("median_readout_error"),
            median_2q_error=d.get("median_2q_error"),
            source=d.get("source"),
            schema_version=d.get("schema", "devqubit.calibration/0.1"),
        )
