# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for Qiskit adapter.

Provides type conversion helpers, version detection, and common
utilities used across the adapter components.
"""

from __future__ import annotations

from typing import Any

import qiskit


# =============================================================================
# Version Utilities
# =============================================================================


def qiskit_version() -> str:
    """
    Get installed Qiskit version.

    Returns
    -------
    str
        Qiskit version string or "unknown" if not available.
    """
    return getattr(qiskit, "__version__", "unknown")


def get_adapter_version() -> str:
    """
    Get adapter version dynamically from package metadata.

    Returns
    -------
    str
        Adapter version or "unknown" if unavailable.
    """
    try:
        from importlib.metadata import version

        return version("devqubit-qiskit")
    except Exception:
        return "unknown"


# =============================================================================
# Backend Utilities
# =============================================================================


def get_backend_name(backend: Any) -> str:
    """
    Extract backend name from a Qiskit backend instance.

    Handles both property and method-based name access patterns
    used across different Qiskit versions.

    Parameters
    ----------
    backend : Any
        Qiskit backend instance.

    Returns
    -------
    str
        Backend name or class name as fallback.
    """
    try:
        name_attr = getattr(backend, "name", None)
        if callable(name_attr):
            name = name_attr()
        else:
            name = name_attr
        if name:
            return str(name)
    except Exception:
        pass
    return backend.__class__.__name__


def extract_job_id(job: Any) -> str | None:
    """
    Extract job ID from a Qiskit job instance.

    Parameters
    ----------
    job : Any
        Qiskit job instance.

    Returns
    -------
    str or None
        Job ID if available, None otherwise.
    """
    if not hasattr(job, "job_id"):
        return None
    try:
        job_id_attr = job.job_id
        if callable(job_id_attr):
            return str(job_id_attr())
        return str(job_id_attr)
    except Exception:
        return None


# =============================================================================
# Type Conversion Utilities
# =============================================================================


def to_float(x: Any) -> float | None:
    """
    Convert value to float, returning None on failure.

    Handles numpy scalars, Decimal, and string representations.

    Parameters
    ----------
    x : Any
        Value to convert.

    Returns
    -------
    float or None
        Converted value or None if conversion fails.
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        pass
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def to_int(x: Any) -> int | None:
    """
    Convert value to int, returning None on failure.

    Parameters
    ----------
    x : Any
        Value to convert.

    Returns
    -------
    int or None
        Converted value or None if conversion fails.
    """
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x).strip())
        except Exception:
            return None


def as_int_tuple(seq: Any) -> tuple[int, ...] | None:
    """
    Convert sequence to tuple of ints.

    Parameters
    ----------
    seq : Any
        Sequence to convert.

    Returns
    -------
    tuple of int or None
        Converted tuple or None if conversion fails.
    """
    if not isinstance(seq, (list, tuple)):
        return None
    out: list[int] = []
    for v in seq:
        iv = to_int(v)
        if iv is None:
            return None
        out.append(iv)
    return tuple(out)


# =============================================================================
# Unit Conversion Utilities
# =============================================================================

# Time to microseconds conversion factors
_TIME_TO_US: dict[str, float] = {
    "s": 1e6,
    "sec": 1e6,
    "secs": 1e6,
    "ms": 1e3,
    "millisec": 1e3,
    "us": 1.0,
    "µs": 1.0,
    "ns": 1e-3,
    "ps": 1e-6,
}

# Duration to nanoseconds conversion factors
_DURATION_TO_NS: dict[str, float] = {
    "s": 1e9,
    "sec": 1e9,
    "ms": 1e6,
    "us": 1e3,
    "µs": 1e3,
    "ns": 1.0,
    "ps": 1e-3,
}

# Frequency to GHz conversion factors
_FREQ_TO_GHZ: dict[str, float] = {
    "hz": 1e-9,
    "khz": 1e-6,
    "mhz": 1e-3,
    "ghz": 1.0,
}


def convert_time_to_us(val: float, unit: str | None) -> float:
    """
    Convert time value to microseconds.

    Parameters
    ----------
    val : float
        Time value.
    unit : str or None
        Unit string (e.g., "s", "ms", "us", "ns").
        If None, assumes microseconds (Qiskit convention for T1/T2).

    Returns
    -------
    float
        Time in microseconds.
    """
    if unit is None:
        return float(val)
    u = str(unit).strip().lower()
    factor = _TIME_TO_US.get(u)
    return float(val) * factor if factor is not None else float(val)


def convert_duration_to_ns(val: float, unit: str | None) -> float:
    """
    Convert duration value to nanoseconds.

    Parameters
    ----------
    val : float
        Duration value.
    unit : str or None
        Unit string. If None, assumes the value is already in nanoseconds.

    Returns
    -------
    float
        Duration in nanoseconds.
    """
    if unit is None:
        return float(val)
    u = str(unit).strip().lower()
    factor = _DURATION_TO_NS.get(u)
    return float(val) * factor if factor is not None else float(val)


def convert_freq_to_ghz(val: float, unit: str | None) -> float:
    """
    Convert frequency value to GHz.

    Parameters
    ----------
    val : float
        Frequency value.
    unit : str or None
        Unit string. If None, uses heuristic based on magnitude.

    Returns
    -------
    float
        Frequency in GHz.
    """
    if unit is None:
        # Heuristic: if it looks like Hz-scale, convert; else assume GHz
        return float(val) * 1e-9 if abs(val) > 1e6 else float(val)
    u = str(unit).strip().lower()
    factor = _FREQ_TO_GHZ.get(u)
    return float(val) * factor if factor is not None else float(val)
