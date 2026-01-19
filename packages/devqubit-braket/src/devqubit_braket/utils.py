# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Common utilities for Braket adapter.

Provides version utilities, bitstring canonicalization, and common helpers
used across the adapter components.
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# Version utilities
# =============================================================================


def braket_version() -> str:
    """
    Get the installed Amazon Braket SDK version.

    Returns
    -------
    str
        Braket SDK version string (e.g., "1.70.0"), or "unknown" if
        Braket is not installed or version cannot be determined.
    """
    try:
        import braket

        return getattr(braket, "__version__", "unknown")
    except ImportError:
        return "unknown"


def get_adapter_version() -> str:
    """
    Get the devqubit-braket adapter version.

    Returns
    -------
    str
        Adapter version string, or "unknown" if not installed.
    """
    try:
        from importlib.metadata import version

        return version("devqubit-braket")
    except Exception:
        return "unknown"


# =============================================================================
# Bitstring canonicalization
# =============================================================================


def reverse_bitstring(bitstring: str) -> str:
    """
    Reverse a bitstring (convert between big-endian and little-endian).

    Braket uses big-endian (qubit 0 = leftmost bit, cbit0_left).
    UEC canonical format is little-endian (qubit 0 = rightmost bit, cbit0_right).

    Parameters
    ----------
    bitstring : str
        Input bitstring (e.g., "011").

    Returns
    -------
    str
        Reversed bitstring (e.g., "110").

    Examples
    --------
    >>> reverse_bitstring("011")
    '110'
    >>> reverse_bitstring("00")
    '00'
    """
    return bitstring[::-1]


def canonicalize_counts(
    counts: dict[str, int],
    *,
    reverse: bool = True,
) -> dict[str, int]:
    """
    Transform measurement counts to canonical UEC format.

    Braket returns counts with big-endian bitstrings (cbit0_left).
    UEC canonical format uses little-endian (cbit0_right, like Qiskit).

    Parameters
    ----------
    counts : dict
        Measurement counts from Braket {bitstring: count}.
    reverse : bool, optional
        Whether to reverse bitstrings to canonical format. Default True.

    Returns
    -------
    dict
        Counts with canonicalized bitstrings.

    Examples
    --------
    >>> canonicalize_counts({"01": 50, "10": 50})
    {'10': 50, '01': 50}
    """
    if not reverse:
        return counts
    return {reverse_bitstring(k): v for k, v in counts.items()}


# =============================================================================
# Device utilities
# =============================================================================


def get_backend_name(device: Any) -> str:
    """
    Extract device name from a Braket device.

    Parameters
    ----------
    device : Any
        Braket device instance.

    Returns
    -------
    str
        Device name, ARN, or class name as fallback.
    """
    for key in ("name", "short_name"):
        try:
            if hasattr(device, key):
                v = getattr(device, key)
                return str(v() if callable(v) else v)
        except Exception:
            pass

    try:
        if hasattr(device, "arn"):
            arn = getattr(device, "arn")
            return str(arn() if callable(arn) else arn)
    except Exception:
        pass

    return device.__class__.__name__


def extract_task_id(task: Any) -> str | None:
    """
    Extract task ID from a Braket task.

    Parameters
    ----------
    task : Any
        Braket task instance.

    Returns
    -------
    str or None
        Task ID if available.
    """
    for key in ("id", "task_id", "arn"):
        try:
            if hasattr(task, key):
                v = getattr(task, key)
                return str(v() if callable(v) else v)
        except Exception:
            continue
    return None


# =============================================================================
# Conversion utilities
# =============================================================================


def to_float(x: Any) -> float | None:
    """
    Convert to float, returning None on failure.

    Parameters
    ----------
    x : Any
        Value to convert.

    Returns
    -------
    float or None
        Float value or None if conversion fails.
    """
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def get_nested(obj: Any, path: tuple[str, ...]) -> Any:
    """
    Get a nested value supporting both attribute and dict-style access.

    Parameters
    ----------
    obj : Any
        Object or dict to traverse.
    path : tuple of str
        Sequence of keys/attributes.

    Returns
    -------
    Any
        Nested value, or None if any key is missing.
    """
    cur: Any = obj
    for key in path:
        if cur is None:
            return None
        try:
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                cur = getattr(cur, key, None)
        except Exception:
            return None
    return cur


def obj_to_dict(x: Any) -> dict[str, Any] | None:
    """
    Convert a Braket/pydantic object to a plain JSON-serializable dict.

    Parameters
    ----------
    x : Any
        Object to convert (pydantic model, dict, or other).

    Returns
    -------
    dict or None
        Plain dict representation, or None if conversion fails.
    """
    if x is None:
        return None
    try:
        if isinstance(x, dict):
            return x
        # pydantic v1 style
        if hasattr(x, "dict") and callable(getattr(x, "dict")):
            from devqubit_engine.utils.serialization import to_jsonable

            return to_jsonable(x.dict())
        # pydantic v2 or custom style
        if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
            from devqubit_engine.utils.serialization import to_jsonable

            return to_jsonable(x.to_dict())
        # Fallback: attempt generic conversion
        from devqubit_engine.utils.serialization import to_jsonable

        return to_jsonable(x)
    except Exception:
        return None
