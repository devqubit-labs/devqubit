# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Deterministic JSON serialization utilities.

This module provides functions for converting arbitrary Python objects
to JSON-serializable format with **deterministic output** suitable for
fingerprinting, hashing, and reproducible logging.

Key guarantees for production use:

- **Deterministic ordering**: `sort_keys=True` for all dicts
- **Stable separators**: Consistent across Python versions
- **Float normalization**: Controllable precision for reproducibility
- **Set/frozenset handling**: Converted to sorted lists
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any


logger = logging.getLogger(__name__)

# Maximum recursion depth to prevent infinite loops
_MAX_DEPTH = 50

# Default float precision for canonical serialization (15 significant digits)
_DEFAULT_FLOAT_PRECISION = 15


def _sort_key(v: Any) -> tuple[str, str]:
    """
    Generate a stable sort key for heterogeneous collections.

    Parameters
    ----------
    v : Any
        Value to generate sort key for.

    Returns
    -------
    tuple of (type_name, str_repr)
        Tuple ensuring stable ordering across types.
    """
    return (type(v).__name__, str(v))


def _normalize_float(
    value: float,
    precision: int = _DEFAULT_FLOAT_PRECISION,
) -> float:
    """
    Normalize a float to a specific precision.

    Parameters
    ----------
    value : float
        Float value to normalize.
    precision : int
        Number of significant digits.

    Returns
    -------
    float
        Normalized float value.
    """
    if not isinstance(value, float):
        return value
    # Handle special values
    if value != value:  # NaN
        return value
    if value == float("inf") or value == float("-inf"):
        return value
    # Round to precision
    if value == 0.0:
        return 0.0
    return float(f"{value:.{precision}g}")


def to_jsonable(
    obj: Any,
    *,
    max_depth: int = _MAX_DEPTH,
    normalize_floats: bool = False,
    float_precision: int = _DEFAULT_FLOAT_PRECISION,
) -> Any:
    """
    Convert arbitrary Python objects to JSON-serializable format.

    Handles numpy arrays/scalars, dataclasses, Pydantic models, and objects
    with common serialization methods. This is the canonical implementation
    used throughout devqubit for consistent serialization.

    Parameters
    ----------
    obj : Any
        Object to convert. Can be any Python type.
    max_depth : int, optional
        Maximum recursion depth to prevent infinite loops in
        self-referential structures. Default is 50.
    normalize_floats : bool, optional
        If True, normalize floats to a specific precision for
        deterministic output. Default is False.
    float_precision : int, optional
        Number of significant digits for float normalization.
        Only used if normalize_floats is True. Default is 15.

    Returns
    -------
    Any
        JSON-serializable representation. Will be one of:
        ``None``, ``str``, ``int``, ``float``, ``bool``, ``list``, or ``dict``.

    Notes
    -----
    The conversion follows this priority:

    1. JSON primitives (None, str, int, float, bool) - returned as-is
    2. NumPy scalars - converted via ``.item()``
    3. NumPy arrays - converted via ``.tolist()``
    4. Dicts - recursively convert values, stringify keys
    5. Lists/tuples - recursively convert elements
    6. Sets/frozensets - convert to **sorted** lists for determinism
    7. Dataclasses - convert via ``dataclasses.asdict()``
    8. Pydantic models - try ``model_dump()``, then ``dict()``
    9. Objects with ``to_dict()`` method
    10. Objects with ``__dict__`` attribute
    11. Fallback to ``repr()`` (truncated to 500 chars)

    """
    if max_depth <= 0:
        logger.debug("Max depth exceeded, truncating: %r", type(obj))
        return {"__truncated__": repr(obj)[:100]}

    # JSON primitives - return as-is (with optional float normalization)
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj

    if isinstance(obj, float):
        if normalize_floats:
            return _normalize_float(obj, float_precision)
        return obj

    # NumPy scalars (check before arrays since scalars also have tolist)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            item = obj.item()
            if normalize_floats and isinstance(item, float):
                return _normalize_float(item, float_precision)
            return item
        except (TypeError, ValueError):
            logger.debug("Failed to convert numpy scalar via .item(): %r", type(obj))

    # NumPy arrays and array-like objects
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return to_jsonable(
                obj.tolist(),
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
        except (TypeError, ValueError):
            logger.debug("Failed to convert via .tolist(): %r", type(obj))

    # Dictionaries - recurse with depth limit
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(
                v,
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
            for k, v in obj.items()
        }

    # Lists and tuples - recurse with depth limit
    if isinstance(obj, (list, tuple)):
        return [
            to_jsonable(
                v,
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
            for v in obj
        ]

    # Sets/frozensets - convert to SORTED list for deterministic output
    if isinstance(obj, (set, frozenset)):
        converted = [
            to_jsonable(
                v,
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
            for v in obj
        ]
        try:
            return sorted(converted)
        except TypeError:
            # Heterogeneous types - use stable sort key
            return sorted(converted, key=_sort_key)

    # Dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_jsonable(
                asdict(obj),
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
        except (TypeError, ValueError):
            logger.debug("Failed to convert dataclass via asdict(): %r", type(obj))

    # Try common serialization methods (Pydantic v2, Pydantic v1, custom)
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return to_jsonable(
                    method(),
                    max_depth=max_depth - 1,
                    normalize_floats=normalize_floats,
                    float_precision=float_precision,
                )
            except Exception:
                logger.debug(
                    "Serialization via .%s() failed for %r", method_name, type(obj)
                )
                continue

    # Try __dict__ for generic objects
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(
                vars(obj),
                max_depth=max_depth - 1,
                normalize_floats=normalize_floats,
                float_precision=float_precision,
            )
        except Exception:
            logger.debug("Failed to serialize via __dict__ for %r", type(obj))

    # Last resort: repr (truncated)
    return {"__repr__": repr(obj)[:500]}


def _default_serializer(obj: Any) -> Any:
    """
    JSON default serializer for json.dumps fallback.

    Parameters
    ----------
    obj : Any
        Object that couldn't be serialized.

    Returns
    -------
    Any
        Serializable representation (dict or string).
    """
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def json_dumps(
    obj: Any,
    *,
    compact: bool = False,
    normalize_floats: bool = False,
    float_precision: int = _DEFAULT_FLOAT_PRECISION,
    indent: int | None = None,
) -> str:
    """
    Serialize to deterministic JSON.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    compact : bool, default=False
        If True, produce minimal output for hashing/fingerprinting.
        Implies normalize_floats=True and no indentation.
    normalize_floats : bool, default=False
        Normalize floats to specific precision.
    float_precision : int, default=15
        Significant digits for float normalization.
    indent : int or None, optional
        JSON indentation level. Default is None (single-line output).
        Ignored when compact=True.
    """
    if compact:
        normalize_floats = True
        indent = None

    return json.dumps(
        to_jsonable(
            obj,
            normalize_floats=normalize_floats,
            float_precision=float_precision,
        ),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else (",", ": "),
        default=_default_serializer,
    )
