# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
JSON serialization utilities.

This module provides functions for converting arbitrary Python objects
to JSON-serializable format, with robust handling for numpy arrays,
dataclasses, Pydantic models, and other common types.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any


logger = logging.getLogger(__name__)

# Maximum recursion depth to prevent infinite loops
_MAX_DEPTH = 50


def to_jsonable(obj: Any, *, max_depth: int = _MAX_DEPTH) -> Any:
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
    6. Dataclasses - convert via ``dataclasses.asdict()``
    7. Pydantic models - try ``model_dump()``, then ``dict()``
    8. Objects with ``to_dict()`` method
    9. Objects with ``__dict__`` attribute
    10. Fallback to ``repr()`` (truncated to 500 chars)
    """
    if max_depth <= 0:
        logger.debug("Max depth exceeded, truncating: %r", type(obj))
        return {"__truncated__": repr(obj)[:100]}

    # JSON primitives - return as-is
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # NumPy scalars (check before arrays since scalars also have tolist)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass

    # NumPy arrays and array-like objects
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except (TypeError, ValueError):
            pass

    # Dictionaries - recurse with depth limit
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_depth=max_depth - 1) for k, v in obj.items()}

    # Lists and tuples - recurse with depth limit
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_depth=max_depth - 1) for v in obj]

    # Sets - convert to sorted list for deterministic output
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(to_jsonable(v, max_depth=max_depth - 1) for v in obj)
        except TypeError:
            # Not sortable, just convert to list
            return [to_jsonable(v, max_depth=max_depth - 1) for v in obj]

    # Dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_jsonable(asdict(obj), max_depth=max_depth - 1)
        except (TypeError, ValueError):
            pass

    # Try common serialization methods (Pydantic v2, Pydantic v1, custom)
    for method_name in ("model_dump", "dict", "to_dict", "to_dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return to_jsonable(method(), max_depth=max_depth - 1)
            except Exception:
                continue

    # Try __dict__ for generic objects
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj), max_depth=max_depth - 1)
        except Exception:
            pass

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


def safe_json_dumps(obj: Any, *, indent: int | None = 2) -> str:
    """
    Serialize to JSON with robust fallback for unknown types.

    Combines :func:`to_jsonable` conversion with ``json.dumps``,
    providing an additional fallback for any types that slip through.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    indent : int or None, optional
        Indentation level for pretty-printing. Default is 2.
        Use None for compact output.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(
        to_jsonable(obj),
        indent=indent,
        default=_default_serializer,
    )
