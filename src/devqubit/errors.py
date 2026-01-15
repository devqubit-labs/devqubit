# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Public exception types.

This module exposes all public exceptions that may be raised by devqubit
operations. Users can catch these to handle specific error conditions.

Exception Types
---------------
Storage errors:
    - ``StorageError`` - Base for storage-related errors
    - ``ObjectNotFoundError`` - Artifact not found in object store
    - ``RunNotFoundError`` - Run not found in registry

Query errors:
    - ``QueryParseError`` - Invalid search query syntax

Envelope errors:
    - ``MissingEnvelopeError`` - Run missing required envelope
    - ``EnvelopeValidationError`` - Invalid envelope data

Catch-All
---------
``DevQubitError`` is a convenience base class defined in this module.
Use it to catch any error from devqubit in a single except block:

>>> from devqubit.errors import DevQubitError, RunNotFoundError
>>> try:
...     run = load_run("nonexistent")
... except RunNotFoundError as e:
...     print(f"Run not found: {e.run_id}")
... except DevQubitError:
...     print("Other devqubit error")

Basic Usage
-----------
>>> from devqubit.errors import RunNotFoundError
>>> try:
...     run = load_run("nonexistent")
... except RunNotFoundError as e:
...     print(f"Run not found: {e.run_id}")

Note
----
The engine exceptions (``StorageError``, ``RunNotFoundError``, etc.) are
re-exported from ``devqubit_engine`` for convenience. ``DevQubitError``
is a separate base class defined here for catch-all patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    # Convenience base exception
    "DevQubitError",
    # Storage errors
    "StorageError",
    "ObjectNotFoundError",
    "RunNotFoundError",
    # Query errors
    "QueryParseError",
    # Envelope errors
    "MissingEnvelopeError",
    "EnvelopeValidationError",
]


class DevQubitError(Exception):
    """
    Convenience base exception for catch-all error handling.

    This exception is NOT the base class for engine exceptions, but can
    be used alongside specific exceptions for comprehensive error handling.

    Examples
    --------
    >>> from devqubit.errors import DevQubitError, RunNotFoundError
    >>> try:
    ...     result = some_devqubit_operation()
    ... except RunNotFoundError:
    ...     print("Specific: run not found")
    ... except DevQubitError:
    ...     print("Fallback: other devqubit error")
    """

    pass


if TYPE_CHECKING:
    from devqubit_engine.query import QueryParseError as _QueryParseError
    from devqubit_engine.storage.errors import (
        ObjectNotFoundError as _ObjectNotFoundError,
    )
    from devqubit_engine.storage.errors import RunNotFoundError as _RunNotFoundError
    from devqubit_engine.storage.errors import StorageError as _StorageError
    from devqubit_engine.uec.errors import (
        EnvelopeValidationError as _EnvelopeValidationError,
    )
    from devqubit_engine.uec.errors import MissingEnvelopeError as _MissingEnvelopeError

    # Re-export with type hints
    StorageError = _StorageError
    ObjectNotFoundError = _ObjectNotFoundError
    RunNotFoundError = _RunNotFoundError
    QueryParseError = _QueryParseError
    MissingEnvelopeError = _MissingEnvelopeError
    EnvelopeValidationError = _EnvelopeValidationError


_LAZY_IMPORTS = {
    "StorageError": ("devqubit_engine.storage.errors", "StorageError"),
    "ObjectNotFoundError": ("devqubit_engine.storage.errors", "ObjectNotFoundError"),
    "RunNotFoundError": ("devqubit_engine.storage.errors", "RunNotFoundError"),
    "QueryParseError": ("devqubit_engine.query", "QueryParseError"),
    "MissingEnvelopeError": ("devqubit_engine.uec.errors", "MissingEnvelopeError"),
    "EnvelopeValidationError": (
        "devqubit_engine.uec.errors",
        "EnvelopeValidationError",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
