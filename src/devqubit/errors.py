# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Public exception types.

This module exposes all public exceptions that may be raised by devqubit
operations.  Users can catch these to handle specific error conditions.

Exception Hierarchy
-------------------
All exceptions inherit from :class:`DevQubitError`::

    DevQubitError
    ├── StorageError
    │   ├── ObjectNotFoundError
    │   └── RunNotFoundError
    ├── RegistryError
    ├── QueryParseError
    ├── CircuitError
    │   ├── LoaderError
    │   └── SerializerError
    ├── MissingEnvelopeError
    └── EnvelopeValidationError

Catch-all pattern:

>>> from devqubit.errors import DevQubitError, RunNotFoundError
>>> try:
...     run = load_run("nonexistent")
... except RunNotFoundError as e:
...     print(f"Run not found: {e.run_id}")
... except DevQubitError:
...     print("Other devqubit error")

Specific error handling:

>>> from devqubit.errors import RunNotFoundError
>>> try:
...     run = load_run("nonexistent")
... except RunNotFoundError as e:
...     print(f"Run not found: {e.run_id}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    # Base exception
    "DevQubitError",
    # Storage errors
    "RegistryError",
    "StorageError",
    "ObjectNotFoundError",
    "RunNotFoundError",
    # Query errors
    "QueryParseError",
    # Circuit errors
    "CircuitError",
    "LoaderError",
    "SerializerError",
    # Envelope errors
    "MissingEnvelopeError",
    "EnvelopeValidationError",
]


if TYPE_CHECKING:
    from devqubit_engine.circuit.registry import CircuitError as _CircuitError
    from devqubit_engine.circuit.registry import LoaderError as _LoaderError
    from devqubit_engine.circuit.registry import SerializerError as _SerializerError
    from devqubit_engine.errors import DevQubitError as _DevQubitError
    from devqubit_engine.query import QueryParseError as _QueryParseError
    from devqubit_engine.storage.errors import (
        ObjectNotFoundError as _ObjectNotFoundError,
    )
    from devqubit_engine.storage.errors import RegistryError as _RegistryError
    from devqubit_engine.storage.errors import RunNotFoundError as _RunNotFoundError
    from devqubit_engine.storage.errors import StorageError as _StorageError
    from devqubit_engine.uec.errors import (
        EnvelopeValidationError as _EnvelopeValidationError,
    )
    from devqubit_engine.uec.errors import MissingEnvelopeError as _MissingEnvelopeError

    DevQubitError = _DevQubitError
    RegistryError = _RegistryError
    StorageError = _StorageError
    ObjectNotFoundError = _ObjectNotFoundError
    RunNotFoundError = _RunNotFoundError
    QueryParseError = _QueryParseError
    CircuitError = _CircuitError
    LoaderError = _LoaderError
    SerializerError = _SerializerError
    MissingEnvelopeError = _MissingEnvelopeError
    EnvelopeValidationError = _EnvelopeValidationError


_LAZY_IMPORTS = {
    "DevQubitError": ("devqubit_engine.errors", "DevQubitError"),
    "RegistryError": ("devqubit_engine.storage.errors", "RegistryError"),
    "StorageError": ("devqubit_engine.storage.errors", "StorageError"),
    "ObjectNotFoundError": ("devqubit_engine.storage.errors", "ObjectNotFoundError"),
    "RunNotFoundError": ("devqubit_engine.storage.errors", "RunNotFoundError"),
    "QueryParseError": ("devqubit_engine.query", "QueryParseError"),
    "CircuitError": ("devqubit_engine.circuit.registry", "CircuitError"),
    "LoaderError": ("devqubit_engine.circuit.registry", "LoaderError"),
    "SerializerError": ("devqubit_engine.circuit.registry", "SerializerError"),
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
