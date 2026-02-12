# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Storage backend access (advanced).

Most users should use the high-level APIs in :mod:`devqubit` and
:mod:`devqubit.runs`. This module is for advanced use cases that
require direct storage access or custom backend implementations.

Factory Functions
-----------------
>>> from devqubit.storage import create_store, create_registry
>>> store = create_store()       # uses global config
>>> registry = create_registry()

>>> store = create_store("s3://bucket/devqubit/objects")
>>> registry = create_registry("s3://bucket/devqubit")

Custom Backends
---------------
Implement the protocol interfaces for custom storage:

>>> from devqubit.storage import ObjectStoreProtocol, RegistryProtocol

Data Types
----------
>>> from devqubit.storage import ArtifactRef, RunSummary, BaselineInfo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    # Factory functions
    "create_store",
    "create_registry",
    # Protocols (for custom implementations)
    "ObjectStoreProtocol",
    "RegistryProtocol",
    # Data types
    "ArtifactRef",
    "RunSummary",
    "BaselineInfo",
]


if TYPE_CHECKING:
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.types import (
        ArtifactRef,
        BaselineInfo,
        ObjectStoreProtocol,
        RegistryProtocol,
        RunSummary,
    )


_LAZY_IMPORTS = {
    "create_store": ("devqubit_engine.storage.factory", "create_store"),
    "create_registry": ("devqubit_engine.storage.factory", "create_registry"),
    "ObjectStoreProtocol": ("devqubit_engine.storage.types", "ObjectStoreProtocol"),
    "RegistryProtocol": ("devqubit_engine.storage.types", "RegistryProtocol"),
    "ArtifactRef": ("devqubit_engine.storage.types", "ArtifactRef"),
    "RunSummary": ("devqubit_engine.storage.types", "RunSummary"),
    "BaselineInfo": ("devqubit_engine.storage.types", "BaselineInfo"),
}


def __getattr__(name: str) -> Any:
    """Lazy-import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available public attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
