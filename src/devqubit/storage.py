# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Storage backend access (advanced).

This module provides direct access to storage backends for advanced use
cases. Most users should use the high-level API in :mod:`devqubit` and
:mod:`devqubit.runs` instead.

Basic Configuration
-------------------
>>> from devqubit.storage import create_store, create_registry
>>> store = create_store()  # Uses global config
>>> registry = create_registry()

Custom Configuration
--------------------
>>> from devqubit import Config
>>> from devqubit.storage import create_store, create_registry
>>> config = Config(root_dir="/custom/path")
>>> store = create_store(config=config)
>>> registry = create_registry(config=config)

Protocols
---------
For creating custom storage backends, implement these protocols:

>>> from devqubit.storage import ObjectStoreProtocol, RegistryProtocol
>>> class MyStore:
...     def put_bytes(self, data: bytes) -> str: ...
...     def get_bytes(self, digest: str) -> bytes: ...
...     # ... etc.

Types
-----
>>> from devqubit.storage import ArtifactRef, RunSummary, BaselineInfo
>>> ref = ArtifactRef(
...     kind="qiskit.qpy.circuits",
...     digest="sha256:abc...",
...     media_type="application/x-qpy",
...     role="program",
... )
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
