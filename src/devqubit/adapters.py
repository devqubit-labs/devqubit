# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Adapter plugin system for SDK integration.

This module provides the public API for adapter discovery and the
protocol that adapters must implement.

Available Adapters
------------------
>>> from devqubit.adapters import list_available_adapters
>>> adapters = list_available_adapters()
>>> print("Installed adapters:", adapters)

Creating an Adapter
-------------------
Adapters are discovered via Python entry points. To create a custom adapter:

1. Implement the AdapterProtocol interface
2. Register via entry point in setup.py or pyproject.toml::

    [project.entry-points."devqubit.adapters"]
    my_sdk = "my_package.adapter:MyAdapter"

Adapter Protocol
----------------
>>> from devqubit.adapters import AdapterProtocol

>>> class MyAdapter:
...     name = "my_sdk"
...
...     def supports_executor(self, executor) -> bool:
...         return isinstance(executor, MySdkBackend)
...
...     def describe_executor(self, executor) -> dict:
...         return {"name": executor.name, "type": "simulator"}
...
...     def wrap_executor(self, executor, tracker):
...         return WrappedExecutor(executor, tracker)

Debugging Adapter Issues
------------------------
>>> from devqubit.adapters import adapter_load_errors
>>> for error in adapter_load_errors():
...     print(f"Failed to load {error.entry_point}: {error.message}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    # Protocol
    "AdapterProtocol",
    # Discovery
    "list_available_adapters",
    "adapter_load_errors",
    "get_adapter_by_name",
    "resolve_adapter",
    # Error type
    "AdapterLoadError",
    # Advanced
    "load_adapters",
    "clear_adapter_cache",
]


if TYPE_CHECKING:
    from devqubit_engine.adapters import (
        AdapterLoadError,
        AdapterProtocol,
        adapter_load_errors,
        clear_adapter_cache,
        get_adapter_by_name,
        list_available_adapters,
        load_adapters,
        resolve_adapter,
    )


_LAZY_IMPORTS = {
    "AdapterProtocol": ("devqubit_engine.adapters", "AdapterProtocol"),
    "AdapterLoadError": ("devqubit_engine.adapters", "AdapterLoadError"),
    "list_available_adapters": ("devqubit_engine.adapters", "list_available_adapters"),
    "adapter_load_errors": ("devqubit_engine.adapters", "adapter_load_errors"),
    "get_adapter_by_name": ("devqubit_engine.adapters", "get_adapter_by_name"),
    "resolve_adapter": ("devqubit_engine.adapters", "resolve_adapter"),
    "load_adapters": ("devqubit_engine.adapters", "load_adapters"),
    "clear_adapter_cache": ("devqubit_engine.adapters", "clear_adapter_cache"),
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
