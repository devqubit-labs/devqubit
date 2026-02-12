# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
SDK adapter plugin system.

Adapters provide SDK-specific integration for automatic circuit capture,
device snapshots, and result logging.  Each installed adapter package
(e.g. ``devqubit-qiskit``) registers itself via a Python entry point and
is discovered automatically when :meth:`Run.wrap` is called.

Discovery
---------
>>> from devqubit.adapters import list_available_adapters
>>> list_available_adapters()
['braket', 'cirq', 'cudaq', 'pennylane', 'qiskit', 'qiskit_runtime']

>>> from devqubit.adapters import resolve_adapter
>>> adapter = resolve_adapter(my_backend)   # auto-detect by executor type

Implementing a Custom Adapter
-----------------------------
1. Implement :class:`AdapterProtocol`.
2. Register via entry point::

       [project.entry-points."devqubit.adapters"]
       my_sdk = "my_package.adapter:MyAdapter"

>>> from devqubit.adapters import AdapterProtocol
>>> class MyAdapter:
...     name = "my_sdk"
...     def supports_executor(self, executor) -> bool: ...
...     def describe_executor(self, executor) -> dict: ...
...     def wrap_executor(self, executor, tracker, **kw): ...

Debugging
---------
>>> from devqubit.adapters import adapter_load_errors
>>> for err in adapter_load_errors():
...     print(f"{err.entry_point}: {err.message}")
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
