# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Configuration management.

The most common configuration symbols (``Config``, ``get_config``,
``set_config``) are re-exported from the top-level :mod:`devqubit`
package.  This submodule adds advanced utilities for credential
redaction and explicit config lifecycle control.

Custom Configuration
--------------------
>>> from devqubit import Config, set_config
>>> from pathlib import Path
>>> set_config(Config(root_dir=Path("/custom/workspace"), capture_git=True))

Redaction
---------
>>> from devqubit.config import RedactionConfig
>>> config = Config(
...     redaction=RedactionConfig(
...         enabled=True,
...         patterns=["MY_SECRET_.*", "CUSTOM_TOKEN"],
...     ),
... )

Resetting
---------
>>> from devqubit.config import reset_config
>>> reset_config()  # next get_config() reloads from environment
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "Config",
    "RedactionConfig",
    "get_config",
    "set_config",
    "reset_config",
    "load_config",
]


if TYPE_CHECKING:
    from devqubit_engine.config import (
        Config,
        RedactionConfig,
        get_config,
        load_config,
        reset_config,
        set_config,
    )


_LAZY_IMPORTS = {
    "Config": ("devqubit_engine.config", "Config"),
    "RedactionConfig": ("devqubit_engine.config", "RedactionConfig"),
    "get_config": ("devqubit_engine.config", "get_config"),
    "set_config": ("devqubit_engine.config", "set_config"),
    "reset_config": ("devqubit_engine.config", "reset_config"),
    "load_config": ("devqubit_engine.config", "load_config"),
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
