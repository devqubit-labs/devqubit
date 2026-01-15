# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Web UI for experiment visualization (optional).

This module provides the web-based user interface for browsing and
visualizing devqubit experiments. It requires the optional ``devqubit[ui]``
extra to be installed.

Installation
------------
To use the UI, install with the ui extra::

    pip install devqubit[ui]

Starting the Server
-------------------
>>> from devqubit.ui import run_server
>>> run_server(port=8080)
Starting devqubit UI at http://localhost:8080

Or from the command line::

    devqubit ui --port 8080

Note
----
If you see an ImportError when importing from this module, install
the UI dependencies with::

    pip install devqubit[ui]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "run_server",
]


if TYPE_CHECKING:
    from devqubit_ui.app import run_server


_LAZY_IMPORTS = {
    "run_server": ("devqubit_ui.app", "run_server"),
}


def __getattr__(name: str) -> Any:
    """
    Lazy import handler with helpful error messages for optional dependencies.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            module = __import__(module_path, fromlist=[attr_name])
        except ImportError as e:
            raise ImportError(
                f"The devqubit UI requires additional dependencies.\n"
                f"Install with: pip install devqubit[ui]\n"
                f"Original error: {e}"
            ) from e
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
