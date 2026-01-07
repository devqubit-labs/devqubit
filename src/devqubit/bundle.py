# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run packaging and sharing utilities.

Basic packing/unpacking is available from the main module:

>>> from devqubit import pack_run, unpack_bundle, Bundle

This submodule provides additional utilities:
- list_bundle_contents: Inspect bundle without unpacking
- replay: Re-execute circuits from a bundle

Packing
-------
>>> from devqubit import pack_run
>>> pack_run("run_id", "experiment.zip")

Unpacking
---------
>>> from devqubit import unpack_bundle
>>> unpack_bundle("experiment.zip")

Reading Bundles
---------------
>>> from devqubit import Bundle
>>> with Bundle("experiment.zip") as bundle:
...     print(bundle.run_id)
...     print(bundle.run_record)

Listing Contents
----------------
>>> from devqubit.bundle import list_bundle_contents
>>> contents = list_bundle_contents("experiment.zip")

Replay
------
>>> from devqubit.bundle import replay
>>> result = replay("experiment.zip", backend="aer_simulator")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "pack_run",
    "unpack_bundle",
    "Bundle",
    "list_bundle_contents",
    "replay",
]


if TYPE_CHECKING:
    from devqubit_engine.bundle.pack import (
        list_bundle_contents,
        pack_run,
        unpack_bundle,
    )
    from devqubit_engine.bundle.reader import Bundle
    from devqubit_engine.bundle.replay import replay


_LAZY_IMPORTS = {
    "pack_run": ("devqubit_engine.bundle.pack", "pack_run"),
    "unpack_bundle": ("devqubit_engine.bundle.pack", "unpack_bundle"),
    "list_bundle_contents": ("devqubit_engine.bundle.pack", "list_bundle_contents"),
    "Bundle": ("devqubit_engine.bundle.reader", "Bundle"),
    "replay": ("devqubit_engine.bundle.replay", "replay"),
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
