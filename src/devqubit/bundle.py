# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Portable run packaging and sharing.

Bundles are self-contained ZIP archives that include a run record, all
referenced artifacts, and a manifest with SHA-256 integrity digests.
They can be shared, archived, imported into another workspace, or used
directly for offline verification.

Packing a Run
-------------
>>> from devqubit.bundle import pack_run
>>> result = pack_run("baseline-v1", "experiment.zip", project="bell-state")
>>> print(f"Packed {result.object_count} objects ({result.total_bytes} bytes)")

Unpacking into a Workspace
--------------------------
>>> from devqubit.bundle import unpack_bundle
>>> result = unpack_bundle("experiment.zip")
>>> print(f"Imported {result.object_count} new objects")

Inspecting Without Importing
----------------------------
>>> from devqubit.bundle import Bundle
>>> with Bundle("experiment.zip") as b:
...     print(b.run_id, b.run_record["project"])

Listing Contents
----------------
>>> from devqubit.bundle import list_bundle_contents
>>> for entry in list_bundle_contents("experiment.zip"):
...     print(entry)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "pack_run",
    "unpack_bundle",
    "Bundle",
    "list_bundle_contents",
    "PackResult",
    "UnpackResult",
]


if TYPE_CHECKING:
    from devqubit_engine.bundle.pack import (
        PackResult,
        UnpackResult,
        list_bundle_contents,
        pack_run,
        unpack_bundle,
    )
    from devqubit_engine.bundle.reader import Bundle


_LAZY_IMPORTS = {
    "pack_run": ("devqubit_engine.bundle.pack", "pack_run"),
    "unpack_bundle": ("devqubit_engine.bundle.pack", "unpack_bundle"),
    "list_bundle_contents": ("devqubit_engine.bundle.pack", "list_bundle_contents"),
    "Bundle": ("devqubit_engine.bundle.reader", "Bundle"),
    "PackResult": ("devqubit_engine.bundle.pack", "PackResult"),
    "UnpackResult": ("devqubit_engine.bundle.pack", "UnpackResult"),
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
