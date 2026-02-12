# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Local-first experiment tracking for quantum computing.

``devqubit`` captures circuits, backend state, and configuration so that
quantum experiments are reproducible, comparable, and easy to share.
Data is accessible via the Python API, CLI, or Web UI.

Getting Started
---------------
Open a tracked run, wrap your backend, and execute as usual.  The SDK
adapter captures circuits, device snapshots, and results automatically:

>>> from devqubit import track
>>> with track(project="bell-state", run_name="baseline-v1") as run:
...     backend = run.wrap(AerSimulator())
...     job = backend.run(circuit, shots=1000)
...     run.log_metric("p00", counts.get("00", 0) / 1000)

Manual logging works the same way without a backend wrapper:

>>> with track(project="my-experiment") as run:
...     run.log_param("shots", 1000)
...     run.log_metric("fidelity", 0.95)

Comparing and Verifying Runs
----------------------------
>>> from devqubit.compare import diff, verify_baseline
>>> result = diff("baseline-v1", "experiment-v2", project="bell-state")
>>> print(result.tvd)
0.023

>>> result = verify_baseline("nightly-run", project="bell-state")
>>> assert result.ok, result.verdict.summary

Navigating Runs
---------------
>>> from devqubit.runs import list_runs, get_baseline
>>> runs = list_runs(project="bell-state", limit=10)
>>> baseline = get_baseline("bell-state")

Bundling and Sharing
--------------------
>>> from devqubit.bundle import pack_run, Bundle
>>> pack_run("run_id", "experiment.zip")
>>> with Bundle("experiment.zip") as b:
...     print(b.run_id)

Public Submodules
-----------------
:mod:`devqubit.runs`
    Run navigation, search, baseline management, and DataFrame export.
:mod:`devqubit.compare`
    Run comparison (``diff``), baseline verification, drift detection.
:mod:`devqubit.bundle`
    Portable run packaging (pack / unpack / inspect).
:mod:`devqubit.ci`
    CI/CD integration helpers (JUnit reports, GitHub annotations).
:mod:`devqubit.config`
    Configuration management.
:mod:`devqubit.uec`
    Uniform Execution Contract snapshot schemas.
:mod:`devqubit.storage`
    Storage backend access (advanced).
:mod:`devqubit.adapters`
    SDK adapter plugin system.
:mod:`devqubit.errors`
    Public exception hierarchy.
:mod:`devqubit.ui`
    Web UI server (requires ``devqubit[ui]``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any


__all__ = [
    "__version__",
    # Core tracking
    "Run",
    "track",
    "wrap_backend",
    # Configuration
    "Config",
    "get_config",
    "set_config",
]


try:
    __version__ = version("devqubit")
except PackageNotFoundError:
    __version__ = "0.0.0"


if TYPE_CHECKING:
    from devqubit_engine.config import Config, get_config, set_config
    from devqubit_engine.tracking.run import Run, track, wrap_backend


_LAZY_IMPORTS = {
    # Core tracking
    "Run": ("devqubit_engine.tracking.run", "Run"),
    "track": ("devqubit_engine.tracking.run", "track"),
    "wrap_backend": ("devqubit_engine.tracking.run", "wrap_backend"),
    # Config
    "Config": ("devqubit_engine.config", "Config"),
    "get_config": ("devqubit_engine.config", "get_config"),
    "set_config": ("devqubit_engine.config", "set_config"),
}


def __getattr__(name: str) -> Any:
    """Lazy-import handler for module-level attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available public attributes (enables IDE autocomplete)."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
