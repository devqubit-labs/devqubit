# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit: Experiment tracking for quantum computing.

Quick Start
-----------
>>> from devqubit import track
>>> with track(project="my_experiment") as run:
...     run.log_param("shots", 1000)
...     run.log_metric("fidelity", 0.95)

With Backend Wrapping
---------------------
>>> from devqubit import track
>>> with track(project="bell_state") as run:
...     backend = run.wrap(AerSimulator())
...     job = backend.run(circuit, shots=1000)

Comparison & Verification
-------------------------
>>> from devqubit.compare import diff, verify_baseline
>>> result = diff("run_id_a", "run_id_b")
>>> print(result.identical)

>>> result = verify_baseline("candidate_run_id", project="my_project")
>>> if result.ok:
...     print("Verification passed!")

Run Navigation
--------------
>>> from devqubit.runs import list_runs, search_runs, get_baseline
>>> runs = list_runs(project="my_project", limit=10)
>>> baseline = get_baseline("my_project")

Bundling
--------
>>> from devqubit.bundle import pack_run, unpack_bundle, Bundle
>>> pack_run("run_id", "experiment.zip")
>>> with Bundle("experiment.zip") as bundle:
...     print(bundle.run_id)

Submodules
----------
- devqubit.runs: Run navigation and baseline management
- devqubit.compare: Comparison, verification, and diff utilities
- devqubit.bundle: Run packaging utilities
- devqubit.ci: CI/CD integration (JUnit, GitHub annotations)
- devqubit.config: Configuration management
- devqubit.uec: UEC snapshot schemas
- devqubit.storage: Storage backends (advanced)
- devqubit.adapters: SDK adapter extension API
- devqubit.errors: Public exception types
- devqubit.ui: Web UI (optional, requires devqubit[ui])
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
    """Lazy import handler for module-level attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes for autocomplete."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
