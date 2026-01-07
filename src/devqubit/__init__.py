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

Comparison
----------
>>> from devqubit import diff
>>> result = diff("run_id_a", "run_id_b")
>>> print(result.identical)

Verification
------------
>>> from devqubit import verify_against_baseline
>>> from devqubit.compare import VerifyPolicy
>>> result = verify_against_baseline(candidate, project="my_project", policy=VerifyPolicy())
>>> assert result.ok

Snapshots
---------
>>> from devqubit.snapshot import ExecutionEnvelope, DeviceSnapshot
>>> envelope = ExecutionEnvelope(device=device_snapshot, ...)

UI
------------
>>> from devqubit import run_server
>>> run_server(port=8080)

Submodules
----------
- devqubit.compare: Comparison utilities (ProgramMatchMode)
- devqubit.ci: CI/CD integration (JUnit, GitHub annotations)
- devqubit.bundle: Run packaging utilities
- devqubit.config: Configuration management
- devqubit.snapshot: UEC snapshot schemas
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any


__all__ = [
    # Version
    "__version__",
    # Core tracking
    "Run",
    "track",
    "wrap_backend",
    # Models
    "RunRecord",
    "ArtifactRef",
    # Comparison
    "diff",
    "diff_runs",
    # Verification
    "verify",
    "verify_against_baseline",
    # Bundle
    "pack_run",
    "unpack_bundle",
    "Bundle",
    # Storage
    "create_store",
    "create_registry",
    # Config
    "Config",
    "get_config",
    "set_config",
    # UI
    "run_server",
]


try:
    __version__ = version("devqubit")
except PackageNotFoundError:
    __version__ = "0.0.0"


if TYPE_CHECKING:
    from devqubit_engine.bundle.pack import pack_run, unpack_bundle
    from devqubit_engine.bundle.reader import Bundle
    from devqubit_engine.compare.diff import diff, diff_runs
    from devqubit_engine.compare.verify import verify, verify_against_baseline
    from devqubit_engine.core.config import Config, get_config, set_config
    from devqubit_engine.core.record import ArtifactRef, RunRecord
    from devqubit_engine.core.tracker import Run, track, wrap_backend
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_ui.app import run_server


_LAZY_IMPORTS = {
    "Run": ("devqubit_engine.core.tracker", "Run"),
    "track": ("devqubit_engine.core.tracker", "track"),
    "wrap_backend": ("devqubit_engine.core.tracker", "wrap_backend"),
    "RunRecord": ("devqubit_engine.core.record", "RunRecord"),
    "ArtifactRef": ("devqubit_engine.core.models", "ArtifactRef"),
    "diff": ("devqubit_engine.compare.diff", "diff"),
    "diff_runs": ("devqubit_engine.compare.diff", "diff_runs"),
    "verify": ("devqubit_engine.compare.verify", "verify"),
    "verify_against_baseline": (
        "devqubit_engine.compare.verify",
        "verify_against_baseline",
    ),
    "pack_run": ("devqubit_engine.bundle.pack", "pack_run"),
    "unpack_bundle": ("devqubit_engine.bundle.pack", "unpack_bundle"),
    "Bundle": ("devqubit_engine.bundle.reader", "Bundle"),
    "create_store": ("devqubit_engine.storage.factory", "create_store"),
    "create_registry": ("devqubit_engine.storage.factory", "create_registry"),
    "Config": ("devqubit_engine.core.config", "Config"),
    "get_config": ("devqubit_engine.core.config", "get_config"),
    "set_config": ("devqubit_engine.core.config", "set_config"),
    "run_server": ("devqubit_ui.app", "run_server"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
