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

Verification (High-Level)
-------------------------
>>> from devqubit import verify_baseline
>>> result = verify_baseline("candidate_run_id", project="my_project")
>>> if result.ok:
...     print("Verification passed!")
>>> else:
...     print(result.verdict.summary)

Verification (Custom Policy)
----------------------------
>>> from devqubit import verify_baseline
>>> from devqubit.compare import VerifyPolicy, ProgramMatchMode
>>> policy = VerifyPolicy(
...     program_match_mode=ProgramMatchMode.STRUCTURAL,
...     noise_factor=1.2,
... )
>>> result = verify_baseline("candidate_run_id", project="my_project", policy=policy)

Run Navigation
--------------
>>> from devqubit.runs import list_runs, search_runs, get_baseline
>>> runs = list_runs(project="my_project", limit=10)
>>> high_fidelity = search_runs("metric.fidelity > 0.95")
>>> baseline = get_baseline("my_project")

Submodules
----------
- devqubit.runs: Run navigation and baseline management
- devqubit.compare: Comparison types (ProgramMatchMode, Verdict, etc.)
- devqubit.ci: CI/CD integration (JUnit, GitHub annotations)
- devqubit.bundle: Run packaging utilities
- devqubit.config: Configuration management
- devqubit.uec: UEC snapshot schemas
- devqubit.storage: Storage backends
- devqubit.adapters: SDK adapter extension API
- devqubit.errors: Public exception types
- devqubit.ui: Web UI (optional, requires devqubit[ui])
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any


__all__ = [
    # Version
    "__version__",
    # Core tracking
    "Run",
    "track",
    "wrap_backend",
    # Comparison
    "diff",
    # Verification
    "verify_baseline",
    # Bundle
    "pack_run",
    "unpack_bundle",
    "Bundle",
    # Config
    "Config",
    "get_config",
    "set_config",
]


try:
    __version__ = version("devqubit")
except PackageNotFoundError:
    __version__ = "0.0.0"


if TYPE_CHECKING:
    from devqubit_engine.bundle.pack import pack_run, unpack_bundle
    from devqubit_engine.bundle.reader import Bundle
    from devqubit_engine.compare.diff import diff
    from devqubit_engine.compare.results import VerifyResult
    from devqubit_engine.compare.verify import VerifyPolicy
    from devqubit_engine.config import Config, get_config, set_config
    from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
    from devqubit_engine.tracking.record import RunRecord
    from devqubit_engine.tracking.run import Run, track, wrap_backend


_LAZY_IMPORTS = {
    # Core tracking
    "Run": ("devqubit_engine.tracking.run", "Run"),
    "track": ("devqubit_engine.tracking.run", "track"),
    "wrap_backend": ("devqubit_engine.tracking.run", "wrap_backend"),
    # Comparison
    "diff": ("devqubit_engine.compare.diff", "diff"),
    # Bundle
    "pack_run": ("devqubit_engine.bundle.pack", "pack_run"),
    "unpack_bundle": ("devqubit_engine.bundle.pack", "unpack_bundle"),
    "Bundle": ("devqubit_engine.bundle.reader", "Bundle"),
    # Config
    "Config": ("devqubit_engine.config", "Config"),
    "get_config": ("devqubit_engine.config", "get_config"),
    "set_config": ("devqubit_engine.config", "set_config"),
}


def verify_baseline(
    candidate: str | Path | RunRecord,
    *,
    project: str,
    policy: VerifyPolicy | dict[str, Any] | None = None,
    store: ObjectStoreProtocol | None = None,
    registry: RegistryProtocol | None = None,
    promote_on_pass: bool = False,
) -> "VerifyResult":
    """
    Verify a candidate run against the stored baseline for a project.

    This is the recommended high-level API for CI/CD verification.
    It automatically loads the candidate run, baseline, and storage
    backends from the global configuration.

    Parameters
    ----------
    candidate : str, Path, or RunRecord
        Candidate to verify. Can be:
        - A run ID (str)
        - A path to a bundle file (Path or str ending in .zip)
        - A RunRecord instance (already loaded)
    project : str
        Project name to look up baseline for.
    policy : VerifyPolicy or dict or None, optional
        Verification policy configuration. Uses defaults if not provided.
        Can be a VerifyPolicy instance or a dict with policy options.
    store : ObjectStoreProtocol or None, optional
        Object store to use. If None, uses the default from config.
        Required when candidate is a RunRecord from a different workspace.
    registry : RegistryProtocol or None, optional
        Registry to use. If None, uses the default from config.
    promote_on_pass : bool, default=False
        If True and verification passes, promote candidate to new baseline.

    Returns
    -------
    VerifyResult
        Verification result with ``ok`` status, ``failures``, ``comparison``,
        and ``verdict`` (root-cause analysis if failed).

    Raises
    ------
    ValueError
        If no baseline is set for the project and ``allow_missing_baseline``
        is False in the policy.
    RunNotFoundError
        If the candidate run does not exist.

    Examples
    --------
    Basic verification:

    >>> from devqubit import verify_baseline
    >>> result = verify_baseline("candidate_run_id", project="my_project")
    >>> if result.ok:
    ...     print("Verification passed!")
    ... else:
    ...     print(f"Failed: {result.failures}")
    ...     print(f"Root cause: {result.verdict.summary}")

    With custom policy:

    >>> from devqubit import verify_baseline
    >>> from devqubit.compare import VerifyPolicy, ProgramMatchMode
    >>> policy = VerifyPolicy(
    ...     program_match_mode=ProgramMatchMode.STRUCTURAL,
    ...     noise_factor=1.2,
    ...     allow_missing_baseline=True,
    ... )
    >>> result = verify_baseline(
    ...     "candidate_run_id",
    ...     project="my_project",
    ...     policy=policy,
    ...     promote_on_pass=True,
    ... )

    With bundle file:

    >>> result = verify_baseline(
    ...     "experiment.zip",
    ...     project="my_project",
    ... )

    CI/CD integration:

    >>> from devqubit import verify_baseline
    >>> from devqubit.ci import write_junit
    >>> result = verify_baseline("candidate_run_id", project="my_project")
    >>> write_junit(result, "results.xml")
    >>> assert result.ok, f"Verification failed: {result.failures}"

    Cross-workspace verification (explicit stores):

    >>> from devqubit.storage import create_store, create_registry
    >>> store = create_store("s3://my-bucket/objects")
    >>> registry = create_registry("s3://my-bucket")
    >>> result = verify_baseline(
    ...     "candidate_run_id",
    ...     project="my_project",
    ...     store=store,
    ...     registry=registry,
    ... )
    """
    from devqubit_engine.bundle.reader import Bundle, is_bundle_path
    from devqubit_engine.compare.verify import (
        verify_against_baseline as _verify_against_baseline,
    )
    from devqubit_engine.config import get_config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.types import ArtifactRef
    from devqubit_engine.tracking.record import RunRecord

    # Get default store/registry from config if not provided
    if store is None or registry is None:
        cfg = get_config()
        if store is None:
            store = create_store(config=cfg)
        if registry is None:
            registry = create_registry(config=cfg)

    # Handle different candidate types
    candidate_record: RunRecord
    candidate_store = store

    # Case 1: Already a RunRecord
    if isinstance(candidate, RunRecord):
        candidate_record = candidate

    # Case 2: Bundle file path
    elif is_bundle_path(candidate):
        with Bundle(Path(candidate)) as bundle:
            record_dict = bundle.run_record
            artifacts = [
                ArtifactRef.from_dict(a)
                for a in record_dict.get("artifacts", [])
                if isinstance(a, dict)
            ]
            candidate_record = RunRecord(record=record_dict, artifacts=artifacts)
            # Use bundle's store for artifacts
            candidate_store = bundle.store

            return _verify_against_baseline(
                candidate_record,
                project=project,
                store=candidate_store,
                registry=registry,
                policy=policy,
                promote_on_pass=promote_on_pass,
            )

    # Case 3: Run ID string
    else:
        candidate_record = registry.load(str(candidate))

    return _verify_against_baseline(
        candidate_record,
        project=project,
        store=candidate_store,
        registry=registry,
        policy=policy,
        promote_on_pass=promote_on_pass,
    )


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
