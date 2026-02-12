# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run comparison, verification, and diff utilities.

This module is the primary entry point for comparing quantum experiment
runs and verifying candidates against baselines in CI/CD pipelines.

Comparing Two Runs
------------------
>>> from devqubit.compare import diff
>>> result = diff("baseline-v1", "experiment-v2", project="bell-state")
>>> print(result.identical)       # True if everything matches
>>> print(result.tvd)             # Total variation distance
>>> print(result.program.structural_match)

Baseline Verification
---------------------
>>> from devqubit.compare import verify_baseline, VerifyPolicy
>>> result = verify_baseline(
...     "nightly-run",
...     project="bell-state",
...     policy=VerifyPolicy(noise_factor=1.2),
... )
>>> assert result.ok

Drift Detection
---------------
>>> from devqubit.compare import DriftThresholds
>>> result = diff("run_a", "run_b", thresholds=DriftThresholds(t1_us=0.15))
>>> if result.device_drift and result.device_drift.significant_drift:
...     print("Significant calibration drift detected")

Formatting
----------
>>> from devqubit.compare import FormatOptions
>>> print(result.format(FormatOptions(max_drifts=3, show_evidence=False)))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any


__all__ = [
    # Core functions
    "diff",
    "verify_baseline",
    # Result types
    "ComparisonResult",
    "VerifyResult",
    # Policy configuration
    "VerifyPolicy",
    "ProgramMatchMode",
    # Drift analysis
    "DriftResult",
    "DriftThresholds",
    "MetricDrift",
    # Formatting
    "FormatOptions",
    # Program comparison
    "ProgramComparison",
]


if TYPE_CHECKING:
    from devqubit_engine.compare.diff import diff
    from devqubit_engine.compare.drift import DriftThresholds
    from devqubit_engine.compare.results import (
        ComparisonResult,
        DriftResult,
        FormatOptions,
        MetricDrift,
        ProgramComparison,
        ProgramMatchMode,
        VerifyResult,
    )
    from devqubit_engine.compare.verify import VerifyPolicy
    from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
    from devqubit_engine.tracking.record import RunRecord


_LAZY_IMPORTS = {
    # Core comparison function
    "diff": ("devqubit_engine.compare.diff", "diff"),
    # Result types
    "ComparisonResult": ("devqubit_engine.compare.results", "ComparisonResult"),
    "VerifyResult": ("devqubit_engine.compare.results", "VerifyResult"),
    # Policy
    "VerifyPolicy": ("devqubit_engine.compare.verify", "VerifyPolicy"),
    "ProgramMatchMode": ("devqubit_engine.compare.results", "ProgramMatchMode"),
    # Drift
    "DriftResult": ("devqubit_engine.compare.results", "DriftResult"),
    "DriftThresholds": ("devqubit_engine.compare.drift", "DriftThresholds"),
    "MetricDrift": ("devqubit_engine.compare.results", "MetricDrift"),
    # Formatting
    "FormatOptions": ("devqubit_engine.compare.results", "FormatOptions"),
    # Program comparison
    "ProgramComparison": ("devqubit_engine.compare.results", "ProgramComparison"),
}


def verify_baseline(
    candidate: str | Path | "RunRecord",
    *,
    project: str,
    policy: "VerifyPolicy | dict[str, Any] | None" = None,
    store: "ObjectStoreProtocol | None" = None,
    registry: "RegistryProtocol | None" = None,
    promote_on_pass: bool = False,
) -> "VerifyResult":
    """
    Verify a candidate run against the project baseline.

    This is the recommended high-level API for CI/CD verification.  It
    resolves the candidate run, loads the baseline, and delegates to the
    engine's verification logic.

    Parameters
    ----------
    candidate : str | Path | RunRecord
        The run to verify.  Accepted forms:

        * **Run ID** — a ULID string (e.g. ``"01HXYZ..."``).
        * **Run name** — resolved within *project*.
        * **Bundle path** — a ``.zip`` file path or :class:`~pathlib.Path`.
        * **RunRecord** — an already-loaded record instance.
    project : str
        Project whose stored baseline the candidate is verified against.
    policy : VerifyPolicy | dict | None, optional
        Verification policy.  Accepts a :class:`VerifyPolicy` instance, a
        plain ``dict`` of policy options, or *None* for defaults.
    store : ObjectStoreProtocol | None, optional
        Object store override.  Defaults to the global configuration.
    registry : RegistryProtocol | None, optional
        Registry override.  Defaults to the global configuration.
    promote_on_pass : bool, default False
        Promote the candidate to the new baseline when verification passes.

    Returns
    -------
    VerifyResult
        Result with :attr:`~VerifyResult.ok` status, :attr:`~VerifyResult.failures`,
        and :attr:`~VerifyResult.comparison`.

    Raises
    ------
    ValueError
        If no baseline is set for *project* and ``allow_missing_baseline``
        is ``False`` in the policy.
    RunNotFoundError
        If the candidate run cannot be found.

    Examples
    --------
    Basic CI gate:

    >>> result = verify_baseline("nightly-run", project="bell-state")
    >>> assert result.ok

    Custom policy with noise tolerance:

    >>> from devqubit.compare import VerifyPolicy, ProgramMatchMode
    >>> policy = VerifyPolicy(
    ...     program_match_mode=ProgramMatchMode.STRUCTURAL,
    ...     noise_factor=1.2,
    ...     allow_missing_baseline=True,
    ... )
    >>> result = verify_baseline(
    ...     "nightly-run", project="bell-state",
    ...     policy=policy, promote_on_pass=True,
    ... )

    Verify from a bundle file:

    >>> result = verify_baseline("experiment.zip", project="bell-state")

    Combine with JUnit output for CI:

    >>> from devqubit.ci import write_junit
    >>> result = verify_baseline("nightly-run", project="bell-state")
    >>> write_junit(result, "results.xml")
    """
    from devqubit_engine.bundle.reader import Bundle, is_bundle_path
    from devqubit_engine.compare.verify import (
        verify_against_baseline as _verify_against_baseline,
    )
    from devqubit_engine.config import get_config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.types import ArtifactRef
    from devqubit_engine.tracking.record import RunRecord, resolve_run_id

    if store is None or registry is None:
        cfg = get_config()
        if store is None:
            store = create_store(config=cfg)
        if registry is None:
            registry = create_registry(config=cfg)

    candidate_record: RunRecord
    candidate_store = store
    # Keep bundle context alive so its temporary store remains accessible
    # during verification.
    bundle_ctx = None

    try:
        if isinstance(candidate, RunRecord):
            candidate_record = candidate

        elif is_bundle_path(candidate):
            bundle_ctx = Bundle(Path(candidate))
            bundle_ctx.__enter__()
            record_dict = bundle_ctx.run_record
            artifacts = [
                ArtifactRef.from_dict(a)
                for a in record_dict.get("artifacts", [])
                if isinstance(a, dict)
            ]
            candidate_record = RunRecord(
                record=record_dict,
                artifacts=artifacts,
            )
            candidate_store = bundle_ctx.store

        else:
            # Resolve name to ID if needed
            run_id = resolve_run_id(str(candidate), project, registry)
            candidate_record = registry.load(run_id)

        return _verify_against_baseline(
            candidate_record,
            project=project,
            store=candidate_store,
            registry=registry,
            policy=policy,
            promote_on_pass=promote_on_pass,
        )
    finally:
        if bundle_ctx is not None:
            bundle_ctx.__exit__(None, None, None)


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
