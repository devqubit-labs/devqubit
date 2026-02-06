# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison, verification, and diff utilities.

This module provides the primary API for comparing runs and verifying
experiments against baselines.

Diff
----
>>> from devqubit.compare import diff
>>> result = diff("run_a", "run_b")
>>> print(result.identical)

>>> # By run name (requires project)
>>> result = diff("baseline-v1", "experiment-v2", project="bell_state")

Verification
------------
>>> from devqubit.compare import verify_baseline
>>> result = verify_baseline("candidate_run_id", project="my_project")
>>> if result.ok:
...     print("Verification passed!")
... else:
...     print(result.verdict.summary)

Custom Policy
-------------
>>> from devqubit.compare import verify_baseline, VerifyPolicy, ProgramMatchMode
>>> policy = VerifyPolicy(
...     program_match_mode=ProgramMatchMode.STRUCTURAL,
...     noise_factor=1.2,
... )
>>> result = verify_baseline(
...     "candidate_run_id",
...     project="my_project",
...     policy=policy,
... )

Verdicts (Root-Cause Analysis)
------------------------------
>>> from devqubit.compare import Verdict, VerdictCategory
>>> if not result.ok:
...     verdict = result.verdict
...     print(f"Category: {verdict.category}")
...     print(f"Action: {verdict.action}")

Drift Detection
---------------
>>> from devqubit.compare import diff, DriftThresholds
>>> thresholds = DriftThresholds(t1_us=0.15, t2_us=0.15)
>>> result = diff("run_a", "run_b", thresholds=thresholds)
>>> if result.device_drift and result.device_drift.significant_drift:
...     print("Significant calibration drift detected!")

Formatting
----------
>>> from devqubit.compare import FormatOptions
>>> opts = FormatOptions(max_drifts=3, show_evidence=False)
>>> print(result.format(opts))
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
    # Verdict types
    "Verdict",
    "VerdictCategory",
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
        Verdict,
        VerdictCategory,
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
    # Verdicts
    "Verdict": ("devqubit_engine.compare.results", "Verdict"),
    "VerdictCategory": ("devqubit_engine.compare.results", "VerdictCategory"),
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
    Verify a candidate run against the stored baseline for a project.

    This is the recommended high-level API for CI/CD verification.
    It automatically loads the candidate run, baseline, and storage
    backends from the global configuration.

    Parameters
    ----------
    candidate : str, Path, or RunRecord
        Candidate to verify. Can be:
        - A run ID (str)
        - A run name within the project (str)
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

    >>> from devqubit.compare import verify_baseline
    >>> result = verify_baseline("candidate_run_id", project="my_project")
    >>> if result.ok:
    ...     print("Verification passed!")
    ... else:
    ...     print(f"Failed: {result.failures}")
    ...     print(f"Root cause: {result.verdict.summary}")

    With custom policy:

    >>> from devqubit.compare import verify_baseline, VerifyPolicy, ProgramMatchMode
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

    >>> result = verify_baseline("experiment.zip", project="my_project")

    CI/CD integration:

    >>> from devqubit.compare import verify_baseline
    >>> from devqubit.ci import write_junit
    >>> result = verify_baseline("candidate_run_id", project="my_project")
    >>> write_junit(result, "results.xml")
    >>> assert result.ok, f"Verification failed: {result.failures}"
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
