# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison and verification types.

This module provides types and utilities for comparing runs and
configuring verification policies.

Result Types
------------
>>> from devqubit import diff
>>> from devqubit.compare import ComparisonResult, VerifyResult
>>> result = diff("run_a", "run_b")
>>> assert isinstance(result, ComparisonResult)
>>> print(result.identical)

Verification Policy
-------------------
>>> from devqubit.compare import VerifyPolicy, ProgramMatchMode
>>> policy = VerifyPolicy(
...     params_must_match=True,
...     program_match_mode=ProgramMatchMode.STRUCTURAL,
...     noise_factor=1.2,
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
>>> from devqubit.compare import DriftThresholds, DriftResult
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

from typing import TYPE_CHECKING, Any


__all__ = [
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


_LAZY_IMPORTS = {
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
