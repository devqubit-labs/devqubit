# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison result types.

This module provides typed dataclasses for comparison, drift, and
verification results. These are the primary data structures returned
by comparison and verification operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from devqubit_engine.compare.types import (
    FormatOptions,
    ProgramMatchMode,
    ProgramMatchStatus,
    VerdictCategory,
)


if TYPE_CHECKING:
    from devqubit_engine.circuit.summary import CircuitDiff
    from devqubit_engine.utils.distributions import NoiseContext


# =============================================================================
# Drift results
# =============================================================================


@dataclass
class MetricDrift:
    """
    Drift information for a single calibration metric.

    Attributes
    ----------
    metric : str
        Metric name (e.g., "median_t1_us").
    value_a : float or None
        Value from baseline snapshot.
    value_b : float or None
        Value from candidate snapshot.
    delta : float or None
        Absolute difference (b - a).
    percent_change : float or None
        Percentage change relative to value_a.
    threshold : float or None
        Threshold used for significance determination.
    significant : bool
        Whether drift exceeds threshold.
    """

    metric: str
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None
    percent_change: float | None = None
    threshold: float | None = None
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {"metric": self.metric, "significant": self.significant}
        if self.value_a is not None:
            d["a"] = self.value_a
        if self.value_b is not None:
            d["b"] = self.value_b
        if self.delta is not None:
            d["delta"] = self.delta
        if self.percent_change is not None:
            d["percent_change"] = self.percent_change
        if self.threshold is not None:
            d["threshold"] = self.threshold
        return d

    def __repr__(self) -> str:
        sig = "!" if self.significant else ""
        return f"MetricDrift({self.metric}{sig}, {self.value_a} -> {self.value_b})"


@dataclass
class DriftResult:
    """
    Complete drift analysis result.

    Attributes
    ----------
    has_calibration_data : bool
        Whether calibration data was available in both snapshots.
    calibration_time_a : str or None
        Baseline calibration timestamp.
    calibration_time_b : str or None
        Candidate calibration timestamp.
    metrics : list of MetricDrift
        Per-metric drift analysis.
    significant_drift : bool
        Whether any metric exceeds its threshold.
    """

    has_calibration_data: bool = False
    calibration_time_a: str | None = None
    calibration_time_b: str | None = None
    metrics: list[MetricDrift] = field(default_factory=list)
    significant_drift: bool = False

    @property
    def calibration_time_changed(self) -> bool:
        """Whether calibration timestamps differ."""
        return self.calibration_time_a != self.calibration_time_b

    @property
    def top_drifts(self) -> list[MetricDrift]:
        """Metrics with significant drift, sorted by magnitude."""
        sig = [m for m in self.metrics if m.significant]
        sig.sort(
            key=lambda m: abs(m.percent_change) if m.percent_change is not None else 0,
            reverse=True,
        )
        return sig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_calibration_data": self.has_calibration_data,
            "calibration_time_changed": self.calibration_time_changed,
            "significant_drift": self.significant_drift,
            "calibration_times": {
                "a": self.calibration_time_a,
                "b": self.calibration_time_b,
            },
            "metrics": {m.metric: m.to_dict() for m in self.metrics},
            "top_drifts": [
                {
                    "metric": m.metric,
                    "delta": m.delta,
                    "percent_change": m.percent_change,
                }
                for m in self.top_drifts[:5]
            ],
        }

    def __repr__(self) -> str:
        if not self.has_calibration_data:
            return "DriftResult(no_data)"
        status = "significant" if self.significant_drift else "within_threshold"
        return f"DriftResult({status}, {len(self.metrics)} metrics)"


# =============================================================================
# Verdict
# =============================================================================


@dataclass
class Verdict:
    """
    Regression verdict with root-cause analysis.

    Attributes
    ----------
    category : VerdictCategory
        Primary suspected cause.
    summary : str
        One-liner explanation.
    evidence : dict
        Supporting data and numbers.
    action : str
        Suggested next step.
    contributing_factors : list of str
        All detected factors.
    """

    category: VerdictCategory
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    contributing_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "action": self.action,
            "contributing_factors": self.contributing_factors,
        }

    def __repr__(self) -> str:
        return f"Verdict({self.category.value})"


# =============================================================================
# Program comparison
# =============================================================================


@dataclass
class ProgramComparison:
    """
    Detailed program comparison result.

    Captures exact (digest), structural (structural_hash), and parametric
    (parametric_hash) matching to support different verification policies.
    Also includes executed hashes for physical circuits (post-compilation).

    Attributes
    ----------
    has_programs : bool
        True if at least one run has program artifacts.
    exact_match : bool
        True if artifact digests are identical (byte-for-byte match).
    structural_match : bool
        True if structural_hash values match (same structure).
    parametric_match : bool
        True if parametric_hash values match (same structure AND params).
    digests_a : list of str
        Program artifact digests from baseline.
    digests_b : list of str
        Program artifact digests from candidate.
    circuit_hash_a : str or None
        Structural hash (structural_hash) from baseline.
    circuit_hash_b : str or None
        Structural hash (structural_hash) from candidate.
    parametric_hash_a : str or None
        Parametric hash from baseline.
    parametric_hash_b : str or None
        Parametric hash from candidate.
    executed_structural_hash_a : str or None
        Executed structural hash from baseline (physical circuit).
    executed_structural_hash_b : str or None
        Executed structural hash from candidate (physical circuit).
    executed_parametric_hash_a : str or None
        Executed parametric hash from baseline (physical circuit).
    executed_parametric_hash_b : str or None
        Executed parametric hash from candidate (physical circuit).
    executed_structural_match : bool
        True if executed structural hashes match.
    executed_parametric_match : bool
        True if executed parametric hashes match.
    hash_available : bool
        True if program hashes were available for comparison.

    Notes
    -----
    Hash semantics:

    - ``structural_hash``: Structure only. Same = same circuit template.
    - ``parametric_hash``: Structure + params. Same = identical execution.
    - ``executed_*_hash``: Hashes for physical (compiled) circuits.

    For manual runs, hashes are unavailable and ``hash_available=False``.
    When no programs are captured, ``has_programs=False`` and match fields
    should be ignored.
    """

    has_programs: bool = False
    exact_match: bool = False
    structural_match: bool = False
    parametric_match: bool = False
    digests_a: list[str] = field(default_factory=list)
    digests_b: list[str] = field(default_factory=list)
    circuit_hash_a: str | None = None
    circuit_hash_b: str | None = None
    parametric_hash_a: str | None = None
    parametric_hash_b: str | None = None
    executed_structural_hash_a: str | None = None
    executed_structural_hash_b: str | None = None
    executed_parametric_hash_a: str | None = None
    executed_parametric_hash_b: str | None = None
    executed_structural_match: bool = False
    executed_parametric_match: bool = False
    hash_available: bool = True

    @property
    def status(self) -> ProgramMatchStatus:
        """
        Get detailed match status.

        Returns
        -------
        ProgramMatchStatus
            Detailed status based on hash comparison.
        """
        if not self.has_programs:
            return ProgramMatchStatus.HASH_UNAVAILABLE
        if not self.hash_available:
            return ProgramMatchStatus.HASH_UNAVAILABLE
        if self.parametric_match:
            return ProgramMatchStatus.FULL_MATCH
        if self.structural_match:
            return ProgramMatchStatus.STRUCTURAL_MATCH_PARAM_MISMATCH
        return ProgramMatchStatus.STRUCTURAL_MISMATCH

    def matches(self, mode: ProgramMatchMode) -> bool:
        """
        Check if programs match according to specified mode.

        Parameters
        ----------
        mode : ProgramMatchMode
            Matching mode to use.

        Returns
        -------
        bool
            True if programs match according to the mode.
            Returns False if no programs are available.
        """
        if not self.has_programs:
            return False
        if mode == ProgramMatchMode.EXACT:
            return self.exact_match
        elif mode == ProgramMatchMode.STRUCTURAL:
            return self.structural_match
        else:  # EITHER
            return self.exact_match or self.structural_match

    @property
    def structural_only_match(self) -> bool:
        """True if structural matches but exact doesn't (different param values)."""
        return self.has_programs and self.structural_match and not self.exact_match

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "has_programs": self.has_programs,
            "exact_match": self.exact_match,
            "structural_match": self.structural_match,
            "parametric_match": self.parametric_match,
            "structural_only_match": self.structural_only_match,
            "status": self.status.value,
            "hash_available": self.hash_available,
            "digests_a": self.digests_a,
            "digests_b": self.digests_b,
            "circuit_hash_a": self.circuit_hash_a,
            "circuit_hash_b": self.circuit_hash_b,
            "parametric_hash_a": self.parametric_hash_a,
            "parametric_hash_b": self.parametric_hash_b,
        }
        if self.executed_structural_hash_a or self.executed_structural_hash_b:
            d["executed_structural_hash_a"] = self.executed_structural_hash_a
            d["executed_structural_hash_b"] = self.executed_structural_hash_b
            d["executed_structural_match"] = self.executed_structural_match
        if self.executed_parametric_hash_a or self.executed_parametric_hash_b:
            d["executed_parametric_hash_a"] = self.executed_parametric_hash_a
            d["executed_parametric_hash_b"] = self.executed_parametric_hash_b
            d["executed_parametric_match"] = self.executed_parametric_match
        return d

    def __repr__(self) -> str:
        return f"ProgramComparison({self.status.value})"


# =============================================================================
# Comparison result
# =============================================================================


@dataclass
class ComparisonResult:
    """
    Complete comparison result between two runs.

    Captures all dimensions of comparison including metadata, parameters,
    metrics, program artifacts, device drift, and result distributions.

    Attributes
    ----------
    run_id_a : str
        Baseline run ID.
    run_id_b : str
        Candidate run ID.
    fingerprint_a : str or None
        Baseline run fingerprint.
    fingerprint_b : str or None
        Candidate run fingerprint.
    identical : bool
        True if runs are identical across all dimensions.
    metadata : dict
        Metadata comparison (project, adapter, backend).
    params : dict
        Parameter comparison result.
    metrics : dict
        Metrics comparison result.
    program : ProgramComparison
        Detailed program comparison with exact and structural matching.
    device_drift : DriftResult or None
        Device calibration drift analysis.
    counts_a : dict or None
        Baseline measurement counts.
    counts_b : dict or None
        Candidate measurement counts.
    tvd : float or None
        Total Variation Distance between distributions.
    noise_context : NoiseContext or None
        Statistical noise analysis.
    circuit_diff : CircuitDiff or None
        Semantic circuit comparison.
    warnings : list of str
        Non-fatal warnings.
    """

    run_id_a: str
    run_id_b: str
    fingerprint_a: str | None = None
    fingerprint_b: str | None = None
    identical: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    program: ProgramComparison = field(default_factory=ProgramComparison)
    device_drift: DriftResult | None = None
    counts_a: dict[str, int] | None = None
    counts_b: dict[str, int] | None = None
    tvd: float | None = None
    noise_context: NoiseContext | None = None
    circuit_diff: CircuitDiff | None = None
    warnings: list[str] = field(default_factory=list)

    def program_matches(self, mode: ProgramMatchMode = ProgramMatchMode.EITHER) -> bool:
        """
        Check if programs match according to specified mode.

        Parameters
        ----------
        mode : ProgramMatchMode, default=EITHER
            Matching mode to use.

        Returns
        -------
        bool
            True if programs match according to the mode.
        """
        return self.program.matches(mode)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        status = "identical" if self.identical else "differ"
        return f"<ComparisonResult {self.run_id_a} vs {self.run_id_b}: {status}>"

    def format(self, opts: FormatOptions | None = None) -> str:
        """
        Format as human-readable text report.

        Parameters
        ----------
        opts : FormatOptions, optional
            Formatting options.

        Returns
        -------
        str
            Formatted text report.
        """
        from devqubit_engine.compare._formatting import format_comparison_result

        return format_comparison_result(self, opts)

    def format_json(self, indent: int = 2) -> str:
        """Format as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def format_summary(self) -> str:
        """Format as brief one-line summary."""
        if self.identical:
            return f"Identical: {self.run_id_a} == {self.run_id_b}"

        issues: list[str] = []
        if not self.params.get("match"):
            issues.append("params")
        if not self.metrics.get("match"):
            issues.append("metrics")
        if not self.program.matches(ProgramMatchMode.EITHER):
            issues.append("program")
        if self.device_drift and self.device_drift.significant_drift:
            issues.append("drift")
        if self.tvd is not None and self.tvd > 0.05:
            issues.append(f"TVD={self.tvd:.3f}")
        if self.circuit_diff and not self.circuit_diff.match:
            issues.append("circuit")

        return f"Differ ({', '.join(issues)}): {self.run_id_a} vs {self.run_id_b}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "run_a": self.run_id_a,
            "run_b": self.run_id_b,
            "identical": self.identical,
            "fingerprints": {"a": self.fingerprint_a, "b": self.fingerprint_b},
            "metadata": self.metadata,
            "params": self.params,
            "metrics": self.metrics,
            "program": self.program.to_dict(),
        }

        if self.device_drift:
            result["device_drift"] = self.device_drift.to_dict()
        else:
            result["device_drift"] = {
                "has_calibration_data": False,
                "significant_drift": False,
            }

        if self.tvd is not None:
            result["tvd"] = self.tvd

        if self.counts_a and self.counts_b:
            result["shots"] = {
                "a": sum(self.counts_a.values()),
                "b": sum(self.counts_b.values()),
            }

        if self.noise_context:
            result["noise_context"] = self.noise_context.to_dict()

        if self.circuit_diff:
            result["circuit_diff"] = self.circuit_diff.to_dict()

        if self.warnings:
            result["warnings"] = self.warnings

        return result


# =============================================================================
# Verification result
# =============================================================================


@dataclass
class VerifyResult:
    """
    Result of verification against a baseline.

    Attributes
    ----------
    ok : bool
        True if all policy checks passed.
    failures : list of str
        Human-readable failure messages.
    comparison : ComparisonResult or None
        Full comparison result.
    baseline_run_id : str or None
        Baseline run ID.
    candidate_run_id : str or None
        Candidate run ID.
    duration_ms : float
        Verification time in milliseconds.
    verdict : Verdict or None
        Root-cause verdict when verification fails.
    """

    ok: bool
    failures: list[str] = field(default_factory=list)
    comparison: ComparisonResult | None = None
    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    duration_ms: float = 0.0
    verdict: Verdict | None = None

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"<VerifyResult {self.candidate_run_id}: {status}>"

    def format(self, opts: FormatOptions | None = None) -> str:
        """
        Format as human-readable text report.

        Parameters
        ----------
        opts : FormatOptions, optional
            Formatting options.

        Returns
        -------
        str
            Formatted text report.
        """
        from devqubit_engine.compare._formatting import format_verify_result

        return format_verify_result(self, opts)

    def format_json(self, indent: int = 2) -> str:
        """Format as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def format_summary(self) -> str:
        """Format as brief one-line summary."""
        if self.ok:
            return (
                f"PASS: {self.candidate_run_id} verified against {self.baseline_run_id}"
            )
        return f"FAIL: {self.candidate_run_id} - {'; '.join(self.failures[:2])}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "ok": self.ok,
            "failures": self.failures,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "duration_ms": self.duration_ms,
        }
        if self.comparison:
            d["comparison"] = self.comparison.to_dict()
        if self.verdict:
            d["verdict"] = self.verdict.to_dict()
        return d
