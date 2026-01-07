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
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from devqubit_engine.circuit.summary import CircuitDiff
from devqubit_engine.utils.distributions import NoiseContext


logger = logging.getLogger(__name__)


class ProgramMatchMode(str, Enum):
    """
    Program matching mode for verification.

    Attributes
    ----------
    EXACT : str
        Require artifact digest equality (strict reproducibility).
        Use when you need byte-for-byte identical circuits.
    STRUCTURAL : str
        Require circuit_hash (structure) equality (variational-friendly).
        Use for Variational Circuits where parameter values differ but structure is same.
    EITHER : str
        Pass if exact OR structural match (recommended default).
        Detects true program changes without breaking variational workflows.
    """

    EXACT = "exact"
    STRUCTURAL = "structural"
    EITHER = "either"


class VerdictCategory(str, Enum):
    """
    Root-cause categories for verification regression.

    Attributes
    ----------
    PROGRAM_CHANGED : str
        The quantum program (circuit) has changed.
    COMPILER_CHANGED : str
        Same circuit compiled differently (depth/2Q gates changed).
    DEVICE_DRIFT : str
        Device calibration has drifted significantly.
    SHOT_NOISE : str
        Difference is consistent with statistical sampling noise.
    MIXED : str
        Multiple contributing factors detected.
    UNKNOWN : str
        No clear root cause identified.
    """

    PROGRAM_CHANGED = "PROGRAM_CHANGED"
    COMPILER_CHANGED = "COMPILER_CHANGED"
    DEVICE_DRIFT = "DEVICE_DRIFT"
    SHOT_NOISE = "SHOT_NOISE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass
class FormatOptions:
    """
    Formatting options for text reports.

    Attributes
    ----------
    max_drifts : int
        Maximum drift metrics to display. Default is 5.
    max_circuit_changes : int
        Maximum circuit changes to display. Default is 10.
    max_param_changes : int
        Maximum parameter changes to display. Default is 10.
    max_metric_changes : int
        Maximum metric changes to display. Default is 10.
    show_evidence : bool
        Show detailed evidence in verdicts. Default is True.
    width : int
        Line width for text output. Default is 70.
    """

    max_drifts: int = 5
    max_circuit_changes: int = 10
    max_param_changes: int = 10
    max_metric_changes: int = 10
    show_evidence: bool = True
    width: int = 70


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
        """Return string representation."""
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
        """Return string representation."""
        if not self.has_calibration_data:
            return "DriftResult(no_data)"
        status = "significant" if self.significant_drift else "within_threshold"
        return f"DriftResult({status}, {len(self.metrics)} metrics)"


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
        """Return string representation."""
        return f"Verdict({self.category.value})"


@dataclass
class ProgramComparison:
    """
    Detailed program comparison result.

    Captures both exact (digest) and structural (circuit_hash) matching
    to support different verification policies.

    Attributes
    ----------
    exact_match : bool
        True if artifact digests are identical (byte-for-byte match).
    structural_match : bool
        True if circuit_hash values match (same structure, possibly different params).
    digests_a : list of str
        Program artifact digests from baseline.
    digests_b : list of str
        Program artifact digests from candidate.
    circuit_hash_a : str or None
        Circuit structure hash from baseline.
    circuit_hash_b : str or None
        Circuit structure hash from candidate.
    """

    exact_match: bool = False
    structural_match: bool = False
    digests_a: list[str] = field(default_factory=list)
    digests_b: list[str] = field(default_factory=list)
    circuit_hash_a: str | None = None
    circuit_hash_b: str | None = None

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
        """
        if mode == ProgramMatchMode.EXACT:
            return self.exact_match
        elif mode == ProgramMatchMode.STRUCTURAL:
            return self.structural_match
        else:  # EITHER
            return self.exact_match or self.structural_match

    @property
    def structural_only_match(self) -> bool:
        """True if structural matches but exact doesn't (different param values)."""
        return self.structural_match and not self.exact_match

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exact_match": self.exact_match,
            "structural_match": self.structural_match,
            "structural_only_match": self.structural_only_match,
            "digests_a": self.digests_a,
            "digests_b": self.digests_b,
            "circuit_hash_a": self.circuit_hash_a,
            "circuit_hash_b": self.circuit_hash_b,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        if self.exact_match:
            return "ProgramComparison(exact_match)"
        elif self.structural_match:
            return "ProgramComparison(structural_match)"
        else:
            return "ProgramComparison(no_match)"


def _format_header(title: str, width: int = 70, char: str = "=") -> list[str]:
    """Format a section header."""
    return [char * width, title, char * width]


def _format_change(key: str, change: dict[str, Any]) -> str:
    """Format a single parameter/metric change."""
    val_a = change.get("a")
    val_b = change.get("b")

    # Calculate percentage change for numeric values
    pct_str = ""
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        if val_a != 0:
            pct = ((val_b - val_a) / abs(val_a)) * 100
            sign = "+" if pct > 0 else ""
            pct_str = f" ({sign}{pct:.1f}%)"

    return f"    {key}: {val_a} -> {val_b}{pct_str}"


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
        """Return formatted text report."""
        return self.format()

    def __repr__(self) -> str:
        """Return string representation."""
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
        if opts is None:
            opts = FormatOptions()

        lines = _format_header("RUN COMPARISON", opts.width)
        lines.extend(
            [
                f"Baseline:  {self.run_id_a}",
                f"Candidate: {self.run_id_b}",
                "",
                f"Overall: {'✓ IDENTICAL' if self.identical else '✗ DIFFER'}",
            ]
        )

        # Metadata section
        if self.metadata:
            lines.extend(["", "-" * opts.width, "Metadata", "-" * opts.width])

            project_match = self.metadata.get("project_match", True)
            if project_match:
                lines.append("  project: ✓")
            else:
                a = self.metadata.get("project_a", "?")
                b = self.metadata.get("project_b", "?")
                lines.append("  project: ✗")
                lines.append(f"    {a} -> {b}")

            backend_match = self.metadata.get("backend_match", True)
            if backend_match:
                lines.append("  backend: ✓")
            else:
                a = self.metadata.get("backend_a", "?")
                b = self.metadata.get("backend_b", "?")
                lines.append("  backend: ✗")
                lines.append(f"    {a} -> {b}")

        # Program section
        lines.extend(["", "-" * opts.width, "Program", "-" * opts.width])
        if self.program.exact_match:
            prog_status = "✓ Match (exact)"
        elif self.program.structural_match:
            prog_status = "✓ Match (structural)"
        else:
            prog_status = "✗ Differ"
        lines.append(f"  {prog_status}")

        # Parameters section
        if self.params:
            lines.extend(["", "-" * opts.width, "Parameters", "-" * opts.width])
            if self.params.get("match", False):
                lines.append("  ✓ Match")
            else:
                changed = self.params.get("changed", {})
                added = self.params.get("added", {})
                removed = self.params.get("removed", {})
                if changed:
                    lines.append("  Changed:")
                    for i, (k, v) in enumerate(changed.items()):
                        if i >= opts.max_param_changes:
                            lines.append(f"    ... and {len(changed) - i} more")
                            break
                        lines.append(_format_change(k, v))
                if removed:
                    lines.append("  Only in baseline:")
                    for i, (k, v) in enumerate(removed.items()):
                        if i >= opts.max_param_changes:
                            lines.append(f"    ... and {len(removed) - i} more")
                            break
                        lines.append(f"    {k}: {v}")
                if added:
                    lines.append("  Only in candidate:")
                    for i, (k, v) in enumerate(added.items()):
                        if i >= opts.max_param_changes:
                            lines.append(f"    ... and {len(added) - i} more")
                            break
                        lines.append(f"    {k}: {v}")

        # Metrics section
        if self.metrics:
            lines.extend(["", "-" * opts.width, "Metrics", "-" * opts.width])
            if self.metrics.get("match", True):
                lines.append("  ✓ Match")
            else:
                changed = self.metrics.get("changed", {})
                added = self.metrics.get("added", {})
                removed = self.metrics.get("removed", {})
                if changed:
                    lines.append("  Changed:")
                    for i, (k, v) in enumerate(changed.items()):
                        if i >= opts.max_metric_changes:
                            lines.append(f"    ... and {len(changed) - i} more")
                            break
                        lines.append(_format_change(k, v))
                if removed:
                    lines.append("  Only in baseline:")
                    for i, (k, v) in enumerate(removed.items()):
                        if i >= opts.max_metric_changes:
                            lines.append(f"    ... and {len(removed) - i} more")
                            break
                        lines.append(f"    {k}: {v}")
                if added:
                    lines.append("  Only in candidate:")
                    for i, (k, v) in enumerate(added.items()):
                        if i >= opts.max_metric_changes:
                            lines.append(f"    ... and {len(added) - i} more")
                            break
                        lines.append(f"    {k}: {v}")

        # Device drift section
        if self.device_drift and self.device_drift.has_calibration_data:
            lines.extend(["", "-" * opts.width, "Device Calibration", "-" * opts.width])
            if self.device_drift.calibration_time_changed:
                lines.append("  Calibration times differ:")
                lines.append(f"    Baseline:  {self.device_drift.calibration_time_a}")
                lines.append(f"    Candidate: {self.device_drift.calibration_time_b}")
            if self.device_drift.significant_drift:
                lines.append("  ✗ Significant drift detected:")
                for i, m in enumerate(self.device_drift.top_drifts):
                    if i >= opts.max_drifts:
                        lines.append(
                            f"    ... and {len(self.device_drift.top_drifts) - i} more"
                        )
                        break
                    pct = f"{m.percent_change:+.1f}%" if m.percent_change else "N/A"
                    lines.append(f"    {m.metric}: {m.value_a} -> {m.value_b} ({pct})")
            else:
                lines.append("  ✓ Drift within thresholds")

        # Results section (TVD + noise)
        if self.tvd is not None or self.noise_context:
            lines.extend(["", "-" * opts.width, "Results", "-" * opts.width])
            if self.tvd is not None:
                lines.append(f"  TVD: {self.tvd:.6f}")
            if self.noise_context:
                lines.append(
                    f"  Expected noise: {self.noise_context.expected_noise:.6f}"
                )
                lines.append(f"  Noise ratio: {self.noise_context.noise_ratio:.2f}x")
                lines.append(f"  Interpretation: {self.noise_context.interpretation()}")

        # Circuit section
        lines.extend(["", "-" * opts.width, "Circuit", "-" * opts.width])
        if self.circuit_diff is None:
            lines.append("  N/A (not captured)")
        elif self.circuit_diff.match:
            lines.append("  ✓ Match")
        else:
            lines.append("  ✗ Differ:")
            if self.circuit_diff.changes:
                if isinstance(self.circuit_diff.changes, dict):
                    for i, (key, change) in enumerate(
                        self.circuit_diff.changes.items()
                    ):
                        if i >= opts.max_circuit_changes:
                            lines.append(
                                f"    ... and {len(self.circuit_diff.changes) - i} more"
                            )
                            break
                        lines.append(
                            f"    {key}: {change.get('a')} -> {change.get('b')}"
                        )
                else:
                    for i, change in enumerate(self.circuit_diff.changes):
                        if i >= opts.max_circuit_changes:
                            lines.append(
                                f"    ... and {len(self.circuit_diff.changes) - i} more"
                            )
                            break
                        lines.append(f"    {change}")

        # Warnings section
        if self.warnings:
            lines.extend(["", "-" * opts.width, "Warnings", "-" * opts.width])
            for w in self.warnings:
                lines.append(f"  ! {w}")

        lines.extend(["", "=" * opts.width])
        return "\n".join(lines)

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
        """Return formatted text report."""
        return self.format()

    def __repr__(self) -> str:
        """Return string representation."""
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
        if opts is None:
            opts = FormatOptions()

        lines = _format_header("VERIFICATION RESULT", opts.width)
        lines.extend(
            [
                f"Baseline:  {self.baseline_run_id or 'N/A'}",
                f"Candidate: {self.candidate_run_id}",
                f"Duration:  {self.duration_ms:.1f}ms",
                "",
                f"Status: {'✓ PASSED' if self.ok else '✗ FAILED'}",
            ]
        )

        # Failures section
        if not self.ok and self.failures:
            lines.extend(["", "-" * opts.width, "Failures", "-" * opts.width])
            for failure in self.failures:
                lines.append(f"  ✗ {failure}")

        # Results section
        if self.comparison:
            comp = self.comparison
            if comp.tvd is not None or comp.noise_context:
                lines.extend(["", "-" * opts.width, "Results", "-" * opts.width])
                if comp.tvd is not None:
                    lines.append(f"  TVD: {comp.tvd:.6f}")
                if comp.noise_context:
                    lines.append(
                        f"  Expected noise: {comp.noise_context.expected_noise:.6f}"
                    )
                    lines.append(
                        f"  Noise ratio: {comp.noise_context.noise_ratio:.2f}x"
                    )
                    lines.append(
                        f"  Interpretation: {comp.noise_context.interpretation()}"
                    )

            # Circuit section (only if differs)
            if comp.circuit_diff and not comp.circuit_diff.match:
                lines.extend(["", "-" * opts.width, "Circuit", "-" * opts.width])
                lines.append("  ✗ Differ:")
                if comp.circuit_diff.changes:
                    if isinstance(comp.circuit_diff.changes, dict):
                        for i, (key, change) in enumerate(
                            comp.circuit_diff.changes.items()
                        ):
                            if i >= opts.max_circuit_changes:
                                lines.append(
                                    f"    ... and {len(comp.circuit_diff.changes) - i} more"
                                )
                                break
                            lines.append(
                                f"    {key}: {change.get('a')} -> {change.get('b')}"
                            )
                    else:
                        for i, change in enumerate(comp.circuit_diff.changes):
                            if i >= opts.max_circuit_changes:
                                lines.append(
                                    f"    ... and {len(comp.circuit_diff.changes) - i} more"
                                )
                                break
                            lines.append(f"    {change}")

            # Device drift section (only if significant)
            if comp.device_drift and comp.device_drift.significant_drift:
                lines.extend(
                    ["", "-" * opts.width, "Device Calibration", "-" * opts.width]
                )
                lines.append("  ✗ Significant drift detected:")
                for i, m in enumerate(comp.device_drift.top_drifts):
                    if i >= opts.max_drifts:
                        lines.append(
                            f"    ... and {len(comp.device_drift.top_drifts) - i} more"
                        )
                        break
                    pct = f"{m.percent_change:+.1f}%" if m.percent_change else "N/A"
                    lines.append(f"    {m.metric}: {m.value_a} -> {m.value_b} ({pct})")

        # Verdict section
        if self.verdict:
            lines.extend(
                ["", "-" * opts.width, "Root Cause Analysis", "-" * opts.width]
            )
            lines.append(f"  Category: {self.verdict.category.value}")
            lines.append(f"  Summary: {self.verdict.summary}")
            lines.append(f"  Action: {self.verdict.action}")
            if self.verdict.contributing_factors:
                lines.append(
                    f"  Factors: {', '.join(self.verdict.contributing_factors)}"
                )

        lines.extend(["", "=" * opts.width])
        return "\n".join(lines)

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
