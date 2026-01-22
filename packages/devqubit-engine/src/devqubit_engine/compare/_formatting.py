# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Text formatting utilities for comparison results.

This module provides human-readable text formatting for comparison and
verification results. The formatting is separated from result dataclasses
to keep them focused on data representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from devqubit_engine.compare.types import FormatOptions


if TYPE_CHECKING:
    from devqubit_engine.compare.results import ComparisonResult, VerifyResult


# ASCII symbols for cross-platform compatibility
_SYM_OK = "[OK]"
_SYM_FAIL = "[X]"
_SYM_WARN = "[!]"


def _format_header(title: str, width: int = 70, char: str = "=") -> list[str]:
    """Format a section header."""
    return [char * width, title, char * width]


def _format_change(key: str, change: dict[str, Any]) -> str:
    """Format a single parameter/metric change with percentage."""
    val_a = change.get("a")
    val_b = change.get("b")

    pct_str = ""
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        if val_a != 0:
            pct = ((val_b - val_a) / abs(val_a)) * 100
            sign = "+" if pct > 0 else ""
            pct_str = f" ({sign}{pct:.1f}%)"

    return f"    {key}: {val_a} => {val_b}{pct_str}"


def _format_dict_changes(
    diff: dict[str, Any],
    opts: FormatOptions,
    label_changed: str = "Changed",
    label_removed: str = "Only in baseline",
    label_added: str = "Only in candidate",
) -> list[str]:
    """Format dictionary comparison changes."""
    lines: list[str] = []
    max_changes = opts.max_param_changes

    changed = diff.get("changed", {})
    added = diff.get("added", {})
    removed = diff.get("removed", {})

    if changed:
        lines.append(f"  {label_changed}:")
        for i, (k, v) in enumerate(changed.items()):
            if i >= max_changes:
                lines.append(f"    ... and {len(changed) - i} more")
                break
            lines.append(_format_change(k, v))

    if removed:
        lines.append(f"  {label_removed}:")
        for i, (k, v) in enumerate(removed.items()):
            if i >= max_changes:
                lines.append(f"    ... and {len(removed) - i} more")
                break
            lines.append(f"    {k}: {v}")

    if added:
        lines.append(f"  {label_added}:")
        for i, (k, v) in enumerate(added.items()):
            if i >= max_changes:
                lines.append(f"    ... and {len(added) - i} more")
                break
            lines.append(f"    {k}: {v}")

    return lines


def _format_circuit_diff(circuit_diff, opts: FormatOptions) -> list[str]:
    """Format circuit diff changes."""
    lines: list[str] = []

    if circuit_diff.changes:
        if isinstance(circuit_diff.changes, dict):
            for i, (key, change) in enumerate(circuit_diff.changes.items()):
                if i >= opts.max_circuit_changes:
                    lines.append(f"    ... and {len(circuit_diff.changes) - i} more")
                    break
                lines.append(f"    {key}: {change.get('a')} => {change.get('b')}")
        else:
            for i, change in enumerate(circuit_diff.changes):
                if i >= opts.max_circuit_changes:
                    lines.append(f"    ... and {len(circuit_diff.changes) - i} more")
                    break
                lines.append(f"    {change}")

    return lines


def _format_drift_metrics(drift_result, opts: FormatOptions) -> list[str]:
    """Format drift metrics."""
    lines: list[str] = []

    for i, m in enumerate(drift_result.top_drifts):
        if i >= opts.max_drifts:
            lines.append(f"    ... and {len(drift_result.top_drifts) - i} more")
            break
        pct = f"{m.percent_change:+.1f}%" if m.percent_change else "N/A"
        lines.append(f"    {m.metric}: {m.value_a} => {m.value_b} ({pct})")

    return lines


def format_comparison_result(
    result: ComparisonResult,
    opts: FormatOptions | None = None,
) -> str:
    """
    Format ComparisonResult as human-readable text report.

    Parameters
    ----------
    result : ComparisonResult
        Comparison result to format.
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
            f"Baseline:  {result.run_id_a}",
            f"Candidate: {result.run_id_b}",
            "",
            f"Overall: {_SYM_OK + ' IDENTICAL' if result.identical else _SYM_FAIL + ' DIFFER'}",
        ]
    )

    # Metadata section
    if result.metadata:
        lines.extend(["", "-" * opts.width, "Metadata", "-" * opts.width])

        project_match = result.metadata.get("project_match", True)
        if project_match:
            lines.append(f"  project: {_SYM_OK}")
        else:
            a = result.metadata.get("project_a", "?")
            b = result.metadata.get("project_b", "?")
            lines.append(f"  project: {_SYM_FAIL}")
            lines.append(f"    {a} => {b}")

        backend_match = result.metadata.get("backend_match", True)
        if backend_match:
            lines.append(f"  backend: {_SYM_OK}")
        else:
            a = result.metadata.get("backend_a", "?")
            b = result.metadata.get("backend_b", "?")
            lines.append(f"  backend: {_SYM_FAIL}")
            lines.append(f"    {a} => {b}")

    # Program section
    lines.extend(["", "-" * opts.width, "Program", "-" * opts.width])
    if not result.program.has_programs:
        lines.append("  N/A (not captured)")
    elif result.program.exact_match:
        lines.append(f"  {_SYM_OK} Match (exact)")
    elif result.program.structural_match:
        lines.append(f"  {_SYM_OK} Match (structural)")
    else:
        lines.append(f"  {_SYM_FAIL} Differ")

    # Parameters section
    if result.params:
        lines.extend(["", "-" * opts.width, "Parameters", "-" * opts.width])
        if result.params.get("match", False):
            lines.append(f"  {_SYM_OK} Match")
        else:
            lines.extend(_format_dict_changes(result.params, opts))

    # Metrics section
    if result.metrics:
        lines.extend(["", "-" * opts.width, "Metrics", "-" * opts.width])
        if result.metrics.get("match", True):
            lines.append(f"  {_SYM_OK} Match")
        else:
            lines.extend(_format_dict_changes(result.metrics, opts))

    # Device drift section
    if result.device_drift and result.device_drift.has_calibration_data:
        lines.extend(["", "-" * opts.width, "Device Calibration", "-" * opts.width])
        if result.device_drift.calibration_time_changed:
            lines.append("  Calibration times differ:")
            lines.append(f"    Baseline:  {result.device_drift.calibration_time_a}")
            lines.append(f"    Candidate: {result.device_drift.calibration_time_b}")
        if result.device_drift.significant_drift:
            lines.append(f"  {_SYM_FAIL} Significant drift detected:")
            lines.extend(_format_drift_metrics(result.device_drift, opts))
        else:
            lines.append(f"  {_SYM_OK} Drift within thresholds")

    # Results section (TVD + noise)
    if result.tvd is not None or result.noise_context:
        lines.extend(["", "-" * opts.width, "Results", "-" * opts.width])
        if result.tvd is not None:
            lines.append(f"  TVD: {result.tvd:.6f}")
        if result.noise_context:
            lines.append(f"  Expected noise: {result.noise_context.expected_noise:.6f}")
            lines.append(f"  Noise ratio: {result.noise_context.noise_ratio:.2f}x")
            lines.append(f"  Interpretation: {result.noise_context.interpretation()}")

    # Circuit section
    lines.extend(["", "-" * opts.width, "Circuit", "-" * opts.width])
    if result.circuit_diff is None:
        lines.append("  N/A (not captured)")
    elif result.circuit_diff.match:
        lines.append(f"  {_SYM_OK} Match")
    else:
        lines.append(f"  {_SYM_FAIL} Differ:")
        lines.extend(_format_circuit_diff(result.circuit_diff, opts))

    # Warnings section
    if result.warnings:
        lines.extend(["", "-" * opts.width, "Warnings", "-" * opts.width])
        for w in result.warnings:
            lines.append(f"  {_SYM_WARN} {w}")

    lines.extend(["", "=" * opts.width])
    return "\n".join(lines)


def format_verify_result(
    result: VerifyResult,
    opts: FormatOptions | None = None,
) -> str:
    """
    Format VerifyResult as human-readable text report.

    Parameters
    ----------
    result : VerifyResult
        Verification result to format.
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
            f"Baseline:  {result.baseline_run_id or 'N/A'}",
            f"Candidate: {result.candidate_run_id}",
            f"Duration:  {result.duration_ms:.1f}ms",
            "",
            f"Status: {_SYM_OK + ' PASSED' if result.ok else _SYM_FAIL + ' FAILED'}",
        ]
    )

    # Failures section
    if not result.ok and result.failures:
        lines.extend(["", "-" * opts.width, "Failures", "-" * opts.width])
        for failure in result.failures:
            lines.append(f"  {_SYM_FAIL} {failure}")

    # Results section
    if result.comparison:
        comp = result.comparison
        if comp.tvd is not None or comp.noise_context:
            lines.extend(["", "-" * opts.width, "Results", "-" * opts.width])
            if comp.tvd is not None:
                lines.append(f"  TVD: {comp.tvd:.6f}")
            if comp.noise_context:
                lines.append(
                    f"  Expected noise: {comp.noise_context.expected_noise:.6f}"
                )
                lines.append(f"  Noise ratio: {comp.noise_context.noise_ratio:.2f}x")
                lines.append(f"  Interpretation: {comp.noise_context.interpretation()}")

        # Circuit section (only if differs)
        if comp.circuit_diff and not comp.circuit_diff.match:
            lines.extend(["", "-" * opts.width, "Circuit", "-" * opts.width])
            lines.append(f"  {_SYM_FAIL} Differ:")
            lines.extend(_format_circuit_diff(comp.circuit_diff, opts))

        # Device drift section (only if significant)
        if comp.device_drift and comp.device_drift.significant_drift:
            lines.extend(["", "-" * opts.width, "Device Calibration", "-" * opts.width])
            lines.append(f"  {_SYM_FAIL} Significant drift detected:")
            lines.extend(_format_drift_metrics(comp.device_drift, opts))

    # Verdict section
    if result.verdict:
        lines.extend(["", "-" * opts.width, "Root Cause Analysis", "-" * opts.width])
        lines.append(f"  Category: {result.verdict.category.value}")
        lines.append(f"  Summary: {result.verdict.summary}")
        lines.append(f"  Action: {result.verdict.action}")
        if result.verdict.contributing_factors:
            lines.append(f"  Factors: {', '.join(result.verdict.contributing_factors)}")

    lines.extend(["", "=" * opts.width])
    return "\n".join(lines)
