# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Text formatting utilities for comparison results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from devqubit_engine.compare.types import FormatOptions


if TYPE_CHECKING:
    from devqubit_engine.compare.results import ComparisonResult, VerifyResult


# Box drawing characters (with ASCII fallback)
_BOX_H = "─"
_BOX_H_BOLD = "═"
_BOX_BULLET = "•"

# Status symbols
_SYM_PASS = "✓"
_SYM_FAIL = "✗"
_SYM_WARN = "!"
_SYM_NA = "–"


def _header(title: str, width: int = 70) -> str:
    """Create centered header with double lines."""
    pad = (width - len(title) - 2) // 2
    return "\n".join(
        [
            _BOX_H_BOLD * width,
            f"{' ' * pad} {title} {' ' * pad}".ljust(width),
            _BOX_H_BOLD * width,
        ]
    )


def _section(title: str, width: int = 70) -> str:
    """Create section separator with single line."""
    return f"\n  {title}\n  {_BOX_H * len(title)}"


def _status_line(passed: bool, width: int = 70) -> str:
    """Create prominent status line."""
    if passed:
        label = f"{_SYM_PASS} IDENTICAL"
    else:
        label = f"{_SYM_FAIL} DIFFER"

    return "\n".join(
        [
            "",
            _BOX_H * width,
            f"  RESULT: {label}",
            _BOX_H * width,
        ]
    )


def _verify_status_line(passed: bool, width: int = 70) -> str:
    """Create prominent verification status line."""
    if passed:
        label = f"{_SYM_PASS} PASSED"
    else:
        label = f"{_SYM_FAIL} FAILED"

    return "\n".join(
        [
            "",
            _BOX_H * width,
            f"  STATUS: {label}",
            _BOX_H * width,
        ]
    )


def _format_id(run_id: str, max_len: int = 20) -> str:
    """Format run ID, truncating if needed."""
    if len(run_id) <= max_len:
        return run_id
    return run_id[: max_len - 3] + "..."


def _format_pct_change(val_a: Any, val_b: Any) -> str:
    """Format percentage change between two values."""
    if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
        return ""
    if val_a == 0:
        return ""
    pct = ((val_b - val_a) / abs(val_a)) * 100
    sign = "+" if pct > 0 else ""
    return f"  ({sign}{pct:.1f}%)"


def _format_value(val: Any) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        if abs(val) < 0.0001 or abs(val) >= 10000:
            return f"{val:.4e}"
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


# =============================================================================
# Summary builders
# =============================================================================


def _build_summary_items(result: ComparisonResult) -> list[str]:
    """Build summary bullet points for comparison."""
    items = []

    # Program status
    if not result.program.has_programs:
        items.append(f"Program:     {_SYM_NA} not captured")
    elif result.program.exact_match:
        items.append(f"Program:     {_SYM_PASS} identical")
    elif result.program.structural_match:
        items.append(f"Program:     {_SYM_PASS} structural match")
    else:
        items.append(f"Program:     {_SYM_FAIL} differ")

    # Parameters status
    if result.params:
        if result.params.get("match", False):
            items.append(f"Parameters:  {_SYM_PASS} match")
        else:
            n_changed = len(result.params.get("changed", {}))
            n_added = len(result.params.get("added", {}))
            n_removed = len(result.params.get("removed", {}))
            parts = []
            if n_changed:
                parts.append(f"{n_changed} changed")
            if n_added:
                parts.append(f"{n_added} added")
            if n_removed:
                parts.append(f"{n_removed} removed")
            items.append(f"Parameters:  {_SYM_FAIL} {', '.join(parts)}")

    # TVD / Results status
    if result.tvd is not None:
        tvd_str = f"TVD = {result.tvd:.4f}"
        if result.noise_context:
            ratio = result.noise_context.noise_ratio
            tvd_str += f" ({ratio:.1f}x noise floor)"
        items.append(f"Results:     {tvd_str}")

    # Device drift
    if result.device_drift and result.device_drift.significant_drift:
        items.append(f"Device:      {_SYM_WARN} calibration drift detected")

    return items


# =============================================================================
# Detail section formatters
# =============================================================================


def _format_metadata_section(result: ComparisonResult, width: int) -> list[str]:
    """Format metadata differences (only if they differ)."""
    lines = []

    project_match = result.metadata.get("project_match", True)
    backend_match = result.metadata.get("backend_match", True)

    if project_match and backend_match:
        return []

    lines.append(_section("METADATA DIFFERENCES", width))

    if not project_match:
        a = result.metadata.get("project_a", "?")
        b = result.metadata.get("project_b", "?")
        lines.append(f"    project: {a} => {b}")

    if not backend_match:
        a = result.metadata.get("backend_a", "?")
        b = result.metadata.get("backend_b", "?")
        lines.append(f"    backend: {a} => {b}")

    return lines


def _format_params_section(
    result: ComparisonResult,
    opts: FormatOptions,
) -> list[str]:
    """Format parameter changes (only if they differ)."""
    if not result.params or result.params.get("match", False):
        return []

    lines = [_section("PARAMETER CHANGES")]
    max_items = opts.max_param_changes

    changed = result.params.get("changed", {})
    added = result.params.get("added", {})
    removed = result.params.get("removed", {})

    count = 0

    for key, change in changed.items():
        if count >= max_items:
            remaining = len(changed) - count + len(added) + len(removed)
            lines.append(f"    ... and {remaining} more")
            break
        val_a = _format_value(change.get("a"))
        val_b = _format_value(change.get("b"))
        pct = _format_pct_change(change.get("a"), change.get("b"))
        lines.append(f"    {key}: {val_a} => {val_b}{pct}")
        count += 1

    if count < max_items and removed:
        for key, val in list(removed.items())[: max_items - count]:
            lines.append(f"    {key}: {_format_value(val)} => (removed)")
            count += 1

    if count < max_items and added:
        for key, val in list(added.items())[: max_items - count]:
            lines.append(f"    {key}: (new) => {_format_value(val)}")
            count += 1

    return lines


def _format_metrics_section(
    result: ComparisonResult,
    opts: FormatOptions,
) -> list[str]:
    """Format metric changes (only if they differ)."""
    if not result.metrics or result.metrics.get("match", True):
        return []

    lines = [_section("METRIC CHANGES")]
    max_items = opts.max_param_changes

    changed = result.metrics.get("changed", {})

    for i, (key, change) in enumerate(changed.items()):
        if i >= max_items:
            lines.append(f"    ... and {len(changed) - i} more")
            break
        val_a = _format_value(change.get("a"))
        val_b = _format_value(change.get("b"))
        pct = _format_pct_change(change.get("a"), change.get("b"))
        lines.append(f"    {key}: {val_a} => {val_b}{pct}")

    return lines


def _format_distribution_section(result: ComparisonResult) -> list[str]:
    """Format distribution analysis (TVD + noise context)."""
    if result.tvd is None:
        return []

    lines = [_section("DISTRIBUTION ANALYSIS")]
    lines.append(f"    TVD:             {result.tvd:.6f}")

    if result.noise_context:
        nc = result.noise_context
        lines.append(f"    Expected noise:  {nc.expected_noise:.6f}")
        lines.append(f"    Noise ratio:     {nc.noise_ratio:.2f}x")
        lines.append(f"    Assessment:      {nc.interpretation()}")

    return lines


def _format_drift_section(
    result: ComparisonResult,
    opts: FormatOptions,
) -> list[str]:
    """Format device drift (only if significant)."""
    drift = result.device_drift
    if not drift or not drift.has_calibration_data:
        return []

    if not drift.significant_drift and not drift.calibration_time_changed:
        return []

    lines = [_section("DEVICE CALIBRATION")]

    if drift.calibration_time_changed:
        lines.append("    Calibration time changed:")
        lines.append(f"      Baseline:  {drift.calibration_time_a or 'unknown'}")
        lines.append(f"      Candidate: {drift.calibration_time_b or 'unknown'}")

    if drift.significant_drift:
        lines.append(f"    Significant drift in {len(drift.top_drifts)} metric(s):")
        for i, m in enumerate(drift.top_drifts):
            if i >= opts.max_drifts:
                lines.append(f"      ... and {len(drift.top_drifts) - i} more")
                break
            pct = f"{m.percent_change:+.1f}%" if m.percent_change else "N/A"
            lines.append(f"      {m.metric}: {m.value_a} => {m.value_b} ({pct})")

    return lines


def _format_circuit_section(
    result: ComparisonResult,
    opts: FormatOptions,
) -> list[str]:
    """Format circuit diff (only if differs)."""
    if result.circuit_diff is None:
        return []

    if result.circuit_diff.match:
        return []

    lines = [_section("CIRCUIT DIFFERENCES")]

    if not result.circuit_diff.changes:
        lines.append("    Circuits differ (no detailed diff available)")
        return lines

    changes = result.circuit_diff.changes
    max_items = opts.max_circuit_changes

    if isinstance(changes, dict):
        for i, (key, change) in enumerate(changes.items()):
            if i >= max_items:
                lines.append(f"    ... and {len(changes) - i} more")
                break
            a = change.get("a", "?")
            b = change.get("b", "?")
            lines.append(f"    {key}: {a} => {b}")
    else:
        for i, change in enumerate(changes):
            if i >= max_items:
                lines.append(f"    ... and {len(changes) - i} more")
                break
            lines.append(f"    {change}")

    return lines


def _format_warnings_section(result: ComparisonResult) -> list[str]:
    """Format warnings (only if present)."""
    if not result.warnings:
        return []

    lines = [_section("WARNINGS")]
    for w in result.warnings:
        lines.append(f"    {_SYM_WARN} {w}")

    return lines


# =============================================================================
# Main formatters
# =============================================================================


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

    w = opts.width
    lines = []

    # Header
    lines.append(_header("RUN COMPARISON", w))
    lines.append("")

    # Run identifiers
    id_a = _format_id(result.run_id_a, 24)
    id_b = _format_id(result.run_id_b, 24)

    proj_a = result.metadata.get("project_a", "")
    proj_b = result.metadata.get("project_b", "")

    proj_suffix_a = f"  (project: {proj_a})" if proj_a else ""
    proj_suffix_b = f"  (project: {proj_b})" if proj_b else ""

    lines.append(f"  Baseline:   {id_a}{proj_suffix_a}")
    lines.append(f"  Candidate:  {id_b}{proj_suffix_b}")

    # Status line
    lines.append(_status_line(result.identical, w))

    # Summary section
    summary_items = _build_summary_items(result)
    if summary_items:
        lines.append(_section("SUMMARY"))
        for item in summary_items:
            lines.append(f"    {_BOX_BULLET} {item}")

    # Detail sections (only shown if relevant)
    lines.extend(_format_metadata_section(result, w))
    lines.extend(_format_params_section(result, opts))
    lines.extend(_format_metrics_section(result, opts))
    lines.extend(_format_distribution_section(result))
    lines.extend(_format_drift_section(result, opts))
    lines.extend(_format_circuit_section(result, opts))
    lines.extend(_format_warnings_section(result))

    # Footer
    lines.append("")
    lines.append(_BOX_H_BOLD * w)

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

    w = opts.width
    lines = []

    # Header
    lines.append(_header("VERIFICATION RESULT", w))
    lines.append("")

    # Run identifiers
    baseline_id = _format_id(result.baseline_run_id or "N/A", 24)
    candidate_id = _format_id(result.candidate_run_id or "N/A", 24)

    lines.append(f"  Baseline:   {baseline_id}")
    lines.append(f"  Candidate:  {candidate_id}")

    # Status line
    lines.append(_verify_status_line(result.ok, w))

    # Failures section (only if failed)
    if not result.ok and result.failures:
        lines.append(_section("FAILURES"))
        for failure in result.failures:
            lines.append(f"    {_SYM_FAIL} {failure}")

    # Results section (TVD + noise)
    if result.comparison:
        comp = result.comparison
        if comp.tvd is not None:
            lines.append(_section("RESULTS"))
            lines.append(f"    TVD:             {comp.tvd:.6f}")
            if comp.noise_context:
                nc = comp.noise_context
                lines.append(f"    Expected noise:  {nc.expected_noise:.6f}")
                lines.append(f"    Noise ratio:     {nc.noise_ratio:.2f}x")
                lines.append(f"    Assessment:      {nc.interpretation()}")

    # Root cause analysis (only if failed with verdict)
    if not result.ok and result.verdict:
        v = result.verdict
        lines.append(_section("ROOT CAUSE ANALYSIS"))
        lines.append(f"    Category:  {v.category.value}")
        lines.append(f"    Summary:   {v.summary}")
        if v.action:
            lines.append(f"    Action:    {v.action}")
        if v.contributing_factors:
            factors = ", ".join(v.contributing_factors[:3])
            if len(v.contributing_factors) > 3:
                factors += f" (+{len(v.contributing_factors) - 3} more)"
            lines.append(f"    Factors:   {factors}")

    # Duration
    lines.append("")
    lines.append(f"  Duration: {result.duration_ms:.1f}ms")

    # Footer
    lines.append(_BOX_H_BOLD * w)

    return "\n".join(lines)
