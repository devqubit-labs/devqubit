# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Text formatting for comparison and verification results.

This module provides human-readable text formatting for ComparisonResult
and VerifyResult objects. It uses the TextRenderer from text_format for
consistent visual output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from devqubit_engine.compare.types import FormatOptions
from devqubit_engine.utils.text_format import (
    StatusLevel,
    TextRenderer,
    format_pct_change,
    format_value,
    truncate_id,
)


if TYPE_CHECKING:
    from devqubit_engine.compare.results import ComparisonResult, VerifyResult


__all__ = [
    "format_comparison_result",
    "format_verify_result",
    "compute_tvd_status",
    "TVDStatus",
]


# =============================================================================
# TVD status evaluation
# =============================================================================


class TVDStatus(NamedTuple):
    """
    TVD evaluation result.

    Attributes
    ----------
    level : StatusLevel
        Overall status level (PASS/WARN/FAIL).
    detail : str
        Human-readable detail string.
    """

    level: StatusLevel
    detail: str


# TVD thresholds
_TVD_NOISE_RATIO_FAIL = 5.0
_TVD_NOISE_RATIO_WARN = 2.0
_TVD_ABSOLUTE_FAIL = 0.1
_TVD_ABSOLUTE_WARN = 0.05


def compute_tvd_status(tvd: float, noise_ratio: float | None = None) -> TVDStatus:
    """
    Compute TVD status based on value and optional noise context.

    Uses noise ratio thresholds when available, otherwise falls back
    to absolute TVD thresholds.

    Parameters
    ----------
    tvd : float
        Total Variation Distance value.
    noise_ratio : float, optional
        Ratio of TVD to expected noise (TVD / expected_noise).

    Returns
    -------
    TVDStatus
        Named tuple with status level and detail string.

    Examples
    --------
    >>> compute_tvd_status(0.02)
    TVDStatus(level=<StatusLevel.PASS: 'pass'>, detail='TVD = 0.0200')

    >>> compute_tvd_status(0.15)
    TVDStatus(level=<StatusLevel.FAIL: 'fail'>, detail='TVD = 0.1500')

    >>> compute_tvd_status(0.03, noise_ratio=6.0)
    TVDStatus(level=<StatusLevel.FAIL: 'fail'>, detail='TVD = 0.0300 (6.0x noise)')
    """
    if noise_ratio is not None:
        detail = f"TVD = {tvd:.4f} ({noise_ratio:.1f}x noise)"
        if noise_ratio > _TVD_NOISE_RATIO_FAIL:
            return TVDStatus(StatusLevel.FAIL, detail)
        if noise_ratio > _TVD_NOISE_RATIO_WARN:
            return TVDStatus(StatusLevel.WARN, detail)
        return TVDStatus(StatusLevel.PASS, detail)

    detail = f"TVD = {tvd:.4f}"
    if tvd > _TVD_ABSOLUTE_FAIL:
        return TVDStatus(StatusLevel.FAIL, detail)
    if tvd > _TVD_ABSOLUTE_WARN:
        return TVDStatus(StatusLevel.WARN, detail)
    return TVDStatus(StatusLevel.PASS, detail)


# =============================================================================
# Result formatter
# =============================================================================


class ResultFormatter:
    """
    Formatter for comparison and verification results.

    Encapsulates all formatting logic for generating human-readable
    text reports from result objects.

    Parameters
    ----------
    opts : FormatOptions, optional
        Formatting options controlling output width and limits.

    Attributes
    ----------
    opts : FormatOptions
        Active formatting options.
    renderer : TextRenderer
        Text renderer instance.
    key_width : int
        Width for keys in change sections.
    id_length : int
        Maximum length for identifiers.
    """

    # Default key width for change sections
    _KEY_WIDTH = 20

    def __init__(self, opts: FormatOptions | None = None) -> None:
        self.opts = opts or FormatOptions()
        self.key_width = min(20, max(14, self.opts.width // 5))
        self.id_length = min(40, self.opts.width * 2 // 5)
        self.renderer = TextRenderer(width=self.opts.width, key_width=self.key_width)

    @property
    def r(self) -> TextRenderer:
        """Shorthand accessor for renderer."""
        return self.renderer

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def format_comparison(self, result: ComparisonResult) -> str:
        """
        Format ComparisonResult as human-readable text.

        Parameters
        ----------
        result : ComparisonResult
            Comparison result to format.

        Returns
        -------
        str
            Formatted text report.
        """
        lines: list[str] = []

        lines += self._header_section(result)
        lines += self._summary_section(result)
        lines += self._metadata_section(result)
        lines += self._params_section(result)
        lines += self._metrics_section(result)
        lines += self._distribution_section(result)
        lines += self._drift_section(result)
        lines += self._circuit_section(result)
        lines += self._warnings_section(result)
        lines += self._footer_section()

        return self.r.render(lines)

    def format_verification(self, result: VerifyResult) -> str:
        """
        Format VerifyResult as human-readable text.

        Parameters
        ----------
        result : VerifyResult
            Verification result to format.

        Returns
        -------
        str
            Formatted text report.
        """
        lines: list[str] = []

        lines += self._verify_header_section(result)
        lines += self._verify_status_section(result)
        lines += self._verify_failures_section(result)
        lines += self._verify_results_section(result)
        lines += self._verify_verdict_section(result)
        lines += self._verify_footer_section(result)

        return self.r.render(lines)

    # -------------------------------------------------------------------------
    # Comparison sections
    # -------------------------------------------------------------------------

    def _header_section(self, result: ComparisonResult) -> list[str]:
        """Build header with run IDs."""
        lines = self.r.header("RUN COMPARISON")
        lines += self.r.blank()

        id_a = truncate_id(result.run_id_a, self.id_length)
        id_b = truncate_id(result.run_id_b, self.id_length)
        proj_a = result.metadata.get("project_a", "")
        proj_b = result.metadata.get("project_b", "")

        baseline = f"{id_a}  [{proj_a}]" if proj_a else id_a
        candidate = f"{id_b}  [{proj_b}]" if proj_b else id_b

        lines += self.r.kv("Baseline:", baseline)
        lines += self.r.kv("Candidate:", candidate)
        lines += self.r.status_line(
            "RESULT",
            result.identical,
            pass_text="IDENTICAL",
            fail_text="DIFFER",
        )

        return lines

    def _summary_section(self, result: ComparisonResult) -> list[str]:
        """Build summary bullet points."""
        items = self._build_summary_items(result)
        if not items:
            return []

        lines = self.r.section("SUMMARY")
        lines += items
        return lines

    def _build_summary_items(self, result: ComparisonResult) -> list[str]:
        """Generate summary status items."""
        lines: list[str] = []

        # Program status
        if not result.program.has_programs:
            lines += self.r.item(StatusLevel.NA, "Program:", "not captured")
        elif result.program.exact_match:
            lines += self.r.item(StatusLevel.PASS, "Program:", "identical")
        elif result.program.structural_match:
            lines += self.r.item(StatusLevel.PASS, "Program:", "structural match")
        else:
            lines += self.r.item(StatusLevel.FAIL, "Program:", "differ")

        # Parameters status
        if result.params:
            if result.params.get("match", False):
                lines += self.r.item(StatusLevel.PASS, "Parameters:", "match")
            else:
                parts = self._count_param_changes(result.params)
                lines += self.r.item(StatusLevel.FAIL, "Parameters:", ", ".join(parts))

        # TVD status
        if result.tvd is not None:
            ratio = result.noise_context.noise_ratio if result.noise_context else None
            status = compute_tvd_status(result.tvd, ratio)
            lines += self.r.item(status.level, "Results:", status.detail)

        # Drift status
        if result.device_drift and result.device_drift.significant_drift:
            lines += self.r.item(
                StatusLevel.WARN, "Device:", "calibration drift detected"
            )

        return lines

    def _count_param_changes(self, params: dict) -> list[str]:
        """Count parameter changes for summary."""
        parts = []
        if n := len(params.get("changed", {})):
            parts.append(f"{n} changed")
        if n := len(params.get("added", {})):
            parts.append(f"{n} added")
        if n := len(params.get("removed", {})):
            parts.append(f"{n} removed")
        return parts

    def _metadata_section(self, result: ComparisonResult) -> list[str]:
        """Build metadata differences section."""
        proj_match = result.metadata.get("project_match", True)
        back_match = result.metadata.get("backend_match", True)

        if proj_match and back_match:
            return []

        lines = self.r.section("METADATA DIFFERENCES")

        if not proj_match:
            a = result.metadata.get("project_a", "?")
            b = result.metadata.get("project_b", "?")
            lines += self.r.kv("project:", f"{a} => {b}")

        if not back_match:
            a = result.metadata.get("backend_a", "?")
            b = result.metadata.get("backend_b", "?")
            lines += self.r.kv("backend:", f"{a} => {b}")

        return lines

    def _params_section(self, result: ComparisonResult) -> list[str]:
        """Build parameter changes section."""
        if not result.params or result.params.get("match", False):
            return []

        lines = self.r.section("PARAMETER CHANGES")

        changed = result.params.get("changed", {})
        added = result.params.get("added", {})
        removed = result.params.get("removed", {})
        max_items = self.opts.max_param_changes

        lines += self._format_changes(changed, added, removed, max_items)
        return lines

    def _metrics_section(self, result: ComparisonResult) -> list[str]:
        """Build metric changes section."""
        if not result.metrics or result.metrics.get("match", True):
            return []

        lines = self.r.section("METRIC CHANGES")
        changed = result.metrics.get("changed", {})

        for i, key in enumerate(sorted(changed)):
            if i >= self.opts.max_metric_changes:
                lines += self.r.overflow(len(changed) - i)
                break

            ch = changed[key]
            a, b = format_value(ch.get("a")), format_value(ch.get("b"))
            pct = format_pct_change(ch.get("a"), ch.get("b"))
            lines += self.r.change(
                truncate_id(key, self._KEY_WIDTH),
                a,
                b,
                suffix=pct,
                key_width=self._KEY_WIDTH,
            )

        return lines

    def _distribution_section(self, result: ComparisonResult) -> list[str]:
        """Build distribution analysis section."""
        if result.tvd is None:
            return []

        lines = self.r.section("DISTRIBUTION ANALYSIS")

        ratio = result.noise_context.noise_ratio if result.noise_context else None
        status = compute_tvd_status(result.tvd, ratio)
        suffix = self._status_suffix(status.level)

        lines += self.r.kv("TVD:", f"{result.tvd:.6f}{suffix}")

        if result.noise_context:
            nc = result.noise_context
            lines += self.r.kv("Expected noise:", f"{nc.expected_noise:.6f}")
            lines += self.r.kv("Noise ratio:", f"{nc.noise_ratio:.2f}x")
            lines += self.r.kv("Assessment:", nc.interpretation())

        return lines

    def _drift_section(self, result: ComparisonResult) -> list[str]:
        """Build device drift section."""
        drift = result.device_drift
        if not drift or not drift.has_calibration_data:
            return []
        if not drift.significant_drift and not drift.calibration_time_changed:
            return []

        lines = self.r.section("DEVICE CALIBRATION")

        if drift.calibration_time_changed:
            lines += self.r.kv("Baseline cal:", drift.calibration_time_a or "unknown")
            lines += self.r.kv("Candidate cal:", drift.calibration_time_b or "unknown")

        if drift.significant_drift:
            lines += self.r.blank()
            lines.append(
                f"  {self.r.style.sym_warn} Significant drift in "
                f"{len(drift.top_drifts)} metric(s):"
            )

            for i, m in enumerate(drift.top_drifts):
                if i >= self.opts.max_drifts:
                    lines += self.r.overflow(len(drift.top_drifts) - i, indent=4)
                    break
                pct = f"{m.percent_change:+.1f}%" if m.percent_change else "N/A"
                lines.append(f"    {m.metric:<18}  {m.value_a} => {m.value_b} ({pct})")

        return lines

    def _circuit_section(self, result: ComparisonResult) -> list[str]:
        """Build circuit differences section."""
        if not result.circuit_diff or result.circuit_diff.match:
            return []

        lines = self.r.section("CIRCUIT DIFFERENCES")
        cd = result.circuit_diff

        # No data available
        if (
            not cd.changed
            and not cd.added_gates
            and not cd.removed_gates
            and not cd.is_clifford_changed
        ):
            lines += self.r.text("Circuits differ (no detailed diff available)")
            return lines

        # Format changed properties
        count = 0
        for key in sorted(cd.changed):
            if count >= self.opts.max_circuit_changes:
                remaining = len(cd.changed) - count
                if cd.added_gates:
                    remaining += 1
                if cd.removed_gates:
                    remaining += 1
                lines += self.r.overflow(remaining)
                return lines

            ch = cd.changed[key]
            label = ch.get("label", key)
            val_a = format_value(ch.get("a"))
            val_b = format_value(ch.get("b"))

            if "pct" in ch:
                pct = ch["pct"]
                suffix = f" ({pct:+.1f}%)"
            else:
                suffix = ""

            lines += self.r.change(
                truncate_id(label, self._KEY_WIDTH),
                val_a,
                val_b,
                suffix=suffix,
                key_width=self._KEY_WIDTH,
            )
            count += 1

        # Format Clifford status change
        if cd.is_clifford_changed:
            lines += self.r.change(
                "Clifford",
                str(cd.is_clifford_a),
                str(cd.is_clifford_b),
                key_width=self._KEY_WIDTH,
            )

        # Format added gates
        if cd.added_gates:
            gates_str = ", ".join(cd.added_gates[:5])
            if len(cd.added_gates) > 5:
                gates_str += f" (+{len(cd.added_gates) - 5} more)"
            lines += self.r.kv("New gates:", gates_str)

        # Format removed gates
        if cd.removed_gates:
            gates_str = ", ".join(cd.removed_gates[:5])
            if len(cd.removed_gates) > 5:
                gates_str += f" (+{len(cd.removed_gates) - 5} more)"
            lines += self.r.kv("Removed gates:", gates_str)

        return lines

    def _warnings_section(self, result: ComparisonResult) -> list[str]:
        """Build warnings section."""
        if not result.warnings:
            return []

        lines = self.r.section("WARNINGS")
        for w in result.warnings:
            lines.append(f"  {self.r.style.sym_warn} {w}")
        return lines

    def _footer_section(self) -> list[str]:
        """Build footer."""
        lines = self.r.blank()
        lines += self.r.footer()
        return lines

    # -------------------------------------------------------------------------
    # Verification sections
    # -------------------------------------------------------------------------

    def _verify_header_section(self, result: VerifyResult) -> list[str]:
        """Build verification header."""
        lines = self.r.header("VERIFICATION RESULT")
        lines += self.r.blank()

        baseline = truncate_id(result.baseline_run_id or "N/A", self.id_length)
        candidate = truncate_id(result.candidate_run_id or "N/A", self.id_length)

        lines += self.r.kv("Baseline:", baseline)
        lines += self.r.kv("Candidate:", candidate)

        return lines

    def _verify_status_section(self, result: VerifyResult) -> list[str]:
        """Build verification status line."""
        return self.r.status_line("STATUS", result.ok)

    def _verify_failures_section(self, result: VerifyResult) -> list[str]:
        """Build failures section."""
        if result.ok or not result.failures:
            return []

        lines = self.r.section("FAILURES")
        for f in result.failures:
            lines.append(f"  {self.r.style.sym_fail} {f}")
        return lines

    def _verify_results_section(self, result: VerifyResult) -> list[str]:
        """Build results section with TVD."""
        if not result.comparison or result.comparison.tvd is None:
            return []

        comp = result.comparison
        lines = self.r.section("RESULTS")

        ratio = comp.noise_context.noise_ratio if comp.noise_context else None
        status = compute_tvd_status(comp.tvd, ratio)
        suffix = self._status_suffix(status.level)

        lines += self.r.kv("TVD:", f"{comp.tvd:.6f}{suffix}")

        if comp.noise_context:
            nc = comp.noise_context
            lines += self.r.kv("Expected noise:", f"{nc.expected_noise:.6f}")
            lines += self.r.kv("Noise ratio:", f"{nc.noise_ratio:.2f}x")
            lines += self.r.kv("Assessment:", nc.interpretation())

        return lines

    def _verify_verdict_section(self, result: VerifyResult) -> list[str]:
        """Build root cause analysis section."""
        if result.ok or not result.verdict:
            return []

        v = result.verdict
        lines = self.r.section("ROOT CAUSE ANALYSIS")

        lines += self.r.kv("Category:", v.category.value)
        lines += self.r.kv("Summary:", v.summary)

        if v.action:
            lines += self.r.kv("Action:", v.action)

        if v.contributing_factors:
            factors = ", ".join(v.contributing_factors[:3])
            if len(v.contributing_factors) > 3:
                factors += f" (+{len(v.contributing_factors) - 3} more)"
            lines += self.r.kv("Factors:", factors)

        return lines

    def _verify_footer_section(self, result: VerifyResult) -> list[str]:
        """Build verification footer with duration."""
        lines = self.r.blank()
        lines += self.r.kv("Duration:", f"{result.duration_ms:.1f}ms")
        lines += self.r.footer()
        return lines

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _format_changes(
        self,
        changed: dict,
        added: dict,
        removed: dict,
        max_items: int,
    ) -> list[str]:
        """Format parameter/metric changes with limits."""
        lines: list[str] = []
        count = 0

        # Changed values
        for key in sorted(changed):
            if count >= max_items:
                remaining = len(changed) - count + len(added) + len(removed)
                lines += self.r.overflow(remaining)
                return lines

            ch = changed[key]
            a, b = format_value(ch.get("a")), format_value(ch.get("b"))
            pct = format_pct_change(ch.get("a"), ch.get("b"))
            lines += self.r.change(
                truncate_id(key, self._KEY_WIDTH),
                a,
                b,
                suffix=pct,
                key_width=self._KEY_WIDTH,
            )
            count += 1

        # Removed values
        for key in sorted(removed):
            if count >= max_items:
                lines += self.r.overflow(
                    len(removed) + len(added) - (count - len(changed))
                )
                return lines

            lines += self.r.change(
                truncate_id(key, self._KEY_WIDTH),
                format_value(removed[key]),
                "(removed)",
                key_width=self._KEY_WIDTH,
            )
            count += 1

        # Added values
        for key in sorted(added):
            if count >= max_items:
                lines += self.r.overflow(
                    len(added) - (count - len(changed) - len(removed))
                )
                return lines

            lines += self.r.change(
                truncate_id(key, self._KEY_WIDTH),
                "(new)",
                format_value(added[key]),
                key_width=self._KEY_WIDTH,
            )
            count += 1

        return lines

    def _status_suffix(self, level: StatusLevel) -> str:
        """Get status symbol suffix for a level."""
        if level == StatusLevel.FAIL:
            return f"  {self.r.style.sym_fail}"
        if level == StatusLevel.WARN:
            return f"  {self.r.style.sym_warn}"
        return ""


# =============================================================================
# Public API functions
# =============================================================================


def format_comparison_result(
    result: ComparisonResult,
    opts: FormatOptions | None = None,
) -> str:
    """
    Format ComparisonResult as human-readable text.

    Parameters
    ----------
    result : ComparisonResult
        Comparison result to format.
    opts : FormatOptions, optional
        Formatting options controlling output width and limits.

    Returns
    -------
    str
        Formatted text report.

    Examples
    --------
    >>> from devqubit_engine.compare.results import ComparisonResult
    >>> result = ComparisonResult(run_id_a="run-001", run_id_b="run-002")
    >>> print(format_comparison_result(result))
    """
    return ResultFormatter(opts).format_comparison(result)


def format_verify_result(
    result: VerifyResult,
    opts: FormatOptions | None = None,
) -> str:
    """
    Format VerifyResult as human-readable text.

    Parameters
    ----------
    result : VerifyResult
        Verification result to format.
    opts : FormatOptions, optional
        Formatting options controlling output width and limits.

    Returns
    -------
    str
        Formatted text report.

    Examples
    --------
    >>> from devqubit_engine.compare.results import VerifyResult
    >>> result = VerifyResult(ok=True, candidate_run_id="run-002")
    >>> print(format_verify_result(result))
    """
    return ResultFormatter(opts).format_verification(result)
