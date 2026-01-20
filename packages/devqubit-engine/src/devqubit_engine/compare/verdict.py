# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Regression verdict and root-cause analysis.

This module provides quantum-aware explanations for verification failures,
using bootstrap-calibrated noise context to distinguish real differences
from shot noise. Verdicts help developers understand why a verification
failed and what action to take.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.circuit.summary import diff_summaries
from devqubit_engine.compare.context import RunContext, extract_circuit_summary
from devqubit_engine.compare.results import ComparisonResult, Verdict
from devqubit_engine.compare.types import VerdictCategory
from devqubit_engine.storage.types import ObjectStoreProtocol
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.uec.api.resolve import resolve_envelope
from devqubit_engine.utils.distributions import compute_noise_context


logger = logging.getLogger(__name__)


# Threshold for significant compiler change (20% in either direction)
_COMPILER_CHANGE_THRESHOLD = 0.20

# Minimum absolute change requirements to avoid false positives on small circuits
_MIN_DEPTH_DELTA = 2
_MIN_2Q_GATE_DELTA = 5


def _detect_program_change(result: ComparisonResult) -> dict[str, Any] | None:
    """
    Check if program artifacts changed.

    Only reports PROGRAM_CHANGED when there is a genuine structural change,
    NOT when parameters differ but structure remains the same (VQE/QAOA case).

    Parameters
    ----------
    result : ComparisonResult
        Comparison result with program comparison data.

    Returns
    -------
    dict or None
        Evidence dict if programs structurally differ, None if they match.
    """
    if result.program.exact_match:
        return None

    # Structural match -> NOT a program change (just parameter variation)
    if result.program.structural_match:
        logger.debug(
            "Program artifacts differ but structure matches (parameter variation)"
        )
        return None

    logger.debug(
        "Program change detected: exact=%s, structural=%s",
        result.program.exact_match,
        result.program.structural_match,
    )

    return {
        "exact_match": result.program.exact_match,
        "structural_match": result.program.structural_match,
        "structural_only_match": result.program.structural_only_match,
        "digests_a": result.program.digests_a[:2] if result.program.digests_a else [],
        "digests_b": result.program.digests_b[:2] if result.program.digests_b else [],
        "circuit_hash_a": result.program.circuit_hash_a,
        "circuit_hash_b": result.program.circuit_hash_b,
    }


def _format_change(val_a: int, val_b: int) -> str:
    """Format a value change with percentage."""
    if val_a == 0:
        return f"{val_a} → {val_b}"
    ratio = val_b / val_a
    pct = (ratio - 1) * 100
    sign = "+" if pct > 0 else ""
    return f"{val_a} → {val_b} ({sign}{pct:.0f}%)"


def _detect_compiler_change(result: ComparisonResult) -> dict[str, Any] | None:
    """
    Check if compilation changed (same program, different decomposition).

    Uses a two-tier detection approach:
    1. Primary: executed_*_hash mismatch when structural_match is True
       (definitive signal that transpilation changed)
    2. Secondary: significant changes (>20%) in depth or 2Q gate count
       (evidence/explanation for what changed)

    Returns
    -------
    dict or None
        Evidence dict if compiler change detected, None otherwise.
    """
    evidence: dict[str, Any] = {}

    # Primary detection: executed hashes differ while logical structure matches
    if result.program.structural_match and not result.program.executed_structural_match:
        evidence["executed_structural_hash_mismatch"] = True
        evidence["logical_structural_hash_a"] = result.program.circuit_hash_a
        evidence["logical_structural_hash_b"] = result.program.circuit_hash_b
        evidence["executed_structural_hash_a"] = (
            result.program.executed_structural_hash_a
        )
        evidence["executed_structural_hash_b"] = (
            result.program.executed_structural_hash_b
        )

        logger.debug(
            "Compiler change detected via executed_structural_hash: "
            "structural_match=%s, executed_structural_match=%s",
            result.program.structural_match,
            result.program.executed_structural_match,
        )

    # Secondary detection/evidence: circuit diff metrics
    if result.circuit_diff and not result.circuit_diff.match:
        summary_a = result.circuit_diff.summary_a
        summary_b = result.circuit_diff.summary_b

        # Check depth change
        if summary_a.depth > 0:
            depth_delta = abs(summary_b.depth - summary_a.depth)
            depth_ratio = summary_b.depth / summary_a.depth
            if (
                abs(depth_ratio - 1.0) > _COMPILER_CHANGE_THRESHOLD
                and depth_delta >= _MIN_DEPTH_DELTA
            ):
                evidence["depth_change"] = _format_change(
                    summary_a.depth, summary_b.depth
                )

        # Check 2Q gate count change
        if summary_a.gate_count_2q > 0:
            gate_delta = abs(summary_b.gate_count_2q - summary_a.gate_count_2q)
            gate_ratio = summary_b.gate_count_2q / summary_a.gate_count_2q
            if (
                abs(gate_ratio - 1.0) > _COMPILER_CHANGE_THRESHOLD
                and gate_delta >= _MIN_2Q_GATE_DELTA
            ):
                evidence["gate_2q_change"] = _format_change(
                    summary_a.gate_count_2q, summary_b.gate_count_2q
                )
        elif summary_b.gate_count_2q >= _MIN_2Q_GATE_DELTA:
            evidence["gate_2q_change"] = _format_change(
                summary_a.gate_count_2q, summary_b.gate_count_2q
            )

    if evidence:
        logger.debug("Compiler change detected: %s", evidence)
        return evidence

    return None


def _detect_device_drift(result: ComparisonResult) -> dict[str, Any] | None:
    """
    Check if device calibration drift is significant.

    Returns
    -------
    dict or None
        Evidence dict if significant drift detected, None otherwise.
    """
    if not result.device_drift or not result.device_drift.significant_drift:
        return None

    top_drifts = result.device_drift.top_drifts[:3]

    logger.debug("Device drift detected: %d significant metrics", len(top_drifts))

    return {
        "top_drifts": [
            {
                "metric": d.metric,
                "change": f"{d.percent_change:.1f}%" if d.percent_change else "N/A",
            }
            for d in top_drifts
        ],
        "calibration_times": {
            "a": result.device_drift.calibration_time_a,
            "b": result.device_drift.calibration_time_b,
        },
    }


def _detect_shot_noise(result: ComparisonResult) -> dict[str, Any] | None:
    """
    Check if difference is consistent with shot noise.

    Uses bootstrap-calibrated noise_p95 and p_value for robust detection.
    Falls back to noise_ratio < 2 heuristic if bootstrap unavailable.

    Returns
    -------
    dict or None
        Evidence dict if likely shot noise, None otherwise.
    """
    if result.noise_context is None or result.tvd is None:
        return None

    ctx = result.noise_context

    # Primary path: use bootstrap-calibrated thresholds
    if ctx.p_value is not None and ctx.noise_p95 > 0:
        if result.tvd > ctx.noise_p95 and ctx.p_value < 0.05:
            logger.debug(
                "Difference exceeds noise threshold: tvd=%.4f > p95=%.4f, p=%.4f",
                result.tvd,
                ctx.noise_p95,
                ctx.p_value,
            )
            return None

        logger.debug(
            "Difference consistent with shot noise: tvd=%.4f, p95=%.4f, p=%.4f",
            result.tvd,
            ctx.noise_p95,
            ctx.p_value,
        )
        return {
            "tvd": result.tvd,
            "noise_p95": ctx.noise_p95,
            "p_value": ctx.p_value,
            "interpretation": ctx.interpretation(),
        }

    # Fallback: use noise_ratio heuristic
    if ctx.noise_ratio >= 2.0:
        logger.debug("Difference exceeds noise ratio threshold: %.2fx", ctx.noise_ratio)
        return None

    logger.debug("Difference consistent with shot noise: ratio=%.2fx", ctx.noise_ratio)

    return {
        "tvd": result.tvd,
        "expected_noise": ctx.expected_noise,
        "noise_ratio": ctx.noise_ratio,
        "interpretation": ctx.interpretation(),
    }


# Verdict information: (summary, action)
_VERDICT_INFO: dict[VerdictCategory, tuple[str, str]] = {
    VerdictCategory.PROGRAM_CHANGED: (
        "Quantum program (circuit) has changed",
        "Review circuit changes; update baseline if intentional",
    ),
    VerdictCategory.COMPILER_CHANGED: (
        "Same circuit compiled differently (depth/2Q gates changed)",
        "Check transpiler settings; consider pinning optimization level",
    ),
    VerdictCategory.DEVICE_DRIFT: (
        "Device calibration has drifted significantly",
        "Re-run on fresh calibration or adjust TVD threshold",
    ),
    VerdictCategory.SHOT_NOISE: (
        "Difference is consistent with statistical sampling noise",
        "Increase shots for tighter comparison, or accept as passing",
    ),
}


# =============================================================================
# Core: Envelope-only verdict building
# =============================================================================


def build_verdict_contexts(
    result: ComparisonResult,
    ctx_a: RunContext,
    ctx_b: RunContext,
) -> Verdict:
    """
    Build regression verdict from comparison result using envelope-only data.

    This is the core verdict function operating entirely on RunContext
    (envelope + store). All analysis (program change, compiler change, drift,
    noise) is performed here without any RunRecord dependencies.

    Parameters
    ----------
    result : ComparisonResult
        Comparison result to analyze.
    ctx_a : RunContext
        Baseline context with resolved envelope.
    ctx_b : RunContext
        Candidate context with resolved envelope.

    Returns
    -------
    Verdict
        Root-cause verdict with evidence and suggested action.

    Notes
    -----
    Factors are checked in priority order:

    1. Program change (highest priority) - only if structural_match is False
    2. Compiler change (if program unchanged) - via executed_*_hash or circuit diff
    3. Device drift
    4. Shot noise (only if no other factors)
    """
    factors: list[str] = []
    all_evidence: dict[str, Any] = {}

    logger.debug(
        "Building verdict for comparison %s vs %s",
        ctx_a.run_id,
        ctx_b.run_id,
    )

    # Ensure circuit diff is populated using envelope-based extraction
    if result.circuit_diff is None:
        summary_a = extract_circuit_summary(ctx_a.envelope, ctx_a.store)
        summary_b = extract_circuit_summary(ctx_b.envelope, ctx_b.store)
        if summary_a and summary_b:
            result.circuit_diff = diff_summaries(summary_a, summary_b)

    # Ensure noise context is computed (with bootstrap)
    if (
        result.noise_context is None
        and result.counts_a is not None
        and result.counts_b is not None
        and result.tvd is not None
    ):
        result.noise_context = compute_noise_context(
            result.counts_a, result.counts_b, result.tvd
        )

    # Check each factor in priority order
    program_evidence = _detect_program_change(result)
    if program_evidence:
        factors.append("PROGRAM_CHANGED")
        all_evidence["program"] = program_evidence

    # Only check compiler change if program didn't change
    if not program_evidence:
        compiler_evidence = _detect_compiler_change(result)
        if compiler_evidence:
            factors.append("COMPILER_CHANGED")
            all_evidence["compiler"] = compiler_evidence

    drift_evidence = _detect_device_drift(result)
    if drift_evidence:
        factors.append("DEVICE_DRIFT")
        all_evidence["device_drift"] = drift_evidence

    # Only consider shot noise if no other factors detected
    noise_evidence = _detect_shot_noise(result)
    if noise_evidence and not factors:
        factors.append("SHOT_NOISE")
        all_evidence["shot_noise"] = noise_evidence

    # Add circuit diff to evidence if available
    if result.circuit_diff:
        all_evidence["circuit_diff"] = {
            "match": result.circuit_diff.match,
            "changes": result.circuit_diff.changes,
            "metrics": result.circuit_diff.metrics,
        }

    if result.noise_context:
        all_evidence["noise_context"] = result.noise_context.to_dict()

    # Build verdict
    if not factors:
        logger.info("No clear root cause identified")
        return Verdict(
            category=VerdictCategory.UNKNOWN,
            summary="No clear root cause identified",
            evidence=all_evidence,
            action="Review run configurations and retry",
            contributing_factors=[],
        )

    if len(factors) == 1:
        category = VerdictCategory(factors[0])
        info = _VERDICT_INFO.get(category)
        if info:
            summary, action = info
        else:
            summary, action = "Unknown factor", "Investigate manually"

        logger.info("Verdict: %s", category.value)

        return Verdict(
            category=category,
            summary=summary,
            evidence=all_evidence,
            action=action,
            contributing_factors=factors,
        )

    # Multiple factors
    logger.info("Multiple factors detected: %s", factors)

    return Verdict(
        category=VerdictCategory.MIXED,
        summary=f"Multiple factors: {', '.join(factors)}",
        evidence=all_evidence,
        action="Address factors in order: program > compiler > device",
        contributing_factors=factors,
    )


# =============================================================================
# Adapters: RunRecord to RunContext conversion
# =============================================================================


def build_verdict(
    result: ComparisonResult,
    run_a: RunRecord,
    run_b: RunRecord,
    store_a: ObjectStoreProtocol,
    store_b: ObjectStoreProtocol,
) -> Verdict:
    """
    Build regression verdict from comparison result.

    This is an adapter function that converts RunRecord inputs to RunContext
    and delegates to build_verdict_contexts. Use this when you have RunRecord
    objects.

    Parameters
    ----------
    result : ComparisonResult
        Comparison result to analyze.
    run_a : RunRecord
        Baseline run record.
    run_b : RunRecord
        Candidate run record.
    store_a : ObjectStoreProtocol
        Object store for baseline artifacts.
    store_b : ObjectStoreProtocol
        Object store for candidate artifacts.

    Returns
    -------
    Verdict
        Root-cause verdict with evidence and suggested action.
    """
    envelope_a = resolve_envelope(run_a, store_a)
    envelope_b = resolve_envelope(run_b, store_b)

    ctx_a = RunContext(
        run_id=run_a.run_id,
        envelope=envelope_a,
        store=store_a,
    )
    ctx_b = RunContext(
        run_id=run_b.run_id,
        envelope=envelope_b,
        store=store_b,
    )

    return build_verdict_contexts(result, ctx_a, ctx_b)
