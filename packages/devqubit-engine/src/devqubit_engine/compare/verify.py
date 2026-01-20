# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Baseline verification for CI/regression testing.

This module provides policy-based verification of a candidate run against
a baseline run, with bootstrap-calibrated noise thresholds.

Architecture
------------
- Core functions (verify_contexts) operate entirely on RunContext/envelope
- Adapter functions (verify, verify_against_baseline) handle RunRecord/registry IO
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from devqubit_engine.compare.context import RunContext
from devqubit_engine.compare.diff import diff_contexts
from devqubit_engine.compare.results import VerifyResult
from devqubit_engine.compare.types import ProgramMatchMode
from devqubit_engine.compare.verdict import build_verdict_contexts
from devqubit_engine.storage.types import ObjectStoreProtocol
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.uec.api.resolve import resolve_envelope


logger = logging.getLogger(__name__)


@runtime_checkable
class BaselineRegistryProtocol(Protocol):
    """Protocol for registries supporting baseline operations."""

    def get_baseline(self, project: str) -> dict[str, Any] | None:
        """
        Get baseline metadata for a project.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        dict or None
            Baseline metadata (must contain 'run_id') or None if not set.
        """
        ...

    def load(self, run_id: str) -> RunRecord:
        """
        Load a run record by ID.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord
            Loaded run record.
        """
        ...

    def set_baseline(self, project: str, run_id: str) -> None:
        """
        Set baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.
        run_id : str
            Run ID to set as baseline.
        """
        ...


def _opt_float(name: str, value: Any) -> float | None:
    """
    Convert value to float or None with validation.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : Any
        Value to convert.

    Returns
    -------
    float or None
        Converted value.

    Raises
    ------
    ValueError
        If value cannot be converted to float.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")


@dataclass
class VerifyPolicy:
    """
    Verification policy configuration.

    Defines the criteria for verification pass/fail decisions.

    Attributes
    ----------
    params_must_match : bool
        Require parameters to match exactly. Default is True.
    program_must_match : bool
        Require program artifacts to match (according to program_match_mode).
        Default is True.
    program_match_mode : ProgramMatchMode
        How to compare program artifacts:
        - EXACT: require byte-for-byte identical artifacts
        - STRUCTURAL: require same circuit structure (ignore parameter values)
        - EITHER: pass if exact OR structural match (recommended default)
        Default is EITHER.
    fingerprint_must_match : bool
        Require run fingerprints to match. Default is False.
    tvd_max : float or None
        Maximum allowed TVD (hard limit). If None, TVD is not checked.
        When both tvd_max and noise_factor are set, the STRICTER (min)
        threshold is used.
    noise_factor : float or None
        If set, fail if tvd > noise_factor * noise_p95.
        Uses bootstrap-calibrated noise_p95 threshold.
        Recommended: 1.0-1.5 for CI gating (since p95 is already conservative).
    noise_alpha : float
        Quantile level for noise_p95 threshold. Default is 0.95.
        Use 0.99 for stricter false positive control in production CI.
    noise_n_boot : int
        Number of bootstrap iterations for noise estimation. Default is 1000.
        Use 1000-3000 for CI gating decisions.
    noise_seed : int or None
        Random seed for reproducible noise estimation.
        Default is 12345 for deterministic results.
    noise_pvalue_cutoff : float or None
        If set, only fail when p_value < cutoff in addition to
        tvd > threshold. This reduces false positives. Default is None.
    allow_missing_baseline : bool
        If True, verification passes when no baseline exists.
        Default is False.

    Notes
    -----
    The ``noise_factor`` multiplies the bootstrap-calibrated `noise_p95`
    threshold. This provides better false positive control:

    - noise_factor=1.0: Use raw p95 threshold (5% false positive rate under H0)
    - noise_factor=1.2: Slightly more lenient (recommended for noisy hardware)
    - noise_factor=1.5: Very lenient (use for exploratory runs)

    For strict production CI, consider using noise_alpha=0.99 combined with
    noise_factor=1.0 and noise_pvalue_cutoff=0.05.

    When both tvd_max and noise_factor are set, the STRICTER threshold wins.
    This ensures tvd_max acts as a true hard limit.
    """

    params_must_match: bool = True
    program_must_match: bool = True
    program_match_mode: ProgramMatchMode = ProgramMatchMode.EITHER
    fingerprint_must_match: bool = False
    tvd_max: float | None = None
    noise_factor: float | None = None
    noise_alpha: float = 0.95
    noise_n_boot: int = 1000
    noise_seed: int | None = 12345
    noise_pvalue_cutoff: float | None = None
    allow_missing_baseline: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifyPolicy:
        """
        Create policy from dictionary.

        Parameters
        ----------
        d : dict
            Configuration dictionary.

        Returns
        -------
        VerifyPolicy
            Configured policy instance.

        Raises
        ------
        ValueError
            If program_match_mode is invalid or numeric fields cannot be parsed.
        """
        mode_raw = d.get("program_match_mode", "either")

        if isinstance(mode_raw, ProgramMatchMode):
            mode = mode_raw
        elif isinstance(mode_raw, str):
            try:
                mode = ProgramMatchMode(mode_raw.lower())
            except ValueError:
                valid_modes = [m.value for m in ProgramMatchMode]
                raise ValueError(
                    f"Invalid program_match_mode={mode_raw!r}. "
                    f"Allowed values: {valid_modes}"
                )
        else:
            raise ValueError(
                f"program_match_mode must be str or ProgramMatchMode, "
                f"got {type(mode_raw).__name__}"
            )

        return cls(
            params_must_match=bool(d.get("params_must_match", True)),
            program_must_match=bool(d.get("program_must_match", True)),
            program_match_mode=mode,
            fingerprint_must_match=bool(d.get("fingerprint_must_match", False)),
            tvd_max=_opt_float("tvd_max", d.get("tvd_max")),
            noise_factor=_opt_float("noise_factor", d.get("noise_factor")),
            noise_alpha=float(d.get("noise_alpha", 0.95)),
            noise_n_boot=int(d.get("noise_n_boot", 1000)),
            noise_seed=(
                int(d["noise_seed"]) if d.get("noise_seed") is not None else 12345
            ),
            noise_pvalue_cutoff=_opt_float(
                "noise_pvalue_cutoff", d.get("noise_pvalue_cutoff")
            ),
            allow_missing_baseline=bool(d.get("allow_missing_baseline", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "params_must_match": self.params_must_match,
            "program_must_match": self.program_must_match,
            "program_match_mode": self.program_match_mode.value,
            "fingerprint_must_match": self.fingerprint_must_match,
            "allow_missing_baseline": self.allow_missing_baseline,
            "noise_alpha": self.noise_alpha,
            "noise_n_boot": self.noise_n_boot,
        }
        if self.tvd_max is not None:
            d["tvd_max"] = self.tvd_max
        if self.noise_factor is not None:
            d["noise_factor"] = self.noise_factor
        if self.noise_seed is not None:
            d["noise_seed"] = self.noise_seed
        if self.noise_pvalue_cutoff is not None:
            d["noise_pvalue_cutoff"] = self.noise_pvalue_cutoff
        return d


def _normalize_policy(policy: VerifyPolicy | dict[str, Any] | None) -> VerifyPolicy:
    """Convert policy input to VerifyPolicy instance."""
    if policy is None:
        return VerifyPolicy()
    if isinstance(policy, dict):
        return VerifyPolicy.from_dict(policy)
    return policy


# =============================================================================
# Core: Envelope-only verification
# =============================================================================


def verify_contexts(
    ctx_baseline: RunContext,
    ctx_candidate: RunContext,
    policy: VerifyPolicy,
) -> VerifyResult:
    """
    Verify a candidate against baseline using envelope-only comparison.

    This is the core verification function operating entirely on RunContext
    (envelope + store). All verification logic (thresholds, fingerprint gating,
    noise logic) is evaluated here without any RunRecord dependencies.

    Parameters
    ----------
    ctx_baseline : RunContext
        Baseline context with resolved envelope.
    ctx_candidate : RunContext
        Candidate context with resolved envelope.
    policy : VerifyPolicy
        Verification policy defining pass/fail criteria.

    Returns
    -------
    VerifyResult
        Verification result with ok status, failures, comparison, and verdict.

    Notes
    -----
    Use the adapter functions (verify, verify_against_baseline) if you have
    RunRecord or run_id inputs. This function is for direct envelope access.
    """
    start = time.perf_counter()

    logger.info(
        "Verifying %s against baseline %s",
        ctx_candidate.run_id,
        ctx_baseline.run_id,
    )

    need_noise_context = bool(policy.noise_factor or policy.noise_pvalue_cutoff)

    comparison = diff_contexts(
        ctx_baseline,
        ctx_candidate,
        include_circuit_diff=False,
        include_noise_context=need_noise_context,
        noise_alpha=policy.noise_alpha,
        noise_n_boot=policy.noise_n_boot,
        noise_seed=policy.noise_seed if policy.noise_seed is not None else 12345,
    )

    failures: list[str] = []

    # Check fingerprint match
    if policy.fingerprint_must_match:
        if comparison.fingerprint_a and comparison.fingerprint_b:
            if comparison.fingerprint_a != comparison.fingerprint_b:
                failures.append(
                    f"fingerprint mismatch: baseline={comparison.fingerprint_a} "
                    f"candidate={comparison.fingerprint_b}"
                )
        else:
            failures.append(
                "fingerprint missing: cannot enforce fingerprint_must_match"
            )

    # Check params match
    if policy.params_must_match and not comparison.params.get("match", False):
        changed = comparison.params.get("changed", {})
        added = comparison.params.get("added", {})
        removed = comparison.params.get("removed", {})
        failures.append(
            f"params differ: {len(changed)} changed, "
            f"{len(removed)} only in baseline, {len(added)} only in candidate"
        )

    # Check program match
    if policy.program_must_match:
        program_ok = comparison.program_matches(policy.program_match_mode)
        if not program_ok:
            mode_desc = {
                ProgramMatchMode.EXACT: "exact artifact match required",
                ProgramMatchMode.STRUCTURAL: "structural match required",
                ProgramMatchMode.EITHER: "no match (neither exact nor structural)",
            }
            failures.append(
                f"program artifacts differ ({mode_desc[policy.program_match_mode]})"
            )

    # Check TVD threshold
    if comparison.tvd is not None:
        effective_threshold: float | None = None
        thresholds_to_apply: list[float] = []

        if policy.tvd_max is not None:
            thresholds_to_apply.append(policy.tvd_max)

        if policy.noise_factor and comparison.noise_context:
            noise_threshold = policy.noise_factor * comparison.noise_context.noise_p95
            thresholds_to_apply.append(noise_threshold)

        if thresholds_to_apply:
            effective_threshold = min(thresholds_to_apply)

        if effective_threshold is not None and comparison.tvd > effective_threshold:
            should_fail = True
            if policy.noise_pvalue_cutoff is not None and comparison.noise_context:
                p_value = comparison.noise_context.p_value
                if p_value is not None and p_value >= policy.noise_pvalue_cutoff:
                    should_fail = False
                    comparison.warnings.append(
                        f"TVD ({comparison.tvd:.6f}) exceeds threshold "
                        f"({effective_threshold:.6f}) but p-value ({p_value:.4f}) "
                        f">= cutoff ({policy.noise_pvalue_cutoff}). Not failing."
                    )

            if should_fail:
                if policy.noise_factor and comparison.noise_context:
                    ctx = comparison.noise_context
                    p_value_info = ""
                    if ctx.p_value is not None:
                        p_value_info = f", p-value={ctx.p_value:.4f}"

                    failures.append(
                        f"TVD too high: {comparison.tvd:.6f} > "
                        f"{effective_threshold:.6f} "
                        f"(noise_factor={policy.noise_factor}x noise_p95 of "
                        f"{ctx.noise_p95:.6f}{p_value_info})"
                    )
                else:
                    failures.append(
                        f"TVD too high: {comparison.tvd:.6f} > {effective_threshold:.6f}"
                    )
    else:
        if policy.tvd_max is not None or policy.noise_factor is not None:
            comparison.warnings.append(
                "TVD check skipped: no measurement counts available "
                "(analytic mode or missing results artifact)"
            )

    # Build verdict if failures
    verdict = None
    if failures:
        verdict = build_verdict_contexts(
            result=comparison,
            ctx_a=ctx_baseline,
            ctx_b=ctx_candidate,
        )

    duration_ms = (time.perf_counter() - start) * 1000

    result = VerifyResult(
        ok=len(failures) == 0,
        failures=failures,
        comparison=comparison,
        baseline_run_id=ctx_baseline.run_id,
        candidate_run_id=ctx_candidate.run_id,
        duration_ms=duration_ms,
        verdict=verdict,
    )

    logger.info(
        "Verification %s in %.1fms",
        "PASSED" if result.ok else f"FAILED ({len(failures)} failures)",
        duration_ms,
    )

    return result


# =============================================================================
# Adapters: RunRecord/registry to RunContext conversion
# =============================================================================


def verify(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    store_baseline: ObjectStoreProtocol,
    store_candidate: ObjectStoreProtocol,
    policy: VerifyPolicy | dict[str, Any] | None = None,
) -> VerifyResult:
    """
    Verify a candidate run against a baseline run.

    This is an adapter function that converts RunRecord inputs to RunContext
    and delegates to verify_contexts. Use this when you have RunRecord objects.

    Parameters
    ----------
    baseline : RunRecord
        Baseline run record.
    candidate : RunRecord
        Candidate run record to verify.
    store_baseline : ObjectStoreProtocol
        Object store for baseline artifacts.
    store_candidate : ObjectStoreProtocol
        Object store for candidate artifacts.
    policy : VerifyPolicy or dict or None
        Verification policy. Uses defaults if not provided.

    Returns
    -------
    VerifyResult
        Verification result with ok status, failures, and comparison.
    """
    pol = _normalize_policy(policy)

    envelope_baseline = resolve_envelope(baseline, store_baseline)
    envelope_candidate = resolve_envelope(candidate, store_candidate)

    ctx_baseline = RunContext(
        run_id=baseline.run_id,
        envelope=envelope_baseline,
        store=store_baseline,
    )
    ctx_candidate = RunContext(
        run_id=candidate.run_id,
        envelope=envelope_candidate,
        store=store_candidate,
    )

    return verify_contexts(ctx_baseline, ctx_candidate, pol)


def verify_against_baseline(
    candidate: RunRecord,
    *,
    project: str,
    store: ObjectStoreProtocol,
    registry: BaselineRegistryProtocol,
    policy: VerifyPolicy | dict[str, Any] | None = None,
    promote_on_pass: bool = False,
) -> VerifyResult:
    """
    Verify a candidate run against the stored baseline for a project.

    Parameters
    ----------
    candidate : RunRecord
        Candidate run record.
    project : str
        Project name to look up baseline for.
    store : ObjectStoreProtocol
        Object store for artifacts.
    registry : BaselineRegistryProtocol
        Registry supporting baseline operations.
    policy : VerifyPolicy or dict or None
        Verification policy.
    promote_on_pass : bool, default=False
        If True and verification passes, promote candidate to new baseline.

    Returns
    -------
    VerifyResult
        Verification result.

    Raises
    ------
    ValueError
        If no baseline is set and allow_missing_baseline is False.
    """
    pol = _normalize_policy(policy)
    baseline_info = registry.get_baseline(project)

    if not baseline_info or not baseline_info.get("run_id"):
        if pol.allow_missing_baseline:
            logger.info("No baseline for project %s, allowing pass", project)
            result = VerifyResult(
                ok=True,
                failures=[],
                comparison=None,
                baseline_run_id=None,
                candidate_run_id=candidate.run_id,
            )
            if promote_on_pass:
                registry.set_baseline(project, candidate.run_id)
                logger.info("Promoted %s to baseline for %s", candidate.run_id, project)
            return result
        raise ValueError(f"No baseline set for project: {project}")

    baseline = registry.load(str(baseline_info["run_id"]))

    result = verify(
        baseline,
        candidate,
        store_baseline=store,
        store_candidate=store,
        policy=pol,
    )

    if result.ok and promote_on_pass:
        registry.set_baseline(project, candidate.run_id)
        logger.info("Promoted %s to baseline for %s", candidate.run_id, project)

    return result


def promote_baseline(
    run_id: str,
    *,
    project: str,
    registry: BaselineRegistryProtocol,
) -> None:
    """
    Promote a run to be the baseline for a project.

    Parameters
    ----------
    run_id : str
        Run ID to promote.
    project : str
        Project name.
    registry : BaselineRegistryProtocol
        Registry to update.

    Raises
    ------
    RunNotFoundError
        If the run does not exist.
    """
    registry.load(run_id)  # Verify run exists
    registry.set_baseline(project, run_id)
    logger.info("Promoted %s to baseline for %s", run_id, project)
