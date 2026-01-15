# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Baseline verification for CI/regression testing.

This module provides policy-based verification of a candidate run against
a baseline run, with bootstrap-calibrated noise thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from devqubit_engine.compare.diff import diff_runs
from devqubit_engine.compare.results import ProgramMatchMode, VerifyResult
from devqubit_engine.compare.verdict import build_verdict
from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.types import ObjectStoreProtocol


logger = logging.getLogger(__name__)


@runtime_checkable
class BaselineRegistryProtocol(Protocol):
    """
    Protocol for registries supporting baseline operations.
    """

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
        Maximum allowed TVD. If None, TVD is not checked.
    noise_factor : float or None
        If set, fail if tvd > noise_factor * noise_p95.
        Uses bootstrap-calibrated noise_p95 threshold.
        Recommended: 1.0-1.5 for CI gating (since p95 is already conservative).
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

    For strict production CI, consider using alpha=0.99 in noise computation
    (via diff_runs kwargs) combined with noise_factor=1.0.
    """

    params_must_match: bool = True
    program_must_match: bool = True
    program_match_mode: ProgramMatchMode = ProgramMatchMode.EITHER
    fingerprint_must_match: bool = False
    tvd_max: float | None = None
    noise_factor: float | None = None
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
        }
        if self.tvd_max is not None:
            d["tvd_max"] = self.tvd_max
        if self.noise_factor is not None:
            d["noise_factor"] = self.noise_factor
        return d


def _normalize_policy(policy: VerifyPolicy | dict[str, Any] | None) -> VerifyPolicy:
    """Convert policy input to VerifyPolicy instance."""
    if policy is None:
        return VerifyPolicy()
    if isinstance(policy, dict):
        return VerifyPolicy.from_dict(policy)
    return policy


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

    Performs comparison and applies policy checks to determine
    pass/fail status. Uses bootstrap-calibrated noise thresholds
    for robust false positive control.

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
    start = time.perf_counter()
    pol = _normalize_policy(policy)

    logger.info(
        "Verifying %s against baseline %s",
        candidate.run_id,
        baseline.run_id,
    )

    # Optimize: only compute noise_context if noise_factor is set
    # Circuit diff is computed on-demand in build_verdict if needed
    comparison = diff_runs(
        baseline,
        candidate,
        store_a=store_baseline,
        store_b=store_candidate,
        include_circuit_diff=False,
        include_noise_context=bool(pol.noise_factor),
    )

    failures: list[str] = []

    # Check fingerprint match
    if pol.fingerprint_must_match:
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
    if pol.params_must_match and not comparison.params.get("match", False):
        changed = comparison.params.get("changed", {})
        added = comparison.params.get("added", {})
        removed = comparison.params.get("removed", {})
        failures.append(
            f"params differ: {len(changed)} changed, "
            f"{len(removed)} only in baseline, {len(added)} only in candidate"
        )

    # Check program match (using policy's match mode)
    if pol.program_must_match:
        program_ok = comparison.program_matches(pol.program_match_mode)
        if not program_ok:
            mode_desc = {
                ProgramMatchMode.EXACT: "exact artifact match required",
                ProgramMatchMode.STRUCTURAL: "structural match required",
                ProgramMatchMode.EITHER: "no match (neither exact nor structural)",
            }
            failures.append(
                f"program artifacts differ ({mode_desc[pol.program_match_mode]})"
            )

    # Check TVD threshold (with bootstrap-calibrated noise option)
    if comparison.tvd is not None:
        effective_threshold = pol.tvd_max

        # Apply noise-aware threshold using bootstrap-calibrated p95
        if pol.noise_factor and comparison.noise_context:
            # Use noise_p95 (bootstrap-calibrated) instead of expected_noise
            noise_threshold = pol.noise_factor * comparison.noise_context.noise_p95

            if effective_threshold is None:
                effective_threshold = noise_threshold
            else:
                effective_threshold = max(effective_threshold, noise_threshold)

        if effective_threshold is not None and comparison.tvd > effective_threshold:
            if pol.noise_factor and comparison.noise_context:
                ctx = comparison.noise_context
                p_value_info = ""
                if ctx.p_value is not None:
                    p_value_info = f", p-value={ctx.p_value:.4f}"

                failures.append(
                    f"TVD too high: {comparison.tvd:.6f} > "
                    f"{effective_threshold:.6f} "
                    f"(noise_factor={pol.noise_factor}x noise_p95 of "
                    f"{ctx.noise_p95:.6f}{p_value_info})"
                )
            else:
                failures.append(
                    f"TVD too high: {comparison.tvd:.6f} > {effective_threshold:.6f}"
                )
    else:
        # TVD not available (no counts / analytic mode)
        if pol.tvd_max is not None or pol.noise_factor is not None:
            comparison.warnings.append(
                "TVD check skipped: no measurement counts available "
                "(analytic mode or missing results artifact)"
            )

    # Build verdict if failures (circuit diff computed on-demand here)
    verdict = None
    if failures:
        verdict = build_verdict(
            result=comparison,
            run_a=baseline,
            run_b=candidate,
            store_a=store_baseline,
            store_b=store_candidate,
        )

    duration_ms = (time.perf_counter() - start) * 1000

    result = VerifyResult(
        ok=len(failures) == 0,
        failures=failures,
        comparison=comparison,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        duration_ms=duration_ms,
        verdict=verdict,
    )

    logger.info(
        "Verification %s in %.1fms",
        "PASSED" if result.ok else f"FAILED ({len(failures)} failures)",
        duration_ms,
    )

    return result


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
