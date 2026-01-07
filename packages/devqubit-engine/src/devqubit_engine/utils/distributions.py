# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Probability distribution utilities for quantum measurement comparison.

This module provides vectorized functions for working with probability
distributions, including normalization, distance metrics, and sampling
noise estimation using parametric bootstrap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    """
    Normalize raw shot counts into probabilities.

    Parameters
    ----------
    counts : dict
        Raw counts mapping outcome bitstrings to shot counts.

    Returns
    -------
    dict
        Probabilities in [0, 1]. Empty dict if total is zero.
    """
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def counts_to_arrays(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """
    Convert two count dictionaries to aligned probability arrays.

    Parameters
    ----------
    counts_a : dict
        First count distribution.
    counts_b : dict
        Second count distribution.

    Returns
    -------
    p_a : ndarray
        Probability array for counts_a.
    p_b : ndarray
        Probability array for counts_b.
    keys : list of str
        Sorted outcome keys (shared order).

    Notes
    -----
    Both arrays have the same length and key order, with zeros
    filled in for missing outcomes.
    """
    all_keys = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    n = len(all_keys)

    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), []

    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    if total_a <= 0:
        p_a = np.zeros(n, dtype=np.float64)
    else:
        p_a = np.array(
            [counts_a.get(k, 0) / total_a for k in all_keys], dtype=np.float64
        )

    if total_b <= 0:
        p_b = np.zeros(n, dtype=np.float64)
    else:
        p_b = np.array(
            [counts_b.get(k, 0) / total_b for k in all_keys], dtype=np.float64
        )

    return p_a, p_b, all_keys


def total_variation_distance(
    p: dict[str, float] | NDArray[np.float64],
    q: dict[str, float] | NDArray[np.float64],
) -> float:
    """
    Compute Total Variation Distance between two distributions.

    The TVD is defined as::

        TVD(p, q) = 0.5 * sum(|p(x) - q(x)|)

    Parameters
    ----------
    p : dict or ndarray
        First probability distribution.
    q : dict or ndarray
        Second probability distribution (must be same type as p).

    Returns
    -------
    float
        TVD in [0, 1]. 0 = identical, 1 = disjoint support.

    Raises
    ------
    TypeError
        If p and q are not both dicts or both ndarrays.
    """
    if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):
        return 0.5 * float(np.sum(np.abs(p - q)))

    if isinstance(p, dict) and isinstance(q, dict):
        all_keys = set(p.keys()) | set(q.keys())
        if not all_keys:
            return 0.0
        total = sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)
        return 0.5 * total

    raise TypeError("Both p and q must be either dicts or ndarrays")


def tvd_from_counts(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> float:
    """
    Compute TVD directly from count dictionaries.

    This is a convenience function that normalizes counts and computes TVD
    in a single vectorized operation.

    Parameters
    ----------
    counts_a : dict
        First count distribution.
    counts_b : dict
        Second count distribution.

    Returns
    -------
    float
        Total Variation Distance in [0, 1].
    """
    p_a, p_b, _ = counts_to_arrays(counts_a, counts_b)
    if len(p_a) == 0:
        return 0.0
    return 0.5 * float(np.sum(np.abs(p_a - p_b)))


def expected_tvd_from_shots(shots: int, num_outcomes: int) -> float:
    """
    Estimate expected TVD from shot noise (heuristic fallback).

    For multinomial sampling with uniform-ish distributions,
    TVD scales as O(sqrt(k/n)) where k = outcomes, n = shots.

    Parameters
    ----------
    shots : int
        Total number of shots (effective shots for two-sample case).
    num_outcomes : int
        Number of distinct outcomes observed.

    Returns
    -------
    float
        Expected TVD upper bound from shot noise.

    Notes
    -----
    The formula is::

        expected_tvd ~ 1.5 * sqrt(k / 2n)

    The 1.5 factor provides a conservative bound (~90th percentile)
    without requiring distributional assumptions.
    """
    if shots <= 0 or num_outcomes <= 0:
        return 1.0

    base_tvd = np.sqrt(num_outcomes / (2.0 * shots))
    return min(1.0, float(1.5 * base_tvd))


def _bootstrap_tvd_null(
    pooled_p: NDArray[np.float64],
    shots_a: int,
    shots_b: int,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Simulate TVD distribution under H0 (same underlying distribution).

    Uses parametric bootstrap: sample from pooled distribution
    to estimate TVD variance from shot noise alone.

    Parameters
    ----------
    pooled_p : ndarray
        Pooled probability distribution (H0 assumption).
    shots_a : int
        Number of shots for first sample.
    shots_b : int
        Number of shots for second sample.
    n_boot : int
        Number of bootstrap iterations.
    rng : numpy.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    ndarray
        Array of simulated TVD values under H0.
    """
    # Generate bootstrap samples: shape (n_boot, k)
    samples_a = rng.multinomial(shots_a, pooled_p, size=n_boot)
    samples_b = rng.multinomial(shots_b, pooled_p, size=n_boot)

    # Normalize to probabilities
    p_a = samples_a / max(shots_a, 1)
    p_b = samples_b / max(shots_b, 1)

    # Compute TVD for each bootstrap iteration
    return 0.5 * np.sum(np.abs(p_a - p_b), axis=1)


@dataclass(frozen=True, slots=True)
class NoiseContext:
    """
    Sampling noise context for TVD comparison.

    Provides bootstrap-calibrated context for interpreting observed TVD.
    The primary decision signal is ``noise_p95`` (quantile-based threshold)
    rather than the simple ratio heuristic.

    Attributes
    ----------
    tvd : float
        Observed Total Variation Distance.
    expected_noise : float
        Expected (mean) TVD from shot noise alone.
    noise_ratio : float
        Ratio of observed TVD to expected noise (tvd / expected_noise).
        Kept for backward compatibility; prefer using p_value.
    shots_a : int
        Shot count from first distribution.
    shots_b : int
        Shot count from second distribution.
    num_outcomes : int
        Number of distinct outcomes observed.
    exceeds_noise : bool
        True if tvd > noise_p95 (bootstrap-calibrated threshold).
    noise_p95 : float
        95th percentile of TVD under H0 (null hypothesis).
        This is the primary threshold for conservative decisions.
        Always in [0, 1].
    p_value : float or None
        Empirical p-value: P(TVD_null >= observed_tvd).
        Computed with +1 correction to avoid p=0.
        None if bootstrap was not performed.
    method : str
        Estimation method: "bootstrap" or "heuristic".
    n_boot : int
        Number of bootstrap iterations (0 if heuristic).
    alpha : float
        Quantile level used for noise_p95 (e.g., 0.95).

    Notes
    -----
    Decision guidelines using bootstrap-calibrated thresholds:

    - tvd <= noise_p95: Consistent with sampling noise (don't reject H0)
    - tvd > noise_p95 AND p_value < 0.05: Likely a real difference
    - p_value >= 0.05: Cannot distinguish from noise

    The bootstrap approach provides better false positive control than
    the simple ratio heuristic, especially for non-uniform distributions.
    """

    tvd: float
    expected_noise: float
    noise_ratio: float
    shots_a: int
    shots_b: int
    num_outcomes: int
    exceeds_noise: bool
    noise_p95: float = 0.0
    p_value: float | None = None
    method: str = "heuristic"
    n_boot: int = 0
    alpha: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "tvd": self.tvd,
            "expected_noise": self.expected_noise,
            "noise_ratio": self.noise_ratio,
            "shots_a": self.shots_a,
            "shots_b": self.shots_b,
            "num_outcomes": self.num_outcomes,
            "exceeds_noise": self.exceeds_noise,
            "noise_p95": self.noise_p95,
            "method": self.method,
            "alpha": self.alpha,
        }
        if self.p_value is not None:
            d["p_value"] = self.p_value
        if self.n_boot > 0:
            d["n_boot"] = self.n_boot
        return d

    def interpretation(self) -> str:
        """
        Return human-readable interpretation of the noise context.

        Returns
        -------
        str
            Plain-English interpretation based on bootstrap results.
        """
        if self.method == "bootstrap" and self.p_value is not None:
            if self.p_value >= 0.10:
                return "Difference is consistent with sampling noise (p >= 0.10)"
            elif self.p_value >= 0.05:
                return "Difference is borderline; consider increasing shots (0.05 <= p < 0.10)"
            elif self.p_value >= 0.01:
                return "Difference likely exceeds sampling noise (0.01 <= p < 0.05)"
            else:
                return "Difference significantly exceeds sampling noise (p < 0.01)"

        # Fallback to ratio-based interpretation
        if self.noise_ratio < 1.5:
            return "Difference is consistent with sampling noise"
        elif self.noise_ratio < 3.0:
            return "Difference is ambiguous; consider increasing shots"
        else:
            return "Difference likely exceeds sampling noise"

    def __repr__(self) -> str:
        """Return string representation."""
        p_str = f", p={self.p_value:.3f}" if self.p_value is not None else ""
        return (
            f"NoiseContext(tvd={self.tvd:.4f}, p95={self.noise_p95:.4f}, "
            f"ratio={self.noise_ratio:.2f}x{p_str}, exceeds={self.exceeds_noise})"
        )


def compute_noise_context(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    tvd: float | None = None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.95,
    seed: int = 12345,
) -> NoiseContext:
    """
    Compute sampling noise context for count comparison.

    Uses parametric bootstrap under H0 (pooled multinomial) to estimate
    the distribution of TVD from shot noise alone. Provides conservative
    thresholds for false positive control.

    Parameters
    ----------
    counts_a : dict
        First count distribution (baseline).
    counts_b : dict
        Second count distribution (candidate).
    tvd : float, optional
        Pre-computed TVD. Computed if not provided.
    n_boot : int, default=1000
        Number of bootstrap iterations. Use 300-500 for fast feedback,
        1000+ for CI gating decisions.
    alpha : float, default=0.95
        Quantile level for noise_p95 threshold. Use 0.99 for stricter
        false positive control in production CI.
    seed : int, default=12345
        Random seed for reproducibility.

    Returns
    -------
    NoiseContext
        Noise context with bootstrap-calibrated thresholds.

    Notes
    -----
    The bootstrap approach:
    1. Pool both distributions under H0 assumption
    2. Simulate many pairs of measurements from pooled distribution
    3. Compute TVD for each simulated pair
    4. Use quantiles of simulated TVD as thresholds

    The p-value uses +1 correction: p = (1 + sum(sims >= tvd)) / (n_boot + 1)
    to avoid p=0 and provide conservative estimates.

    This is more robust than the simple O(sqrt(k/n)) heuristic,
    especially for non-uniform distributions.
    """
    shots_a = sum(counts_a.values())
    shots_b = sum(counts_b.values())
    num_outcomes = len(set(counts_a.keys()) | set(counts_b.keys()))

    if tvd is None:
        tvd = tvd_from_counts(counts_a, counts_b)

    # Compute heuristic as fallback / lower bound
    if shots_a > 0 and shots_b > 0:
        effective_shots = int(2.0 * shots_a * shots_b / (shots_a + shots_b))
    else:
        effective_shots = max(shots_a, shots_b, 1)

    heuristic = expected_tvd_from_shots(effective_shots, max(num_outcomes, 1))

    # Default values (heuristic fallback)
    method = "heuristic"
    noise_mean = heuristic
    noise_p95 = min(1.0, 2.0 * heuristic)  # Conservative fallback
    p_value: float | None = None
    actual_n_boot = 0

    # Attempt bootstrap if we have sufficient data
    if shots_a > 0 and shots_b > 0 and num_outcomes > 0:
        try:
            p_a, p_b, _ = counts_to_arrays(counts_a, counts_b)

            # Pool distributions under H0
            pooled = (p_a * shots_a + p_b * shots_b) / float(shots_a + shots_b)
            pooled = pooled / pooled.sum()  # Ensure normalization

            rng = np.random.default_rng(seed)
            sims = _bootstrap_tvd_null(pooled, shots_a, shots_b, n_boot=n_boot, rng=rng)

            bootstrap_mean = float(np.mean(sims))
            bootstrap_p95 = float(np.quantile(sims, alpha))

            # +1 correction for p-value (standard bootstrap practice)
            # This ensures p-value is never exactly 0 and provides
            # more conservative estimates for small n_boot
            bootstrap_pval = float((1 + np.sum(sims >= tvd)) / (n_boot + 1))

            # Conservative approach: never let bootstrap underestimate noise
            # This protects against pathological cases with unobserved outcomes
            noise_mean = max(bootstrap_mean, heuristic)

            # Clamp to [0, 1] range (TVD is always in this range)
            noise_p95 = min(1.0, max(bootstrap_p95, 1.5 * heuristic))

            p_value = bootstrap_pval
            method = "bootstrap"
            actual_n_boot = n_boot

            logger.debug(
                "Bootstrap noise estimation: mean=%.6f, p95=%.6f, p_value=%.4f",
                noise_mean,
                noise_p95,
                p_value,
            )

        except Exception as e:
            logger.debug("Bootstrap failed, using heuristic: %s", e)
            # Keep heuristic values

    # Compute ratio for backward compatibility
    if noise_mean > 0:
        noise_ratio = tvd / noise_mean
    else:
        noise_ratio = float("inf") if tvd > 0 else 0.0

    # Primary decision: use p95 threshold (not ratio)
    exceeds_noise = tvd > noise_p95

    logger.debug(
        "Noise context [%s]: tvd=%.6f, p95=%.6f, exceeds=%s",
        method,
        tvd,
        noise_p95,
        exceeds_noise,
    )

    return NoiseContext(
        tvd=tvd,
        expected_noise=noise_mean,
        noise_ratio=noise_ratio,
        shots_a=shots_a,
        shots_b=shots_b,
        num_outcomes=num_outcomes,
        exceeds_noise=exceeds_noise,
        noise_p95=noise_p95,
        p_value=p_value,
        method=method,
        n_boot=actual_n_boot,
        alpha=alpha,
    )
