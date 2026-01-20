# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device calibration drift detection and analysis.

This module provides configurable thresholds and utilities for detecting
significant calibration drift between device snapshots. Drift detection
helps identify when hardware changes may affect experiment comparability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from devqubit_engine.compare.results import DriftResult, MetricDrift
from devqubit_engine.uec.models.calibration import DeviceCalibration
from devqubit_engine.uec.models.device import DeviceSnapshot


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftThresholds:
    """
    Configurable thresholds for drift detection.

    All thresholds are expressed as fractional changes (e.g., 0.10 = 10%).
    When a metric changes by more than its threshold, it is flagged as
    significant drift.

    Minimum absolute thresholds (min_abs_*) provide protection against
    false positives when baseline values are small. A change must exceed
    BOTH the relative threshold AND the absolute minimum to be flagged.

    Attributes
    ----------
    t1_us : float
        Threshold for median T1 relaxation time. Default is 0.10 (10%).
    t2_us : float
        Threshold for median T2 coherence time. Default is 0.10 (10%).
    readout_error : float
        Threshold for readout error rate. Default is 0.20 (20%).
    q2_error : float
        Threshold for two-qubit gate error. Default is 0.20 (20%).
    min_abs_readout_error : float
        Minimum absolute change for readout error. Default is 0.003.
    min_abs_q2_error : float
        Minimum absolute change for 2Q gate error. Default is 0.001.
    """

    t1_us: float = 0.10
    t2_us: float = 0.10
    readout_error: float = 0.20
    q2_error: float = 0.20
    min_abs_readout_error: float = 0.003
    min_abs_q2_error: float = 0.001

    def get_threshold(self, metric: str) -> float | None:
        """
        Get threshold for a metric name.

        Parameters
        ----------
        metric : str
            Metric name (e.g., "median_t1_us" or "t1_us").

        Returns
        -------
        float or None
            Threshold value, or None if metric not recognized.
        """
        # Strip common prefixes
        key = metric.replace("median_", "").replace("2q_", "q2_")
        return getattr(self, key, None)

    def get_min_abs(self, metric: str) -> float | None:
        """
        Get minimum absolute threshold for a metric name.

        Parameters
        ----------
        metric : str
            Metric name (e.g., "median_readout_error").

        Returns
        -------
        float or None
            Minimum absolute threshold, or None if not applicable.
        """
        # Map metric names to min_abs fields
        key = metric.replace("median_", "").replace("2q_", "q2_")
        min_abs_key = f"min_abs_{key}"
        return getattr(self, min_abs_key, None)

    def to_dict(self) -> dict[str, float]:
        """
        Return thresholds as dictionary.

        Returns
        -------
        dict
            Threshold values keyed by metric name.
        """
        return {
            "median_t1_us": self.t1_us,
            "median_t2_us": self.t2_us,
            "median_readout_error": self.readout_error,
            "median_2q_error": self.q2_error,
            "min_abs_readout_error": self.min_abs_readout_error,
            "min_abs_q2_error": self.min_abs_q2_error,
        }


#: Default thresholds for drift detection.
DEFAULT_THRESHOLDS = DriftThresholds()

#: Metrics to check for drift.
_DRIFT_METRICS = (
    "median_t1_us",
    "median_t2_us",
    "median_readout_error",
    "median_2q_error",
)


def _compute_metric_drift(
    metric: str,
    val_a: float | None,
    val_b: float | None,
    threshold: float | None,
    min_abs: float | None = None,
) -> MetricDrift:
    """
    Compute drift for a single metric.

    Parameters
    ----------
    metric : str
        Metric name.
    val_a : float or None
        Baseline value.
    val_b : float or None
        Candidate value.
    threshold : float or None
        Significance threshold (relative change).
    min_abs : float or None
        Minimum absolute change required (for error rate metrics).

    Returns
    -------
    MetricDrift
        Drift result for this metric.
    """
    drift = MetricDrift(
        metric=metric,
        value_a=val_a,
        value_b=val_b,
        threshold=threshold,
    )

    if val_a is None or val_b is None:
        return drift

    drift.delta = val_b - val_a
    abs_delta = abs(drift.delta)

    if val_a != 0.0:
        frac_change = abs_delta / abs(val_a)
        drift.percent_change = frac_change * 100.0
        if threshold is not None:
            # Check relative threshold
            exceeds_rel = frac_change > threshold
            # Check minimum absolute threshold (if applicable)
            exceeds_abs = min_abs is None or abs_delta > min_abs
            # Both conditions must be met
            drift.significant = exceeds_rel and exceeds_abs
    elif val_b != 0.0:
        # val_a is zero but val_b is not
        drift.percent_change = float("inf")
        # For zero baseline, only check min_abs if available
        if min_abs is not None:
            drift.significant = abs_delta > min_abs
        else:
            drift.significant = True

    return drift


def compute_drift(
    snapshot_a: DeviceSnapshot,
    snapshot_b: DeviceSnapshot,
    thresholds: DriftThresholds | None = None,
) -> DriftResult:
    """
    Compute device drift between two snapshots.

    Compares calibration metrics from two device snapshots and flags
    any that exceed configured thresholds.

    Parameters
    ----------
    snapshot_a : DeviceSnapshot
        Baseline device snapshot.
    snapshot_b : DeviceSnapshot
        Candidate device snapshot.
    thresholds : DriftThresholds, optional
        Drift thresholds. Uses DEFAULT_THRESHOLDS if not provided.

    Returns
    -------
    DriftResult
        Complete drift analysis with per-metric results.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    result = DriftResult()

    cal_a = snapshot_a.calibration
    cal_b = snapshot_b.calibration

    if cal_a is None and cal_b is None:
        logger.debug("No calibration data in either snapshot")
        return result

    result.has_calibration_data = True
    result.calibration_time_a = cal_a.calibration_time if cal_a else None
    result.calibration_time_b = cal_b.calibration_time if cal_b else None

    for metric in _DRIFT_METRICS:
        val_a = getattr(cal_a, metric, None) if cal_a else None
        val_b = getattr(cal_b, metric, None) if cal_b else None
        threshold = thresholds.get_threshold(metric)
        min_abs = thresholds.get_min_abs(metric)

        drift = _compute_metric_drift(metric, val_a, val_b, threshold, min_abs)
        result.metrics.append(drift)

        if drift.significant:
            result.significant_drift = True
            logger.debug(
                "Significant drift in %s: %.2f%% (threshold: %.0f%%)",
                metric,
                drift.percent_change or 0,
                (threshold or 0) * 100,
            )

    if result.significant_drift:
        logger.info(
            "Device drift detected: %d metrics exceed thresholds",
            len(result.top_drifts),
        )

    return result


def compare_calibrations(
    cal_a: DeviceCalibration,
    cal_b: DeviceCalibration,
    thresholds: DriftThresholds | None = None,
) -> dict[str, Any]:
    """
    Compare two calibration objects directly.

    Convenience function that wraps calibrations in temporary snapshots
    and returns drift analysis as a dictionary.

    Parameters
    ----------
    cal_a : DeviceCalibration
        Baseline calibration.
    cal_b : DeviceCalibration
        Candidate calibration.
    thresholds : DriftThresholds, optional
        Drift thresholds.

    Returns
    -------
    dict
        Drift analysis result as dictionary.
    """
    snapshot_a = DeviceSnapshot(
        captured_at="",
        backend_name="",
        backend_type="",
        provider="",
        calibration=cal_a,
    )
    snapshot_b = DeviceSnapshot(
        captured_at="",
        backend_name="",
        backend_type="",
        provider="",
        calibration=cal_b,
    )

    return compute_drift(snapshot_a, snapshot_b, thresholds).to_dict()
