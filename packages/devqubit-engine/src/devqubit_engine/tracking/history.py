# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Metric history read-path for time-series data.

Provides streaming and tabular access to logged metric series, suitable
for plotting overlays, long-format exports, and downstream analytics.

The core API is iterator-based (no heavy dependencies).  Optional
``to_dataframe()`` converts to ``pandas.DataFrame`` for convenience.

Examples
--------
Iterate over a single metric's history:

>>> from devqubit_engine.tracking.history import iter_metric_points
>>> for pt in iter_metric_points(registry, run_id, "train/loss"):
...     print(pt["step"], pt["value"])

Build a long-format table across multiple runs:

>>> from devqubit_engine.tracking.history import metric_history_long
>>> rows = metric_history_long(registry, run_ids=["abc", "def"], keys=["loss"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Sequence


if TYPE_CHECKING:
    import pandas as pd
    from devqubit_engine.storage.types import RegistryProtocol

logger = logging.getLogger(__name__)


def iter_metric_points(
    registry: RegistryProtocol,
    run_id: str,
    key: str,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield metric points for *key* from a single run.

    Points are returned in step-ascending order.  When the registry
    exposes a ``iter_metric_points`` method (e.g. ``LocalRegistry``
    with the ``metric_points`` table), that fast-path is used.
    Otherwise, the full ``metric_series`` from the run record is
    iterated.

    Parameters
    ----------
    registry : RegistryProtocol
        Registry to query.
    run_id : str
        Run identifier.
    key : str
        Metric name (e.g. ``"train/loss"``).
    start_step : int, optional
        Inclusive lower bound on step.
    end_step : int, optional
        Inclusive upper bound on step.

    Yields
    ------
    dict
        ``{"value": float, "step": int, "timestamp": str}``
    """
    # Fast path: dedicated metric_points table
    if hasattr(registry, "iter_metric_points"):
        yield from registry.iter_metric_points(
            run_id,
            key,
            start_step=start_step,
            end_step=end_step,
        )
        return

    # Fallback: read from record JSON
    record = registry.load(run_id)
    series = record.metric_series.get(key, [])
    sorted_pts = sorted(series, key=lambda p: p.get("step", 0))

    for pt in sorted_pts:
        s = pt.get("step", 0)
        if start_step is not None and s < start_step:
            continue
        if end_step is not None and s > end_step:
            break
        yield pt


def metric_history_long(
    registry: RegistryProtocol,
    *,
    run_ids: Sequence[str] | None = None,
    project: str | None = None,
    group_id: str | None = None,
    keys: Sequence[str] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    max_points: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build a long-format table of metric history across runs.

    Each row is ``{"run_id", "project", "key", "step", "timestamp",
    "value", ...}``.  When *max_points* is set and the result set
    exceeds that limit, uniform down-sampling is applied per
    (run_id, key) group.

    Parameters
    ----------
    registry : RegistryProtocol
        Registry to query.
    run_ids : sequence of str, optional
        Explicit list of run IDs.  When ``None``, *project* or
        *group_id* must be provided.
    project : str, optional
        Filter runs by project name.
    group_id : str, optional
        Filter runs by group ID.
    keys : sequence of str, optional
        Metric names to include.  ``None`` means all available keys.
    start_step : int, optional
        Inclusive lower bound on step.
    end_step : int, optional
        Inclusive upper bound on step.
    max_points : int, optional
        Maximum total points to return.  Applies per (run_id, key) via
        uniform stride down-sampling.

    Returns
    -------
    list of dict
        Long-format rows suitable for plotting or DataFrame conversion.

    Raises
    ------
    ValueError
        If neither *run_ids* nor *project*/*group_id* is provided.
    """
    if run_ids is None:
        if project is None and group_id is None:
            raise ValueError("Provide run_ids, project, or group_id to select runs.")
        summaries = registry.list_runs(
            project=project,
            group_id=group_id,
            limit=10_000,
        )
        run_ids = [s["run_id"] for s in summaries if s.get("run_id")]

    keys_set = set(keys) if keys else None
    rows: list[dict[str, Any]] = []

    for rid in run_ids:
        record = registry.load(rid)
        proj_name = record.project

        # Prefer metric_points table (fast, works for in-progress runs).
        # Fall back to record JSON for legacy records.
        if hasattr(registry, "load_metric_series"):
            series_dict = registry.load_metric_series(rid)
            if not series_dict:
                series_dict = record.metric_series
        else:
            series_dict = record.metric_series

        for mkey, points in series_dict.items():
            if keys_set and mkey not in keys_set:
                continue

            sorted_pts = sorted(points, key=lambda p: p.get("step", 0))

            # Step-range filter
            filtered = _filter_step_range(sorted_pts, start_step, end_step)

            for pt in filtered:
                rows.append(
                    {
                        "run_id": rid,
                        "project": proj_name,
                        "key": mkey,
                        "step": pt.get("step", 0),
                        "timestamp": pt.get("timestamp", ""),
                        "value": pt["value"],
                    }
                )

    # Down-sample if needed
    if max_points and len(rows) > max_points:
        rows = _downsample_rows(rows, max_points)

    return rows


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert long-format rows to a ``pandas.DataFrame``.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`metric_history_long`.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns ``run_id``, ``project``,
        ``key``, ``step``, ``timestamp``, ``value``.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame export. "
            "Install it with: pip install pandas"
        ) from None

    return pd.DataFrame(rows)


def _filter_step_range(
    points: list[dict[str, Any]],
    start_step: int | None,
    end_step: int | None,
) -> list[dict[str, Any]]:
    """
    Filter a sorted point list to ``[start_step, end_step]``.

    Parameters
    ----------
    points : list of dict
        Sorted (by step) metric points.
    start_step : int or None
        Inclusive lower bound.
    end_step : int or None
        Inclusive upper bound.

    Returns
    -------
    list of dict
        Filtered subset.
    """
    if start_step is None and end_step is None:
        return points

    result: list[dict[str, Any]] = []
    for pt in points:
        s = pt.get("step", 0)
        if start_step is not None and s < start_step:
            continue
        if end_step is not None and s > end_step:
            break
        result.append(pt)
    return result


def _downsample_rows(
    rows: list[dict[str, Any]],
    max_points: int,
) -> list[dict[str, Any]]:
    """
    Uniformly down-sample rows to at most *max_points*.

    Preserves the first and last point of each (run_id, key) group and
    evenly distributes the remaining budget.

    Parameters
    ----------
    rows : list of dict
        Full row set.
    max_points : int
        Target maximum row count.

    Returns
    -------
    list of dict
        Down-sampled rows.
    """
    # Group by (run_id, key)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        gk = (r["run_id"], r["key"])
        groups.setdefault(gk, []).append(r)

    n_groups = len(groups) or 1
    budget_per_group = max(2, max_points // n_groups)

    result: list[dict[str, Any]] = []
    for pts in groups.values():
        if len(pts) <= budget_per_group:
            result.extend(pts)
        else:
            stride = max(1, (len(pts) - 1) // (budget_per_group - 1))
            sampled = pts[::stride]
            if pts[-1] is not sampled[-1]:
                sampled.append(pts[-1])
            result.extend(sampled[:budget_per_group])

    return result
