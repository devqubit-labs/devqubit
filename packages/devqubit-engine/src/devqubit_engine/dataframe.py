# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Convert runs to tabular DataFrames.

Provides :func:`runs_to_dataframe` which loads runs from the registry,
flattens nested params/metrics/tags into dot-prefixed columns, and
returns a ``pandas.DataFrame``.

Examples
--------
>>> from devqubit_engine.dataframe import runs_to_dataframe
>>> df = runs_to_dataframe(registry, project="vqe", limit=200)
>>> df.columns
Index(['run_id', 'run_name', 'project', ..., 'param.shots', 'metric.fidelity'])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pandas as pd
    from devqubit_engine.storage.types import RegistryProtocol

logger = logging.getLogger(__name__)

# Column order for standard (non-dynamic) fields in DataFrames.
STANDARD_COLUMNS: list[str] = [
    "run_id",
    "run_name",
    "project",
    "adapter",
    "status",
    "created_at",
    "ended_at",
    "group_id",
    "group_name",
    "backend_name",
]


def _require_pandas() -> Any:
    """Import pandas or raise a clear error."""
    try:
        import pandas

        return pandas
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame export. "
            "Install it with: pip install pandas"
        ) from None


def _flatten_record(record: Any) -> dict[str, Any]:
    """Flatten a RunRecord into a single dict row.

    Standard columns are always present (possibly ``None``).
    Dynamic columns use dotted prefixes: ``param.*``, ``metric.*``, ``tag.*``.
    """
    row: dict[str, Any] = {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": record.created_at or None,
        "ended_at": record.ended_at,
        "group_id": record.group_id,
        "group_name": record.group_name,
        "backend_name": record.backend_name,
    }

    for key, value in record.params.items():
        row[f"param.{key}"] = value

    for key, value in record.metrics.items():
        row[f"metric.{key}"] = value

    for key, value in record.tags.items():
        row[f"tag.{key}"] = value

    return row


def runs_to_dataframe(
    registry: RegistryProtocol,
    *,
    project: str | None = None,
    group_id: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Load runs from the registry and return a flat DataFrame.

    Each run becomes one row.  Standard fields (``run_id``, ``project``,
    ``status``, …) are always present.  Params, metrics, and tags are
    expanded into ``param.*``, ``metric.*``, and ``tag.*`` columns.

    Parameters
    ----------
    registry : RegistryProtocol
        Registry to query.
    project : str, optional
        Filter by project name.
    group_id : str, optional
        Filter by group ID (for sweep/experiment aggregation).
    status : str, optional
        Filter by status (``FINISHED``, ``FAILED``, ``RUNNING``, ``KILLED``).
    since : str, optional
        Only include runs created after this ISO 8601 datetime.
    until : str, optional
        Only include runs created before this ISO 8601 datetime.
    limit : int, default 1000
        Maximum number of runs to load.

    Returns
    -------
    pandas.DataFrame
        One row per run, columns sorted: standard fields first,
        then ``metric.*``, ``param.*``, ``tag.*`` alphabetically.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.

    Examples
    --------
    >>> df = runs_to_dataframe(registry, project="vqe")
    >>> df[["run_id", "metric.fidelity", "param.shots"]].head()

    >>> # Group aggregation
    >>> df = runs_to_dataframe(registry, group_id="sweep_20260115")
    >>> df.groupby("param.optimization_level")["metric.fidelity"].mean()
    """
    pd = _require_pandas()

    # Fetch run summaries with pagination
    rows: list[dict[str, Any]] = []
    offset = 0
    batch_size = min(limit, 500)

    while len(rows) < limit:
        remaining = min(batch_size, limit - len(rows))
        summaries = registry.list_runs(
            limit=remaining,
            offset=offset,
            project=project,
            group_id=group_id,
            status=status,
            created_after=since,
            created_before=until,
        )
        if not summaries:
            break

        for summary in summaries:
            run_id = summary.get("run_id", "")
            if not run_id:
                continue
            try:
                record = registry.load(run_id)
                rows.append(_flatten_record(record))
            except Exception as exc:
                logger.warning("Skipping run %s: %s", run_id, exc)

        offset += len(summaries)
        if len(summaries) < remaining:
            break

    if not rows:
        # Return empty DataFrame with standard columns
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(rows)

    # Sort columns: standard fields first, then dynamic fields alphabetically
    standard_set = set(STANDARD_COLUMNS)
    present_standard = [c for c in STANDARD_COLUMNS if c in df.columns]
    dynamic = sorted(c for c in df.columns if c not in standard_set)
    df = df[present_standard + dynamic]

    return df
