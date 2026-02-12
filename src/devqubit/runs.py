# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run navigation and baseline management.

High-level functions for loading, listing, searching, and managing
quantum experiment runs. All functions resolve storage backends from
the global configuration automatically; pass an explicit ``registry``
to override.

Loading Runs
------------
>>> from devqubit.runs import load_run
>>> run = load_run("baseline-v1", project="bell-state")
>>> print(run.project, run.status)

Listing and Searching
---------------------
>>> from devqubit.runs import list_runs, search_runs
>>> list_runs(project="bell-state", limit=10)
>>> search_runs("metric.fidelity > 0.95", sort_by="metric.fidelity")

Baseline Management
-------------------
>>> from devqubit.runs import get_baseline, set_baseline, clear_baseline
>>> set_baseline("bell-state", "run_abc123")
>>> get_baseline("bell-state")
>>> clear_baseline("bell-state")

DataFrame Export
----------------
>>> from devqubit.runs import runs_to_dataframe
>>> df = runs_to_dataframe(project="vqe", limit=200)
>>> df[["run_id", "metric.fidelity", "param.shots"]].head()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd


__all__ = [
    # Run loading
    "load_run",
    "load_run_or_none",
    "run_exists",
    # Run listing
    "list_runs",
    "search_runs",
    "count_runs",
    # Project/group navigation
    "list_projects",
    "list_groups",
    "list_runs_in_group",
    # Baseline management
    "get_baseline",
    "set_baseline",
    "clear_baseline",
    # DataFrame export
    "runs_to_dataframe",
    # Types (for annotation convenience)
    "RunRecord",
    "RunSummary",
    "BaselineInfo",
]


if TYPE_CHECKING:
    from devqubit_engine.storage.types import (
        BaselineInfo,
        RegistryProtocol,
        RunSummary,
    )
    from devqubit_engine.tracking.record import RunRecord


# Lazy imports
_LAZY_IMPORTS = {
    "RunRecord": ("devqubit_engine.tracking.record", "RunRecord"),
    "RunSummary": ("devqubit_engine.storage.types", "RunSummary"),
    "BaselineInfo": ("devqubit_engine.storage.types", "BaselineInfo"),
}


def _get_registry() -> "RegistryProtocol":
    """Return the registry from the global configuration."""
    from devqubit_engine.config import get_config
    from devqubit_engine.storage.factory import create_registry

    return create_registry(config=get_config())


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def load_run(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: RegistryProtocol | None = None,
) -> RunRecord:
    """Load a run by ID or name.

    The first argument may be either a run ID (ULID) or a human-readable
    run name.  When *project* is provided, the function first tries to
    load by ID, then falls back to a name lookup within the project.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name.  Required when loading by name.
    registry : RegistryProtocol, optional
        Registry override.  Uses global config when *None*.

    Returns
    -------
    RunRecord

    Raises
    ------
    RunNotFoundError
        If the run cannot be found.

    Examples
    --------
    >>> load_run("01HXYZ...")
    >>> load_run("bell-experiment-v2", project="bell-state")
    """
    reg = registry if registry is not None else _get_registry()

    # Fast path: try by ID
    record = reg.load_or_none(run_id_or_name)
    if record is not None:
        return record

    # Fallback: name lookup within the project
    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        if runs:
            return reg.load(runs[0]["run_id"])

        from devqubit_engine.storage.errors import RunNotFoundError

        raise RunNotFoundError(
            f"No run with name {run_id_or_name!r} in project {project!r}"
        )

    from devqubit_engine.storage.errors import RunNotFoundError

    raise RunNotFoundError(run_id_or_name)


def load_run_or_none(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: RegistryProtocol | None = None,
) -> RunRecord | None:
    """
    Load a run by ID or name, returning *None* if not found.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name.  Required when loading by name.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    RunRecord or None

    Examples
    --------
    >>> run = load_run_or_none("maybe-exists", project="bell-state")
    >>> if run is not None:
    ...     print(run.status)
    """
    reg = registry if registry is not None else _get_registry()

    record = reg.load_or_none(run_id_or_name)
    if record is not None:
        return record

    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        if runs:
            return reg.load_or_none(runs[0]["run_id"])

    return None


def run_exists(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: RegistryProtocol | None = None,
) -> bool:
    """
    Check whether a run exists by ID or name.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name.  Required when checking by name.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    bool

    Examples
    --------
    >>> if run_exists("nightly-v1", project="bell-state"):
    ...     print("Run found")
    """
    reg = registry if registry is not None else _get_registry()

    if reg.exists(run_id_or_name):
        return True

    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        return len(runs) > 0

    return False


# ---------------------------------------------------------------------------
# Run listing and search
# ---------------------------------------------------------------------------


def list_runs(
    *,
    project: str | None = None,
    name: str | None = None,
    adapter: str | None = None,
    status: str | None = None,
    backend_name: str | None = None,
    fingerprint: str | None = None,
    git_commit: str | None = None,
    group_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    registry: RegistryProtocol | None = None,
) -> list[RunSummary]:
    """
    List runs with optional filtering.

    Returns lightweight summaries ordered by ``created_at`` descending.
    Use :func:`load_run` to fetch the full record for a given run.

    Parameters
    ----------
    project : str, optional
        Filter by project name.
    name : str, optional
        Filter by run name (exact match).
    adapter : str, optional
        Filter by adapter (e.g. ``"qiskit"``, ``"pennylane"``).
    status : str, optional
        Filter by status (``"RUNNING"``, ``"FINISHED"``, ``"FAILED"``,
        ``"KILLED"``).
    backend_name : str, optional
        Filter by backend name.
    fingerprint : str, optional
        Filter by run fingerprint.
    git_commit : str, optional
        Filter by git commit SHA.
    group_id : str, optional
        Filter by group ID.
    limit : int, default 100
        Maximum results to return.
    offset : int, default 0
        Number of results to skip (pagination).
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    list[RunSummary]

    Examples
    --------
    >>> for r in list_runs(project="bell-state", limit=5):
    ...     print(r["run_id"], r["status"])
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_runs(
        project=project,
        name=name,
        adapter=adapter,
        status=status,
        backend_name=backend_name,
        fingerprint=fingerprint,
        git_commit=git_commit,
        group_id=group_id,
        limit=limit,
        offset=offset,
    )


def search_runs(
    query: str,
    *,
    sort_by: str | None = None,
    descending: bool = True,
    limit: int = 100,
    offset: int = 0,
    registry: RegistryProtocol | None = None,
) -> list[RunRecord]:
    """
    Search runs using a query expression.

    Supports filtering by parameters, metrics, tags, and top-level fields
    with a simple expression syntax.

    Query Syntax
    ~~~~~~~~~~~~~
    ::

        field op value [and field op value ...]

    **Fields:** ``params.*``, ``metric.*`` / ``metrics.*``,
    ``tag.*`` / ``tags.*``, ``project``, ``adapter``, ``status``,
    ``backend``, ``fingerprint``.

    **Operators:** ``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``,
    ``~`` (case-insensitive substring), ``exists``.

    Parameters
    ----------
    query : str
        Query expression.
    sort_by : str, optional
        Sort field (e.g. ``"metric.fidelity"``).
    descending : bool, default True
        Sort in descending order.
    limit : int, default 100
        Maximum results.
    offset : int, default 0
        Results to skip.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    list[RunRecord]

    Examples
    --------
    >>> search_runs("metric.fidelity > 0.95", sort_by="metric.fidelity")
    >>> search_runs("params.shots >= 1000 and status = FINISHED")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.search_runs(
        query,
        sort_by=sort_by,
        descending=descending,
        limit=limit,
        offset=offset,
    )


def count_runs(
    *,
    project: str | None = None,
    status: str | None = None,
    registry: RegistryProtocol | None = None,
) -> int:
    """
    Count runs matching the given filters.

    Parameters
    ----------
    project : str, optional
        Filter by project.
    status : str, optional
        Filter by status.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    int

    Examples
    --------
    >>> total = count_runs(project="bell-state")
    >>> finished = count_runs(project="bell-state", status="FINISHED")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.count_runs(project=project, status=status)


# ---------------------------------------------------------------------------
# Project and group navigation
# ---------------------------------------------------------------------------


def list_projects(
    *,
    registry: RegistryProtocol | None = None,
) -> list[str]:
    """
    List all project names.

    Returns
    -------
    list[str]
        Sorted project names.

    Examples
    --------
    >>> list_projects()
    ['bell-state', 'vqe-hydrogen']
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_projects()


def list_groups(
    *,
    project: str | None = None,
    limit: int = 100,
    offset: int = 0,
    registry: RegistryProtocol | None = None,
) -> list[dict[str, Any]]:
    """
    List run groups with optional project filtering.

    Parameters
    ----------
    project : str, optional
        Filter by project.
    limit : int, default 100
        Maximum results.
    offset : int, default 0
        Results to skip.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    list[dict]
        Group summaries with ``group_id``, ``group_name``, ``run_count``.

    Examples
    --------
    >>> for g in list_groups(project="bell-state"):
    ...     print(f"{g['group_name']}: {g['run_count']} runs")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_groups(project=project, limit=limit, offset=offset)


def list_runs_in_group(
    group_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
    registry: RegistryProtocol | None = None,
) -> list[RunSummary]:
    """
    List runs belonging to a specific group.

    Parameters
    ----------
    group_id : str
        Group identifier.
    limit : int, default 100
        Maximum results.
    offset : int, default 0
        Results to skip.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    list[RunSummary]

    Examples
    --------
    >>> for r in list_runs_in_group("sweep_20260115"):
    ...     print(r["run_id"])
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_runs_in_group(group_id, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def get_baseline(
    project: str,
    *,
    registry: RegistryProtocol | None = None,
) -> BaselineInfo | None:
    """
    Get the baseline run for a project.

    Parameters
    ----------
    project : str
        Project name.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    BaselineInfo or None
        Dict with ``project``, ``run_id``, ``set_at``; or *None* when no
        baseline is configured.

    Examples
    --------
    >>> baseline = get_baseline("bell-state")
    >>> if baseline:
    ...     print(baseline["run_id"])
    """
    reg = registry if registry is not None else _get_registry()
    return reg.get_baseline(project)


def set_baseline(
    project: str,
    run_id: str,
    *,
    registry: RegistryProtocol | None = None,
) -> None:
    """
    Set the baseline run for a project.

    The baseline serves as the reference for verification in CI/CD
    pipelines (see :func:`~devqubit.compare.verify_baseline`).

    Parameters
    ----------
    project : str
        Project name.
    run_id : str
        Run ID to promote as baseline.
    registry : RegistryProtocol, optional
        Registry override.

    Raises
    ------
    RunNotFoundError
        If *run_id* does not exist.

    Examples
    --------
    >>> set_baseline("bell-state", "01HXYZ...")
    """
    reg = registry if registry is not None else _get_registry()
    reg.load(run_id)  # verify existence
    reg.set_baseline(project, run_id)


def clear_baseline(
    project: str,
    *,
    registry: RegistryProtocol | None = None,
) -> bool:
    """
    Clear the baseline for a project.

    Parameters
    ----------
    project : str
        Project name.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    bool
        *True* if a baseline was cleared, *False* if none was set.

    Examples
    --------
    >>> clear_baseline("bell-state")
    True
    """
    reg = registry if registry is not None else _get_registry()
    return reg.clear_baseline(project)


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def runs_to_dataframe(
    *,
    project: str | None = None,
    group_id: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 1000,
    registry: RegistryProtocol | None = None,
) -> pd.DataFrame:
    """
    Export runs as a flat pandas DataFrame.

    Standard columns (``run_id``, ``project``, ``status``, ...) are always
    present.  Parameters, metrics, and tags are expanded into
    ``param.*``, ``metric.*``, and ``tag.*`` columns respectively.

    Parameters
    ----------
    project : str, optional
        Filter by project.
    group_id : str, optional
        Filter by group ID.
    status : str, optional
        Filter by status.
    since : str, optional
        Include runs created after this ISO 8601 datetime.
    until : str, optional
        Include runs created before this ISO 8601 datetime.
    limit : int, default 1000
        Maximum runs to load.
    registry : RegistryProtocol, optional
        Registry override.

    Returns
    -------
    pd.DataFrame
        One row per run.

    Examples
    --------
    >>> df = runs_to_dataframe(project="vqe")
    >>> df[["run_id", "metric.fidelity", "param.shots"]].head()

    >>> df = runs_to_dataframe(group_id="sweep_20260115")
    >>> df.groupby("param.optimization_level")["metric.fidelity"].mean()
    """
    from devqubit_engine.dataframe import runs_to_dataframe as _engine_to_df

    reg = registry if registry is not None else _get_registry()
    return _engine_to_df(
        reg,
        project=project,
        group_id=group_id,
        status=status,
        since=since,
        until=until,
        limit=limit,
    )


def __getattr__(name: str) -> Any:
    """Lazy-import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available public attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
