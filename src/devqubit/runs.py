# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run navigation and baseline management.

This module provides high-level functions for loading, listing, and
searching runs, as well as managing project baselines. All functions
use the global configuration automatically.

Loading Runs
------------
>>> from devqubit.runs import load_run
>>> run = load_run("abc123")
>>> print(run.project, run.status)

Listing Runs
------------
>>> from devqubit.runs import list_runs
>>> recent = list_runs(project="my_project", limit=10)
>>> for summary in recent:
...     print(summary["run_id"], summary["status"])

Searching Runs
--------------
>>> from devqubit.runs import search_runs
>>> high_fidelity = search_runs(
...     "metric.fidelity > 0.95",
...     sort_by="metric.fidelity",
...     limit=10,
... )

Baseline Management
-------------------
>>> from devqubit.runs import get_baseline, set_baseline, clear_baseline
>>> baseline = get_baseline("my_project")
>>> if baseline:
...     print(f"Baseline: {baseline['run_id']}")
>>> set_baseline("my_project", "run_abc123")
>>> clear_baseline("my_project")

DataFrame Export
----------------
>>> from devqubit.runs import runs_to_dataframe
>>> df = runs_to_dataframe(project="vqe", limit=200)
>>> df[["run_id", "metric.fidelity", "param.shots"]].head()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


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
    import pandas as pd
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
    """Get registry from global config."""
    from devqubit_engine.config import get_config
    from devqubit_engine.storage.factory import create_registry

    return create_registry(config=get_config())


def load_run(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: "RegistryProtocol | None" = None,
) -> "RunRecord":
    """
    Load a run by ID or name.

    When ``project`` is provided, the first argument can be either a run ID
    or a run name. The function first attempts to load by ID, then falls
    back to searching by name within the project.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name. Required when loading by name.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    RunRecord
        The loaded run record.

    Raises
    ------
    RunNotFoundError
        If the run does not exist.

    Examples
    --------
    >>> from devqubit.runs import load_run
    >>> # Load by ID
    >>> run = load_run("01HXYZ...")
    >>> # Load by name within a project
    >>> run = load_run("bell-experiment-v2", project="bell_state")
    """
    reg = registry if registry is not None else _get_registry()

    # Try loading by ID first (fast path, backward compatible)
    record = reg.load_or_none(run_id_or_name)
    if record is not None:
        return record

    # If project provided, try loading by name
    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        if runs:
            return reg.load(runs[0]["run_id"])

        from devqubit_engine.storage.errors import RunNotFoundError

        raise RunNotFoundError(
            f"No run with name {run_id_or_name!r} in project {project!r}"
        )

    # No project provided and ID not found
    from devqubit_engine.storage.errors import RunNotFoundError

    raise RunNotFoundError(run_id_or_name)


def load_run_or_none(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: "RegistryProtocol | None" = None,
) -> "RunRecord | None":
    """
    Load a run by ID or name, returning None if not found.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name. Required when loading by name.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    RunRecord or None
        The run record, or None if not found.

    Examples
    --------
    >>> from devqubit.runs import load_run_or_none
    >>> run = load_run_or_none("maybe_exists")
    >>> if run is not None:
    ...     print(run.status)

    >>> # Try loading by name
    >>> run = load_run_or_none("my-experiment", project="bell_state")
    """
    reg = registry if registry is not None else _get_registry()

    # Try loading by ID first
    record = reg.load_or_none(run_id_or_name)
    if record is not None:
        return record

    # If project provided, try loading by name
    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        if runs:
            return reg.load_or_none(runs[0]["run_id"])

    return None


def run_exists(
    run_id_or_name: str,
    *,
    project: str | None = None,
    registry: "RegistryProtocol | None" = None,
) -> bool:
    """
    Check if a run exists by ID or name.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str, optional
        Project name. Required when checking by name.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    bool
        True if the run exists.

    Examples
    --------
    >>> from devqubit.runs import run_exists
    >>> if run_exists("abc123"):
    ...     print("Run found!")

    >>> # Check by name
    >>> if run_exists("my-experiment", project="bell_state"):
    ...     print("Run found!")
    """
    reg = registry if registry is not None else _get_registry()

    # Try by ID first
    if reg.exists(run_id_or_name):
        return True

    # If project provided, try by name
    if project is not None:
        runs = reg.list_runs(project=project, name=run_id_or_name, limit=1)
        return len(runs) > 0

    return False


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
    registry: "RegistryProtocol | None" = None,
) -> "list[RunSummary]":
    """
    List runs with optional filtering.

    Returns lightweight run summaries suitable for display and navigation.
    For full run details, use :func:`load_run`.

    Parameters
    ----------
    project : str, optional
        Filter by project name.
    name : str, optional
        Filter by run name (exact match).
    adapter : str, optional
        Filter by adapter name (e.g., "qiskit", "pennylane").
    status : str, optional
        Filter by run status ("RUNNING", "FINISHED", "FAILED", "KILLED").
    backend_name : str, optional
        Filter by backend name.
    fingerprint : str, optional
        Filter by run fingerprint.
    git_commit : str, optional
        Filter by git commit SHA.
    group_id : str, optional
        Filter by group ID.
    limit : int, default=100
        Maximum number of results.
    offset : int, default=0
        Number of results to skip (for pagination).
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    list of RunSummary
        Matching runs, ordered by created_at descending.

    Examples
    --------
    >>> from devqubit.runs import list_runs
    >>> # List recent runs for a project
    >>> runs = list_runs(project="bell_state", limit=10)
    >>> for r in runs:
    ...     print(f"{r['run_id']}: {r['status']}")

    >>> # List only finished runs
    >>> finished = list_runs(status="FINISHED", limit=50)

    >>> # Find run by name
    >>> runs = list_runs(project="bell_state", name="nightly-check")
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
    registry: "RegistryProtocol | None" = None,
) -> "list[RunRecord]":
    """
    Search runs using a query expression.

    Supports filtering by params, metrics, tags, and top-level fields
    using a simple query syntax.

    Query Syntax
    ------------
    Conditions are joined with AND::

        field op value [and field op value ...]

    Supported fields:
        - ``params.*``: Experiment parameters
        - ``metric.*`` or ``metrics.*``: Logged metrics
        - ``tag.*`` or ``tags.*``: String tags
        - ``project``, ``adapter``, ``status``, ``backend``, ``fingerprint``

    Supported operators:
        - ``=``, ``!=``: Equality
        - ``>``, ``>=``, ``<``, ``<=``: Numeric comparison
        - ``~``: Contains (case-insensitive substring)
        - ``exists``: Field exists

    Parameters
    ----------
    query : str
        Query expression.
    sort_by : str, optional
        Field to sort by (e.g., "metric.fidelity").
    descending : bool, default=True
        Sort in descending order.
    limit : int, default=100
        Maximum number of results.
    offset : int, default=0
        Number of results to skip.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    list of RunRecord
        Matching run records.

    Examples
    --------
    >>> from devqubit.runs import search_runs
    >>> # Find high-fidelity runs
    >>> results = search_runs(
    ...     "metric.fidelity > 0.95",
    ...     sort_by="metric.fidelity",
    ...     limit=10,
    ... )

    >>> # Find runs with specific params
    >>> results = search_runs("params.shots >= 1000 and status = FINISHED")

    >>> # Find runs by tag
    >>> results = search_runs("tags.experiment ~ bell")
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
    registry: "RegistryProtocol | None" = None,
) -> int:
    """
    Count runs matching filters.

    Parameters
    ----------
    project : str, optional
        Filter by project name.
    status : str, optional
        Filter by run status.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    int
        Number of matching runs.

    Examples
    --------
    >>> from devqubit.runs import count_runs
    >>> total = count_runs(project="my_project")
    >>> finished = count_runs(project="my_project", status="FINISHED")
    >>> print(f"{finished}/{total} runs finished")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.count_runs(project=project, status=status)


def list_projects(
    *,
    registry: "RegistryProtocol | None" = None,
) -> list[str]:
    """
    List all unique project names.

    Returns
    -------
    list of str
        Sorted list of project names.

    Examples
    --------
    >>> from devqubit.runs import list_projects
    >>> projects = list_projects()
    >>> print("Available projects:", projects)
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_projects()


def list_groups(
    *,
    project: str | None = None,
    limit: int = 100,
    offset: int = 0,
    registry: "RegistryProtocol | None" = None,
) -> list[dict[str, Any]]:
    """
    List run groups with optional project filtering.

    Parameters
    ----------
    project : str, optional
        Filter by project name.
    limit : int, default=100
        Maximum number of results.
    offset : int, default=0
        Number of results to skip.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    list of dict
        Group summaries with group_id, group_name, and run_count.

    Examples
    --------
    >>> from devqubit.runs import list_groups
    >>> groups = list_groups(project="my_project")
    >>> for g in groups:
    ...     print(f"{g['group_name']}: {g['run_count']} runs")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_groups(project=project, limit=limit, offset=offset)


def list_runs_in_group(
    group_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
    registry: "RegistryProtocol | None" = None,
) -> "list[RunSummary]":
    """
    List runs belonging to a specific group.

    Parameters
    ----------
    group_id : str
        Group identifier.
    limit : int, default=100
        Maximum number of results.
    offset : int, default=0
        Number of results to skip.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    list of RunSummary
        Runs in the group, ordered by created_at descending.

    Examples
    --------
    >>> from devqubit.runs import list_runs_in_group
    >>> runs = list_runs_in_group("group_abc123")
    >>> for r in runs:
    ...     print(r["run_id"])
    """
    reg = registry if registry is not None else _get_registry()
    return reg.list_runs_in_group(group_id, limit=limit, offset=offset)


def get_baseline(
    project: str,
    *,
    registry: "RegistryProtocol | None" = None,
) -> "BaselineInfo | None":
    """
    Get the baseline run for a project.

    Parameters
    ----------
    project : str
        Project name.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    BaselineInfo or None
        Baseline info (project, run_id, set_at), or None if no baseline set.

    Examples
    --------
    >>> from devqubit.runs import get_baseline
    >>> baseline = get_baseline("my_project")
    >>> if baseline:
    ...     print(f"Baseline run: {baseline['run_id']}")
    ...     print(f"Set at: {baseline['set_at']}")
    ... else:
    ...     print("No baseline configured")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.get_baseline(project)


def set_baseline(
    project: str,
    run_id: str,
    *,
    registry: "RegistryProtocol | None" = None,
) -> None:
    """
    Set the baseline run for a project.

    The baseline is used as the reference point for verification
    in CI/CD pipelines.

    Parameters
    ----------
    project : str
        Project name.
    run_id : str
        Run ID to set as baseline.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Raises
    ------
    RunNotFoundError
        If the specified run does not exist.

    Examples
    --------
    >>> from devqubit.runs import set_baseline
    >>> set_baseline("my_project", "run_abc123")
    >>> print("Baseline updated!")
    """
    reg = registry if registry is not None else _get_registry()
    # Verify run exists
    reg.load(run_id)
    reg.set_baseline(project, run_id)


def clear_baseline(
    project: str,
    *,
    registry: "RegistryProtocol | None" = None,
) -> bool:
    """
    Clear the baseline for a project.

    Parameters
    ----------
    project : str
        Project name.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    bool
        True if a baseline was cleared, False if none existed.

    Examples
    --------
    >>> from devqubit.runs import clear_baseline
    >>> if clear_baseline("my_project"):
    ...     print("Baseline cleared")
    ... else:
    ...     print("No baseline was set")
    """
    reg = registry if registry is not None else _get_registry()
    return reg.clear_baseline(project)


def runs_to_dataframe(
    *,
    project: str | None = None,
    group_id: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 1000,
    registry: "RegistryProtocol | None" = None,
) -> "pd.DataFrame":
    """
    Load runs into a pandas DataFrame with flat columns.

    Standard columns are always present (``run_id``, ``project``, ``status``,
    …).  Params, metrics, and tags are expanded into ``param.*``,
    ``metric.*``, ``tag.*`` columns.

    Requires ``pandas`` — install with ``pip install devqubit[pandas]``.

    Parameters
    ----------
    project : str, optional
        Filter by project name.
    group_id : str, optional
        Filter by group ID (for sweep/experiment aggregation).
    status : str, optional
        Filter by status.
    since : str, optional
        Only include runs created after this ISO 8601 datetime.
    until : str, optional
        Only include runs created before this ISO 8601 datetime.
    limit : int, default 1000
        Maximum number of runs to load.
    registry : RegistryProtocol, optional
        Custom registry. Uses global config if not provided.

    Returns
    -------
    pandas.DataFrame
        One row per run.

    Examples
    --------
    >>> from devqubit.runs import runs_to_dataframe
    >>> df = runs_to_dataframe(project="vqe")
    >>> df[["run_id", "metric.fidelity", "param.shots"]].head()

    >>> # Group aggregation
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
    """Lazy import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
