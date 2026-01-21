# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run listing, search, and group CLI commands.

This module provides commands for listing, searching, viewing, and
deleting runs, as well as managing projects and run groups.

Commands
--------
list
    List recent runs with filters.
search
    Search runs using query expressions.
show
    Show detailed run information.
delete
    Delete a run.
projects
    List all projects.
groups
    Manage run groups (sweeps, experiments).
"""

from __future__ import annotations

from typing import Any

import click
from devqubit_engine.cli._utils import (
    echo,
    print_json,
    print_table,
    root_from_ctx,
    safe_get,
    truncate_id,
)


def register(cli: click.Group) -> None:
    """Register run commands with CLI."""
    cli.add_command(list_runs)
    cli.add_command(search_runs)
    cli.add_command(show_run)
    cli.add_command(delete_run)
    cli.add_command(list_projects)
    cli.add_command(groups_group)


def _get_run_tags(run_record: Any) -> dict[str, Any]:
    """Safely extract tags from a run record."""
    data = run_record.record.get("data") if hasattr(run_record, "record") else {}
    if not isinstance(data, dict):
        return {}
    tags = data.get("tags")
    return tags if isinstance(tags, dict) else {}


@click.command("list")
@click.option("--limit", "-n", type=int, default=20, show_default=True)
@click.option("--project", "-p", default=None, help="Filter by project.")
@click.option("--adapter", "-a", default=None, help="Filter by adapter.")
@click.option("--status", "-s", default=None, help="Filter by status.")
@click.option("--backend", "-b", default=None, help="Filter by backend name.")
@click.option("--group", "-g", default=None, help="Filter by group ID.")
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Filter by tag (key or key=value).",
)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def list_runs(
    ctx: click.Context,
    limit: int,
    project: str | None,
    adapter: str | None,
    status: str | None,
    backend: str | None,
    group: str | None,
    tags: tuple[str, ...],
    fmt: str,
) -> None:
    """
    List recent runs.

    Examples:
        devqubit list
        devqubit list --limit 50 --project myproject
        devqubit list --status COMPLETED --backend ibm_brisbane
        devqubit list --tag experiment=bell --tag validated
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    # When filtering by tags, fetch more runs initially then filter
    filter_kwargs: dict[str, Any] = {
        "limit": limit if not tags else 1000,
        "project": project,
        "adapter": adapter,
        "status": status,
    }
    if backend and hasattr(registry, "list_runs"):
        filter_kwargs["backend_name"] = backend
    if group:
        filter_kwargs["group_id"] = group

    runs = registry.list_runs(**filter_kwargs)

    # Apply tag filters if specified
    if tags:
        tag_filters: dict[str, str | None] = {}
        for tag in tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tag_filters[key.strip()] = value.strip()
            else:
                tag_filters[tag.strip()] = None

        filtered_runs = []
        for r in runs:
            run_id = r.get("run_id", "")
            if not run_id:
                continue
            try:
                run_record = registry.load(run_id)
                run_tags = _get_run_tags(run_record)
                match = all(
                    key in run_tags
                    and (expected is None or run_tags.get(key) == expected)
                    for key, expected in tag_filters.items()
                )
                if match:
                    filtered_runs.append(r)
                    if len(filtered_runs) >= limit:
                        break
            except Exception:
                continue
        runs = filtered_runs

    if fmt == "json":
        print_json([dict(r) for r in runs])
        return

    if not runs:
        echo("No runs found.")
        return

    headers = ["Run ID", "Project", "Adapter", "Status", "Created"]
    rows = []
    for r in runs:
        proj = r.get("project")
        if isinstance(proj, dict):
            proj_name = proj.get("name", "")
        else:
            proj_name = str(proj) if proj else ""

        info = r.get("info", {}) or {}
        adapter_name = r.get("adapter") or info.get("adapter", "")
        status_val = r.get("status") or info.get("status", "")
        created = str(r.get("created_at", ""))[:19]

        rows.append(
            [
                truncate_id(r.get("run_id", "")),
                proj_name[:20],
                str(adapter_name),
                str(status_val),
                created,
            ]
        )

    print_table(headers, rows, f"Recent Runs ({len(runs)})")


@click.command("search")
@click.argument("query")
@click.option("--limit", "-n", type=int, default=20, show_default=True)
@click.option("--project", "-p", default=None, help="Filter by project first.")
@click.option(
    "--sort",
    "-s",
    default=None,
    help="Sort by field (e.g., metric.fidelity).",
)
@click.option("--asc", is_flag=True, help="Sort ascending (default: descending).")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def search_runs(
    ctx: click.Context,
    query: str,
    limit: int,
    project: str | None,
    sort: str | None,
    asc: bool,
    fmt: str,
) -> None:
    """
    Search runs using query expression.

    Query syntax: field op value [and field op value ...]
    Fields: params.*, metric.*, tags.*, project, adapter, status, backend
    Operators: =, !=, >, >=, <, <=, ~ (contains)

    Examples:
        devqubit search "metric.fidelity > 0.95"
        devqubit search "params.shots = 1000 and tags.device ~ ibm"
        devqubit search "status = COMPLETED" --sort metric.fidelity
    """
    from devqubit_engine.config import Config
    from devqubit_engine.query import QueryParseError, parse_query
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    # Validate query syntax
    try:
        parse_query(query)
    except QueryParseError as e:
        raise click.ClickException(f"Invalid query: {e}") from e

    try:
        results = registry.search_runs(
            query, sort_by=sort, descending=not asc, limit=limit
        )
        # Filter by project if specified
        if project:
            results = [r for r in results if r.project == project]
    except Exception as e:
        raise click.ClickException(f"Search failed: {e}") from e

    if fmt == "json":
        print_json([r.to_dict() for r in results])
        return

    if not results:
        echo("No matching runs found.")
        return

    headers = ["Run ID", "Project", "Status", "Created"]
    if sort and sort.startswith("metric."):
        metric_name = sort.split(".", 1)[1]
        headers.append(metric_name[:12])

    rows = []
    for r in results:
        row = [
            truncate_id(r.run_id),
            r.project[:20] if r.project else "",
            r.status or "",
            r.created_at[:19] if r.created_at else "",
        ]
        if sort and sort.startswith("metric."):
            metric_name = sort.split(".", 1)[1]
            metrics = safe_get(r.record, "data", "metrics", default={})
            val = metrics.get(metric_name)
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val) if val is not None else "-")
        rows.append(row)

    print_table(headers, rows, f"Search Results ({len(results)})")


@click.command("show")
@click.argument("run_id")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def show_run(ctx: click.Context, run_id: str, fmt: str) -> None:
    """Show detailed run information."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    if fmt == "json":
        print_json(run_record.to_dict())
        return

    record = run_record.record
    info = record.get("info", {}) or {}

    echo(f"Run ID:      {run_record.run_id}")
    echo(f"Project:     {run_record.project or '-'}")
    echo(f"Adapter:     {run_record.adapter or '-'}")
    echo(f"Status:      {run_record.status or '-'}")
    echo(f"Created:     {run_record.created_at or '-'}")
    echo(f"Ended:       {info.get('ended_at') or '-'}")

    group_id = record.get("group_id")
    if group_id:
        group_name = record.get("group_name", "")
        suffix = f" ({group_name})" if group_name else ""
        echo(f"Group:       {group_id}{suffix}")

    parent_id = record.get("parent_run_id")
    if parent_id:
        echo(f"Parent:      {parent_id}")

    backend = record.get("backend", {}) or {}
    if backend:
        echo(f"Backend:     {backend.get('name', 'unknown')}")
        if backend.get("provider"):
            echo(f"Provider:    {backend['provider']}")

    fps = run_record.fingerprints
    if fps:
        run_fp = fps.get("run", "")
        if run_fp:
            echo(f"Fingerprint: {run_fp[:16]}...")
        else:
            echo("Fingerprint: -")

    prov = record.get("provenance", {}) or {}
    git = prov.get("git", {}) if isinstance(prov, dict) else {}
    if git:
        commit = git.get("commit", "")[:8] if git.get("commit") else ""
        branch = git.get("branch", "")
        dirty = " (dirty)" if git.get("dirty") else ""
        if commit or branch:
            echo(f"Git:         {branch}@{commit}{dirty}")

    data = record.get("data", {}) or {}
    params = data.get("params", {}) or {}
    if params:
        echo(f"Params:      {len(params)} parameter(s)")

    metrics = data.get("metrics", {}) or {}
    if metrics:
        echo(f"Metrics:     {len(metrics)} metric(s)")
        for i, (k, v) in enumerate(metrics.items()):
            if i >= 5:
                echo(f"  ... and {len(metrics) - 5} more")
                break
            if isinstance(v, float):
                echo(f"  {k}: {v:.6f}")
            else:
                echo(f"  {k}: {v}")

    artifacts = run_record.artifacts
    if artifacts:
        echo(f"Artifacts:   {len(artifacts)} artifact(s)")
    else:
        echo("Artifacts:   0")


@click.command("delete")
@click.argument("run_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def delete_run(ctx: click.Context, run_id: str, yes: bool) -> None:
    """Delete a run and its artifacts."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    if not registry.exists(run_id):
        raise click.ClickException(f"Run not found: {run_id}")

    if not yes:
        if not click.confirm(f"Delete run {run_id}? This cannot be undone."):
            echo("Cancelled.")
            return

    ok = registry.delete(run_id)
    if not ok:
        raise click.ClickException(f"Failed to delete run {run_id}")

    echo(f"Deleted run {run_id}")


@click.command("projects")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def list_projects(ctx: click.Context, fmt: str) -> None:
    """List all projects."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    projects = registry.list_projects()

    if fmt == "json":
        result = []
        for p in projects:
            count = registry.count_runs(project=p)
            baseline = registry.get_baseline(p)
            result.append(
                {
                    "name": p,
                    "run_count": count,
                    "baseline_run_id": baseline.get("run_id") if baseline else None,
                }
            )
        print_json(result)
        return

    if not projects:
        echo("No projects found.")
        return

    headers = ["Project", "Runs", "Baseline"]
    rows = []
    for p in projects:
        count = registry.count_runs(project=p)
        baseline = registry.get_baseline(p)
        baseline_str = truncate_id(baseline["run_id"]) if baseline else "-"
        rows.append([p, count, baseline_str])

    print_table(headers, rows, "Projects")


# =============================================================================
# Groups commands
# =============================================================================


@click.group("groups")
def groups_group() -> None:
    """Manage run groups (sweeps, experiments)."""
    pass


@groups_group.command("list")
@click.option("--project", "-p", default=None, help="Filter by project.")
@click.option("--limit", "-n", type=int, default=20, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def groups_list(
    ctx: click.Context,
    project: str | None,
    limit: int,
    fmt: str,
) -> None:
    """List run groups."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    groups = registry.list_groups(project=project, limit=limit)

    if fmt == "json":
        print_json(groups)
        return

    if not groups:
        echo("No groups found.")
        return

    headers = ["Group ID", "Name", "Runs", "Last Run"]
    rows = []
    for g in groups:
        group_id = g.get("group_id", "")
        group_name = g.get("group_name") or ""
        run_count = g.get("run_count", 0)
        last_created = str(g.get("last_created", ""))[:19]

        rows.append(
            [
                truncate_id(group_id, 20),
                group_name[:20],
                run_count,
                last_created,
            ]
        )

    print_table(headers, rows, f"Run Groups ({len(groups)})")


@groups_group.command("show")
@click.argument("group_id")
@click.option("--limit", "-n", type=int, default=50, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def groups_show(
    ctx: click.Context,
    group_id: str,
    limit: int,
    fmt: str,
) -> None:
    """Show runs in a group."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    runs = registry.list_runs_in_group(group_id, limit=limit)

    if fmt == "json":
        print_json([dict(r) for r in runs])
        return

    if not runs:
        echo(f"No runs found in group: {group_id}")
        return

    headers = ["Run ID", "Status", "Created"]
    rows = []
    for r in runs:
        rows.append(
            [
                truncate_id(r.get("run_id", "")),
                r.get("status", ""),
                str(r.get("created_at", ""))[:19],
            ]
        )

    print_table(headers, rows, f"Runs in {truncate_id(group_id, 20)} ({len(runs)})")
