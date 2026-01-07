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
from devqubit_engine.cli._utils import echo, print_json, print_table, root_from_ctx


def register(cli: click.Group) -> None:
    """Register run commands with CLI."""
    cli.add_command(list_runs)
    cli.add_command(search_runs)
    cli.add_command(show_run)
    cli.add_command(delete_run)
    cli.add_command(list_projects)
    cli.add_command(groups_group)


@click.command("list")
@click.option("--limit", "-n", type=int, default=20, show_default=True)
@click.option("--project", "-p", default=None, help="Filter by project.")
@click.option("--adapter", "-a", default=None, help="Filter by adapter.")
@click.option("--status", "-s", default=None, help="Filter by status.")
@click.option("--backend", "-b", default=None, help="Filter by backend name.")
@click.option("--group", "-g", default=None, help="Filter by group ID.")
@click.option(
    "--tag", "-t", "tags", multiple=True, help="Filter by tag (key or key=value)."
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
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

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
                tag_filters[key] = value
            else:
                tag_filters[tag] = None

        filtered_runs = []
        for r in runs:
            run_id = r.get("run_id", "")
            try:
                run_record = registry.load(run_id)
                run_tags = run_record.record.get("data", {}).get("tags", {})
                match = all(
                    key in run_tags and (expected is None or run_tags[key] == expected)
                    for key, expected in tag_filters.items()
                )
                if match:
                    filtered_runs.append(r)
                    if len(filtered_runs) >= limit:
                        break
            except Exception:
                pass
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
        proj_name = proj.get("name", "") if isinstance(proj, dict) else str(proj or "")
        adapter_name = r.get("adapter") or r.get("info", {}).get("adapter", "")
        status_val = r.get("status") or r.get("info", {}).get("status", "")

        rows.append(
            [
                (r.get("run_id", "")[:12] + "...") if r.get("run_id") else "",
                proj_name[:20],
                str(adapter_name),
                str(status_val),
                str(r.get("created_at", ""))[:19],
            ]
        )

    print_table(headers, rows, f"Recent Runs ({len(runs)})")


@click.command("search")
@click.argument("query")
@click.option("--limit", "-n", type=int, default=20, show_default=True)
@click.option("--project", "-p", default=None, help="Filter by project first.")
@click.option(
    "--sort", "-s", default=None, help="Sort by field (e.g., metric.fidelity)."
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
    from devqubit_engine.core.config import Config
    from devqubit_engine.query import QueryParseError, parse_query
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        parse_query(query)
    except QueryParseError as e:
        raise click.ClickException(f"Invalid query: {e}") from e

    try:
        results = registry.search_runs(
            query, sort_by=sort, descending=not asc, limit=limit, project=project
        )
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
        headers.append(sort.split(".", 1)[1][:12])

    rows = []
    for r in results:
        row = [
            r.run_id[:12] + "...",
            r.project[:20] if r.project else "",
            r.status or "",
            r.created_at[:19] if r.created_at else "",
        ]
        if sort and sort.startswith("metric."):
            metric_name = sort.split(".", 1)[1]
            metrics = r.record.get("data", {}).get("metrics", {})
            val = metrics.get(metric_name)
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val or "-"))
        rows.append(row)

    print_table(headers, rows, f"Search Results ({len(results)})")


@click.command("show")
@click.argument("run_id")
@click.option(
    "--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty"
)
@click.pass_context
def show_run(ctx: click.Context, run_id: str, fmt: str) -> None:
    """Show detailed run information."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry
    from devqubit_engine.storage.protocols import RunNotFoundError

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
    echo(f"Run ID:      {run_record.run_id}")
    echo(f"Project:     {run_record.project}")
    echo(f"Adapter:     {run_record.adapter}")
    echo(f"Status:      {run_record.status}")
    echo(f"Created:     {run_record.created_at}")
    echo(f"Ended:       {record.get('info', {}).get('ended_at', '')}")

    if record.get("group_id"):
        group_name = record.get("group_name") or ""
        echo(
            f"Group:       {record['group_id']}"
            + (f" ({group_name})" if group_name else "")
        )
    if record.get("parent_run_id"):
        echo(f"Parent:      {record['parent_run_id']}")

    backend = record.get("backend", {})
    if backend:
        echo(f"Backend:     {backend.get('name', 'unknown')}")
        if backend.get("provider"):
            echo(f"Provider:    {backend.get('provider')}")

    fps = run_record.fingerprints
    if fps:
        run_fp = fps.get("run", "")
        echo(f"Fingerprint: {run_fp[:16]}..." if run_fp else "Fingerprint: -")

    prov = record.get("provenance", {})
    git = prov.get("git", {}) if isinstance(prov, dict) else {}
    if git:
        commit = git.get("commit", "")[:8] if git.get("commit") else ""
        branch = git.get("branch", "")
        dirty = " (dirty)" if git.get("dirty") else ""
        echo(f"Git:         {branch}@{commit}{dirty}")

    params = record.get("data", {}).get("params", {})
    if params:
        echo(f"Params:      {len(params)} parameters")

    metrics = record.get("data", {}).get("metrics", {})
    if metrics:
        echo(f"Metrics:     {len(metrics)} metrics")
        for k, v in list(metrics.items())[:5]:
            echo(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        if len(metrics) > 5:
            echo(f"  ... and {len(metrics) - 5} more")

    if run_record.artifacts:
        echo(f"Artifacts:   {len(run_record.artifacts)} artifacts")


@click.command("delete")
@click.argument("run_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def delete_run(ctx: click.Context, run_id: str, yes: bool) -> None:
    """Delete a run."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    if not registry.exists(run_id):
        raise click.ClickException(f"Run not found: {run_id}")

    if not yes and not click.confirm(f"Delete run {run_id}?"):
        echo("Cancelled.")
        return

    ok = registry.delete(run_id)
    if not ok:
        raise click.ClickException(f"Failed to delete run {run_id}")

    echo(f"Deleted run {run_id}")


@click.command("projects")
@click.pass_context
def list_projects(ctx: click.Context) -> None:
    """List all projects."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    projects = registry.list_projects()
    if not projects:
        echo("No projects found.")
        return

    headers = ["Project", "Runs", "Baseline"]
    rows = []
    for p in projects:
        count = registry.count_runs(project=p)
        baseline = registry.get_baseline(p)
        baseline_str = baseline["run_id"][:12] + "..." if baseline else "-"
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
    from devqubit_engine.core.config import Config
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
    rows = [
        [
            g["group_id"][:20] + ("..." if len(g["group_id"]) > 20 else ""),
            (g.get("group_name") or "")[:20],
            g["run_count"],
            g.get("last_created", "")[:19],
        ]
        for g in groups
    ]
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
    from devqubit_engine.core.config import Config
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
    rows = [
        [
            r.get("run_id", "")[:12] + "...",
            r.get("status", ""),
            str(r.get("created_at", ""))[:19],
        ]
        for r in runs
    ]
    print_table(headers, rows, f"Runs in {group_id} ({len(runs)})")
