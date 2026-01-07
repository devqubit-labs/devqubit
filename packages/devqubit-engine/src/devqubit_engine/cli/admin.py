# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Administrative CLI commands.

This module provides administrative commands for storage management,
baseline configuration, system configuration display, and web UI launch.

Command Groups
--------------
storage
    Garbage collection, pruning, and health checks.
baseline
    Manage project baselines for verification.
config
    Display current configuration.
ui
    Launch local web UI.
"""

from __future__ import annotations

import click
from devqubit_engine.cli._utils import echo, print_json, print_table, root_from_ctx


def register(cli: click.Group) -> None:
    """Register admin commands with CLI."""
    cli.add_command(storage_group)
    cli.add_command(baseline_group)
    cli.add_command(config_cmd)
    cli.add_command(ui_command)


# =============================================================================
# Storage commands
# =============================================================================


@click.group("storage")
def storage_group() -> None:
    """Storage management commands."""
    pass


@storage_group.command("gc")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without deleting.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.option(
    "--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty"
)
@click.pass_context
def storage_gc(ctx: click.Context, dry_run: bool, yes: bool, fmt: str) -> None:
    """
    Garbage collect unreferenced objects.

    Identifies and optionally removes objects in the object store that
    are not referenced by any run records.
    """
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.gc import gc_run

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    stats = gc_run(store, registry, dry_run=True)

    if fmt == "json":
        print_json(
            {
                "objects_total": stats.objects_total,
                "objects_referenced": stats.objects_referenced,
                "objects_deleted": stats.objects_deleted if not dry_run else 0,
                "bytes_reclaimable": stats.bytes_reclaimable,
                "bytes_reclaimed": stats.bytes_reclaimed if not dry_run else 0,
                "dry_run": dry_run,
            }
        )
        if dry_run:
            return

    if dry_run or stats.objects_total == stats.objects_referenced:
        echo(f"Objects total:      {stats.objects_total}")
        echo(f"Objects referenced: {stats.objects_referenced}")
        echo(f"Objects orphaned:   {stats.objects_total - stats.objects_referenced}")
        echo(f"Reclaimable:        {stats.bytes_reclaimable:,} bytes")
        if dry_run:
            echo("\nDry run - no objects deleted.")
        else:
            echo("\nNo orphaned objects to delete.")
        return

    if not yes:
        orphaned = stats.objects_total - stats.objects_referenced
        if not click.confirm(
            f"Delete {orphaned} orphaned objects ({stats.bytes_reclaimable:,} bytes)?"
        ):
            echo("Cancelled.")
            return

    stats = gc_run(store, registry, dry_run=False)
    echo(f"Deleted {stats.objects_deleted} objects ({stats.bytes_reclaimed:,} bytes)")


@storage_group.command("prune")
@click.option(
    "--status", "-s", default="FAILED", help="Status to prune (default: FAILED)."
)
@click.option("--older-than", type=int, default=30, help="Days old (default: 30).")
@click.option("--keep-latest", type=int, default=5, help="Keep N latest (default: 5).")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without deleting.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def storage_prune(
    ctx: click.Context,
    status: str,
    older_than: int,
    keep_latest: int,
    dry_run: bool,
    yes: bool,
) -> None:
    """
    Prune old runs by status.

    Removes runs matching the specified status that are older than the
    threshold, while keeping the N most recent matching runs.
    """
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry
    from devqubit_engine.storage.gc import prune_runs

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    stats = prune_runs(
        registry,
        status=status,
        older_than_days=older_than,
        keep_latest=keep_latest,
        dry_run=True,
    )

    if dry_run or stats["runs_deleted"] == 0:
        echo(f"Runs matching:  {stats['runs_matched']}")
        echo(f"Runs to delete: {stats['runs_deleted']}")
        echo(f"Runs protected: {stats['runs_protected']}")
        if dry_run:
            echo("\nDry run - no runs deleted.")
        return

    if not yes:
        if not click.confirm(f"Delete {stats['runs_deleted']} {status} runs?"):
            echo("Cancelled.")
            return

    stats = prune_runs(
        registry,
        status=status,
        older_than_days=older_than,
        keep_latest=keep_latest,
        dry_run=False,
    )
    echo(f"Deleted {stats['runs_deleted']} runs")


@storage_group.command("health")
@click.pass_context
def storage_health(ctx: click.Context) -> None:
    """
    Check workspace health.

    Reports on total runs, objects, and identifies any integrity issues
    such as orphaned or missing objects.
    """
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.gc import check_workspace_health

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    health = check_workspace_health(store, registry)

    echo(f"Total runs:        {health['total_runs']}")
    echo(f"Total objects:     {health['total_objects']}")
    echo(f"Referenced objects:{health['referenced_objects']}")
    echo(f"Orphaned objects:  {health['orphaned_objects']}")
    echo(f"Missing objects:   {health['missing_objects']}")

    if health["missing_objects"] > 0:
        echo("\n⚠ Some runs reference missing objects!")
    elif health["orphaned_objects"] > 0:
        echo("\nRun 'devqubit storage gc' to reclaim space.")
    else:
        echo("\n✓ Workspace is healthy.")


# =============================================================================
# Baseline commands
# =============================================================================


@click.group("baseline")
def baseline_group() -> None:
    """Manage project baselines for verification."""
    pass


@baseline_group.command("set")
@click.argument("project")
@click.argument("run_id")
@click.pass_context
def baseline_set(ctx: click.Context, project: str, run_id: str) -> None:
    """Set baseline run for a project."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry
    from devqubit_engine.storage.protocols import RunNotFoundError
    from devqubit_engine.utils.time_utils import utc_now_iso

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    registry.set_baseline(project, run_id, utc_now_iso())
    echo(f"Baseline set: {project} → {run_id}")


@baseline_group.command("get")
@click.argument("project")
@click.option(
    "--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty"
)
@click.pass_context
def baseline_get(ctx: click.Context, project: str, fmt: str) -> None:
    """Get baseline run for a project."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    baseline = registry.get_baseline(project)

    if fmt == "json":
        print_json(baseline)
        return

    if not baseline:
        echo(f"No baseline set for project: {project}")
        return

    echo(f"Project:  {baseline['project']}")
    echo(f"Run ID:   {baseline['run_id']}")
    echo(f"Set at:   {baseline['set_at']}")


@baseline_group.command("clear")
@click.argument("project")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.pass_context
def baseline_clear(ctx: click.Context, project: str, yes: bool) -> None:
    """Clear baseline for a project."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    if not yes and not click.confirm(f"Clear baseline for {project}?"):
        echo("Cancelled.")
        return

    if registry.clear_baseline(project):
        echo(f"Baseline cleared for: {project}")
    else:
        echo(f"No baseline set for: {project}")


@baseline_group.command("list")
@click.pass_context
def baseline_list(ctx: click.Context) -> None:
    """List all project baselines."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    projects = registry.list_projects()
    if not projects:
        echo("No projects found.")
        return

    headers = ["Project", "Baseline Run", "Set At"]
    rows = []
    for proj in projects:
        baseline = registry.get_baseline(proj)
        if baseline:
            rows.append(
                [
                    proj,
                    baseline["run_id"][:12] + "...",
                    baseline["set_at"][:19],
                ]
            )

    if not rows:
        echo("No baselines set.")
        return

    print_table(headers, rows, "Project Baselines")


# =============================================================================
# Config command
# =============================================================================


@click.command("config")
@click.pass_context
def config_cmd(ctx: click.Context) -> None:
    """Show current configuration."""
    from devqubit_engine.core.config import Config

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)

    echo(f"Home:              {root}")
    echo(f"Storage URL:       {config.storage_url}")
    echo(f"Registry URL:      {config.registry_url}")
    echo(f"Capture pip:       {config.capture_pip}")
    echo(f"Capture git:       {config.capture_git}")
    echo(f"Validate records:  {config.validate}")
    echo(f"Redaction enabled: {config.redaction.enabled}")


# =============================================================================
# UI command
# =============================================================================


def _is_ui_available() -> bool:
    """Check if the devqubit-ui package is installed via entry points."""
    from importlib.metadata import entry_points

    eps = entry_points()
    ui_eps = eps.select(group="devqubit.components", name="ui")
    return len(list(ui_eps)) > 0


@click.command("ui")
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", "-p", default=8080, type=int, help="Port to listen on.")
@click.option("--workspace", "-w", default=None, help="Workspace directory.")
@click.option("--debug", is_flag=True, help="Enable debug mode.")
def ui_command(host: str, port: int, workspace: str | None, debug: bool) -> None:
    """
    Launch local web UI.

    Requires the devqubit-ui package to be installed.

    Examples:
        devqubit ui
        devqubit ui --port 9000
        devqubit ui --workspace /path/to/.devqubit
    """

    if not _is_ui_available():
        echo("Error: The web UI requires the devqubit-ui package.", err=True)
        echo("", err=True)
        echo("Install it with one of:", err=True)
        echo("  pip install devqubit[ui]", err=True)
        echo("  pip install devqubit-ui", err=True)
        raise SystemExit(1)

    from devqubit_ui import run_server

    echo(f"Starting devqubit UI at http://{host}:{port}")
    if workspace:
        echo(f"Workspace: {workspace}")

    run_server(host=host, port=port, workspace=workspace, debug=debug)
