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
from devqubit_engine.cli._utils import (
    echo,
    print_json,
    print_table,
    resolve_run,
    root_from_ctx,
)


def register(cli: click.Group) -> None:
    """Register admin commands with CLI."""
    cli.add_command(storage_group)
    cli.add_command(baseline_group)
    cli.add_command(config_cmd)
    cli.add_command(ui_command)
    cli.add_command(doctor_command)
    cli.add_command(gc_command)


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
@click.option("--project", "-p", default=None, help="Limit to specific project.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def storage_gc(
    ctx: click.Context,
    dry_run: bool,
    yes: bool,
    project: str | None,
    fmt: str,
) -> None:
    """
    Garbage collect unreferenced objects.

    Identifies and optionally removes objects in the object store that
    are not referenced by any run records.
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.gc import gc_run

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    # For dry-run or when we need confirmation, scan first
    if dry_run or not yes:
        stats = gc_run(store, registry, project=project, dry_run=True)
        objects_total = stats.referenced_objects + stats.unreferenced_objects

        if fmt == "json":
            result = {
                "objects_total": objects_total,
                "objects_referenced": stats.referenced_objects,
                "objects_orphaned": stats.unreferenced_objects,
                "bytes_reclaimable": stats.bytes_reclaimable,
                "dry_run": dry_run,
            }
            if project:
                result["project"] = project
            if stats.errors:
                result["errors"] = stats.errors
            print_json(result)
            return

        echo(f"Objects total:      {objects_total}")
        echo(f"Objects referenced: {stats.referenced_objects}")
        echo(f"Objects orphaned:   {stats.unreferenced_objects}")
        echo(f"Reclaimable:        {stats.bytes_reclaimable:,} bytes")

        if stats.errors:
            echo(f"\nWarnings ({len(stats.errors)}):")
            for err in stats.errors[:5]:
                echo(f"  - {err}")
            if len(stats.errors) > 5:
                echo(f"  ... and {len(stats.errors) - 5} more")

        if dry_run:
            echo("\nDry run - no objects deleted.")
            return

        if stats.unreferenced_objects == 0:
            echo("\nNo orphaned objects to delete.")
            return

        if not click.confirm(
            f"\nDelete {stats.unreferenced_objects} orphaned objects "
            f"({stats.bytes_reclaimable:,} bytes)?"
        ):
            echo("Cancelled.")
            return

    # Actually delete (either --yes was passed, or user confirmed)
    stats = gc_run(store, registry, project=project, dry_run=False)

    if fmt == "json":
        result = {
            "objects_deleted": stats.objects_deleted,
            "bytes_reclaimed": stats.bytes_reclaimed,
            "dry_run": False,
        }
        if project:
            result["project"] = project
        if stats.errors:
            result["errors"] = stats.errors
        print_json(result)
        return

    echo(f"Deleted {stats.objects_deleted} objects ({stats.bytes_reclaimed:,} bytes)")
    if stats.errors:
        echo(f"\nEncountered {len(stats.errors)} errors during deletion.")


@storage_group.command("prune")
@click.option(
    "--status", "-s", default="FAILED", help="Status to prune (default: FAILED)."
)
@click.option("--older-than", type=int, default=30, help="Days old (default: 30).")
@click.option("--keep-latest", type=int, default=5, help="Keep N latest (default: 5).")
@click.option("--project", "-p", default=None, help="Limit to specific project.")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without deleting.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def storage_prune(
    ctx: click.Context,
    status: str,
    older_than: int,
    keep_latest: int,
    project: str | None,
    dry_run: bool,
    yes: bool,
    fmt: str,
) -> None:
    """
    Prune old runs by status.

    Removes runs matching the specified status that are older than the
    threshold, while keeping the N most recent matching runs per project.
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry
    from devqubit_engine.storage.gc import prune_runs

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    # For dry-run or when we need confirmation, scan first
    if dry_run or not yes:
        stats = prune_runs(
            registry,
            project=project,
            status=status,
            older_than_days=older_than,
            keep_latest=keep_latest,
            dry_run=True,
        )

        if fmt == "json":
            result = {
                "status_filter": status,
                "older_than_days": older_than,
                "keep_latest": keep_latest,
                "runs_scanned": stats.runs_scanned,
                "runs_to_prune": stats.runs_pruned,
                "dry_run": dry_run,
            }
            if project:
                result["project"] = project
            if stats.errors:
                result["errors"] = stats.errors
            print_json(result)
            return

        echo(f"Status filter:  {status}")
        echo(f"Older than:     {older_than} days")
        echo(f"Keep latest:    {keep_latest} per project")
        if project:
            echo(f"Project:        {project}")
        echo(f"Runs scanned:   {stats.runs_scanned}")
        echo(f"Runs to prune:  {stats.runs_pruned}")

        if stats.errors:
            echo(f"\nWarnings ({len(stats.errors)}):")
            for err in stats.errors[:5]:
                echo(f"  - {err}")
            if len(stats.errors) > 5:
                echo(f"  ... and {len(stats.errors) - 5} more")

        if dry_run:
            echo("\nDry run - no runs deleted.")
            return

        if stats.runs_pruned == 0:
            echo("\nNo runs match pruning criteria.")
            return

        if not click.confirm(f"\nDelete {stats.runs_pruned} {status} runs?"):
            echo("Cancelled.")
            return

    # Actually prune
    stats = prune_runs(
        registry,
        project=project,
        status=status,
        older_than_days=older_than,
        keep_latest=keep_latest,
        dry_run=False,
    )

    if fmt == "json":
        result = {
            "runs_deleted": stats.runs_pruned,
            "dry_run": False,
        }
        if stats.errors:
            result["errors"] = stats.errors
        print_json(result)
        return

    echo(f"Pruned {stats.runs_pruned} runs.")
    if stats.errors:
        echo(f"\nEncountered {len(stats.errors)} errors during deletion.")


@storage_group.command("health")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def storage_health(ctx: click.Context, fmt: str) -> None:
    """Check storage health and integrity."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.gc import check_workspace_health

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    health = check_workspace_health(store, registry)

    if fmt == "json":
        print_json(health)
        return

    echo(f"Workspace:        {root}")
    echo(f"Total runs:       {health['total_runs']}")
    echo(f"Total objects:    {health['total_objects']}")
    echo(f"Referenced:       {health['referenced_objects']}")
    echo(f"Orphaned:         {health['orphaned_objects']}")
    echo(f"Missing:          {health['missing_objects']}")

    if health["errors"]:
        echo(f"\nErrors ({len(health['errors'])}):")
        for err in health["errors"][:10]:
            echo(f"  - {err}")
        if len(health["errors"]) > 10:
            echo(f"  ... and {len(health['errors']) - 10} more")

    echo("")
    if health["missing_objects"] > 0:
        echo("[!] Some runs reference missing objects!")
        echo("    Data integrity may be compromised.")
    elif health["orphaned_objects"] > 0:
        echo("[*] Run 'devqubit storage gc' to reclaim space from orphaned objects.")
    elif not health["healthy"]:
        echo("[!] Workspace has issues - check errors above.")
    else:
        echo("[OK] Workspace is healthy.")


# =============================================================================
# Baseline commands
# =============================================================================


@click.group("baseline")
def baseline_group() -> None:
    """Manage project baselines for verification."""
    pass


@baseline_group.command("set")
@click.argument("project")
@click.argument("run_id_or_name")
@click.pass_context
def baseline_set(ctx: click.Context, project: str, run_id_or_name: str) -> None:
    """
    Set baseline run for a project.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    the project argument is used for name resolution.

    Examples:
        devqubit baseline set myproject abc123
        devqubit baseline set bell_state baseline-v1
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    # Resolve run (supports ID or name within project)
    run_record = resolve_run(run_id_or_name, registry, project)

    registry.set_baseline(project, run_record.run_id)
    echo(f"Baseline set: {project} -> {run_record.run_id}")


@baseline_group.command("get")
@click.argument("project")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def baseline_get(ctx: click.Context, project: str, fmt: str) -> None:
    """Get baseline run for a project."""
    from devqubit_engine.config import Config
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
    from devqubit_engine.config import Config
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
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def baseline_list(ctx: click.Context, fmt: str) -> None:
    """List all project baselines."""
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    projects = registry.list_projects()

    if fmt == "json":
        baselines = []
        for proj in projects:
            baseline = registry.get_baseline(proj)
            if baseline:
                baselines.append(baseline)
        print_json(baselines)
        return

    if not projects:
        echo("No projects found.")
        return

    headers = ["Project", "Baseline Run", "Set At"]
    rows = []
    for proj in projects:
        baseline = registry.get_baseline(proj)
        if baseline:
            run_id = baseline["run_id"]
            display_id = run_id[:12] + "..." if len(run_id) > 15 else run_id
            set_at = baseline["set_at"][:19] if baseline["set_at"] else "-"
            rows.append([proj, display_id, set_at])

    if not rows:
        echo("No baselines set.")
        return

    print_table(headers, rows, "Project Baselines")


# =============================================================================
# Config command
# =============================================================================


@click.command("config")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def config_cmd(ctx: click.Context, fmt: str) -> None:
    """Show current configuration."""
    from devqubit_engine.config import load_config

    # Use load_config to respect environment variables
    config = load_config()

    # Allow CLI --home to override
    cli_root = root_from_ctx(ctx)
    if cli_root != config.root_dir:
        # CLI override was provided
        from devqubit_engine.config import Config

        config = Config(root_dir=cli_root)

    if fmt == "json":
        print_json(config.to_dict())
        return

    echo(f"Home:              {config.root_dir}")
    echo(f"Storage URL:       {config.storage_url}")
    echo(f"Registry URL:      {config.registry_url}")
    echo(f"Capture pip:       {config.capture_pip}")
    echo(f"Capture git:       {config.capture_git}")
    echo(f"Validate records:  {config.validate}")
    echo(f"Redaction enabled: {config.redaction.enabled}")
    if config.redaction.enabled:
        echo(f"Redaction patterns: {len(config.redaction.patterns)} configured")


# =============================================================================
# UI command
# =============================================================================


def _is_ui_available() -> bool:
    """Check if the devqubit-ui package is installed via entry points."""
    from importlib.metadata import entry_points

    eps = entry_points()
    # Handle both Python 3.9 (returns dict) and 3.10+ (SelectableGroups)
    try:
        ui_eps = eps.select(group="devqubit.components", name="ui")
        return len(list(ui_eps)) > 0
    except AttributeError:
        # Python 3.9 fallback
        group = eps.get("devqubit.components", [])
        return any(ep.name == "ui" for ep in group)


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
        raise SystemExit(1)

    from devqubit.ui import run_server

    echo(f"Starting devqubit UI at http://{host}:{port}")
    if workspace:
        echo(f"Workspace: {workspace}")

    run_server(
        host=host,
        port=port,
        workspace=workspace,
        debug=debug,
    )


# =============================================================================
# Doctor command
# =============================================================================


@click.command("doctor")
@click.option(
    "--older-than",
    default="6h",
    show_default=True,
    help="Age threshold (e.g. 6h, 1d, 30m).",
)
@click.option("--project", "-p", default=None, help="Limit to specific project.")
@click.option("--auto", is_flag=True, help="Recover stale runs without prompting.")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without modifying.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def doctor_command(
    ctx: click.Context,
    older_than: str,
    project: str | None,
    auto: bool,
    dry_run: bool,
    fmt: str,
) -> None:
    """
    Diagnose and recover stale runs.

    Finds runs stuck in RUNNING state that are older than the given
    threshold and marks them as KILLED.  This is useful after process
    crashes, SSH disconnects, or OOM kills during hardware jobs.

    Examples:
        devqubit doctor --dry-run
        devqubit doctor --auto --older-than 6h
        devqubit doctor --auto --older-than 1d --project nightly-ci
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    hours = _parse_duration_to_hours(older_than)

    if not hasattr(registry, "recover_stale_runs"):
        raise click.ClickException(
            "Current registry backend does not support stale run recovery."
        )

    stale = registry.recover_stale_runs(
        older_than_hours=hours,
        project=project,
        dry_run=True,
    )

    if fmt == "json":
        if dry_run or not auto:
            print_json({"stale_runs": stale, "count": len(stale), "dry_run": True})
            if dry_run or not stale:
                return
        if not auto:
            if not click.confirm(f"Recover {len(stale)} stale run(s)?"):
                echo("Cancelled.")
                return
        recovered = registry.recover_stale_runs(
            older_than_hours=hours,
            project=project,
            dry_run=False,
        )
        print_json(
            {"recovered_runs": recovered, "count": len(recovered), "dry_run": False}
        )
        return

    if not stale:
        echo("No stale runs found.")
        return

    echo(f"Found {len(stale)} stale run(s) (RUNNING for >{older_than}):\n")

    headers = ["Run ID", "Project", "Created"]
    rows = [
        [
            r["run_id"][:12] + "...",
            r["project"][:20],
            str(r["created_at"])[:19],
        ]
        for r in stale
    ]
    print_table(headers, rows, "Stale Runs")

    if dry_run:
        echo("\nDry run - no runs modified.")
        return

    if not auto:
        if not click.confirm(f"\nRecover {len(stale)} stale run(s)? (mark as KILLED)"):
            echo("Cancelled.")
            return

    recovered = registry.recover_stale_runs(
        older_than_hours=hours,
        project=project,
        dry_run=False,
    )
    echo(f"\nRecovered {len(recovered)} run(s).")


def _parse_duration_to_hours(value: str) -> float:
    """
    Parse a human-readable duration string to hours.

    Parameters
    ----------
    value : str
        Duration string (e.g. ``"6h"``, ``"1d"``, ``"30m"``).

    Returns
    -------
    float
        Duration in hours.

    Raises
    ------
    click.BadParameter
        If the format is not recognized.
    """
    import re

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hdm])", value.strip().lower())
    if not match:
        raise click.BadParameter(
            f"Invalid duration: {value!r}. Use format like 6h, 1d, or 30m."
        )
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "h":
        return amount
    if unit == "d":
        return amount * 24.0
    if unit == "m":
        return amount / 60.0
    raise click.BadParameter(f"Unknown duration unit: {unit!r}")


# =============================================================================
# Top-level GC command
# =============================================================================


@click.command("gc")
@click.option(
    "--older-than",
    default=None,
    help="Delete runs older than this (e.g. 90d, 24h).  Without this flag, "
    "only orphaned objects are cleaned — no runs are deleted.",
)
@click.option(
    "--keep-last",
    type=int,
    default=10,
    show_default=True,
    help="Always keep N most recent runs per project.",
)
@click.option("--project", "-p", default=None, help="Limit to specific project.")
@click.option(
    "--status",
    "-s",
    default=None,
    help="Only prune runs with this status (e.g. FAILED, KILLED).",
)
@click.option("--dry-run", "-n", is_flag=True, help="Preview without deleting.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def gc_command(
    ctx: click.Context,
    older_than: str | None,
    keep_last: int,
    project: str | None,
    status: str | None,
    dry_run: bool,
    yes: bool,
    fmt: str,
) -> None:
    """
    Clean up old runs and orphaned objects.

    Combines run pruning and object garbage collection in one step.
    Without --older-than, only orphaned objects are cleaned (no runs deleted).

    \b
    Examples:
        devqubit gc --dry-run
        devqubit gc --older-than 90d --project nightly-ci
        devqubit gc --older-than 30d --status FAILED --yes
        devqubit gc --older-than 7d --keep-last 50
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store
    from devqubit_engine.storage.gc import gc_run, prune_runs

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    # Phase 1: prune runs (only if --older-than provided)
    prune_stats = None
    if older_than is not None:
        hours = _parse_duration_to_hours(older_than)
        older_than_days = hours / 24.0

        prune_stats = prune_runs(
            registry,
            project=project,
            status=status,
            older_than_days=int(older_than_days) if older_than_days >= 1 else 1,
            keep_latest=keep_last,
            dry_run=True,
        )

        if fmt == "pretty":
            if prune_stats.runs_pruned > 0:
                echo(f"Runs to delete:   {prune_stats.runs_pruned}")
            else:
                echo("No runs match pruning criteria.")

    # Phase 2: scan orphaned objects
    gc_stats = gc_run(store, registry, project=project, dry_run=True)

    if fmt == "json":

        result: dict = {
            "orphaned_objects": gc_stats.unreferenced_objects,
            "bytes_reclaimable": gc_stats.bytes_reclaimable,
            "dry_run": dry_run,
        }
        if prune_stats is not None:
            result["runs_to_prune"] = prune_stats.runs_pruned
        print_json(result)
        if dry_run:
            return
    else:
        echo(f"Orphaned objects: {gc_stats.unreferenced_objects}")
        if gc_stats.bytes_reclaimable > 0:
            echo(f"Reclaimable:      {gc_stats.bytes_reclaimable:,} bytes")

    nothing_to_do = (
        prune_stats is None or prune_stats.runs_pruned == 0
    ) and gc_stats.unreferenced_objects == 0

    if dry_run:
        if fmt == "pretty":
            echo("\nDry run — nothing modified.")
        return

    if nothing_to_do:
        if fmt == "pretty":
            echo("\nNothing to clean up.")
        return

    if not yes:
        parts = []
        if prune_stats and prune_stats.runs_pruned > 0:
            parts.append(f"{prune_stats.runs_pruned} run(s)")
        if gc_stats.unreferenced_objects > 0:
            parts.append(f"{gc_stats.unreferenced_objects} orphaned object(s)")
        if not click.confirm(f"\nDelete {' and '.join(parts)}?"):
            echo("Cancelled.")
            return

    # Execute
    runs_deleted = 0
    if prune_stats and prune_stats.runs_pruned > 0:
        actual = prune_runs(
            registry,
            project=project,
            status=status,
            older_than_days=int(_parse_duration_to_hours(older_than) / 24.0) or 1,
            keep_latest=keep_last,
            dry_run=False,
        )
        runs_deleted = actual.runs_pruned

    # Re-scan after pruning (may have created new orphans)
    actual_gc = gc_run(store, registry, project=project, dry_run=False)

    if fmt == "json":
        result = {
            "runs_deleted": runs_deleted,
            "objects_deleted": actual_gc.objects_deleted,
            "bytes_reclaimed": actual_gc.bytes_reclaimed,
            "dry_run": False,
        }
        print_json(result)
    else:
        if runs_deleted:
            echo(f"\nDeleted {runs_deleted} run(s).")
        echo(
            f"Reclaimed {actual_gc.objects_deleted} object(s) "
            f"({actual_gc.bytes_reclaimed:,} bytes)."
        )
