# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact and tag CLI commands.

This module provides commands for browsing run artifacts, viewing their
contents, and managing run tags.

Command Groups
--------------
artifacts
    List, show, and inspect artifacts in runs.
tag
    Add, remove, and list tags on runs.
"""

from __future__ import annotations

import click
from devqubit_engine.cli._utils import echo, print_json, print_table, root_from_ctx


def register(cli: click.Group) -> None:
    """Register artifact commands with CLI."""
    cli.add_command(artifacts_group)
    cli.add_command(tag_group)


# =============================================================================
# Artifacts commands
# =============================================================================


@click.group("artifacts")
def artifacts_group() -> None:
    """Browse run artifacts."""
    pass


@artifacts_group.command("list")
@click.argument("run_id")
@click.option(
    "--role", "-r", default=None, help="Filter by role (program, results, etc)."
)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def artifacts_list(
    ctx: click.Context,
    run_id: str,
    role: str | None,
    fmt: str,
) -> None:
    """List artifacts in a run."""
    from devqubit_engine.artifacts import list_artifacts
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    artifacts = list_artifacts(run_record, role=role, store=store)

    if fmt == "json":
        print_json([a.to_dict() for a in artifacts])
        return

    if not artifacts:
        echo("No artifacts found.")
        return

    headers = ["#", "Role", "Kind", "Size"]
    rows = [
        [a.index, a.role, a.kind[:30], f"{a.size:,}" if a.size else "-"]
        for a in artifacts
    ]
    print_table(headers, rows, f"Artifacts ({len(artifacts)})")


@artifacts_group.command("show")
@click.argument("run_id")
@click.argument("selector")
@click.option("--raw", is_flag=True, help="Output raw bytes to stdout.")
@click.pass_context
def artifacts_show(
    ctx: click.Context,
    run_id: str,
    selector: str,
    raw: bool,
) -> None:
    """
    Show artifact content.

    SELECTOR can be: index (0, 1, ...), kind substring, or role:kind pattern.

    Examples:
        devqubit artifacts show abc123 0
        devqubit artifacts show abc123 counts
        devqubit artifacts show abc123 program:openqasm3
        devqubit artifacts show abc123 results --raw > output.json
    """
    from devqubit_engine.artifacts import (
        get_artifact,
        get_artifact_bytes,
        get_artifact_text,
    )
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    # Parse selector as int if possible
    try:
        selector_val: str | int = int(selector)
    except ValueError:
        selector_val = selector

    art = get_artifact(run_record, selector_val)
    if not art:
        raise click.ClickException(f"Artifact not found: {selector}")

    if raw:
        data = get_artifact_bytes(run_record, selector_val, store)
        if data:
            click.echo(data, nl=False)
        return

    text = get_artifact_text(run_record, selector_val, store)
    if text:
        echo(f"# {art.role}/{art.kind} ({art.digest[:20]}...)\n")
        echo(text)
    else:
        echo(f"Binary artifact: {art.role}/{art.kind}")
        echo(f"Digest: {art.digest}")
        echo(f"Media type: {art.media_type}")
        echo("Use --raw to output binary content.")


@artifacts_group.command("counts")
@click.argument("run_id")
@click.option("--top", "-k", type=int, default=10, help="Show top K outcomes.")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.pass_context
def artifacts_counts(
    ctx: click.Context,
    run_id: str,
    top: int,
    fmt: str,
) -> None:
    """Show measurement counts from a run."""
    from devqubit_engine.artifacts import format_counts_table, get_counts
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    counts = get_counts(run_record, store)
    if not counts:
        raise click.ClickException("No counts found in run.")

    if fmt == "json":
        print_json(
            {
                "total_shots": counts.total_shots,
                "num_outcomes": counts.num_outcomes,
                "counts": counts.counts,
                "probabilities": counts.probabilities,
            }
        )
        return

    echo(format_counts_table(counts, top_k=top))


# =============================================================================
# Tag commands
# =============================================================================


@click.group("tag")
def tag_group() -> None:
    """Manage run tags."""
    pass


@tag_group.command("add")
@click.argument("run_id")
@click.argument("tags", nargs=-1, required=True)
@click.pass_context
def tag_add(ctx: click.Context, run_id: str, tags: tuple[str, ...]) -> None:
    """
    Add tags to a run.

    Tags can be specified as key=value pairs or just keys (value defaults to "true").


    Examples:
        devqubit tag add abc123 experiment=bell
        devqubit tag add abc123 validated production
    """
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    record = run_record.record
    run_tags = record.get("data", {}).get("tags", {})

    for tag in tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
        else:
            key, value = tag, "true"
        run_tags[key] = value

    record.setdefault("data", {})["tags"] = run_tags
    registry.save(record)
    echo(f"Added {len(tags)} tag(s) to {run_id}")


@tag_group.command("remove")
@click.argument("run_id")
@click.argument("keys", nargs=-1, required=True)
@click.pass_context
def tag_remove(ctx: click.Context, run_id: str, keys: tuple[str, ...]) -> None:
    """
    Remove tags from a run.


    Examples:
        devqubit tag remove abc123 experiment
        devqubit tag remove abc123 temp debug
    """
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    record = run_record.record
    run_tags = record.get("data", {}).get("tags", {})
    removed = sum(1 for key in keys if run_tags.pop(key, None) is not None)

    record.setdefault("data", {})["tags"] = run_tags
    registry.save(record)
    echo(f"Removed {removed} tag(s) from {run_id}")


@tag_group.command("list")
@click.argument("run_id")
@click.pass_context
def tag_list(ctx: click.Context, run_id: str) -> None:
    """List tags on a run."""
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)

    try:
        run_record = registry.load(run_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Run not found: {run_id}") from e

    run_tags = run_record.record.get("data", {}).get("tags", {})

    if not run_tags:
        echo("No tags.")
        return

    for key, value in sorted(run_tags.items()):
        echo(f"  {key}={value}")
