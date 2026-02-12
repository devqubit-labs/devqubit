# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact and tag CLI commands.

Commands for browsing run artifacts, viewing their contents, and managing tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from devqubit_engine.cli._utils import (
    echo,
    format_artifacts_table,
    print_json,
    resolve_run,
    root_from_ctx,
)


if TYPE_CHECKING:
    from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
    from devqubit_engine.tracking.record import RunRecord


def register(cli: click.Group) -> None:
    """Register artifact commands with CLI."""
    cli.add_command(artifacts_group)
    cli.add_command(tag_group)


def _get_storage(
    ctx: click.Context,
) -> tuple[RegistryProtocol, ObjectStoreProtocol]:
    """
    Create registry and store from CLI context.

    Parameters
    ----------
    ctx : click.Context
        CLI context with root directory.

    Returns
    -------
    tuple
        (registry, store) instances.
    """
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)
    return registry, store


def _load_run(
    ctx: click.Context,
    run_id_or_name: str,
    project: str | None = None,
) -> RunRecord:
    """
    Load run record by ID or name.

    Parameters
    ----------
    ctx : click.Context
        CLI context.
    run_id_or_name : str
        Run identifier or name.
    project : str, optional
        Project name (required when using run name).

    Returns
    -------
    RunRecord
        Loaded run record.

    Raises
    ------
    click.ClickException
        If run is not found.
    """
    registry, _ = _get_storage(ctx)
    return resolve_run(run_id_or_name, registry, project)


def _parse_selector(selector: str) -> str | int:
    """Parse selector string, converting to int if numeric."""
    try:
        return int(selector)
    except ValueError:
        return selector


# =============================================================================
# Artifacts commands
# =============================================================================


@click.group("artifacts")
def artifacts_group() -> None:
    """Browse run artifacts."""


@artifacts_group.command("list")
@click.argument("run_id_or_name")
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.option("--role", "-r", help="Filter by role (program, results, etc).")
@click.option("--kind", "-k", help="Filter by kind substring.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
@click.pass_context
def artifacts_list(
    ctx: click.Context,
    run_id_or_name: str,
    project: str | None,
    role: str | None,
    kind: str | None,
    fmt: str,
) -> None:
    """
    List artifacts in a run.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit artifacts list abc123
        devqubit artifacts list my-experiment --project bell_state
        devqubit artifacts list abc123 --role program
    """
    from devqubit_engine.storage.artifacts.lookup import list_artifacts

    registry, store = _get_storage(ctx)
    run_record = resolve_run(run_id_or_name, registry, project)

    artifacts = list_artifacts(
        run_record,
        role=role,
        kind_contains=kind,
        store=store,
    )

    if fmt == "json":
        print_json([a.to_dict() for a in artifacts])
        return

    if not artifacts:
        echo("No artifacts found.")
        return

    echo(format_artifacts_table(artifacts))


@artifacts_group.command("show")
@click.argument("run_id_or_name")
@click.argument("selector")
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.option("--raw", is_flag=True, help="Output raw bytes to stdout.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def artifacts_show(
    ctx: click.Context,
    run_id_or_name: str,
    selector: str,
    project: str | None,
    raw: bool,
    fmt: str,
) -> None:
    """
    Show artifact content.

    SELECTOR can be: index (0, 1, ...), kind substring, or role:kind pattern.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit artifacts show abc123 0
        devqubit artifacts show abc123 counts
        devqubit artifacts show my-run program:openqasm3 --project bell_state
        devqubit artifacts show abc123 results --raw > output.json
    """
    from devqubit_engine.storage.artifacts.lookup import (
        get_artifact,
        get_artifact_bytes,
        get_artifact_text,
    )

    registry, store = _get_storage(ctx)
    run_record = resolve_run(run_id_or_name, registry, project)

    selector_val = _parse_selector(selector)

    art = get_artifact(run_record, selector_val)
    if not art:
        raise click.ClickException(f"Artifact not found: {selector}")

    # Raw binary output (bypasses format)
    if raw:
        data = get_artifact_bytes(run_record, selector_val, store)
        if data:
            click.echo(data, nl=False)
        return

    text = get_artifact_text(run_record, selector_val, store)

    if fmt == "json":
        result = {
            "role": art.role,
            "kind": art.kind,
            "digest": art.digest,
            "media_type": art.media_type,
            "content": text,
            "is_binary": text is None,
        }
        if art.meta:
            result["meta"] = art.meta
        print_json(result)
        return

    # Pretty format
    if text:
        echo(f"# {art.role}/{art.kind} ({art.digest[:20]}...)\n")
        echo(text)
    else:
        echo(f"Binary artifact: {art.role}/{art.kind}")
        echo(f"Digest: {art.digest}")
        echo(f"Media type: {art.media_type}")
        echo("\nUse --raw to output binary content.")


# =============================================================================
# Tag commands
# =============================================================================


@click.group("tag")
def tag_group() -> None:
    """Manage run tags."""


def _validate_tag_key(key: str) -> str:
    """
    Validate and normalize a tag key.

    Raises
    ------
    click.ClickException
        If key is invalid.
    """
    key = key.strip()
    if not key:
        raise click.ClickException("Tag key cannot be empty")
    if len(key) > 128:
        raise click.ClickException(f"Tag key too long (max 128): {key[:32]}...")
    # Allow alphanumeric, underscore, hyphen, dot
    import re

    if not re.match(r"^[\w.\-]+$", key):
        raise click.ClickException(
            f"Invalid tag key '{key}': use only alphanumeric, underscore, hyphen, dot"
        )
    return key


@tag_group.command("add")
@click.argument("run_id_or_name")
@click.argument("tags", nargs=-1, required=True)
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.pass_context
def tag_add(
    ctx: click.Context,
    run_id_or_name: str,
    tags: tuple[str, ...],
    project: str | None,
) -> None:
    """
    Add tags to a run.

    Tags can be key=value pairs or just keys (value defaults to "true").

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit tag add abc123 experiment=bell
        devqubit tag add my-run validated production --project bell_state
    """
    registry, _ = _get_storage(ctx)
    run_record = resolve_run(run_id_or_name, registry, project)

    # Get current tags
    current_tags = dict(run_record.tags)

    added = 0
    for tag in tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
            key = _validate_tag_key(key)
            value = value.strip()
        else:
            key = _validate_tag_key(tag)
            value = "true"

        current_tags[key] = value
        added += 1

    # Update record
    record = run_record.record
    record.setdefault("data", {})["tags"] = current_tags
    registry.save(record)

    echo(f"Added {added} tag(s) to {run_record.run_id}")


@tag_group.command("remove")
@click.argument("run_id_or_name")
@click.argument("keys", nargs=-1, required=True)
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.pass_context
def tag_remove(
    ctx: click.Context,
    run_id_or_name: str,
    keys: tuple[str, ...],
    project: str | None,
) -> None:
    """
    Remove tags from a run.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit tag remove abc123 experiment
        devqubit tag remove my-run temp debug --project bell_state
    """
    registry, _ = _get_storage(ctx)
    run_record = resolve_run(run_id_or_name, registry, project)

    # Get current tags
    current_tags = dict(run_record.tags)

    removed = 0
    not_found = []
    for key in keys:
        key = key.strip()
        if key in current_tags:
            del current_tags[key]
            removed += 1
        else:
            not_found.append(key)

    # Update record
    record = run_record.record
    record.setdefault("data", {})["tags"] = current_tags
    registry.save(record)

    if removed > 0:
        echo(f"Removed {removed} tag(s) from {run_record.run_id}")
    if not_found:
        echo(f"Tags not found: {', '.join(not_found)}")


@tag_group.command("list")
@click.argument("run_id_or_name")
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def tag_list(
    ctx: click.Context,
    run_id_or_name: str,
    project: str | None,
    fmt: str,
) -> None:
    """
    List tags on a run.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit tag list abc123
        devqubit tag list my-experiment --project bell_state
    """
    run_record = _load_run(ctx, run_id_or_name, project)
    run_tags = run_record.tags

    if fmt == "json":
        print_json(run_tags)
        return

    if not run_tags:
        echo("No tags.")
        return

    for key, value in sorted(run_tags.items()):
        echo(f"  {key}={value}")
