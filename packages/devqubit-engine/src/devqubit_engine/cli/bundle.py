# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Bundle CLI commands.

This module provides commands for packing runs into portable bundles,
unpacking bundles into workspaces, and inspecting bundle contents.

Commands
--------
pack
    Pack a run into a self-contained bundle.
unpack
    Unpack a bundle into a workspace.
info
    Show bundle info without extracting.
"""

from __future__ import annotations

from pathlib import Path

import click
from devqubit_engine.cli._utils import echo, print_json, root_from_ctx


def register(cli: click.Group) -> None:
    """Register bundle commands with CLI."""
    cli.add_command(pack_run_cmd)
    cli.add_command(unpack_bundle_cmd)
    cli.add_command(info_bundle)


@click.command("pack")
@click.argument("run_id")
@click.option("--out", "-o", type=click.Path(path_type=Path), help="Output file path.")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file.")
@click.pass_context
def pack_run_cmd(
    ctx: click.Context,
    run_id: str,
    out: Path | None,
    force: bool,
) -> None:
    """
    Pack a run into a bundle.

    Creates a self-contained ZIP archive containing the run record and
    all referenced artifacts, suitable for sharing or archiving.

    Examples:
        devqubit pack abc123
        devqubit pack abc123 -o experiment.zip
        devqubit pack abc123 -o experiment.zip --force
    """
    from devqubit_engine.bundle.pack import pack_run
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    out = out or Path(f"{run_id}.devqubit.zip")

    if out.exists() and not force:
        raise click.ClickException(f"File exists: {out}. Use --force to overwrite.")

    config = Config(root_dir=root)
    store = create_store(config=config)
    registry = create_registry(config=config)

    try:
        result = pack_run(
            run_id=run_id,
            output_path=out,
            store=store,
            registry=registry,
        )
    except Exception as e:
        raise click.ClickException(f"Pack failed: {e}") from e

    echo(f"Packed run {run_id} to {out}")
    if not ctx.obj.get("quiet"):
        echo(f"  Artifacts: {result.artifact_count}")
        echo(f"  Objects:   {result.object_count}")
        if result.missing_objects:
            echo(f"  Missing:   {len(result.missing_objects)}")


@click.command("unpack")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--to", "-t", "dest", type=click.Path(path_type=Path), help="Destination workspace."
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing run.")
@click.option(
    "--verify/--no-verify", default=True, show_default=True, help="Verify digests."
)
@click.pass_context
def unpack_bundle_cmd(
    ctx: click.Context,
    bundle: Path,
    dest: Path | None,
    force: bool,
    verify: bool,
) -> None:
    """
    Unpack a bundle into a workspace.

    Extracts the run record and all artifacts from a bundle into the
    specified workspace (or current workspace).

    Examples:
        devqubit unpack experiment.zip
        devqubit unpack experiment.zip --to /path/to/workspace
        devqubit unpack experiment.zip --force --no-verify
    """
    from devqubit_engine.bundle.pack import unpack_bundle
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    dest = dest or root
    dest.mkdir(parents=True, exist_ok=True)

    config = Config(root_dir=dest)
    dest_store = create_store(config=config)
    dest_registry = create_registry(config=config)

    try:
        result = unpack_bundle(
            bundle_path=bundle,
            dest_store=dest_store,
            dest_registry=dest_registry,
            overwrite=force,
            verify_digests=verify,
        )
    except FileExistsError as e:
        raise click.ClickException(
            f"Run already exists. Use --force to overwrite.\n  {e}"
        ) from e
    except Exception as e:
        raise click.ClickException(f"Unpack failed: {e}") from e

    echo(f"Unpacked to {dest}")
    if not ctx.obj.get("quiet"):
        echo(f"  Run ID:    {result.run_id}")
        echo(f"  Artifacts: {result.artifact_count}")
        echo(f"  Objects:   {result.object_count}")


@click.command("info")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty"
)
def info_bundle(bundle: Path, fmt: str) -> None:
    """
    Show bundle info without extracting.

    Displays metadata about the bundle including run ID, project,
    adapter, and artifact counts.

    Examples:
        devqubit info experiment.zip
        devqubit info experiment.zip --format json
    """
    from devqubit_engine.bundle.pack import list_bundle_contents

    try:
        contents = list_bundle_contents(bundle)
    except Exception as e:
        raise click.ClickException(f"Failed: {e}") from e

    if fmt == "json":
        print_json(contents)
        return

    manifest = contents.get("manifest", {})

    echo(f"Bundle:      {bundle.name}")
    echo(f"Run ID:      {contents.get('run_id', 'unknown')}")
    echo(f"Project:     {contents.get('project', 'unknown')}")
    echo(f"Adapter:     {contents.get('adapter', 'unknown')}")
    echo(f"Artifacts:   {contents.get('artifact_count', 0)}")
    echo(f"Objects:     {len(contents.get('objects', []))}")

    if manifest.get("backend_name"):
        echo(f"Backend:     {manifest['backend_name']}")
    if manifest.get("fingerprint"):
        echo(f"Fingerprint: {manifest['fingerprint'][:16]}...")
    if manifest.get("git_commit"):
        echo(f"Git commit:  {manifest['git_commit'][:8]}")
    if manifest.get("created_at"):
        echo(f"Packed at:   {manifest['created_at'][:19]}")
