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
from devqubit_engine.cli._utils import echo, is_quiet, print_json, root_from_ctx


def register(cli: click.Group) -> None:
    """Register bundle commands with CLI."""
    cli.add_command(pack_cmd)
    cli.add_command(unpack_cmd)
    cli.add_command(info_cmd)


@click.command("pack")
@click.argument("run_id_or_name")
@click.option(
    "--project", "-p", default=None, help="Project name (required when using run name)."
)
@click.option("--out", "-o", type=click.Path(path_type=Path), help="Output file path.")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def pack_cmd(
    ctx: click.Context,
    run_id_or_name: str,
    project: str | None,
    out: Path | None,
    force: bool,
    fmt: str,
) -> None:
    """
    Pack a run into a bundle.

    Creates a self-contained ZIP archive containing the run record and
    all referenced artifacts, suitable for sharing or archiving.

    RUN_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    Examples:
        devqubit pack abc123
        devqubit pack abc123 -o experiment.zip
        devqubit pack my-experiment --project bell_state -o experiment.zip
        devqubit pack abc123 -o experiment.zip --force
    """
    from devqubit_engine.bundle.pack import pack_run
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    output_path = out or Path(f"{run_id_or_name}.devqubit.zip")

    if output_path.exists() and not force:
        raise click.ClickException(
            f"File exists: {output_path}. Use --force to overwrite."
        )

    config = Config(root_dir=root)
    store = create_store(config=config)
    registry = create_registry(config=config)

    try:
        result = pack_run(
            run_id_or_name,
            output_path=output_path,
            store=store,
            registry=registry,
            project=project,
        )
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            if project:
                raise click.ClickException(
                    f"Run not found: '{run_id_or_name}' (looked up as ID and as name in project '{project}')"
                ) from e
            raise click.ClickException(
                f"Run not found: '{run_id_or_name}'. Use --project to look up by name."
            ) from e
        raise click.ClickException(f"Pack failed: {e}") from e

    if fmt == "json":
        print_json(
            {
                "run_id": result.run_id,
                "output_path": str(output_path),
                "artifact_count": result.artifact_count,
                "object_count": result.object_count,
                "missing_objects": result.missing_objects,
                "size_bytes": (
                    output_path.stat().st_size if output_path.exists() else None
                ),
            }
        )
        return

    echo(f"Packed run {result.run_id} to {output_path}")
    if not is_quiet(ctx):
        echo(f"  Artifacts: {result.artifact_count}")
        echo(f"  Objects:   {result.object_count}")
        if result.missing_objects:
            echo(f"  Missing:   {len(result.missing_objects)}")
            for digest in result.missing_objects[:3]:
                echo(f"    - {digest[:20]}...")
            if len(result.missing_objects) > 3:
                echo(f"    ... and {len(result.missing_objects) - 3} more")


@click.command("unpack")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--to",
    "-t",
    "dest",
    type=click.Path(path_type=Path),
    help="Destination workspace.",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing run.")
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help="Verify digests.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.pass_context
def unpack_cmd(
    ctx: click.Context,
    bundle: Path,
    dest: Path | None,
    force: bool,
    verify: bool,
    fmt: str,
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
    from devqubit_engine.bundle.reader import is_bundle_path
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    # Validate bundle before proceeding
    if not is_bundle_path(bundle):
        raise click.ClickException(
            f"Not a valid devqubit bundle: {bundle}\n"
            "Bundle must be a ZIP file containing manifest.json and run.json"
        )

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

    if fmt == "json":
        print_json(
            {
                "run_id": result.run_id,
                "bundle_path": str(result.bundle_path),
                "destination": str(dest),
                "artifact_count": result.artifact_count,
                "object_count": result.object_count,
                "skipped_objects": result.skipped_objects,
                "missing_objects": result.missing_objects,
                "total_objects": result.total_objects,
            }
        )
        return

    echo(f"Unpacked to {dest}")
    if not is_quiet(ctx):
        echo(f"  Run ID:    {result.run_id}")
        echo(f"  Artifacts: {result.artifact_count}")
        echo(f"  Objects:   {result.object_count}")
        if result.skipped_objects:
            echo(f"  Skipped:   {len(result.skipped_objects)} (already exist)")


@click.command("info")
@click.argument("bundle", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
)
@click.option(
    "--objects", "show_objects", is_flag=True, help="List all object digests."
)
@click.option(
    "--artifacts", "show_artifacts", is_flag=True, help="List artifact details."
)
def info_cmd(
    bundle: Path,
    fmt: str,
    show_objects: bool,
    show_artifacts: bool,
) -> None:
    """
    Show bundle info without extracting.

    Displays metadata about the bundle including run ID, project,
    adapter, and artifact counts.

    Examples:
        devqubit info experiment.zip
        devqubit info experiment.zip --format json
        devqubit info experiment.zip --artifacts
    """
    from devqubit_engine.bundle.reader import Bundle, is_bundle_path

    if not is_bundle_path(bundle):
        raise click.ClickException(
            f"Not a valid devqubit bundle: {bundle}\n"
            "Bundle must be a ZIP file containing manifest.json and run.json"
        )

    try:
        with Bundle(bundle) as b:
            manifest = b.manifest
            run_record = b.run_record
            objects = b.list_objects()
            artifacts = run_record.get("artifacts", []) or []

            contents = {
                "bundle_path": str(bundle),
                "run_id": b.run_id,
                "project": b.get_project(),
                "adapter": b.get_adapter(),
                "artifact_count": len(artifacts),
                "object_count": len(objects),
                "manifest": manifest,
            }

            if show_objects:
                contents["objects"] = objects

            if show_artifacts:
                contents["artifacts"] = artifacts

    except Exception as e:
        raise click.ClickException(f"Failed to read bundle: {e}") from e

    if fmt == "json":
        print_json(contents)
        return

    echo(f"Bundle:      {bundle.name}")
    echo(f"Run ID:      {contents['run_id'] or 'unknown'}")
    echo(f"Project:     {contents['project'] or 'unknown'}")
    echo(f"Adapter:     {contents['adapter'] or 'unknown'}")
    echo(f"Artifacts:   {contents['artifact_count']}")
    echo(f"Objects:     {contents['object_count']}")

    # Additional manifest info
    if manifest.get("format_version"):
        echo(f"Format:      v{manifest['format_version']}")
    if manifest.get("backend_name"):
        echo(f"Backend:     {manifest['backend_name']}")
    if manifest.get("fingerprint"):
        fp = manifest["fingerprint"]
        echo(f"Fingerprint: {fp[:16]}..." if len(fp) > 16 else f"Fingerprint: {fp}")
    if manifest.get("git_commit"):
        echo(f"Git commit:  {manifest['git_commit'][:8]}")
    if manifest.get("created_at"):
        echo(f"Packed at:   {manifest['created_at'][:19]}")

    # Run record info
    info = run_record.get("info", {})
    if isinstance(info, dict):
        if info.get("status"):
            echo(f"Status:      {info['status']}")
        if info.get("run_name"):
            echo(f"Run name:    {info['run_name']}")

    # Optional detailed listings
    if show_artifacts and artifacts:
        echo("\nArtifacts:")
        for i, art in enumerate(artifacts):
            if isinstance(art, dict):
                kind = art.get("kind", "unknown")
                role = art.get("role", "unknown")
                digest = art.get("digest", "")[:16]
                echo(f"  [{i}] {role}/{kind} ({digest}...)")

    if show_objects and objects:
        echo("\nObjects:")
        for digest in objects[:20]:
            echo(f"  {digest}")
        if len(objects) > 20:
            echo(f"  ... and {len(objects) - 20} more")
