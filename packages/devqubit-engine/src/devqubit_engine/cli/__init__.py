"""
devqubit CLI module.

This module provides the command-line interface for devqubit, organized
into logical command groups for managing quantum experiment runs,
artifacts, comparisons, bundles, and administrative tasks.

Usage
-----
::

    devqubit [OPTIONS] COMMAND [ARGS]...

Command Groups
--------------
runs
    List, search, show, delete runs; manage projects and groups.
artifacts
    Browse artifacts; manage tags.
compare
    Diff runs, verify against baselines, replay experiments.
bundle
    Pack, unpack, and inspect bundles.
admin
    Storage management, baselines, configuration, web UI.

Examples
--------
::

    # List recent runs
    devqubit list --limit 10

    # Search by metrics
    devqubit search "metric.fidelity > 0.95"

    # Compare two runs
    devqubit diff abc123 def456

    # Pack a run for sharing
    devqubit pack abc123 -o experiment.zip

    # Launch web UI
    devqubit ui --port 8080
"""

from __future__ import annotations

from pathlib import Path

import click
import devqubit_engine.cli.admin as admin
import devqubit_engine.cli.artifacts as artifacts
import devqubit_engine.cli.bundle as bundle
import devqubit_engine.cli.compare as compare
import devqubit_engine.cli.runs as runs


@click.group()
@click.option(
    "--root",
    "-r",
    type=click.Path(path_type=Path),
    envvar="DEVQUBIT_HOME",
    default=None,
    help="Workspace root directory (default: ~/.devqubit).",
)
@click.option("--quiet", "-q", is_flag=True, help="Less output.")
@click.version_option(package_name="devqubit", prog_name="devqubit")
@click.pass_context
def cli(ctx: click.Context, root: Path | None, quiet: bool) -> None:
    """devqubit - Quantum experiment tracking."""
    ctx.ensure_object(dict)
    ctx.obj["root"] = root or (Path.home() / ".devqubit")
    ctx.obj["quiet"] = quiet


# Register all command modules
runs.register(cli)
artifacts.register(cli)
compare.register(cli)
bundle.register(cli)
admin.register(cli)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
