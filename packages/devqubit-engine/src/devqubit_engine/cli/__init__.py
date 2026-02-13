# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

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
    Diff runs, verify against baselines.
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

    # Compare runs by name
    devqubit diff baseline-v1 experiment-v2 --project bell_state

    # Verify against baseline
    devqubit verify abc123 --project myproject

    # Pack a run for sharing
    devqubit pack abc123 -o experiment.zip

    # Pack by name
    devqubit pack my-experiment --project bell_state -o experiment.zip

    # Launch web UI
    devqubit ui --port 8080
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


def _get_default_root() -> Path:
    """Get default workspace root directory."""
    return Path.home() / ".devqubit"


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
@click.option("--debug", is_flag=True, hidden=True, help="Enable debug logging.")
@click.version_option(package_name="devqubit", prog_name="devqubit")
@click.pass_context
def cli(ctx: click.Context, root: Path | None, quiet: bool, debug: bool) -> None:
    """devqubit - Quantum experiment tracking."""
    import logging

    # Initialize context object
    ctx.ensure_object(dict)
    ctx.obj["root"] = root or _get_default_root()
    ctx.obj["quiet"] = quiet
    ctx.obj["debug"] = debug

    # Configure logging
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    elif not quiet:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )


def _register_commands() -> None:
    """Register all command modules with CLI."""
    # Import here to avoid circular imports and speed up --help
    import devqubit_engine.cli.admin as admin
    import devqubit_engine.cli.artifacts as artifacts
    import devqubit_engine.cli.bundle as bundle
    import devqubit_engine.cli.compare as compare
    import devqubit_engine.cli.runs as runs

    runs.register(cli)
    artifacts.register(cli)
    compare.register(cli)
    bundle.register(cli)
    admin.register(cli)


# Register commands
_register_commands()


def main() -> None:
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as e:
        # Only show full traceback in debug mode
        if "--debug" in sys.argv:
            raise
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
