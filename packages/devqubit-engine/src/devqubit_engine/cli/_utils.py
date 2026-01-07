# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Shared CLI utilities.

This module provides common helper functions used across CLI commands
for consistent output formatting and context management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import click


def echo(msg: str, *, err: bool = False) -> None:
    """
    Print message to stdout or stderr.

    Parameters
    ----------
    msg : str
        Message to print.
    err : bool, default=False
        If True, print to stderr instead of stdout.
    """
    click.echo(msg, err=err)


def print_json(obj: Any) -> None:
    """
    Print object as formatted JSON.

    Parameters
    ----------
    obj : Any
        Object to serialize and print. Non-serializable objects
        are converted to strings.
    """
    click.echo(json.dumps(obj, indent=2, default=str))


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    title: str = "",
) -> None:
    """
    Print formatted ASCII table.

    Parameters
    ----------
    headers : sequence of str
        Column headers.
    rows : sequence of sequence
        Table rows. Each row should have same length as headers.
    title : str, optional
        Table title to display above the table.
    """
    if title:
        echo(f"\n{title}\n{'=' * len(title)}")

    if not rows:
        echo("(empty)")
        return

    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    # Print header
    echo(fmt.format(*headers))
    echo(fmt.format(*["-" * w for w in widths]))

    # Print rows
    for row in rows:
        echo(fmt.format(*[str(c) for c in row]))


def root_from_ctx(ctx: click.Context) -> Path:
    """
    Get workspace root from click context.

    Creates the directory if it doesn't exist.

    Parameters
    ----------
    ctx : click.Context
        Click context containing obj["root"].

    Returns
    -------
    Path
        Workspace root directory path.
    """
    root: Path = ctx.obj["root"]
    root.mkdir(parents=True, exist_ok=True)
    return root
