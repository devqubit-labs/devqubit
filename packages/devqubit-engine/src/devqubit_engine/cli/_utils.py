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
from devqubit_engine.storage.artifacts.counts import CountsInfo
from devqubit_engine.storage.artifacts.lookup import ArtifactInfo


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


def format_counts_table(counts: CountsInfo, top_k: int = 10) -> str:
    """Format measurement counts as ASCII table."""
    lines = [
        f"Total shots: {counts.total_shots:,}",
        f"Unique outcomes: {counts.num_outcomes}",
        "",
        f"{'Outcome':<20} {'Count':>10} {'Prob':>10}",
        "-" * 42,
    ]

    for bitstring, count, prob in counts.top_k(top_k):
        lines.append(f"{bitstring:<20} {count:>10,} {prob:>10.4f}")

    if counts.num_outcomes > top_k:
        lines.append(f"... and {counts.num_outcomes - top_k} more outcomes")

    return "\n".join(lines)


def format_artifacts_table(artifacts: list[ArtifactInfo]) -> str:
    """Format artifact list as ASCII table."""
    if not artifacts:
        return "No artifacts found."

    lines = [
        f"{'#':<4} {'Role':<16} {'Kind':<30} {'Size':>10}",
        "-" * 62,
    ]

    for a in artifacts:
        size_str = f"{a.size:,}" if a.size else "-"
        kind_display = a.kind[:30] if len(a.kind) <= 30 else a.kind[:27] + "..."
        lines.append(f"{a.index:<4} {a.role:<16} {kind_display:<30} {size_str:>10}")

    lines.append("")
    lines.append(f"Total: {len(artifacts)} artifact(s)")

    return "\n".join(lines)
