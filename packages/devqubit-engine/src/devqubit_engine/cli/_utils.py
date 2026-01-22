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
from devqubit_engine.storage.types import RegistryProtocol
from devqubit_engine.tracking.record import RunRecord


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


def echo_err(msg: str) -> None:
    """
    Print error message to stderr.

    Parameters
    ----------
    msg : str
        Error message to print.
    """
    click.echo(msg, err=True)


def print_json(obj: Any, indent: int = 2) -> None:
    """
    Print object as formatted JSON.

    Parameters
    ----------
    obj : Any
        Object to serialize and print. Non-serializable objects
        are converted to strings.
    indent : int, default=2
        Indentation level for JSON formatting.
    """
    click.echo(json.dumps(obj, indent=indent, default=str))


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
        # Pad row if shorter than headers
        padded_row = list(row) + [""] * (len(headers) - len(row))
        echo(fmt.format(*[str(c) for c in padded_row[: len(headers)]]))


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

    Raises
    ------
    click.ClickException
        If context is not properly initialized.
    """
    if ctx.obj is None:
        raise click.ClickException(
            "CLI context not initialized. This is a bug - please report it."
        )

    root = ctx.obj.get("root")
    if root is None:
        # Fallback to default
        root = Path.home() / ".devqubit"

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_quiet(ctx: click.Context) -> bool:
    """
    Check if quiet mode is enabled in context.

    Parameters
    ----------
    ctx : click.Context
        Click context.

    Returns
    -------
    bool
        True if quiet mode is enabled.
    """
    if ctx.obj is None:
        return False
    return bool(ctx.obj.get("quiet", False))


def resolve_run(
    run_id_or_name: str,
    registry: RegistryProtocol,
    project: str | None = None,
) -> RunRecord:
    """
    Resolve run by ID or name and load it.

    Uses resolve_run_id from engine for ID-first resolution:
    1. Try as run ID (fast path)
    2. If project provided and ID not found, try as name within project

    Parameters
    ----------
    run_id_or_name : str
        Run ID or run name.
    registry : RegistryProtocol
        Registry instance.
    project : str, optional
        Project name. Required when using run name.

    Returns
    -------
    RunRecord
        Loaded run record.

    Raises
    ------
    click.ClickException
        If run is not found.
    """
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.tracking.record import resolve_run_id

    resolved_id = resolve_run_id(run_id_or_name, project, registry)

    try:
        return registry.load(resolved_id)
    except RunNotFoundError:
        if project:
            raise click.ClickException(
                f"Run not found: '{run_id_or_name}' "
                f"(looked up as ID and as name in project '{project}')"
            )
        raise click.ClickException(
            f"Run not found: '{run_id_or_name}'. " f"Use --project to look up by name."
        )


def format_counts_table(counts: CountsInfo, top_k: int = 10) -> str:
    """
    Format measurement counts as ASCII table.

    Parameters
    ----------
    counts : CountsInfo
        Counts information object.
    top_k : int, default=10
        Number of top outcomes to display.

    Returns
    -------
    str
        Formatted table string.
    """
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
    """
    Format artifact list as ASCII table.

    Parameters
    ----------
    artifacts : list of ArtifactInfo
        List of artifact info objects.

    Returns
    -------
    str
        Formatted table string.
    """
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


def truncate_id(run_id: str, length: int = 12) -> str:
    """
    Truncate a run ID for display.

    Parameters
    ----------
    run_id : str
        Full run ID.
    length : int, default=12
        Maximum length before truncation.

    Returns
    -------
    str
        Truncated ID with "..." suffix if needed.
    """
    if not run_id:
        return ""
    if len(run_id) <= length:
        return run_id
    return run_id[:length] + "..."


def safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary.

    Parameters
    ----------
    d : dict or None
        Dictionary to traverse.
    *keys : str
        Keys to follow in sequence.
    default : Any, default=None
        Value to return if any key is missing.

    Returns
    -------
    Any
        The value at the nested key path, or default.
    """
    if d is None:
        return default

    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default

    return current
