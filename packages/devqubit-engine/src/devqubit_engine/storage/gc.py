# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Storage garbage collection and workspace hygiene.

This module provides tools for:

- Cleaning up unreferenced objects (garbage collection)
- Pruning old or failed runs
- Checking workspace health
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from devqubit_engine.storage.types import (
    ObjectStoreProtocol,
    RegistryProtocol,
    RunSummary,
)


logger = logging.getLogger(__name__)


@dataclass
class GCStats:
    """
    Garbage collection statistics.

    Attributes
    ----------
    referenced_objects : int
        Number of objects referenced by at least one run.
    unreferenced_objects : int
        Number of orphaned objects (not referenced by any run).
    bytes_reclaimable : int
        Total bytes that can be reclaimed from orphaned objects.
    runs_scanned : int
        Number of runs scanned for artifact references.
    objects_deleted : int
        Number of objects actually deleted (after gc_run with dry_run=False).
    bytes_reclaimed : int
        Total bytes actually reclaimed.
    errors : list of str
        Error messages encountered during collection.
    """

    referenced_objects: int = 0
    unreferenced_objects: int = 0
    bytes_reclaimable: int = 0
    runs_scanned: int = 0
    objects_deleted: int = 0
    bytes_reclaimed: int = 0
    errors: list[str] = field(default_factory=list)


def _collect_referenced_digests(
    registry: RegistryProtocol,
    *,
    project: str | None = None,
    errors: list[str] | None = None,
) -> set[str]:
    """
    Collect all artifact digests referenced by runs.

    Parameters
    ----------
    registry : RegistryProtocol
        Run registry.
    project : str, optional
        Limit to specific project.
    errors : list of str, optional
        List to append error messages to.

    Returns
    -------
    set of str
        All referenced digests.
    """
    referenced: set[str] = set()

    # Paginate through all runs
    offset = 0
    batch_size = 100

    while True:
        summaries = registry.list_runs(
            limit=batch_size,
            offset=offset,
            project=project,
        )

        if not summaries:
            break

        for summary in summaries:
            run_id = summary.get("run_id", "")
            try:
                record = registry.load(run_id)
                if record.artifacts is None:
                    if errors is not None:
                        errors.append(f"Run {run_id}: artifacts is None")
                    continue

                for art in record.artifacts:
                    if art.digest.startswith("sha256:"):
                        referenced.add(art.digest)

            except Exception as e:
                if errors is not None:
                    errors.append(f"Run {run_id}: failed to load: {e}")
                logger.warning("Failed to load run %s: %s", run_id, e)
                continue

        offset += len(summaries)

    logger.debug("Collected %d referenced digests from runs", len(referenced))
    return referenced


def gc_plan(
    store: ObjectStoreProtocol,
    registry: RegistryProtocol,
    *,
    project: str | None = None,
) -> GCStats:
    """
    Plan garbage collection without deleting anything.

    Scans all runs to find referenced objects, then identifies
    orphaned objects that could be safely deleted.

    Parameters
    ----------
    store : ObjectStoreProtocol
        Object store to scan.
    registry : RegistryProtocol
        Run registry.
    project : str, optional
        Limit scan to a specific project.

    Returns
    -------
    GCStats
        Statistics about what would be collected.

    Examples
    --------
    >>> stats = gc_plan(store, registry)
    >>> print(f"Referenced: {stats.referenced_objects}")
    >>> print(f"Orphaned: {stats.unreferenced_objects}")
    >>> print(f"Reclaimable: {stats.bytes_reclaimable:,} bytes")
    """
    logger.info("Planning garbage collection")
    stats = GCStats()

    # Collect referenced digests
    referenced = _collect_referenced_digests(
        registry, project=project, errors=stats.errors
    )
    stats.referenced_objects = len(referenced)
    stats.runs_scanned = registry.count_runs(project=project)

    # Find unreferenced objects
    for digest in store.list_digests():
        if digest not in referenced:
            stats.unreferenced_objects += 1
            try:
                stats.bytes_reclaimable += store.get_size(digest)
            except Exception:
                pass

    logger.info(
        "GC plan: %d referenced, %d orphaned, %d bytes reclaimable",
        stats.referenced_objects,
        stats.unreferenced_objects,
        stats.bytes_reclaimable,
    )

    return stats


def gc_run(
    store: ObjectStoreProtocol,
    registry: RegistryProtocol,
    *,
    project: str | None = None,
    dry_run: bool = True,
) -> GCStats:
    """
    Run garbage collection to remove orphaned objects.

    Parameters
    ----------
    store : ObjectStoreProtocol
        Object store to clean.
    registry : RegistryProtocol
        Run registry.
    project : str, optional
        Limit to a specific project.
    dry_run : bool, optional
        If True (default), only report what would be deleted without
        actually deleting anything.

    Returns
    -------
    GCStats
        Statistics about the collection.

    Examples
    --------
    >>> # Dry run first
    >>> stats = gc_run(store, registry, dry_run=True)
    >>> print(f"Would delete {stats.unreferenced_objects} objects")

    >>> # Actually delete
    >>> stats = gc_run(store, registry, dry_run=False)
    >>> print(f"Deleted {stats.objects_deleted} objects")
    """
    logger.info("Running garbage collection (dry_run=%s)", dry_run)
    stats = GCStats()

    # Collect referenced digests
    referenced = _collect_referenced_digests(
        registry, project=project, errors=stats.errors
    )
    stats.referenced_objects = len(referenced)
    stats.runs_scanned = registry.count_runs(project=project)

    # Find and optionally delete unreferenced objects
    to_delete: list[tuple[str, int]] = []  # (digest, size)

    for digest in store.list_digests():
        if digest not in referenced:
            stats.unreferenced_objects += 1
            try:
                size = store.get_size(digest)
                stats.bytes_reclaimable += size
                to_delete.append((digest, size))
            except Exception as e:
                stats.errors.append(f"Error sizing {digest}: {e}")

    if not dry_run:
        for digest, size in to_delete:
            try:
                if store.delete(digest):
                    stats.objects_deleted += 1
                    stats.bytes_reclaimed += size
            except Exception as e:
                stats.errors.append(f"Error deleting {digest}: {e}")

        logger.info(
            "GC complete: deleted %d objects, reclaimed %d bytes",
            stats.objects_deleted,
            stats.bytes_reclaimed,
        )
    else:
        logger.info(
            "GC dry run: would delete %d objects, reclaim %d bytes",
            stats.unreferenced_objects,
            stats.bytes_reclaimable,
        )

    return stats


@dataclass
class PruneStats:
    """
    Run pruning statistics.

    Attributes
    ----------
    runs_scanned : int
        Number of runs scanned.
    runs_pruned : int
        Number of runs deleted (or would be deleted in dry run).
    artifacts_orphaned : int
        Number of artifact objects that became orphaned (not tracked).
    errors : list of str
        Error messages encountered during pruning.
    """

    runs_scanned: int = 0
    runs_pruned: int = 0
    artifacts_orphaned: int = 0
    errors: list[str] = field(default_factory=list)


def prune_runs(
    registry: RegistryProtocol,
    *,
    project: str | None = None,
    status: str | None = None,
    older_than_days: int | None = None,
    keep_latest: int = 10,
    dry_run: bool = True,
) -> PruneStats:
    """
    Prune old or failed runs from the registry.

    Parameters
    ----------
    registry : RegistryProtocol
        Run registry to prune.
    project : str, optional
        Limit to a specific project.
    status : str, optional
        Only prune runs with this status (e.g., "FAILED", "KILLED").
    older_than_days : int, optional
        Only prune runs older than this many days.
    keep_latest : int, optional
        Always keep at least this many runs per project, regardless
        of age or status. Default is 10.
    dry_run : bool, optional
        If True (default), only report what would be pruned without
        actually deleting anything.

    Returns
    -------
    PruneStats
        Statistics about the pruning operation.

    Raises
    ------
    ValueError
        If keep_latest is negative.

    Examples
    --------
    >>> # Prune failed runs older than 30 days, keep 5 per project
    >>> stats = prune_runs(
    ...     registry,
    ...     status="FAILED",
    ...     older_than_days=30,
    ...     keep_latest=5,
    ...     dry_run=False,
    ... )
    """
    logger.info(
        "Pruning runs (project=%s, status=%s, older_than=%s days, keep=%d, dry_run=%s)",
        project,
        status,
        older_than_days,
        keep_latest,
        dry_run,
    )

    stats = PruneStats()

    if keep_latest < 0:
        raise ValueError(f"keep_latest must be >= 0, got {keep_latest}")

    # Get runs to consider
    summaries = registry.list_runs(
        limit=10000,  # Reasonable max for MVP
        project=project,
        status=status,
    )
    stats.runs_scanned = len(summaries)

    # Calculate age cutoff
    cutoff: datetime | None = None
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    # Group by project for keep_latest logic
    by_project: dict[str, list[RunSummary]] = {}
    for s in summaries:
        proj = s.get("project", "")
        by_project.setdefault(proj, []).append(s)

    to_prune: list[str] = []

    for proj, runs in by_project.items():
        # Sort by created_at descending (newest first)
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        # Skip the first keep_latest runs
        for run in runs[keep_latest:]:
            run_id = run.get("run_id", "")
            created_at = run.get("created_at", "")

            # Check age cutoff
            if cutoff is not None and created_at:
                try:
                    run_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if run_time >= cutoff:
                        continue  # Run is not old enough
                except (ValueError, TypeError):
                    pass

            if run_id:
                to_prune.append(run_id)

    if not dry_run:
        for run_id in to_prune:
            try:
                if registry.delete(run_id):
                    stats.runs_pruned += 1
            except Exception as e:
                stats.errors.append(f"Error deleting run {run_id}: {e}")

        logger.info("Pruned %d runs", stats.runs_pruned)
    else:
        stats.runs_pruned = len(to_prune)
        logger.info("Would prune %d runs (dry run)", stats.runs_pruned)

    return stats


def check_workspace_health(
    store: ObjectStoreProtocol,
    registry: RegistryProtocol,
) -> dict[str, Any]:
    """
    Check workspace health and return diagnostics.

    Performs a comprehensive health check including:

    - Counting runs and objects
    - Finding orphaned objects
    - Finding missing objects (referenced but not in store)
    - Checking for corrupt or unloadable runs

    Parameters
    ----------
    store : ObjectStoreProtocol
        Object store to check.
    registry : RegistryProtocol
        Run registry to check.

    Returns
    -------
    dict
        Health report containing:

        - ``total_runs``: Number of runs in registry
        - ``total_objects``: Number of objects in store
        - ``referenced_objects``: Number of objects referenced by runs
        - ``orphaned_objects``: Number of unreferenced objects
        - ``missing_objects``: Number of referenced but missing objects
        - ``projects``: List of project names
        - ``status_counts``: Dict of status -> count
        - ``errors``: List of error messages
        - ``healthy``: True if no missing objects and no errors

    Examples
    --------
    >>> health = check_workspace_health(store, registry)
    >>> if not health["healthy"]:
    ...     print("Issues found:")
    ...     for error in health["errors"]:
    ...         print(f"  - {error}")
    ...     if health["missing_objects"] > 0:
    ...         print(f"  - {health['missing_objects']} missing objects")
    """
    logger.info("Checking workspace health")

    # Collect all objects in store
    all_objects: set[str] = set()
    for digest in store.list_digests():
        all_objects.add(digest)

    # Scan runs for references
    referenced: set[str] = set()
    missing: set[str] = set()
    status_counts: dict[str, int] = {}
    errors: list[str] = []

    offset = 0
    batch_size = 100
    total_runs = 0

    while True:
        summaries = registry.list_runs(limit=batch_size, offset=offset)
        if not summaries:
            break

        for summary in summaries:
            total_runs += 1
            run_id = summary.get("run_id", "")
            status = summary.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

            try:
                record = registry.load(run_id)
                if record.artifacts is None:
                    errors.append(f"Run {run_id}: artifacts is None")
                    continue

                for art in record.artifacts:
                    digest = art.digest
                    referenced.add(digest)
                    if digest not in all_objects:
                        missing.add(digest)

            except Exception as e:
                errors.append(f"Run {run_id}: failed to load: {e}")
                continue

        offset += len(summaries)

    orphaned = all_objects - referenced

    health = {
        "total_runs": total_runs,
        "total_objects": len(all_objects),
        "referenced_objects": len(referenced),
        "orphaned_objects": len(orphaned),
        "missing_objects": len(missing),
        "projects": registry.list_projects(),
        "status_counts": status_counts,
        "errors": errors,
        "healthy": len(missing) == 0 and len(errors) == 0,
    }

    if health["healthy"]:
        logger.info("Workspace health check passed")
    else:
        logger.warning(
            "Workspace health check failed: %d missing objects, %d errors",
            len(missing),
            len(errors),
        )

    return health
