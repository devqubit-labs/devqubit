# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Service layer for devqubit UI.

Provides a thin abstraction between route handlers and the underlying
registry / store.  Keeps routes focused on HTTP concerns, improves
testability, and prepares a clean seam for hub-backed remote APIs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from devqubit_engine.storage.types import (
    ArtifactRef,
    BaselineInfo,
    ObjectStoreProtocol,
    RegistryProtocol,
    RunSummary,
)


if TYPE_CHECKING:
    from devqubit_engine.tracking.record import RunRecord


logger = logging.getLogger(__name__)

MAX_ARTIFACT_PREVIEW_SIZE = 2 * 1024 * 1024
"""Maximum artifact size (bytes) to load into memory for preview."""

_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]")
"""Replace any character outside the allowlist to prevent header injection."""


# ---------------------------------------------------------------------------
# Record serialization
# ---------------------------------------------------------------------------


def serialize_record(record: RunRecord) -> dict[str, Any]:
    """
    Full run record for detail views.

    Includes artifacts, fingerprints, backend info, and errors.
    """
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "ended_at": record.ended_at,
        "fingerprints": record.fingerprints,
        "group_id": record.group_id,
        "group_name": record.group_name,
        "backend": record.record.get("backend", {}),
        "data": record.record.get("data", {}),
        "artifacts": [
            {
                "kind": a.kind,
                "role": a.role,
                "media_type": a.media_type,
                "digest": a.digest,
            }
            for a in (record.artifacts or [])
        ],
        "errors": record.record.get("info", {}).get("errors", []),
    }


def serialize_record_summary(record: RunRecord) -> dict[str, Any]:
    """Compact run summary for lists, diffs, and search results."""
    ended_at = getattr(record, "ended_at", None)
    if ended_at is None and hasattr(record, "record"):
        ended_at = record.record.get("info", {}).get("ended_at")

    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": getattr(record, "adapter", None),
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "ended_at": ended_at,
    }


def extract_record_data(record: RunRecord | dict[str, Any]) -> dict[str, Any]:
    """
    Extract the raw data dictionary from a run record.

    Handles ``to_dict()``, ``.record`` attribute, and plain dicts.
    Validates the result at each step so that duck-typing mismatches
    (e.g. a non-callable ``to_dict``) fall through gracefully.
    """
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    if hasattr(record, "record") and isinstance(record.record, dict):
        return record.record
    if isinstance(record, dict):
        return record
    raise ValueError(f"Cannot convert record to dict: {type(record)}")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ArtifactContent:
    """Container for artifact content with metadata."""

    data: bytes | None
    size: int
    is_text: bool
    is_json: bool
    preview_available: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


class RunService:
    """
    Run-related operations.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    store : ObjectStoreProtocol, optional
        The object store (needed for artifacts).
    """

    def __init__(
        self,
        registry: RegistryProtocol,
        store: ObjectStoreProtocol | None = None,
    ) -> None:
        self._registry = registry
        self._store = store

    # -- queries -----------------------------------------------------------

    def list_runs(
        self,
        project: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List runs with optional filtering and offset-based pagination."""
        kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if project:
            kwargs["project"] = project
        if status:
            kwargs["status"] = status
        return [dict(s) for s in self._registry.list_runs(**kwargs)]

    def count_runs(
        self,
        project: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count runs matching the given filters."""
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        if status:
            kwargs["status"] = status
        return self._registry.count_runs(**kwargs)

    def search_runs(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search runs using query syntax."""
        logger.debug("Searching runs: %s", query)
        records = self._registry.search_runs(query, limit=limit)
        return [serialize_record_summary(r) for r in records]

    def get_run(self, run_id: str) -> RunRecord:
        """
        Load a run by ID.

        Exceptions from the registry propagate to the caller so that
        routers can map them to the appropriate HTTP status.
        """
        return self._registry.load(run_id)

    # -- mutations ---------------------------------------------------------

    def delete_run(self, run_id: str) -> bool:
        """Delete a run.  Returns ``True`` if it existed."""
        logger.info("Deleting run: %s", run_id)
        return self._registry.delete(run_id)

    def set_baseline(self, project: str, run_id: str) -> None:
        """Set a run as the baseline for *project*."""
        logger.info("Setting baseline for project %s: %s", project, run_id)
        self._registry.set_baseline(project, run_id)

    def get_baseline(self, project: str) -> BaselineInfo | None:
        """Get the baseline run for *project*."""
        return self._registry.get_baseline(project)

    def list_projects(self) -> list[str]:
        """List all project names."""
        return self._registry.list_projects()


class ArtifactService:
    """
    Artifact retrieval with size-limited previews.

    Parameters
    ----------
    registry : RegistryProtocol
    store : ObjectStoreProtocol
    max_preview_size : int
        Artifacts larger than this are not loaded for preview.
    """

    def __init__(
        self,
        registry: RegistryProtocol,
        store: ObjectStoreProtocol,
        max_preview_size: int = MAX_ARTIFACT_PREVIEW_SIZE,
    ) -> None:
        self._registry = registry
        self._store = store
        self._max_preview_size = max_preview_size

    def get_artifact_metadata(
        self,
        run_id: str,
        idx: int,
    ) -> tuple[RunRecord, ArtifactRef]:
        """Return ``(record, artifact)`` or raise ``KeyError``/``IndexError``."""
        try:
            record = self._registry.load(run_id)
        except Exception as exc:
            raise KeyError(f"Run not found: {run_id}") from exc
        if idx < 0 or idx >= len(record.artifacts):
            raise IndexError(f"Artifact index {idx} out of range")
        return record, record.artifacts[idx]

    def get_artifact_content(self, run_id: str, idx: int) -> ArtifactContent:
        """Load artifact content with size guard."""
        _, artifact = self.get_artifact_metadata(run_id, idx)

        try:
            size = self._store.get_size(artifact.digest)
        except AttributeError:
            size = None
        except Exception as exc:
            logger.warning("Failed to get artifact size: %s", exc)
            size = None

        is_text = (
            artifact.media_type.startswith("text/")
            or artifact.media_type == "application/json"
        )
        is_json = artifact.media_type == "application/json" or artifact.kind.endswith(
            ".json"
        )

        if size is not None and size > self._max_preview_size:
            return ArtifactContent(
                data=None,
                size=size,
                is_text=is_text,
                is_json=is_json,
                preview_available=False,
            )

        try:
            data = self._store.get_bytes(artifact.digest)
            actual_size = len(data)
            if actual_size > self._max_preview_size:
                return ArtifactContent(
                    data=None,
                    size=actual_size,
                    is_text=is_text,
                    is_json=is_json,
                    preview_available=False,
                )
            return ArtifactContent(
                data=data,
                size=actual_size,
                is_text=is_text,
                is_json=is_json,
                preview_available=True,
            )
        except Exception as exc:
            logger.warning("Failed to load artifact %s: %s", artifact.digest[:16], exc)
            return ArtifactContent(
                data=None,
                size=size or 0,
                is_text=is_text,
                is_json=is_json,
                preview_available=False,
                error=str(exc),
            )

    def get_artifact_raw(self, run_id: str, idx: int) -> tuple[bytes, str, str]:
        """Return ``(data, media_type, filename)`` for download."""
        _, artifact = self.get_artifact_metadata(run_id, idx)
        try:
            data = self._store.get_bytes(artifact.digest)
        except Exception as exc:
            raise RuntimeError(f"Failed to load artifact: {exc}") from exc
        filename = _UNSAFE_FILENAME_CHARS.sub("_", artifact.kind) or "artifact"
        return data, artifact.media_type, filename


class ProjectService:
    """Project listing with stats."""

    def __init__(self, registry: RegistryProtocol) -> None:
        self._registry = registry

    def list_projects_with_stats(self) -> list[dict[str, Any]]:
        """List all projects with run counts and baseline info."""
        projects = self._registry.list_projects()
        result = []
        for proj in projects:
            run_count = self._registry.count_runs(project=proj)
            baseline = self._registry.get_baseline(proj)
            result.append({"name": proj, "run_count": run_count, "baseline": baseline})
        return result


class GroupService:
    """Run group operations."""

    def __init__(self, registry: RegistryProtocol) -> None:
        self._registry = registry

    def list_groups(self, project: str | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        return self._registry.list_groups(**kwargs)

    def get_group_runs(self, group_id: str) -> list[RunSummary]:
        return self._registry.list_runs_in_group(group_id)


class DiffService:
    """Run comparison."""

    def __init__(
        self,
        registry: RegistryProtocol,
        store: ObjectStoreProtocol,
    ) -> None:
        self._registry = registry
        self._store = store

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> tuple[RunRecord, RunRecord, dict[str, Any]]:
        """
        Compare two runs.

        Returns ``(record_a, record_b, report_dict)``.
        Raises ``KeyError`` if either run is missing.
        """
        try:
            record_a = self._registry.load(run_id_a)
        except Exception as exc:
            raise KeyError(f"Run A not found: {run_id_a}") from exc
        try:
            record_b = self._registry.load(run_id_b)
        except Exception as exc:
            raise KeyError(f"Run B not found: {run_id_b}") from exc

        from devqubit_engine.compare.diff import diff_runs

        logger.debug("Comparing runs: %s vs %s", run_id_a, run_id_b)
        result = diff_runs(
            record_a,
            record_b,
            store_a=self._store,
            store_b=self._store,
        )
        return record_a, record_b, result.to_dict()
