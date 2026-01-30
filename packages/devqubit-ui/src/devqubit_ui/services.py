# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Service layer for devqubit UI.

This module provides a thin abstraction between route handlers and the
underlying registry/store. It simplifies router code, improves testability,
and makes future migration (e.g., to a remote API) easier.

The services here are intentionally simple - they're not a full domain layer,
just enough to keep routes focused on HTTP concerns.

Examples
--------
Using services in a route:

>>> from devqubit_ui.services import RunService
>>>
>>> @router.get("/runs/{run_id}")
>>> async def run_detail(run_id: str, registry: RegistryDep):
...     service = RunService(registry)
...     run = service.get_run(run_id)
...     return JSONResponse(content={"run": run})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol


logger = logging.getLogger(__name__)

# Maximum artifact size to load into memory for preview (2 MB)
MAX_ARTIFACT_PREVIEW_SIZE = 2 * 1024 * 1024


@dataclass
class ArtifactContent:
    """
    Container for artifact content with metadata.

    Attributes
    ----------
    data : bytes or None
        Raw artifact bytes, or None if too large for preview.
    size : int
        Total size in bytes.
    is_text : bool
        Whether content is text-based.
    is_json : bool
        Whether content is JSON.
    preview_available : bool
        Whether preview data is available (size <= MAX_ARTIFACT_PREVIEW_SIZE).
    error : str or None
        Error message if loading failed.
    """

    data: bytes | None
    size: int
    is_text: bool
    is_json: bool
    preview_available: bool
    error: str | None = None


class RunService:
    """
    Service for run-related operations.

    Encapsulates run listing, filtering, retrieval, and deletion logic.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    store : ObjectStoreProtocol, optional
        The object store instance (needed for artifacts).

    Examples
    --------
    >>> service = RunService(registry, store)
    >>> runs = service.list_runs(project="vqe", limit=50)
    >>> run = service.get_run("abc123")
    >>> service.delete_run("abc123")
    """

    def __init__(
        self,
        registry: "RegistryProtocol",
        store: "ObjectStoreProtocol | None" = None,
    ) -> None:
        self._registry = registry
        self._store = store

    def list_runs(
        self,
        project: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List runs with optional filtering.

        Parameters
        ----------
        project : str, optional
            Filter by project name.
        status : str, optional
            Filter by run status.
        limit : int, default=50
            Maximum number of runs to return.

        Returns
        -------
        list[dict[str, Any]]
            List of run summaries as dictionaries.
        """
        kwargs: dict[str, Any] = {"limit": limit}
        if project:
            kwargs["project"] = project
        if status:
            kwargs["status"] = status

        run_summaries = self._registry.list_runs(**kwargs)
        return [dict(s) for s in run_summaries]

    def search_runs(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Search runs using query syntax.

        Parameters
        ----------
        query : str
            Search query (e.g., "metric.fidelity > 0.9").
        limit : int, default=50
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            List of matching runs as dictionaries.
        """
        logger.debug("Searching runs with query: %s", query)
        records = self._registry.search_runs(query, limit=limit)
        return [self._record_to_dict(r) for r in records]

    def get_run(self, run_id: str) -> Any:
        """
        Get a run by ID.

        Parameters
        ----------
        run_id : str
            The run identifier.

        Returns
        -------
        RunRecord
            The run record.

        Raises
        ------
        KeyError
            If run not found.
        """
        try:
            return self._registry.load(run_id)
        except Exception as e:
            logger.warning("Run not found: %s", run_id)
            raise KeyError(f"Run not found: {run_id}") from e

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run by ID.

        Parameters
        ----------
        run_id : str
            The run identifier.

        Returns
        -------
        bool
            True if run was deleted, False if it didn't exist.
        """
        logger.info("Deleting run: %s", run_id)
        return self._registry.delete(run_id)

    def get_baseline(self, project: str) -> dict[str, Any] | None:
        """
        Get the baseline run for a project.

        Parameters
        ----------
        project : str
            The project name.

        Returns
        -------
        dict or None
            Baseline run info, or None if not set.
        """
        return self._registry.get_baseline(project)

    def set_baseline(self, project: str, run_id: str) -> None:
        """
        Set a run as the baseline for a project.

        Parameters
        ----------
        project : str
            The project name.
        run_id : str
            The run ID to set as baseline.
        """
        logger.info("Setting baseline for project %s: %s", project, run_id)
        self._registry.set_baseline(project, run_id)

    def list_projects(self) -> list[str]:
        """
        List all project names.

        Returns
        -------
        list[str]
            List of project names.
        """
        return self._registry.list_projects()

    @staticmethod
    def _record_to_dict(record: Any) -> dict[str, Any]:
        """Convert RunRecord to dictionary."""
        return {
            "run_id": record.run_id,
            "project": record.project,
            "adapter": record.adapter,
            "status": record.status,
            "created_at": record.created_at,
            "ended_at": record.record.get("info", {}).get("ended_at"),
        }


class ArtifactService:
    """
    Service for artifact operations.

    Handles artifact retrieval with size limits to prevent memory issues.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    store : ObjectStoreProtocol
        The object store instance.
    max_preview_size : int, default=MAX_ARTIFACT_PREVIEW_SIZE
        Maximum size (bytes) for artifact preview. Larger artifacts
        require download.

    Examples
    --------
    >>> service = ArtifactService(registry, store)
    >>> content = service.get_artifact_content("run123", 0)
    >>> if content.preview_available:
    ...     print(content.data.decode())
    """

    def __init__(
        self,
        registry: "RegistryProtocol",
        store: "ObjectStoreProtocol",
        max_preview_size: int = MAX_ARTIFACT_PREVIEW_SIZE,
    ) -> None:
        self._registry = registry
        self._store = store
        self._max_preview_size = max_preview_size

    def get_artifact_metadata(self, run_id: str, idx: int) -> tuple[Any, Any]:
        """
        Get artifact metadata without loading content.

        Parameters
        ----------
        run_id : str
            The run identifier.
        idx : int
            Zero-based artifact index.

        Returns
        -------
        tuple[RunRecord, Artifact]
            The run record and artifact metadata.

        Raises
        ------
        KeyError
            If run not found.
        IndexError
            If artifact index out of range.
        """
        try:
            record = self._registry.load(run_id)
        except Exception as e:
            raise KeyError(f"Run not found: {run_id}") from e

        if idx < 0 or idx >= len(record.artifacts):
            raise IndexError(f"Artifact index {idx} out of range")

        return record, record.artifacts[idx]

    def get_artifact_content(self, run_id: str, idx: int) -> ArtifactContent:
        """
        Get artifact content with size safety.

        If the artifact is larger than max_preview_size, returns metadata
        only without loading content into memory.

        Parameters
        ----------
        run_id : str
            The run identifier.
        idx : int
            Zero-based artifact index.

        Returns
        -------
        ArtifactContent
            Container with content (if small enough) and metadata.

        Notes
        -----
        For large artifacts (> 5MB by default), use `get_artifact_raw()`
        with streaming response instead.
        """
        _, artifact = self.get_artifact_metadata(run_id, idx)

        # Check size first
        try:
            size = self._store.get_size(artifact.digest)
        except AttributeError:
            # Fallback if get_size not available
            size = None
        except Exception as e:
            logger.warning("Failed to get artifact size: %s", e)
            size = None

        # Determine content type
        is_text = (
            artifact.media_type.startswith("text/")
            or artifact.media_type == "application/json"
        )
        is_json = artifact.media_type == "application/json" or artifact.kind.endswith(
            ".json"
        )

        # If size known and too large, don't load
        if size is not None and size > self._max_preview_size:
            return ArtifactContent(
                data=None,
                size=size,
                is_text=is_text,
                is_json=is_json,
                preview_available=False,
            )

        # Load content
        try:
            data = self._store.get_bytes(artifact.digest)
            actual_size = len(data)

            # Double-check size after loading
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

        except Exception as e:
            logger.warning("Failed to load artifact %s: %s", artifact.digest[:16], e)
            return ArtifactContent(
                data=None,
                size=size or 0,
                is_text=is_text,
                is_json=is_json,
                preview_available=False,
                error=str(e),
            )

    def get_artifact_raw(self, run_id: str, idx: int) -> tuple[bytes, str, str]:
        """
        Get raw artifact bytes for download.

        Parameters
        ----------
        run_id : str
            The run identifier.
        idx : int
            Zero-based artifact index.

        Returns
        -------
        tuple[bytes, str, str]
            Tuple of (data, media_type, filename).

        Raises
        ------
        KeyError
            If run not found.
        IndexError
            If artifact index out of range.
        RuntimeError
            If artifact data cannot be loaded.
        """
        _, artifact = self.get_artifact_metadata(run_id, idx)

        try:
            data = self._store.get_bytes(artifact.digest)
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact: {e}") from e

        # Sanitize filename
        filename = artifact.kind.replace("/", "_").replace("\\", "_")

        return data, artifact.media_type, filename


class ProjectService:
    """
    Service for project operations.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    """

    def __init__(self, registry: "RegistryProtocol") -> None:
        self._registry = registry

    def list_projects_with_stats(self) -> list[dict[str, Any]]:
        """
        List all projects with run counts and baseline info.

        Returns
        -------
        list[dict[str, Any]]
            List of project info dictionaries with keys:
            name, run_count, baseline.
        """
        projects = self._registry.list_projects()
        result = []

        for proj in projects:
            run_count = self._registry.count_runs(project=proj)
            baseline = self._registry.get_baseline(proj)
            result.append(
                {
                    "name": proj,
                    "run_count": run_count,
                    "baseline": baseline,
                }
            )

        return result


class GroupService:
    """
    Service for run group operations.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    """

    def __init__(self, registry: "RegistryProtocol") -> None:
        self._registry = registry

    def list_groups(self, project: str | None = None) -> list[Any]:
        """
        List run groups with optional project filter.

        Parameters
        ----------
        project : str, optional
            Filter groups by project.

        Returns
        -------
        list
            List of group info objects.
        """
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        return self._registry.list_groups(**kwargs)

    def get_group_runs(self, group_id: str) -> list[Any]:
        """
        Get all runs in a group.

        Parameters
        ----------
        group_id : str
            The group identifier.

        Returns
        -------
        list
            List of runs in the group.
        """
        return self._registry.list_runs_in_group(group_id)


class DiffService:
    """
    Service for run comparison operations.

    Parameters
    ----------
    registry : RegistryProtocol
        The run registry instance.
    store : ObjectStoreProtocol
        The object store instance.
    """

    def __init__(
        self,
        registry: "RegistryProtocol",
        store: "ObjectStoreProtocol",
    ) -> None:
        self._registry = registry
        self._store = store

    def compare_runs(self, run_id_a: str, run_id_b: str) -> tuple[Any, Any, dict]:
        """
        Compare two runs and generate a diff report.

        Parameters
        ----------
        run_id_a : str
            ID of the baseline run (Run A).
        run_id_b : str
            ID of the candidate run (Run B).

        Returns
        -------
        tuple[RunRecord, RunRecord, dict]
            Tuple of (record_a, record_b, report_dict).

        Raises
        ------
        KeyError
            If either run is not found.
        """
        try:
            record_a = self._registry.load(run_id_a)
        except Exception as e:
            raise KeyError(f"Run A not found: {run_id_a}") from e

        try:
            record_b = self._registry.load(run_id_b)
        except Exception as e:
            raise KeyError(f"Run B not found: {run_id_b}") from e

        # Use devqubit's diff_runs function
        from devqubit_engine.compare.diff import diff_runs

        logger.debug("Comparing runs: %s vs %s", run_id_a, run_id_b)
        result = diff_runs(
            record_a,
            record_b,
            store_a=self._store,
            store_b=self._store,
        )

        return record_a, record_b, result.to_dict()
