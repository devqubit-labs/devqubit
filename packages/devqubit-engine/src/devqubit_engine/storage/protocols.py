# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Storage protocol definitions.

This module defines the abstract interfaces (protocols) for storage backends
in devqubit. All storage implementations must conform to these protocols.
"""

from __future__ import annotations

from typing import (
    Any,
    Iterator,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from devqubit_engine.core.record import RunRecord


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class ObjectNotFoundError(StorageError):
    """Raised when a requested object doesn't exist in the store."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__(f"Object not found: {digest}")


class RunNotFoundError(StorageError):
    """Raised when a requested run doesn't exist in the registry."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run not found: {run_id}")


class RunSummary(TypedDict, total=False):
    """
    Summary of a run for listing operations.

    This is a lightweight representation used in list operations,
    containing only the most commonly needed fields.

    Attributes
    ----------
    run_id : str
        Unique run identifier.
    project : str
        Project name.
    adapter : str
        SDK adapter name.
    status : str
        Run status (RUNNING, FINISHED, FAILED, KILLED).
    created_at : str
        ISO 8601 creation timestamp.
    ended_at : str or None
        ISO 8601 end timestamp, if finished.
    group_id : str or None
        Group identifier for related runs.
    group_name : str or None
        Human-readable group name.
    parent_run_id : str or None
        Parent run ID for lineage tracking.
    """

    run_id: str
    project: str
    adapter: str
    status: str
    created_at: str
    ended_at: str | None
    group_id: str | None
    group_name: str | None
    parent_run_id: str | None


class BaselineInfo(TypedDict, total=False):
    """
    Baseline run information for a project.

    A baseline is a reference run used for comparison in drift
    detection and regression testing.

    Attributes
    ----------
    project : str
        Project name.
    run_id : str
        Run ID designated as baseline.
    set_at : str
        ISO 8601 timestamp when baseline was set.
    """

    project: str
    run_id: str
    set_at: str


@runtime_checkable
class ObjectStoreProtocol(Protocol):
    """
    Protocol for content-addressed blob storage.

    Object stores provide content-addressed storage where objects are
    identified by the SHA-256 hash of their contents. This enables
    deduplication and integrity verification.
    """

    def put_bytes(self, data: bytes) -> str:
        """
        Store bytes and return content digest.

        Parameters
        ----------
        data : bytes
            Data to store.

        Returns
        -------
        str
            Content digest in format ``sha256:<64-hex-chars>``.
        """
        ...

    def get_bytes(self, digest: str) -> bytes:
        """
        Retrieve bytes by digest.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bytes
            Stored data.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        """
        ...

    def get_bytes_or_none(self, digest: str) -> bytes | None:
        """
        Retrieve bytes by digest, returning None if not found.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bytes or None
            Stored data, or None if object doesn't exist.
        """
        ...

    def exists(self, digest: str) -> bool:
        """
        Check if object exists.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bool
            True if object exists.
        """
        ...

    def delete(self, digest: str) -> bool:
        """
        Delete object by digest.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bool
            True if object was deleted, False if it didn't exist.
        """
        ...

    def list_digests(self, prefix: str | None = None) -> Iterator[str]:
        """
        List stored digests.

        Parameters
        ----------
        prefix : str, optional
            Filter by digest prefix (e.g., "sha256:ab").

        Yields
        ------
        str
            Content digests.
        """
        ...

    def get_size(self, digest: str) -> int:
        """
        Get size of a stored object in bytes.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        int
            Size in bytes.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        """
        ...


@runtime_checkable
class RegistryProtocol(Protocol):
    """
    Protocol for run metadata registry.

    The registry stores run metadata (parameters, metrics, status, etc.)
    and provides querying capabilities. Artifact blobs are stored
    separately in an ObjectStore.
    """

    def save(self, record: dict[str, Any]) -> None:
        """
        Save or update a run record.

        Parameters
        ----------
        record : dict
            Run record with required 'run_id' field.
        """
        ...

    def load(self, run_id: str) -> "RunRecord":
        """
        Load a run record by ID.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord
            Run record wrapper.

        Raises
        ------
        RunNotFoundError
            If run does not exist.
        """
        ...

    def load_or_none(self, run_id: str) -> "RunRecord | None":
        """
        Load a run record or return None if not found.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord or None
            Run record or None if not found.
        """
        ...

    def exists(self, run_id: str) -> bool:
        """
        Check if run exists.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run exists.
        """
        ...

    def delete(self, run_id: str) -> bool:
        """
        Delete a run record.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run was deleted, False if it didn't exist.
        """
        ...

    def list_runs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        project: str | None = None,
        adapter: str | None = None,
        status: str | None = None,
        backend_name: str | None = None,
        fingerprint: str | None = None,
        git_commit: str | None = None,
        group_id: str | None = None,
    ) -> list[RunSummary]:
        """
        List runs with optional filtering.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.
        project : str, optional
            Filter by project name.
        adapter : str, optional
            Filter by adapter name.
        status : str, optional
            Filter by run status.
        backend_name : str, optional
            Filter by backend name.
        fingerprint : str, optional
            Filter by run fingerprint.
        git_commit : str, optional
            Filter by git commit SHA.
        group_id : str, optional
            Filter by group ID.

        Returns
        -------
        list of RunSummary
            Matching runs, ordered by created_at descending.
        """
        ...

    def search_runs(
        self,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> list["RunRecord"]:
        """
        Search runs using a query expression.

        Parameters
        ----------
        query : str
            Query expression (e.g., "metric.fidelity > 0.95").
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.
        sort_by : str, optional
            Field to sort by (e.g., "metric.fidelity").
        descending : bool, optional
            Sort in descending order. Default is True.

        Returns
        -------
        list of RunRecord
            Matching run records.
        """
        ...

    def list_projects(self) -> list[str]:
        """
        List all unique project names.

        Returns
        -------
        list of str
            Sorted list of project names.
        """
        ...

    def list_groups(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List run groups with optional project filtering.

        Parameters
        ----------
        project : str, optional
            Filter by project name.
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.

        Returns
        -------
        list of dict
            Group summaries with group_id, group_name, and run_count.
        """
        ...

    def list_runs_in_group(
        self,
        group_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunSummary]:
        """
        List runs belonging to a specific group.

        Parameters
        ----------
        group_id : str
            Group identifier.
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.

        Returns
        -------
        list of RunSummary
            Runs in the group, ordered by created_at descending.
        """
        ...

    def count_runs(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
    ) -> int:
        """
        Count runs matching filters.

        Parameters
        ----------
        project : str, optional
            Filter by project name.
        status : str, optional
            Filter by run status.

        Returns
        -------
        int
            Number of matching runs.
        """
        ...

    def set_baseline(self, project: str, run_id: str) -> None:
        """
        Set baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.
        run_id : str
            Run identifier to use as baseline.
        """
        ...

    def get_baseline(self, project: str) -> BaselineInfo | None:
        """
        Get baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        BaselineInfo or None
            Baseline info, or None if no baseline set.
        """
        ...

    def clear_baseline(self, project: str) -> bool:
        """
        Clear baseline for a project.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        bool
            True if baseline was cleared, False if none existed.
        """
        ...
