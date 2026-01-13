# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Remote storage backend for Google Cloud Storage.

This requires optional dependencies:
    pip install devqubit-engine[gcs]

Usage
-----
>>> from devqubit_engine.storage import create_store
>>> store = create_store("gs://my-bucket/prefix")
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterator, List

from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.protocols import (
    BaselineInfo,
    ObjectNotFoundError,
    RunNotFoundError,
    RunSummary,
)
from devqubit_engine.uec.types import ArtifactRef
from devqubit_engine.utils.time_utils import utc_now_iso


class GCSStore:
    """
    Google Cloud Storage-backed content-addressed object store.

    Parameters
    ----------
    bucket : str
        GCS bucket name.
    prefix : str
        Key prefix for all objects.
    project : str, optional
        GCP project ID.
    kwargs : Any
        Extra keyword args reserved for future use.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install devqubit-engine[gcs]"
            )

        self.prefix = prefix.strip("/")
        client = storage.Client(project=project) if project else storage.Client()
        self._bucket = client.bucket(bucket)

    def _blob_name(self, digest: str) -> str:
        """
        Convert a digest into a GCS blob name.

        Parameters
        ----------
        digest : str
            Content digest in format "sha256:<64 hex chars>".

        Returns
        -------
        str
            Blob name.

        Raises
        ------
        ValueError
            If digest format is invalid.
        """
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"Invalid digest: {digest!r}")

        hex_part = digest[7:].strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", hex_part) is None:
            raise ValueError(
                f"Invalid digest value (expected 64 hex chars): {digest!r}"
            )

        parts = [self.prefix, "sha256", hex_part[:2], hex_part]
        return "/".join(p for p in parts if p)

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
            Content digest in format "sha256:<hex>".

        Notes
        -----
        Uses a generation-match precondition (0) to avoid an extra exists() call
        and to prevent overwriting an existing live object. If the object already
        exists, Cloud Storage returns 412 Precondition Failed and we treat it as
        success for CAS semantics.
        """
        from google.api_core.exceptions import PreconditionFailed

        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"
        blob_name = self._blob_name(digest)

        blob = self._bucket.blob(blob_name)
        try:
            # Only create if it doesn't exist (single RPC, no TOCTOU race)
            blob.upload_from_string(data, if_generation_match=0)
        except PreconditionFailed:
            # Object already exists: CAS semantics -> ok
            pass

        return digest

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
            Stored bytes.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        """
        from google.cloud.exceptions import NotFound

        blob_name = self._blob_name(digest)
        blob = self._bucket.blob(blob_name)
        try:
            return blob.download_as_bytes()
        except NotFound:
            raise ObjectNotFoundError(digest)

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
        blob_name = self._blob_name(digest)
        blob = self._bucket.blob(blob_name)
        return blob.exists()

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
        from google.cloud.exceptions import NotFound

        blob_name = self._blob_name(digest)
        blob = self._bucket.blob(blob_name)
        try:
            blob.delete()
            return True
        except NotFound:
            return False

    def list_digests(self, prefix: str | None = None) -> Iterator[str]:
        """
        List all stored digests.

        Parameters
        ----------
        prefix : str, optional
            Filter by digest prefix (e.g., "sha256:ab").

        Yields
        ------
        str
            Content digests.
        """
        gcs_prefix = f"{self.prefix}/sha256/" if self.prefix else "sha256/"
        hex_re = re.compile(r"^[0-9a-f]{64}$")
        prefix_norm = prefix.lower() if isinstance(prefix, str) else None

        for blob in self._bucket.list_blobs(prefix=gcs_prefix):
            hex_part = blob.name.split("/")[-1].strip().lower()
            if hex_re.fullmatch(hex_part) is None:
                continue

            digest = f"sha256:{hex_part}"
            if prefix_norm is None or digest.startswith(prefix_norm):
                yield digest

    def get_size(self, digest: str) -> int:
        """
        Get size of stored object in bytes.

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
        from google.cloud.exceptions import NotFound

        blob_name = self._blob_name(digest)
        blob = self._bucket.blob(blob_name)
        try:
            blob.reload()
            return blob.size or 0
        except NotFound:
            raise ObjectNotFoundError(digest)


class GCSRegistry:
    """
    GCS-backed run registry using JSON files.

    Notes
    -----
    This implementation stores each run record as a JSON object at:

        <prefix>/runs/<run_id>.json

    and each project baseline as:

        <prefix>/baselines/<project>.json

    Parameters
    ----------
    bucket : str
        GCS bucket name.
    prefix : str
        Key prefix for all objects.
    project : str, optional
        GCP project ID.
    kwargs : Any
        Extra keyword args reserved for future use.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install devqubit-engine[gcs]"
            )

        self.prefix = prefix.strip("/")
        client = storage.Client(project=project) if project else storage.Client()
        self._bucket = client.bucket(bucket)

    def _run_blob(self, run_id: str) -> str:
        """
        Compute the blob name for a run record.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str
            Blob name for the run record.
        """
        parts = [self.prefix, "runs", f"{run_id}.json"]
        return "/".join(p for p in parts if p)

    def _baseline_blob(self, project: str) -> str:
        """
        Compute the blob name for a baseline record.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        str
            Blob name for the baseline record.
        """
        parts = [self.prefix, "baselines", f"{project}.json"]
        return "/".join(p for p in parts if p)

    def _runs_prefix(self) -> str:
        """
        Blob prefix for run record objects.

        Returns
        -------
        str
            GCS prefix for runs.
        """
        return f"{self.prefix}/runs/" if self.prefix else "runs/"

    def save(self, record: Dict[str, Any]) -> None:
        """
        Save or update a run record.

        Parameters
        ----------
        record : dict
            Run record. Must include 'run_id'.

        Raises
        ------
        ValueError
            If 'run_id' is missing.
        """
        run_id = record.get("run_id")
        if not run_id:
            raise ValueError("Record must have 'run_id'")

        blob_name = self._run_blob(run_id)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(record, default=str),
            content_type="application/json",
        )

    def _load_dict(self, run_id: str) -> Dict[str, Any]:
        """
        Load a run record as a raw dictionary.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        dict
            Run record as dictionary.

        Raises
        ------
        RunNotFoundError
            If run record does not exist.
        """
        from google.cloud.exceptions import NotFound

        blob_name = self._run_blob(run_id)
        blob = self._bucket.blob(blob_name)
        try:
            return json.loads(blob.download_as_bytes().decode("utf-8"))
        except NotFound:
            raise RunNotFoundError(run_id)

    def load(self, run_id: str) -> RunRecord:
        """
        Load a run record by ID.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord
            Run record.

        Raises
        ------
        RunNotFoundError
            If run record does not exist.
        """
        record_dict = self._load_dict(run_id)
        artifacts = [
            ArtifactRef.from_dict(a)
            for a in record_dict.get("artifacts", [])
            if isinstance(a, dict)
        ]
        return RunRecord(record=record_dict, artifacts=artifacts)

    def load_or_none(self, run_id: str) -> RunRecord | None:
        """
        Load a run record or return None.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord or None
            Run record if found, otherwise None.
        """
        try:
            return self.load(run_id)
        except RunNotFoundError:
            return None

    def exists(self, run_id: str) -> bool:
        """
        Check if a run record exists.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run record exists.
        """
        blob_name = self._run_blob(run_id)
        blob = self._bucket.blob(blob_name)
        return blob.exists()

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
            True if deleted, False if it didn't exist.
        """
        from google.cloud.exceptions import NotFound

        blob_name = self._run_blob(run_id)
        blob = self._bucket.blob(blob_name)
        try:
            blob.delete()
            return True
        except NotFound:
            return False

    def _summarize_record(self, run_id: str, record: Dict[str, Any]) -> RunSummary:
        """
        Convert a full record to a RunSummary.

        Parameters
        ----------
        run_id : str
            Run identifier.
        record : dict
            Full run record.

        Returns
        -------
        RunSummary
            Summary info.
        """
        proj = record.get("project", {})
        proj_name = proj.get("name", "") if isinstance(proj, dict) else str(proj)
        rec_adapter = record.get("adapter", "")
        info = record.get("info", {})
        rec_status = info.get("status", "") if isinstance(info, dict) else ""
        ended_at = info.get("ended_at") if isinstance(info, dict) else None

        # Extract group info
        group_id = record.get("group_id")
        group_name = record.get("group_name")
        parent_run_id = record.get("parent_run_id")

        return RunSummary(
            run_id=run_id,
            project=proj_name,
            adapter=rec_adapter,
            status=rec_status,
            created_at=record.get("created_at", ""),
            ended_at=ended_at,
            group_id=group_id,
            group_name=group_name,
            parent_run_id=parent_run_id,
        )

    def _record_matches(
        self,
        record: Dict[str, Any],
        *,
        project: str | None,
        adapter: str | None,
        status: str | None,
        backend_name: str | None,
        fingerprint: str | None,
        git_commit: str | None,
        group_id: str | None = None,
    ) -> bool:
        """
        Apply filters to a record.

        Parameters
        ----------
        record : dict
            Full run record.
        project, adapter, status, backend_name, fingerprint, git_commit, group_id
            Filter fields.

        Returns
        -------
        bool
            True if record matches all provided filters.
        """
        proj = record.get("project", {})
        proj_name = proj.get("name", "") if isinstance(proj, dict) else str(proj)
        if project and proj_name != project:
            return False

        rec_adapter = record.get("adapter", "")
        if adapter and rec_adapter != adapter:
            return False

        info = record.get("info", {})
        rec_status = info.get("status", "") if isinstance(info, dict) else ""
        if status and rec_status != status:
            return False

        backend = record.get("backend") or {}
        rec_backend_name = backend.get("name") if isinstance(backend, dict) else None
        if backend_name and rec_backend_name != backend_name:
            return False

        provenance = record.get("provenance") or {}
        git = provenance.get("git") if isinstance(provenance, dict) else None
        rec_git_commit = git.get("commit") if isinstance(git, dict) else None
        if git_commit and rec_git_commit != git_commit:
            return False

        fps = record.get("fingerprints") or {}
        rec_fp = fps.get("run") if isinstance(fps, dict) else None
        if fingerprint and rec_fp != fingerprint:
            return False

        if group_id and record.get("group_id") != group_id:
            return False

        return True

    def _iter_run_ids(self) -> Iterator[str]:
        """
        Iterate all run IDs present in the bucket/prefix.

        Yields
        ------
        str
            Run IDs.
        """
        runs_prefix = self._runs_prefix()
        for blob in self._bucket.list_blobs(prefix=runs_prefix):
            name = blob.name
            if name.endswith(".json"):
                yield name.split("/")[-1].replace(".json", "")

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
    ) -> List[RunSummary]:
        """
        List runs with optional filtering.

        Parameters
        ----------
        limit : int
            Maximum number of results.
        offset : int
            Number of results to skip.
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
            Filter by git commit.
        group_id : str, optional
            Filter by group ID.

        Returns
        -------
        list of RunSummary
            Matching runs ordered by created_at descending.
        """
        all_runs: List[RunSummary] = []

        for run_id in self._iter_run_ids():
            try:
                record = self._load_dict(run_id)
            except Exception:
                continue

            if not self._record_matches(
                record,
                project=project,
                adapter=adapter,
                status=status,
                backend_name=backend_name,
                fingerprint=fingerprint,
                git_commit=git_commit,
                group_id=group_id,
            ):
                continue

            all_runs.append(self._summarize_record(run_id, record))

        all_runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return all_runs[offset : offset + limit]

    def list_projects(self) -> List[str]:
        """
        List all unique project names.

        Returns
        -------
        list of str
            Sorted unique project names.
        """
        projects: set[str] = set()
        for run in self.list_runs(limit=10**9, offset=0):
            proj = run.get("project")
            if proj:
                projects.add(proj)
        return sorted(projects)

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
            Count of matching runs.
        """
        return len(
            self.list_runs(
                limit=10**9,
                offset=0,
                project=project,
                status=status,
            )
        )

    def search_runs(
        self,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> List[RunRecord]:
        """
        Search runs using a query expression.

        Parameters
        ----------
        query : str
            Query expression (e.g., "metric.fidelity > 0.95 and params.shots = 1000").
        limit : int
            Maximum results to return.
        offset : int
            Number of results to skip.
        sort_by : str, optional
            Field to sort by.
        descending : bool
            Sort in descending order.

        Returns
        -------
        list of RunRecord
            Matching run records.

        Notes
        -----
        Remote search iterates all runs and filters in memory.
        For large datasets, consider using a local cache or hosted search service.
        """
        from devqubit_engine.query import (
            _resolve_field,
            matches_query,
            parse_query,
        )

        parsed = parse_query(query)
        results: List[RunRecord] = []

        for run_id in self._iter_run_ids():
            try:
                record_dict = self._load_dict(run_id)
            except Exception:
                continue

            # Apply query
            if matches_query(record_dict, parsed):
                artifacts = [
                    ArtifactRef.from_dict(a)
                    for a in record_dict.get("artifacts", [])
                    if isinstance(a, dict)
                ]
                results.append(RunRecord(record=record_dict, artifacts=artifacts))

        # Sort
        if sort_by:

            def sort_key(rec: RunRecord) -> Any:
                found, val = _resolve_field(rec.record, sort_by)
                if not found or val is None:
                    return (1, 0)
                try:
                    return (0, float(val))
                except (ValueError, TypeError):
                    return (0, str(val))

            results.sort(key=sort_key, reverse=descending)
        else:
            results.sort(
                key=lambda r: r.record.get("created_at", ""),
                reverse=descending,
            )

        # Apply limit/offset
        return results[offset : offset + limit]

    def list_groups(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List all run groups.

        Parameters
        ----------
        project : str, optional
            Filter by project.
        limit : int
            Maximum groups to return.
        offset : int
            Number of groups to skip.

        Returns
        -------
        list of dict
            Group summaries with group_id, group_name, run_count, first_run, last_run.
        """
        groups: Dict[str, Dict[str, Any]] = {}

        for run_id in self._iter_run_ids():
            try:
                record = self._load_dict(run_id)
            except Exception:
                continue

            # Apply project filter
            if project:
                proj = record.get("project", {})
                proj_name = (
                    proj.get("name", "") if isinstance(proj, dict) else str(proj)
                )
                if proj_name != project:
                    continue

            group_id = record.get("group_id")
            if not group_id:
                continue

            # Extract project name for storage
            proj = record.get("project", {})
            proj_name = proj.get("name", "") if isinstance(proj, dict) else str(proj)

            if group_id not in groups:
                groups[group_id] = {
                    "group_id": group_id,
                    "group_name": record.get("group_name"),
                    "project": proj_name,
                    "run_count": 0,
                    "first_run": record.get("created_at", ""),
                    "last_run": record.get("created_at", ""),
                }

            groups[group_id]["run_count"] += 1
            created = record.get("created_at", "")
            if created < groups[group_id]["first_run"]:
                groups[group_id]["first_run"] = created
            if created > groups[group_id]["last_run"]:
                groups[group_id]["last_run"] = created

        # Sort by last_run descending
        result = sorted(
            groups.values(),
            key=lambda g: g["last_run"],
            reverse=True,
        )
        return result[offset : offset + limit]

    def list_runs_in_group(
        self,
        group_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RunSummary]:
        """
        List all runs in a group.

        Parameters
        ----------
        group_id : str
            Group identifier.
        limit : int
            Maximum results.
        offset : int
            Results to skip.

        Returns
        -------
        list of RunSummary
            Runs in the group.
        """
        return self.list_runs(limit=limit, offset=offset, group_id=group_id)

    def set_baseline(self, project: str, run_id: str) -> None:
        """
        Set baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.
        run_id : str
            Run identifier to set as baseline.
        """
        blob_name = self._baseline_blob(project)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(
                {
                    "project": project,
                    "run_id": run_id,
                    "set_at": utc_now_iso(),
                }
            ),
            content_type="application/json",
        )

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
            Baseline info if set, else None.
        """
        from google.cloud.exceptions import NotFound

        blob_name = self._baseline_blob(project)
        blob = self._bucket.blob(blob_name)
        try:
            data = json.loads(blob.download_as_bytes().decode("utf-8"))
            return BaselineInfo(**data)
        except NotFound:
            return None
        except Exception:
            return None

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
            True if baseline existed and was deleted, otherwise False.
        """
        from google.cloud.exceptions import NotFound

        blob_name = self._baseline_blob(project)
        blob = self._bucket.blob(blob_name)
        try:
            blob.delete()
            return True
        except NotFound:
            return False
