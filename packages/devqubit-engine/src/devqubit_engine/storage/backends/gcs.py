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

Notes
-----
The GCSRegistry uses a local SQLite index for efficient querying.
The index is stored in ``~/.cache/devqubit/gcs_index/<bucket>_<prefix>.db``
and is automatically synchronized with GCS.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import re
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from devqubit_engine.storage.errors import ObjectNotFoundError, RunNotFoundError
from devqubit_engine.storage.types import (
    ArtifactRef,
    BaselineInfo,
    RunSummary,
)
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.utils.common import utc_now_iso


logger = logging.getLogger(__name__)


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
    **kwargs
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
        """
        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"
        blob_name = self._blob_name(digest)

        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data)

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
        try:
            return self.get_bytes(digest)
        except ObjectNotFoundError:
            return None

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
        try:
            blob_name = self._blob_name(digest)
            blob = self._bucket.blob(blob_name)
            return blob.exists()
        except ValueError:
            return False

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

        try:
            blob_name = self._blob_name(digest)
            blob = self._bucket.blob(blob_name)
            if not blob.exists():
                return False
            blob.delete()
            return True
        except (ValueError, NotFound):
            return False

    def list_digests(self, prefix: str | None = None) -> Iterator[str]:
        """
        List all stored digests.

        Parameters
        ----------
        prefix : str, optional
            Filter returned digests by this digest prefix (e.g., "sha256:ab").

        Yields
        ------
        str
            Content digests.
        """
        blob_prefix = f"{self.prefix}/sha256/" if self.prefix else "sha256/"
        hex_re = re.compile(r"^[0-9a-f]{64}$")
        prefix_norm = prefix.lower() if isinstance(prefix, str) else None

        for blob in self._bucket.list_blobs(prefix=blob_prefix):
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


# -----------------------------------------------------------------------------
# Local index schema for GCSRegistry
# -----------------------------------------------------------------------------

_INDEX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    adapter TEXT NOT NULL,
    status TEXT DEFAULT 'RUNNING',
    created_at TEXT NOT NULL,
    ended_at TEXT,
    run_name TEXT,
    backend_name TEXT,
    provider TEXT,
    git_commit TEXT,
    fingerprint TEXT,
    program_fingerprint TEXT,
    group_id TEXT,
    group_name TEXT,
    parent_run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_adapter ON runs(adapter);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_backend ON runs(backend_name);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON runs(fingerprint);
CREATE INDEX IF NOT EXISTS idx_runs_git_commit ON runs(git_commit);
CREATE INDEX IF NOT EXISTS idx_runs_group_id ON runs(group_id);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);

CREATE TABLE IF NOT EXISTS baselines (
    project TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    set_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class GCSRegistry:
    """
    GCS-backed run registry using JSON files with local SQLite index.

    This implementation stores each run record as a JSON object at:
    ``<prefix>/runs/<run_id>.json``. A local SQLite index is maintained
    for efficient querying without fetching all records from GCS.

    Parameters
    ----------
    bucket : str
        GCS bucket name.
    prefix : str
        Key prefix for all objects.
    project : str, optional
        GCP project ID.
    cache_dir : Path, optional
        Directory for local index cache. Defaults to
        ``~/.cache/devqubit/gcs_index``.
    auto_sync : bool, optional
        Whether to sync index on initialization. Default is True.
    **kwargs
        Extra keyword args reserved for future use.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        cache_dir: Path | None = None,
        auto_sync: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install devqubit-engine[gcs]"
            )

        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        client = storage.Client(project=project) if project else storage.Client()
        self._bucket = client.bucket(bucket)

        # Setup local index
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "devqubit" / "gcs_index"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create unique index name based on bucket and prefix
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", self.prefix) or "root"
        index_name = f"{bucket}_{safe_prefix}.db"
        self._index_path = cache_dir / index_name

        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_index()

        if auto_sync:
            self._sync_index()

        # Ensure connections are cleaned up when the process exits.
        weak_self = weakref.ref(self)
        atexit.register(GCSRegistry._atexit_close, weak_self)

        logger.debug(
            "GCSRegistry initialized: bucket=%s, prefix=%s, index=%s",
            bucket,
            prefix,
            self._index_path,
        )

    @staticmethod
    def _atexit_close(weak_self: weakref.ref[GCSRegistry]) -> None:
        """Close all connections at interpreter shutdown."""
        self = weak_self()
        if self is not None:
            self.close_all()

    def _create_index_connection(self) -> sqlite3.Connection:
        """Create a new index database connection."""
        conn = sqlite3.connect(str(self._index_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        with self._connections_lock:
            self._connections.add(conn)
        return conn

    @contextmanager
    def _get_index_connection(self):
        """Get a thread-local index database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_index_connection()
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_index(self) -> None:
        """Initialize the local index database schema."""
        with self._get_index_connection() as conn:
            conn.executescript(_INDEX_SCHEMA_SQL)

    def _sync_index(self) -> None:
        """
        Synchronize local index with GCS.

        This is an incremental sync that only fetches new or updated records.
        """
        logger.debug("Syncing GCS index for %s/%s", self.bucket_name, self.prefix)

        # Get all run IDs from GCS
        gcs_run_ids = set(self._iter_run_ids_from_gcs())

        # Get all run IDs from local index
        with self._get_index_connection() as conn:
            rows = conn.execute("SELECT run_id FROM runs").fetchall()
            local_run_ids = {row["run_id"] for row in rows}

        # Find new runs in GCS
        new_run_ids = gcs_run_ids - local_run_ids

        # Find deleted runs (in index but not in GCS)
        deleted_run_ids = local_run_ids - gcs_run_ids

        # Remove deleted runs from index
        if deleted_run_ids:
            with self._get_index_connection() as conn:
                for run_id in deleted_run_ids:
                    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            logger.debug("Removed %d deleted runs from index", len(deleted_run_ids))

        # Index new runs
        if new_run_ids:
            for run_id in new_run_ids:
                try:
                    record = self._load_dict_from_gcs(run_id)
                    self._index_record(record)
                except Exception as e:
                    logger.warning("Failed to index run %s: %s", run_id, e)
            logger.debug("Indexed %d new runs", len(new_run_ids))

        logger.debug(
            "Index sync complete: %d total, %d new, %d deleted",
            len(gcs_run_ids),
            len(new_run_ids),
            len(deleted_run_ids),
        )

    def _index_record(self, record: dict[str, Any]) -> None:
        """
        Add or update a record in the local index.

        Parameters
        ----------
        record : dict
            Run record to index.
        """
        proj = record.get("project", {})
        project_name = proj.get("name", "") if isinstance(proj, dict) else str(proj)

        info = record.get("info", {})
        status = info.get("status", "RUNNING") if isinstance(info, dict) else "RUNNING"
        ended_at = info.get("ended_at") if isinstance(info, dict) else None
        run_name = info.get("run_name") if isinstance(info, dict) else None

        backend = record.get("backend") or {}
        backend_name = backend.get("name") if isinstance(backend, dict) else None
        provider = backend.get("provider") if isinstance(backend, dict) else None

        provenance = record.get("provenance") or {}
        git = provenance.get("git") if isinstance(provenance, dict) else None
        git_commit = git.get("commit") if isinstance(git, dict) else None

        fingerprints = record.get("fingerprints") or {}
        fingerprint = (
            fingerprints.get("run") if isinstance(fingerprints, dict) else None
        )
        program_fp = (
            fingerprints.get("program") if isinstance(fingerprints, dict) else None
        )

        with self._get_index_connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, project, adapter, status, created_at, ended_at,
                    run_name, backend_name, provider, git_commit,
                    fingerprint, program_fingerprint,
                    group_id, group_name, parent_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    project = excluded.project,
                    adapter = excluded.adapter,
                    status = excluded.status,
                    ended_at = excluded.ended_at,
                    run_name = excluded.run_name,
                    backend_name = excluded.backend_name,
                    provider = excluded.provider,
                    git_commit = excluded.git_commit,
                    fingerprint = excluded.fingerprint,
                    program_fingerprint = excluded.program_fingerprint,
                    group_id = excluded.group_id,
                    group_name = excluded.group_name,
                    parent_run_id = excluded.parent_run_id
                """,
                (
                    record["run_id"],
                    project_name,
                    record.get("adapter", ""),
                    status,
                    record.get("created_at", ""),
                    ended_at,
                    run_name,
                    backend_name,
                    provider,
                    git_commit,
                    fingerprint,
                    program_fp,
                    record.get("group_id"),
                    record.get("group_name"),
                    record.get("parent_run_id"),
                ),
            )

    def _run_blob_name(self, run_id: str) -> str:
        """
        Compute the GCS blob name for a run record.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str
            GCS blob name for the run record.
        """
        parts = [self.prefix, "runs", f"{run_id}.json"]
        return "/".join(p for p in parts if p)

    def _baseline_blob_name(self, project: str) -> str:
        """
        Compute the GCS blob name for a baseline record.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        str
            GCS blob name for the baseline record.
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

    def save(self, record: dict[str, Any]) -> None:
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

        blob_name = self._run_blob_name(run_id)
        blob = self._bucket.blob(blob_name)
        data = json.dumps(record, default=str)
        blob.upload_from_string(data, content_type="application/json")

        # Update local index
        self._index_record(record)

        logger.debug("Saved run to GCS: %s", run_id)

    def _load_dict_from_gcs(self, run_id: str) -> dict[str, Any]:
        """
        Load a run record as a raw dictionary directly from GCS.

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

        blob_name = self._run_blob_name(run_id)
        blob = self._bucket.blob(blob_name)
        try:
            data = blob.download_as_bytes()
            return json.loads(data.decode("utf-8"))
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
        record_dict = self._load_dict_from_gcs(run_id)
        artifacts = [
            ArtifactRef.from_dict(a)
            for a in record_dict.get("artifacts", [])
            if isinstance(a, dict)
        ]
        rec = RunRecord(record=record_dict, artifacts=artifacts)
        rec.mark_finalized()
        return rec

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

        Uses local index for fast lookup, falls back to GCS check.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run record exists.
        """
        # Check local index first (fast)
        with self._get_index_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                return True

        # Fall back to GCS check
        blob_name = self._run_blob_name(run_id)
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

        blob_name = self._run_blob_name(run_id)
        blob = self._bucket.blob(blob_name)

        if not blob.exists():
            return False

        try:
            blob.delete()
        except NotFound:
            return False

        # Remove from local index
        with self._get_index_connection() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

        logger.debug("Deleted run from GCS: %s", run_id)
        return True

    def _iter_run_ids_from_gcs(self) -> Iterator[str]:
        """
        Iterate all run IDs present in the bucket/prefix.

        Yields
        ------
        str
            Run IDs.
        """
        runs_prefix = self._runs_prefix()
        for blob in self._bucket.list_blobs(prefix=runs_prefix):
            if blob.name.endswith(".json"):
                yield blob.name.split("/")[-1].replace(".json", "")

    def list_runs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        project: str | None = None,
        name: str | None = None,
        adapter: str | None = None,
        status: str | None = None,
        backend_name: str | None = None,
        fingerprint: str | None = None,
        git_commit: str | None = None,
        group_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[RunSummary]:
        """
        List runs with optional filtering.

        Uses local SQLite index for efficient querying.

        Parameters
        ----------
        limit : int
            Maximum number of results. Default is 100.
        offset : int
            Number of results to skip. Default is 0.
        project : str, optional
            Filter by project name.
        name : str, optional
            Filter by run name (exact match).
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
        created_after : str, optional
            ISO 8601 lower bound (exclusive) on ``created_at``.
        created_before : str, optional
            ISO 8601 upper bound (exclusive) on ``created_at``.

        Returns
        -------
        list of RunSummary
            Matching runs ordered by created_at descending.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)
        if name:
            conditions.append("run_name = ?")
            params.append(name)
        if adapter:
            conditions.append("adapter = ?")
            params.append(adapter)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if backend_name:
            conditions.append("backend_name = ?")
            params.append(backend_name)
        if fingerprint:
            conditions.append("fingerprint = ?")
            params.append(fingerprint)
        if git_commit:
            conditions.append("git_commit = ?")
            params.append(git_commit)
        if group_id:
            conditions.append("group_id = ?")
            params.append(group_id)
        if created_after:
            conditions.append("created_at > ?")
            params.append(created_after)
        if created_before:
            conditions.append("created_at < ?")
            params.append(created_before)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        with self._get_index_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, run_name, project, adapter, status, created_at, ended_at,
                       group_id, group_name, parent_run_id
                FROM runs {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()

        return [
            RunSummary(
                run_id=row["run_id"],
                run_name=row["run_name"],
                project=row["project"],
                adapter=row["adapter"],
                status=row["status"],
                created_at=row["created_at"],
                ended_at=row["ended_at"],
                group_id=row["group_id"],
                group_name=row["group_name"],
                parent_run_id=row["parent_run_id"],
            )
            for row in rows
        ]

    def list_projects(self) -> list[str]:
        """
        List all unique project names.

        Returns
        -------
        list of str
            Sorted unique project names.
        """
        with self._get_index_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM runs ORDER BY project"
            ).fetchall()
        return [row["project"] for row in rows]

    def count_runs(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
    ) -> int:
        """
        Count runs matching filters.

        Uses local index for efficient counting.

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
        conditions: list[str] = []
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_index_connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM runs {where_clause}", params
            ).fetchone()
        return row["cnt"] if row else 0

    def search_runs(
        self,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> list[RunRecord]:
        """
        Search runs using a query expression.

        Parameters
        ----------
        query : str
            Query expression (e.g., "metric.fidelity > 0.95 and params.shots = 1000").
        limit : int
            Maximum results to return. Default is 100.
        offset : int
            Number of results to skip. Default is 0.
        sort_by : str, optional
            Field to sort by.
        descending : bool
            Sort in descending order. Default is True.

        Returns
        -------
        list of RunRecord
            Matching run records.

        Notes
        -----
        Uses local index to get candidate run IDs, then fetches and
        filters full records from GCS.
        """
        from devqubit_engine.query import (
            _resolve_field,
            matches_query,
            parse_query,
        )

        parsed = parse_query(query)

        # Get candidate run IDs from index (ordered by created_at)
        with self._get_index_connection() as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC"
            ).fetchall()
        candidate_ids = [row["run_id"] for row in rows]

        # If no sort_by: stream and stop early
        if sort_by is None:
            results: list[RunRecord] = []
            matched = 0
            for run_id in candidate_ids:
                try:
                    record_dict = self._load_dict_from_gcs(run_id)
                    if matches_query(record_dict, parsed):
                        if matched >= offset:
                            artifacts = [
                                ArtifactRef.from_dict(a)
                                for a in record_dict.get("artifacts", [])
                                if isinstance(a, dict)
                            ]
                            rec = RunRecord(record=record_dict, artifacts=artifacts)
                            rec.mark_finalized()
                            results.append(rec)
                            if len(results) >= limit:
                                break
                        matched += 1
                except Exception:
                    logger.debug("Failed to load run during search from GCS")
                    continue
            return results

        # If sort_by requested: collect all matches, then sort
        all_matches: list[RunRecord] = []
        for run_id in candidate_ids:
            try:
                record_dict = self._load_dict_from_gcs(run_id)
                if matches_query(record_dict, parsed):
                    artifacts = [
                        ArtifactRef.from_dict(a)
                        for a in record_dict.get("artifacts", [])
                        if isinstance(a, dict)
                    ]
                    rec = RunRecord(record=record_dict, artifacts=artifacts)
                    rec.mark_finalized()
                    all_matches.append(rec)
            except Exception:
                logger.debug("Failed to load run during sorted search from GCS")
                continue

        def sort_key(rec: RunRecord) -> Any:
            found, val = _resolve_field(rec.record, sort_by)
            if not found or val is None:
                return (1, 0)
            try:
                return (0, float(val))
            except (ValueError, TypeError):
                return (0, str(val))

        all_matches.sort(key=sort_key, reverse=descending)
        return all_matches[offset : offset + limit]

    def list_groups(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List all run groups.

        Parameters
        ----------
        project : str, optional
            Filter by project.
        limit : int
            Maximum groups to return. Default is 100.
        offset : int
            Number of groups to skip. Default is 0.

        Returns
        -------
        list of dict
            Group summaries with group_id, group_name, run_count, first_run, last_run.
        """
        conditions: list[str] = ["group_id IS NOT NULL"]
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        params.extend([limit, offset])

        with self._get_index_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT group_id,
                       MAX(group_name) as group_name,
                       project,
                       COUNT(*) as run_count,
                       MIN(created_at) as first_run,
                       MAX(created_at) as last_run
                FROM runs
                {where_clause}
                GROUP BY project, group_id
                ORDER BY last_run DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()

        return [
            {
                "group_id": row["group_id"],
                "group_name": row["group_name"],
                "project": row["project"],
                "run_count": row["run_count"],
                "first_run": row["first_run"],
                "last_run": row["last_run"],
            }
            for row in rows
        ]

    def list_runs_in_group(
        self,
        group_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunSummary]:
        """
        List all runs in a group.

        Parameters
        ----------
        group_id : str
            Group identifier.
        limit : int
            Maximum results. Default is 100.
        offset : int
            Results to skip. Default is 0.

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
        blob_name = self._baseline_blob_name(project)
        blob = self._bucket.blob(blob_name)
        baseline_data = {
            "project": project,
            "run_id": run_id,
            "set_at": utc_now_iso(),
        }
        blob.upload_from_string(
            json.dumps(baseline_data), content_type="application/json"
        )

        # Update local index
        with self._get_index_connection() as conn:
            conn.execute(
                """
                INSERT INTO baselines (project, run_id, set_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project) DO UPDATE SET
                    run_id = excluded.run_id,
                    set_at = excluded.set_at
                """,
                (project, run_id, baseline_data["set_at"]),
            )

        logger.info("Set baseline for %s: %s", project, run_id)

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
        # Try local index first
        with self._get_index_connection() as conn:
            row = conn.execute(
                "SELECT project, run_id, set_at FROM baselines WHERE project = ?",
                (project,),
            ).fetchone()
            if row:
                return BaselineInfo(
                    project=row["project"],
                    run_id=row["run_id"],
                    set_at=row["set_at"],
                )

        # Fall back to GCS
        from google.cloud.exceptions import NotFound

        blob_name = self._baseline_blob_name(project)
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

        blob_name = self._baseline_blob_name(project)
        blob = self._bucket.blob(blob_name)

        if not blob.exists():
            return False

        try:
            blob.delete()
        except NotFound:
            return False

        # Remove from local index
        with self._get_index_connection() as conn:
            conn.execute("DELETE FROM baselines WHERE project = ?", (project,))

        logger.info("Cleared baseline for %s", project)
        return True

    def sync_index(self) -> None:
        """
        Manually synchronize local index with GCS.

        Call this if you suspect the index is out of sync, e.g., after
        external modifications to GCS.
        """
        self._sync_index()

    def close(self) -> None:
        """
        Close the current thread's index database connection.

        Safe to call multiple times.  For cross-thread cleanup use
        :meth:`close_all`.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with self._connections_lock:
                self._connections.discard(conn)
            try:
                conn.close()
            except Exception:
                logger.debug("Failed to close index connection during cleanup")
            self._local.conn = None

    def close_all(self) -> None:
        """
        Close every tracked index connection across all threads.

        Intended for interpreter shutdown and explicit cleanup in tests.
        """
        with self._connections_lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    logger.debug("Failed to close index connection during close_all")
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self) -> GCSRegistry:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager, closing all connections."""
        self.close_all()
