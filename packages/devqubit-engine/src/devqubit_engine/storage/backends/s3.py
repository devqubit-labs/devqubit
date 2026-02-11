# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Remote storage backend for Amazon S3.

This requires optional dependencies:
    pip install devqubit-engine[s3]

Usage
-----
>>> from devqubit_engine.storage import create_store
>>> store = create_store("s3://my-bucket/prefix")

Notes
-----
The S3Registry uses a local SQLite index for efficient querying.
The index is stored in ``~/.cache/devqubit/s3_index/<bucket>_<prefix>.db``
and is automatically synchronized with S3.
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


class S3Store:
    """
    S3-backed content-addressed object store.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix for all objects.
    region : str, optional
        AWS region.
    endpoint_url : str, optional
        Custom endpoint (for MinIO, LocalStack).
    max_attempts : int, optional
        Maximum retry attempts. Default is 10.
    **kwargs
        Extra keyword args reserved for future use.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        max_attempts: int = 10,
        **kwargs: Any,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install devqubit-engine[s3]"
            )

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        client_kwargs: dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        # Configure retry policy for resilience
        client_kwargs["config"] = BotoConfig(
            retries={
                "mode": "standard",
                "max_attempts": max_attempts,
            }
        )

        self._client = boto3.client("s3", **client_kwargs)

    def _key(self, digest: str) -> str:
        """
        Convert a digest to an S3 key.

        Parameters
        ----------
        digest : str
            Content digest in format "sha256:<64 hex chars>".

        Returns
        -------
        str
            S3 object key.

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
        For CAS (content-addressed storage), overwriting the same key
        with identical content is idempotent and safe. We skip the
        exists() check to avoid an extra HEAD request round-trip.
        """
        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"
        key = self._key(digest)

        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
        )

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
        from botocore.exceptions import ClientError

        key = self._key(digest)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise ObjectNotFoundError(digest)
            raise

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

        Raises
        ------
        Exception
            Re-raises unexpected AWS errors (e.g., permission issues).
        """
        from botocore.exceptions import ClientError

        key = self._key(digest)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            # Permission / auth / throttling should not masquerade as "doesn't exist".
            raise

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
        from botocore.exceptions import ClientError

        key = self._key(digest)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise

        self._client.delete_object(Bucket=self.bucket, Key=key)
        return True

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
        s3_prefix = f"{self.prefix}/sha256/" if self.prefix else "sha256/"
        paginator = self._client.get_paginator("list_objects_v2")

        hex_re = re.compile(r"^[0-9a-f]{64}$")
        prefix_norm = prefix.lower() if isinstance(prefix, str) else None

        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                hex_part = str(obj.get("Key", "")).split("/")[-1].strip().lower()
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
        from botocore.exceptions import ClientError

        key = self._key(digest)
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=key)
            return int(response["ContentLength"])
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise ObjectNotFoundError(digest)
            raise


# -----------------------------------------------------------------------------
# Local index schema for S3Registry
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


class S3Registry:
    """
    S3-backed run registry using JSON files with local SQLite index.

    This implementation stores each run record as a JSON object at:
    ``<prefix>/runs/<run_id>.json``. A local SQLite index is maintained
    for efficient querying without fetching all records from S3.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix for all objects.
    region : str, optional
        AWS region.
    endpoint_url : str, optional
        Custom endpoint (for MinIO, LocalStack).
    max_attempts : int, optional
        Maximum retry attempts. Default is 10.
    cache_dir : Path, optional
        Directory for local index cache. Defaults to
        ``~/.cache/devqubit/s3_index``.
    auto_sync : bool, optional
        Whether to sync index on initialization. Default is True.
    **kwargs
        Extra keyword args reserved for future use.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        max_attempts: int = 10,
        cache_dir: Path | None = None,
        auto_sync: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install devqubit-engine[s3]"
            )

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        client_kwargs: dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        client_kwargs["config"] = BotoConfig(
            retries={
                "mode": "standard",
                "max_attempts": max_attempts,
            }
        )

        self._client = boto3.client("s3", **client_kwargs)

        # Setup local index
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "devqubit" / "s3_index"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create unique index name based on bucket and prefix
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", self.prefix) or "root"
        index_name = f"{self.bucket}_{safe_prefix}.db"
        self._index_path = cache_dir / index_name

        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_index()

        if auto_sync:
            self._sync_index()

        # Ensure connections are cleaned up when the process exits.
        weak_self = weakref.ref(self)
        atexit.register(S3Registry._atexit_close, weak_self)

        logger.debug(
            "S3Registry initialized: bucket=%s, prefix=%s, index=%s",
            bucket,
            prefix,
            self._index_path,
        )

    @staticmethod
    def _atexit_close(weak_self: weakref.ref[S3Registry]) -> None:
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
        Synchronize local index with S3.

        This is an incremental sync that only fetches new or updated records.
        """
        logger.debug("Syncing S3 index for %s/%s", self.bucket, self.prefix)

        # Get all run IDs from S3
        s3_run_ids = set(self._iter_run_ids_from_s3())

        # Get all run IDs from local index
        with self._get_index_connection() as conn:
            rows = conn.execute("SELECT run_id FROM runs").fetchall()
            local_run_ids = {row["run_id"] for row in rows}

        # Find new runs in S3
        new_run_ids = s3_run_ids - local_run_ids

        # Find deleted runs (in index but not in S3)
        deleted_run_ids = local_run_ids - s3_run_ids

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
                    record = self._load_dict_from_s3(run_id)
                    self._index_record(record)
                except Exception as e:
                    logger.warning("Failed to index run %s: %s", run_id, e)
            logger.debug("Indexed %d new runs", len(new_run_ids))

        logger.debug(
            "Index sync complete: %d total, %d new, %d deleted",
            len(s3_run_ids),
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

    def _run_key(self, run_id: str) -> str:
        """
        Compute the S3 key for a run record.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str
            S3 key for the run record.
        """
        parts = [self.prefix, "runs", f"{run_id}.json"]
        return "/".join(p for p in parts if p)

    def _baseline_key(self, project: str) -> str:
        """
        Compute the S3 key for a baseline record.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        str
            S3 key for the baseline record.
        """
        parts = [self.prefix, "baselines", f"{project}.json"]
        return "/".join(p for p in parts if p)

    def _runs_prefix(self) -> str:
        """
        Key prefix for run record objects.

        Returns
        -------
        str
            S3 prefix for runs.
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

        key = self._run_key(run_id)
        data = json.dumps(record, default=str).encode("utf-8")
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

        # Update local index
        self._index_record(record)

        logger.debug("Saved run to S3: %s", run_id)

    def _load_dict_from_s3(self, run_id: str) -> dict[str, Any]:
        """
        Load a run record as a raw dictionary directly from S3.

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
        from botocore.exceptions import ClientError

        key = self._run_key(run_id)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise RunNotFoundError(run_id)
            raise

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
        record_dict = self._load_dict_from_s3(run_id)
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

        Uses local index for fast lookup, falls back to S3 check.

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

        # Fall back to S3 check
        from botocore.exceptions import ClientError

        key = self._run_key(run_id)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise

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
        from botocore.exceptions import ClientError

        key = self._run_key(run_id)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise

        self._client.delete_object(Bucket=self.bucket, Key=key)

        # Remove from local index
        with self._get_index_connection() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

        logger.debug("Deleted run from S3: %s", run_id)
        return True

    def _iter_run_ids_from_s3(self) -> Iterator[str]:
        """
        Iterate all run IDs present in the bucket/prefix.

        Yields
        ------
        str
            Run IDs.
        """
        runs_prefix = self._runs_prefix()
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=runs_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith(".json"):
                    yield key.split("/")[-1].replace(".json", "")

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
        filters full records from S3.
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
                    record_dict = self._load_dict_from_s3(run_id)
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
                    continue
            return results

        # If sort_by requested: collect all matches, then sort
        all_matches: list[RunRecord] = []
        for run_id in candidate_ids:
            try:
                record_dict = self._load_dict_from_s3(run_id)
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
        key = self._baseline_key(project)
        baseline_data = {
            "project": project,
            "run_id": run_id,
            "set_at": utc_now_iso(),
        }
        data = json.dumps(baseline_data).encode("utf-8")
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

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

        # Fall back to S3
        from botocore.exceptions import ClientError

        key = self._baseline_key(project)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return BaselineInfo(**data)
        except ClientError:
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
        from botocore.exceptions import ClientError

        key = self._baseline_key(project)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError:
            return False

        self._client.delete_object(Bucket=self.bucket, Key=key)

        # Remove from local index
        with self._get_index_connection() as conn:
            conn.execute("DELETE FROM baselines WHERE project = ?", (project,))

        logger.info("Cleared baseline for %s", project)
        return True

    def sync_index(self) -> None:
        """
        Manually synchronize local index with S3.

        Call this if you suspect the index is out of sync, e.g., after
        external modifications to S3.
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
                pass
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
                    pass
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self) -> S3Registry:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager, closing all connections."""
        self.close_all()
