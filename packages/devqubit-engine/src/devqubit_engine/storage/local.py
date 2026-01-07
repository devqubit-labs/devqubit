# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Local filesystem storage implementations.

This module provides local filesystem-backed implementations of the
storage protocols:

- :class:`LocalStore` - Content-addressed blob storage using SHA-256
- :class:`LocalRegistry` - SQLite-backed run metadata registry
- :class:`LocalWorkspace` - Combined store + registry workspace
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from devqubit_engine.core.record import RunRecord
from devqubit_engine.core.types import ArtifactRef
from devqubit_engine.storage.protocols import (
    BaselineInfo,
    ObjectNotFoundError,
    RunNotFoundError,
    RunSummary,
)
from devqubit_engine.utils.time_utils import utc_now_iso


logger = logging.getLogger(__name__)

_SQLITE_MAX_VARIABLE_NUMBER = 900


class LocalStore:
    """
    Filesystem-backed content-addressed object store.

    Objects are stored using their SHA-256 hash as the filename,
    organized into sharded directories (first 2 hex chars) for
    filesystem efficiency.

    Storage layout::

        root/
        └── sha256/
            ├── ab/
            │   └── abcd1234...  (full 64-char hex)
            └── cd/
                └── cdef5678...

    Parameters
    ----------
    root : Path
        Root directory for object storage.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug("LocalStore initialized at %s", self.root)

    def _digest_path(self, digest: str) -> Path:
        """
        Get filesystem path for a digest.

        Parameters
        ----------
        digest : str
            Content digest in format ``sha256:<64-hex-chars>``.

        Returns
        -------
        Path
            Filesystem path for storing/retrieving the object.

        Raises
        ------
        ValueError
            If digest format or value is invalid.
        """
        if not digest.startswith("sha256:"):
            raise ValueError(f"Invalid digest format: {digest}")

        hex_part = digest[7:]
        if len(hex_part) != 64:
            raise ValueError(f"Invalid digest length: {digest}")

        # Canonicalize + validate (prevents path traversal via separators etc.)
        hex_part = hex_part.lower()
        if re.fullmatch(r"[0-9a-f]{64}", hex_part) is None:
            raise ValueError(f"Invalid digest value (non-hex): {digest}")

        return self.root / "sha256" / hex_part[:2] / hex_part

    def put_bytes(self, data: bytes) -> str:
        """
        Store bytes and return content digest.

        Uses atomic write (write to temp file, then rename) to ensure
        data integrity even on system crashes.

        Parameters
        ----------
        data : bytes
            Data to store.

        Returns
        -------
        str
            Content digest in format ``sha256:<64-hex-chars>``.

        Notes
        -----
        If an object with the same digest already exists, this is a no-op
        and returns the existing digest (content-addressed deduplication).
        """
        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"
        path = self._digest_path(digest)

        # Deduplication: if already exists, return immediately
        if path.exists():
            logger.debug("Object already exists: %s", digest[:24])
            return digest

        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file + rename
        fd, tmp_path = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            logger.debug("Stored object: %s (%d bytes)", digest[:24], len(data))
        except Exception:
            # Best-effort cleanup
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

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
            Stored data.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        """
        path = self._digest_path(digest)
        if not path.exists():
            raise ObjectNotFoundError(digest)
        return path.read_bytes()

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
            return self._digest_path(digest).exists()
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
        try:
            path = self._digest_path(digest)
            if path.exists():
                path.unlink()
                logger.debug("Deleted object: %s", digest[:24])
                return True
            return False
        except ValueError:
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
        sha256_dir = self.root / "sha256"
        if not sha256_dir.exists():
            return

        # Match canonical hex (we'll lowercase filenames for canonical output)
        hex_re = re.compile(r"^[0-9a-f]{64}$")
        prefix_norm = prefix.lower() if prefix is not None else None

        for shard in sha256_dir.iterdir():
            if not shard.is_dir():
                continue

            for obj_file in shard.iterdir():
                if not obj_file.is_file():
                    continue

                hex_part = obj_file.name.lower()
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
        path = self._digest_path(digest)
        if not path.exists():
            raise ObjectNotFoundError(digest)
        return path.stat().st_size


_SCHEMA_SQL = """
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
    parent_run_id TEXT,
    record_json TEXT NOT NULL
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

-- Tag index for fast tag-based queries
CREATE TABLE IF NOT EXISTS run_tags (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_tags_key ON run_tags(key);
CREATE INDEX IF NOT EXISTS idx_run_tags_value ON run_tags(key, value);
"""


class LocalRegistry:
    """
    SQLite-backed run registry.

    Provides efficient storage and querying of run metadata with
    indexed fields for common query patterns.

    Parameters
    ----------
    root : Path
        Root directory for the registry database.
    timeout : float, optional
        Database connection timeout in seconds. Default is 30.0.
    """

    def __init__(self, root: Path, timeout: float = 30.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "registry.db"
        self._timeout = timeout
        self._init_db()
        logger.debug("LocalRegistry initialized at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with standard settings.

        Returns
        -------
        sqlite3.Connection
            Connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(self.db_path, timeout=self._timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _extract_fields(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Extract indexed fields from a run record.

        Parameters
        ----------
        record : dict
            Full run record.

        Returns
        -------
        dict
            Flattened fields for indexing and storage.
        """
        project = record.get("project", {})
        project_name = (
            project.get("name", "") if isinstance(project, dict) else str(project)
        )

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

        return {
            "run_id": record["run_id"],
            "project": project_name,
            "adapter": record.get("adapter", ""),
            "status": status,
            "created_at": record.get("created_at", ""),
            "ended_at": ended_at,
            "run_name": run_name,
            "backend_name": backend_name,
            "provider": provider,
            "git_commit": git_commit,
            "fingerprint": fingerprint,
            "program_fingerprint": program_fp,
            "group_id": record.get("group_id"),
            "group_name": record.get("group_name"),
            "parent_run_id": record.get("parent_run_id"),
            "record_json": json.dumps(record, default=str),
        }

    def _extract_tags(self, record: dict[str, Any]) -> dict[str, str]:
        """
        Extract tags from run record for indexing.

        Parameters
        ----------
        record : dict
            Full run record.

        Returns
        -------
        dict
            Tags dictionary (key -> value).
        """
        data = record.get("data", {})
        if not isinstance(data, dict):
            return {}
        tags = data.get("tags", {})
        if not isinstance(tags, dict):
            return {}
        return {str(k): str(v) for k, v in tags.items()}

    def save(self, record: dict[str, Any]) -> None:
        """
        Save or update a run record.

        Parameters
        ----------
        record : dict
            Run record with required 'run_id' field.

        Raises
        ------
        ValueError
            If record is missing 'run_id'.
        """
        if "run_id" not in record:
            raise ValueError("Record must have 'run_id' field")

        fields = self._extract_fields(record)
        tags = self._extract_tags(record)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, project, adapter, status, created_at, ended_at,
                    run_name, backend_name, provider, git_commit,
                    fingerprint, program_fingerprint,
                    group_id, group_name, parent_run_id, record_json
                )
                VALUES (
                    :run_id, :project, :adapter, :status, :created_at, :ended_at,
                    :run_name, :backend_name, :provider, :git_commit,
                    :fingerprint, :program_fingerprint,
                    :group_id, :group_name, :parent_run_id, :record_json
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    project = :project,
                    adapter = :adapter,
                    status = :status,
                    ended_at = :ended_at,
                    run_name = :run_name,
                    backend_name = :backend_name,
                    provider = :provider,
                    git_commit = :git_commit,
                    fingerprint = :fingerprint,
                    program_fingerprint = :program_fingerprint,
                    group_id = :group_id,
                    group_name = :group_name,
                    parent_run_id = :parent_run_id,
                    record_json = :record_json
                """,
                fields,
            )

            # Update tag index
            conn.execute("DELETE FROM run_tags WHERE run_id = ?", (fields["run_id"],))
            for key, value in tags.items():
                conn.execute(
                    "INSERT OR REPLACE INTO run_tags (run_id, key, value) VALUES (?, ?, ?)",
                    (fields["run_id"], key, value),
                )

        logger.debug("Saved run: %s", fields["run_id"])

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
            Run record wrapper.

        Raises
        ------
        RunNotFoundError
            If run does not exist.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT record_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

        if row is None:
            raise RunNotFoundError(run_id)

        record_dict = json.loads(row["record_json"])
        artifacts = [
            ArtifactRef.from_dict(a)
            for a in record_dict.get("artifacts", [])
            if isinstance(a, dict)
        ]
        return RunRecord(record=record_dict, artifacts=artifacts)

    def load_or_none(self, run_id: str) -> RunRecord | None:
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
        try:
            return self.load(run_id)
        except RunNotFoundError:
            return None

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
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return row is not None

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
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

        if cursor.rowcount > 0:
            logger.debug("Deleted run: %s", run_id)
            return True
        return False

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
        conditions: list[str] = []
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)
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

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, project, adapter, status, created_at, ended_at,
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

        This method loads full records and applies in-memory filtering.
        For tag-based queries, it uses the tag index for faster lookups.

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
        from devqubit_engine.query import (
            _resolve_field,
            matches_query,
            parse_query,
        )

        parsed = parse_query(query)

        # Separate tag conditions for index optimization
        tag_conditions = []
        for cond in parsed.conditions:
            if cond.field.startswith("tag.") or cond.field.startswith("tags."):
                tag_conditions.append(cond)

        candidate_ids: set[str] | None = None

        # Use tag index if we have tag conditions
        if tag_conditions:
            with self._get_connection() as conn:
                for cond in tag_conditions:
                    key = cond.field.split(".", 1)[1]
                    if cond.op.value == "=":
                        rows = conn.execute(
                            "SELECT run_id FROM run_tags WHERE key = ? AND value = ?",
                            (key, str(cond.value)),
                        ).fetchall()
                    elif cond.op.value == "~":
                        rows = conn.execute(
                            "SELECT run_id FROM run_tags WHERE key = ? AND value LIKE ?",
                            (key, f"%{cond.value}%"),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT run_id FROM run_tags WHERE key = ?",
                            (key,),
                        ).fetchall()

                    ids = {row["run_id"] for row in rows}
                    if candidate_ids is None:
                        candidate_ids = ids
                    else:
                        candidate_ids &= ids

        def _ordered_run_ids_by_created_at(ids: set[str]) -> list[str]:
            """Fetch created_at for ids (chunked), then sort globally by created_at DESC."""
            if not ids:
                return []

            pairs: list[tuple[str, str]] = []
            id_list = list(ids)

            with self._get_connection() as conn:
                for i in range(0, len(id_list), _SQLITE_MAX_VARIABLE_NUMBER):
                    chunk = id_list[i : i + _SQLITE_MAX_VARIABLE_NUMBER]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT run_id, created_at FROM runs WHERE run_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        created_at = row["created_at"] or ""
                        run_id = row["run_id"]
                        pairs.append((created_at, run_id))

            pairs.sort(key=lambda t: t[0], reverse=True)
            return [run_id for _, run_id in pairs]

        def _iter_run_ids_in_default_order(batch_size: int = 100):
            """Yield run_ids in default order (created_at DESC)."""
            if candidate_ids is not None:
                ordered = _ordered_run_ids_by_created_at(candidate_ids)
                for run_id in ordered:
                    yield run_id
                return

            # No candidate restriction: page through runs
            current_offset = 0
            while True:
                summaries = self.list_runs(limit=batch_size, offset=current_offset)
                if not summaries:
                    break
                for s in summaries:
                    yield s["run_id"]
                current_offset += len(summaries)

        # If no explicit sort_by: stream and stop early after offset+limit matches
        if sort_by is None:
            results: list[RunRecord] = []
            matched = 0
            for run_id in _iter_run_ids_in_default_order():
                try:
                    record = self.load(run_id)
                    if matches_query(record.record, parsed):
                        if matched >= offset:
                            results.append(record)
                            if len(results) >= limit:
                                break
                        matched += 1
                except Exception:
                    continue
            return results

        # If sort_by requested: collect all matches, then sort, then slice
        all_matches: list[RunRecord] = []
        for run_id in _iter_run_ids_in_default_order():
            try:
                record = self.load(run_id)
                if matches_query(record.record, parsed):
                    all_matches.append(record)
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
        List unique run groups.

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
            Group summaries with group_id, group_name, project, run_count,
            first_run, and last_run timestamps.
        """
        conditions: list[str] = ["group_id IS NOT NULL"]
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        params.extend([limit, offset])

        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT run_id, project, adapter, status, created_at, ended_at,
                       group_id, group_name, parent_run_id
                FROM runs
                WHERE group_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (group_id, limit, offset),
            ).fetchall()

        return [
            RunSummary(
                run_id=row["run_id"],
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
            Sorted list of project names.
        """
        with self._get_connection() as conn:
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
        conditions: list[str] = []
        params: list[Any] = []

        if project:
            conditions.append("project = ?")
            params.append(project)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM runs {where_clause}", params
            ).fetchone()
        return row["cnt"] if row else 0

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
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO baselines (project, run_id, set_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project) DO UPDATE SET
                    run_id = excluded.run_id,
                    set_at = excluded.set_at
                """,
                (project, run_id, utc_now_iso()),
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
            Baseline info, or None if no baseline set.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT project, run_id, set_at FROM baselines WHERE project = ?",
                (project,),
            ).fetchone()

        if not row:
            return None

        return BaselineInfo(
            project=row["project"],
            run_id=row["run_id"],
            set_at=row["set_at"],
        )

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
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM baselines WHERE project = ?", (project,))

        if cursor.rowcount > 0:
            logger.info("Cleared baseline for %s", project)
            return True
        return False


class LocalWorkspace:
    """
    Combined local storage workspace with store and registry.

    Provides a convenient way to create both an object store and
    registry rooted at the same directory.

    Parameters
    ----------
    root : Path
        Root directory for the workspace.

    Examples
    --------
    >>> workspace = LocalWorkspace(Path("/tmp/devqubit"))
    >>> workspace.store.put_bytes(b"data")
    'sha256:...'
    >>> workspace.registry.save({"run_id": "...", ...})
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._store = LocalStore(self.root / "objects")
        self._registry = LocalRegistry(self.root)
        logger.debug("LocalWorkspace initialized at %s", self.root)

    @property
    def store(self) -> LocalStore:
        """
        Get the object store.

        Returns
        -------
        LocalStore
            Local object store instance.
        """
        return self._store

    @property
    def registry(self) -> LocalRegistry:
        """
        Get the registry.

        Returns
        -------
        LocalRegistry
            Local registry instance.
        """
        return self._registry

    def close(self) -> None:
        """Close the workspace (no-op for local storage)."""
        pass

    def __enter__(self) -> LocalWorkspace:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LocalWorkspace(root={self.root!r})"
