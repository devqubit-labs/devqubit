# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Bundle reading and inspection.

This module provides lightweight helpers to detect and read devqubit bundle
files without extracting them. A bundle is a ZIP archive that includes
required metadata files (``manifest.json``, ``run.json``) and a
content-addressed object store under ``objects/sha256/``.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterator

from devqubit_engine.storage.errors import ObjectNotFoundError


logger = logging.getLogger(__name__)

# Maximum allowed uncompressed size for metadata files (10 MB)
_MAX_METADATA_SIZE = 10 * 1024 * 1024

# Maximum allowed uncompressed size for objects (1 GB)
_MAX_OBJECT_SIZE = 1024 * 1024 * 1024


def validate_digest(digest: str) -> str | None:
    """
    Validate digest format and return hex part.

    Parameters
    ----------
    digest : str
        Digest to validate.

    Returns
    -------
    str or None
        Lowercase hex part if valid, None otherwise.
    """
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None

    hex_part = digest[7:].strip().lower()
    if len(hex_part) != 64:
        return None

    try:
        int(hex_part, 16)
    except ValueError:
        return None

    return hex_part


def digest_to_path(hex_part: str) -> str:
    """Convert hex digest to bundle object path."""
    return f"objects/sha256/{hex_part[:2]}/{hex_part}"


def is_bundle_path(path: Any) -> bool:
    """
    Check if a path points to a devqubit bundle.

    A devqubit bundle is detected by content (not extension). The file must:

    - Exist and be a regular file
    - Be a valid ZIP archive
    - Contain both ``manifest.json`` and ``run.json`` at the archive root

    Parameters
    ----------
    path : Any
        Candidate path-like value. Only ``str`` and ``pathlib.Path`` are
        considered valid inputs.

    Returns
    -------
    bool
        True if the file appears to be a devqubit bundle, otherwise False.
    """
    if not isinstance(path, (str, Path)):
        return False

    p = Path(path)
    if not (p.exists() and p.is_file()):
        return False

    try:
        if not zipfile.is_zipfile(p):
            return False
        with zipfile.ZipFile(p, "r") as zf:
            names = set(zf.namelist())
        return {"manifest.json", "run.json"}.issubset(names)
    except (zipfile.BadZipFile, OSError):
        return False


class BundleStore:
    """
    Read-only content-addressed object store backed by a ZIP bundle.

    Objects are stored under ``objects/sha256/<prefix>/<hex>`` and addressed
    by ``sha256:<hex>`` digests.

    This class provides a partial implementation of ObjectStoreProtocol,
    supporting read-only operations only.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZIP file handle for the bundle (read mode).

    Notes
    -----
    This class caches the ZIP namelist for O(1) existence checks.
    The ZipFile handle must remain open for the lifetime of this object.

    Examples
    --------
    >>> with zipfile.ZipFile("bundle.zip", "r") as zf:
    ...     store = BundleStore(zf)
    ...     if store.exists("sha256:abc..."):
    ...         data = store.get_bytes("sha256:abc...")
    """

    def __init__(self, zf: zipfile.ZipFile) -> None:
        self._zf = zf
        # Cache namelist for O(1) exists() checks
        self._names = frozenset(zf.namelist())
        # Cache ZipInfo for size lookups
        self._info_cache: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            self._info_cache[info.filename] = info

    def get_bytes(self, digest: str) -> bytes:
        """
        Retrieve raw bytes for an object by digest.

        Parameters
        ----------
        digest : str
            Object identifier in the form ``sha256:<64 hex chars>``.

        Returns
        -------
        bytes
            Raw object bytes.

        Raises
        ------
        ValueError
            If digest format is invalid.
        ObjectNotFoundError
            If the object is not present in the bundle.
        """
        hex_part = validate_digest(digest)
        if hex_part is None:
            raise ValueError(f"Invalid digest format: {digest!r}")

        path = digest_to_path(hex_part)

        # Check for zip bomb before reading
        info = self._info_cache.get(path)
        if info is not None and info.file_size > _MAX_OBJECT_SIZE:
            raise ValueError(
                f"Object too large: {info.file_size} bytes " f"(max {_MAX_OBJECT_SIZE})"
            )

        try:
            return self._zf.read(path)
        except KeyError as e:
            raise ObjectNotFoundError(f"sha256:{hex_part}") from e

    def get_bytes_or_none(self, digest: str) -> bytes | None:
        """
        Retrieve bytes by digest, returning None if not found.

        Parameters
        ----------
        digest : str
            Object identifier in the form ``sha256:<64 hex chars>``.

        Returns
        -------
        bytes or None
            Raw object bytes, or None if not found or invalid digest.
        """
        try:
            return self.get_bytes(digest)
        except (ObjectNotFoundError, ValueError):
            return None

    def exists(self, digest: str) -> bool:
        """
        Check if an object exists in the bundle.

        Parameters
        ----------
        digest : str
            Object identifier in the form ``sha256:<64 hex chars>``.

        Returns
        -------
        bool
            True if the object exists in the bundle.
            Invalid digests return False.
        """
        hex_part = validate_digest(digest)
        if hex_part is None:
            return False

        path = digest_to_path(hex_part)
        return path in self._names

    def get_size(self, digest: str) -> int:
        """
        Get the uncompressed size of an object.

        Parameters
        ----------
        digest : str
            Object identifier in the form ``sha256:<64 hex chars>``.

        Returns
        -------
        int
            Uncompressed size in bytes.

        Raises
        ------
        ValueError
            If digest format is invalid.
        ObjectNotFoundError
            If the object is not present in the bundle.
        """
        hex_part = validate_digest(digest)
        if hex_part is None:
            raise ValueError(f"Invalid digest format: {digest!r}")

        path = digest_to_path(hex_part)
        info = self._info_cache.get(path)

        if info is None:
            raise ObjectNotFoundError(f"sha256:{hex_part}")

        return info.file_size

    def list_digests(self, prefix: str | None = None) -> Iterator[str]:
        """
        Iterate over all stored object digests in the bundle.

        Parameters
        ----------
        prefix : str, optional
            Filter by digest prefix (e.g., "sha256:ab").

        Yields
        ------
        str
            Digests in the form ``sha256:<64 hex chars>``.
        """
        for name in self._names:
            if not name.startswith("objects/sha256/"):
                continue

            parts = name.split("/")
            if len(parts) != 4:
                continue

            hex_part = parts[-1].strip().lower()
            if len(hex_part) != 64:
                continue

            try:
                int(hex_part, 16)
            except ValueError:
                logger.debug("Skipping non-hex digest: %s", hex_part[:24])
                continue

            digest = f"sha256:{hex_part}"

            if prefix is not None and not digest.startswith(prefix):
                continue

            yield digest

    # Alias for compatibility
    list_objects = list_digests


class Bundle:
    """
    Reader for devqubit bundle (.zip) files.

    Provides context manager interface for safe resource management.
    Lazily loads manifest and run record on first access.

    Parameters
    ----------
    path : str or Path
        Path to the bundle file.

    Examples
    --------
    >>> with Bundle("my_run.zip") as bundle:
    ...     print(f"Run: {bundle.run_id}")
    ...     print(f"Adapter: {bundle.manifest.get('adapter')}")
    ...     for digest in bundle.list_objects():
    ...         data = bundle.store.get_bytes(digest)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zf: zipfile.ZipFile | None = None
        self._manifest: dict[str, Any] | None = None
        self._run_record: dict[str, Any] | None = None
        self._store: BundleStore | None = None

    def open(self) -> Bundle:
        """
        Open the bundle for reading and validate required files.

        Returns
        -------
        Bundle
            This instance (for fluent usage).

        Raises
        ------
        FileNotFoundError
            If the bundle file does not exist.
        ValueError
            If the file is not a valid devqubit bundle.
        """
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        self._zf = zipfile.ZipFile(self.path, "r")
        names = set(self._zf.namelist())

        if not {"manifest.json", "run.json"}.issubset(names):
            self._zf.close()
            self._zf = None
            raise ValueError(
                f"Not a devqubit bundle: {self.path} "
                "(missing manifest.json or run.json)"
            )

        logger.debug("Opened bundle: %s", self.path)
        return self

    def close(self) -> None:
        """Close the underlying ZIP file and clear cached state."""
        if self._zf:
            self._zf.close()
            self._zf = None
            logger.debug("Closed bundle: %s", self.path)

        self._manifest = None
        self._run_record = None
        self._store = None

    def __enter__(self) -> Bundle:
        """Open the bundle for context manager usage."""
        return self.open()

    def __exit__(self, *args: Any) -> None:
        """Close the bundle when exiting context manager."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        status = "open" if self._zf else "closed"
        return f"Bundle({self.path!r}, {status})"

    def _read_json(self, filename: str, max_size: int = _MAX_METADATA_SIZE) -> Any:
        """
        Read and parse a JSON file from the bundle.

        Parameters
        ----------
        filename : str
            Name of file in bundle.
        max_size : int
            Maximum allowed uncompressed size.

        Returns
        -------
        Any
            Parsed JSON content.

        Raises
        ------
        RuntimeError
            If bundle is not open.
        ValueError
            If file is too large or not valid JSON.
        """
        if self._zf is None:
            raise RuntimeError("Bundle not open")

        # Check size before reading
        try:
            info = self._zf.getinfo(filename)
            if info.file_size > max_size:
                raise ValueError(
                    f"{filename} too large: {info.file_size} bytes (max {max_size})"
                )
        except KeyError as e:
            raise ValueError(f"Missing required file: {filename}") from e

        data = self._zf.read(filename)
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}") from e

    @property
    def manifest(self) -> dict[str, Any]:
        """
        Bundle manifest loaded from ``manifest.json``.

        Returns
        -------
        dict
            Parsed manifest containing format version, run metadata,
            and object inventory.

        Raises
        ------
        RuntimeError
            If the bundle is not open.
        """
        if self._manifest is None:
            self._manifest = self._read_json("manifest.json")
        return self._manifest

    @property
    def run_record(self) -> dict[str, Any]:
        """
        Run record loaded from ``run.json``.

        Returns
        -------
        dict
            Complete run record including artifacts, metrics, etc.

        Raises
        ------
        RuntimeError
            If the bundle is not open.
        """
        if self._run_record is None:
            self._run_record = self._read_json("run.json")
        return self._run_record

    @property
    def run_id(self) -> str:
        """
        Run identifier from the manifest.

        Returns
        -------
        str
            Run ID, or empty string if not present.
        """
        return str(self.manifest.get("run_id", ""))

    @property
    def store(self) -> BundleStore:
        """
        Read-only object store for content-addressed artifacts.

        Returns
        -------
        BundleStore
            Store view over the open bundle.

        Raises
        ------
        RuntimeError
            If the bundle is not open.
        """
        if self._zf is None:
            raise RuntimeError("Bundle not open")
        if self._store is None:
            self._store = BundleStore(self._zf)
        return self._store

    def list_objects(self) -> list[str]:
        """
        List all object digests stored in the bundle.

        Returns
        -------
        list of str
            Digests in the form ``sha256:<64 hex chars>``.
        """
        return list(self.store.list_digests())

    def get_artifact_kinds(self) -> list[str]:
        """
        Get artifact kinds declared in the run record.

        Returns
        -------
        list of str
            Values of ``artifact["kind"]`` for each artifact entry.
            Returns empty list if artifacts section is missing or invalid.
        """
        arts = self.run_record.get("artifacts", []) or []
        if not isinstance(arts, list):
            return []
        return [a.get("kind", "") for a in arts if isinstance(a, dict)]

    def get_project(self) -> str:
        """
        Get project name from run record.

        Returns
        -------
        str
            Project name, or empty string if not present.
        """
        project = self.run_record.get("project", {})
        if isinstance(project, dict):
            return project.get("name", "")
        return str(project) if project else ""

    def get_adapter(self) -> str:
        """
        Get adapter name from run record.

        Returns
        -------
        str
            Adapter name, or empty string if not present.
        """
        return self.run_record.get("adapter", "")

    def get_status(self) -> str:
        """
        Get run status from run record.

        Returns
        -------
        str
            Run status (RUNNING, FINISHED, FAILED, KILLED),
            or "UNKNOWN" if not present.
        """
        info = self.run_record.get("info", {})
        if isinstance(info, dict):
            return info.get("status", "UNKNOWN")
        return "UNKNOWN"

    def get_created_at(self) -> str | None:
        """
        Get run creation timestamp.

        Returns
        -------
        str or None
            ISO 8601 timestamp, or None if not present.
        """
        return self.run_record.get("created_at")

    def validate_objects(self) -> tuple[list[str], list[str]]:
        """
        Validate that all artifact digests exist in bundle.

        Returns
        -------
        tuple of (list, list)
            (present_digests, missing_digests)
        """
        present: list[str] = []
        missing: list[str] = []

        arts = self.run_record.get("artifacts", []) or []
        if not isinstance(arts, list):
            return present, missing

        for art in arts:
            if not isinstance(art, dict):
                continue
            digest = art.get("digest", "")
            if not digest:
                continue

            if self.store.exists(digest):
                present.append(digest)
            else:
                missing.append(digest)

        return present, missing
