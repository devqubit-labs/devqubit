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

from devqubit_engine.storage.protocols import ObjectNotFoundError


logger = logging.getLogger(__name__)


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
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"Invalid digest format: {digest!r}")

        hex_part = digest[7:].strip().lower()
        if len(hex_part) != 64:
            raise ValueError(f"Invalid digest length: {digest!r}")

        try:
            int(hex_part, 16)
        except ValueError as e:
            raise ValueError(f"Invalid digest hex: {digest!r}") from e

        path = f"objects/sha256/{hex_part[:2]}/{hex_part}"

        try:
            return self._zf.read(path)
        except KeyError as e:
            raise ObjectNotFoundError(f"sha256:{hex_part}") from e

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
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            return False

        hex_part = digest[7:].strip().lower()
        if len(hex_part) != 64:
            return False

        try:
            int(hex_part, 16)
        except ValueError:
            return False

        path = f"objects/sha256/{hex_part[:2]}/{hex_part}"
        return path in self._names

    def list_objects(self) -> Iterator[str]:
        """
        Iterate over all stored object digests in the bundle.

        Yields
        ------
        str
            Digests in the form ``sha256:<64 hex chars>``.
        """
        for name in self._names:
            if name.startswith("objects/sha256/") and len(name.split("/")) == 4:
                hex_part = name.split("/")[-1].strip().lower()
                if len(hex_part) != 64:
                    continue
                try:
                    int(hex_part, 16)
                except ValueError:
                    continue
                yield f"sha256:{hex_part}"


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
            if self._zf is None:
                raise RuntimeError("Bundle not open")
            self._manifest = json.loads(self._zf.read("manifest.json").decode("utf-8"))
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
            if self._zf is None:
                raise RuntimeError("Bundle not open")
            self._run_record = json.loads(self._zf.read("run.json").decode("utf-8"))
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
        return list(self.store.list_objects())

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
