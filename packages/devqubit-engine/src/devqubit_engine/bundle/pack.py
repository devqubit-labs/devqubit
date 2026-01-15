# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Pack and unpack operations for devqubit bundles.

This module provides functions for creating portable bundle archives from
runs and restoring them to a workspace. Bundles are self-contained ZIP
archives that include run metadata and all referenced artifact objects.

Bundle Format
-------------
A devqubit bundle is a ZIP archive containing:

- ``manifest.json`` - Bundle metadata and object inventory
- ``run.json`` - Complete run record
- ``objects/sha256/<xx>/<hash>`` - Content-addressed artifact objects

Examples
--------
>>> from devqubit_engine.bundle.pack import pack_run, unpack_bundle

>>> # Pack a run into a bundle
>>> result = pack_run("01HX...", "/path/to/output.zip")
>>> print(f"Packed {result.object_count} objects")

>>> # Unpack a bundle into workspace
>>> result = unpack_bundle("/path/to/bundle.zip")
>>> print(f"Restored run {result.run_id}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.protocols import ObjectStoreProtocol, RegistryProtocol
from devqubit_engine.utils.common import utc_now_iso


logger = logging.getLogger(__name__)

# Bundle format version
BUNDLE_FORMAT_VERSION = "devqubit.bundle/0.1"


@dataclass
class PackResult:
    """
    Result of a pack operation.

    Attributes
    ----------
    run_id : str
        Run identifier that was packed.
    bundle_path : Path
        Output bundle file path.
    artifact_count : int
        Number of artifacts in the run record.
    object_count : int
        Number of unique objects written to the bundle.
    missing_objects : list of str
        Digests of objects that could not be found in the store.
    """

    run_id: str
    bundle_path: Path
    artifact_count: int
    object_count: int
    missing_objects: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PackResult(run_id={self.run_id!r}, "
            f"objects={self.object_count}, missing={len(self.missing_objects)})"
        )


@dataclass
class UnpackResult:
    """
    Result of an unpack operation.

    Attributes
    ----------
    run_id : str
        Run identifier that was unpacked.
    artifact_count : int
        Number of artifacts in the run record.
    object_count : int
        Number of new objects written to the store.
    """

    run_id: str
    artifact_count: int
    object_count: int

    def __repr__(self) -> str:
        """Return string representation."""
        return f"UnpackResult(run_id={self.run_id!r}, objects={self.object_count})"


def pack_run(
    run_id: str,
    output_path: str | Path,
    *,
    store: ObjectStoreProtocol | None = None,
    registry: RegistryProtocol | None = None,
    strict: bool = False,
) -> PackResult:
    """
    Pack a run into a portable bundle.

    Creates a self-contained ZIP archive with all run metadata and
    referenced artifact objects.

    Parameters
    ----------
    run_id : str
        Run identifier to pack.
    output_path : str or Path
        Output bundle file path.
    store : ObjectStoreProtocol, optional
        Object store to read artifacts from. Uses default if not provided.
    registry : RegistryProtocol, optional
        Registry to read run record from. Uses default if not provided.
    strict : bool, optional
        If True, fail if any referenced artifact object is missing.
        Default is False (missing objects are recorded but don't fail).

    Returns
    -------
    PackResult
        Pack operation result with statistics.

    Raises
    ------
    FileNotFoundError
        If strict=True and any referenced objects are missing.
    RunNotFoundError
        If the run does not exist in the registry.

    Notes
    -----
    The bundle is written atomically: data is first written to a temporary
    file, then moved into place. This prevents partially-written bundles
    if the operation is interrupted.
    """
    store = store or create_store()
    registry = registry or create_registry()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Packing run %s to %s", run_id, output_path)

    # Load run record
    run_record = registry.load(run_id)
    record = run_record.to_dict()
    artifacts = record.get("artifacts", []) or []
    if not isinstance(artifacts, list):
        artifacts = []

    # Collect unique digests referenced by artifacts
    digests: list[str] = []
    seen: set[str] = set()
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        digest = art.get("digest", "")
        if (
            isinstance(digest, str)
            and digest.startswith("sha256:")
            and digest not in seen
        ):
            seen.add(digest)
            digests.append(digest)

    logger.debug(
        "Found %d unique digests in %d artifacts", len(digests), len(artifacts)
    )

    # Preflight missing objects in strict mode
    missing: list[str] = []
    if strict:
        for d in digests:
            try:
                if not store.exists(d):
                    missing.append(d)
            except Exception:
                missing.append(d)
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} objects while packing run {run_id} "
                f"(first: {missing[0]})"
            )

    # Atomic write via temp file
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp.zip",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    written: list[str] = []
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write run record
            zf.writestr("run.json", json.dumps(record, indent=2, default=str))

            # Write referenced objects
            for digest in digests:
                hex_part = digest[len("sha256:") :]
                obj_path = f"objects/sha256/{hex_part[:2]}/{hex_part}"

                try:
                    data = store.get_bytes(digest)
                except Exception as e:
                    logger.warning("Missing object %s: %s", digest[:24], e)
                    missing.append(digest)
                    continue

                zf.writestr(obj_path, data)
                written.append(digest)

            # Build manifest
            manifest = _build_manifest(record, run_id, written, missing)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Atomic move into place
        os.replace(tmp_path, output_path)
        logger.info(
            "Packed run %s: %d objects, %d missing",
            run_id,
            len(written),
            len(missing),
        )

    finally:
        # Clean up temp file if it still exists
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return PackResult(
        run_id=run_id,
        bundle_path=output_path,
        artifact_count=len(artifacts),
        object_count=len(written),
        missing_objects=missing,
    )


def _build_manifest(
    record: dict[str, Any],
    run_id: str,
    written: list[str],
    missing: list[str],
) -> dict[str, Any]:
    """
    Build bundle manifest from record and object lists.

    Parameters
    ----------
    record : dict
        Run record dictionary.
    run_id : str
        Run identifier.
    written : list of str
        Digests that were written to the bundle.
    missing : list of str
        Digests that were missing.

    Returns
    -------
    dict
        Manifest dictionary.
    """
    fingerprints = record.get("fingerprints") or {}
    provenance = record.get("provenance") or {}
    git = provenance.get("git") if isinstance(provenance, dict) else None
    backend = record.get("backend") or {}
    project = record.get("project", {})
    project_name = (
        project.get("name", "") if isinstance(project, dict) else str(project)
    )
    artifacts = record.get("artifacts", []) or []

    return {
        "format": BUNDLE_FORMAT_VERSION,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "project": project_name,
        "adapter": record.get("adapter", ""),
        "backend_name": backend.get("name") if isinstance(backend, dict) else None,
        "git_commit": git.get("commit") if isinstance(git, dict) else None,
        "fingerprint": (
            fingerprints.get("run") if isinstance(fingerprints, dict) else None
        ),
        "program_fingerprint": (
            fingerprints.get("program") if isinstance(fingerprints, dict) else None
        ),
        "artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "object_count": len(written),
        "objects": written,
        "missing_objects": missing,
    }


def unpack_bundle(
    bundle_path: str | Path,
    *,
    dest_store: ObjectStoreProtocol | None = None,
    dest_registry: RegistryProtocol | None = None,
    overwrite: bool = False,
    verify_digests: bool = True,
    skip_existing_objects: bool = True,
) -> UnpackResult:
    """
    Unpack a bundle into a workspace.

    Restores the run record to the registry and artifact objects to
    the object store.

    Parameters
    ----------
    bundle_path : str or Path
        Path to bundle file.
    dest_store : ObjectStoreProtocol, optional
        Destination object store. Uses default if not provided.
    dest_registry : RegistryProtocol, optional
        Destination registry. Uses default if not provided.
    overwrite : bool, optional
        If True, overwrite existing run record. Default is False.
    verify_digests : bool, optional
        If True, verify object digests after reading. Default is True.
    skip_existing_objects : bool, optional
        If True, skip objects that already exist in store. Default is True.

    Returns
    -------
    UnpackResult
        Unpack operation result with statistics.

    Raises
    ------
    FileExistsError
        If run already exists and overwrite=False.
    ValueError
        If digest verification fails or run_id is missing from bundle.
    FileNotFoundError
        If bundle file does not exist.
    """
    store = dest_store or create_store()
    registry = dest_registry or create_registry()
    bundle_path = Path(bundle_path)

    logger.info("Unpacking bundle %s", bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        record = json.loads(zf.read("run.json").decode("utf-8"))
        run_id = record.get("run_id")

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Bundle run.json is missing a valid 'run_id'")

        if not overwrite and registry.exists(run_id):
            raise FileExistsError(f"Run {run_id} already exists (use overwrite=True)")

        artifacts = record.get("artifacts", []) or []
        if not isinstance(artifacts, list):
            artifacts = []

        obj_count = 0
        seen: set[str] = set()

        for art in artifacts:
            if not isinstance(art, dict):
                continue
            digest = art.get("digest", "")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                continue

            hex_part = digest[7:].strip().lower()
            if len(hex_part) != 64:
                logger.warning("Invalid digest length in run record: %r", digest)
                continue
            try:
                int(hex_part, 16)
            except ValueError:
                logger.warning("Invalid digest hex in run record: %r", digest)
                continue

            digest = f"sha256:{hex_part}"

            # Dedupe - same digest may be referenced by multiple artifacts
            if digest in seen:
                continue
            seen.add(digest)

            # Skip if already in store
            if skip_existing_objects:
                try:
                    if store.exists(digest):
                        logger.debug("Skipping existing object: %s", digest[:24])
                        continue
                except Exception:
                    pass

            obj_path = f"objects/sha256/{hex_part[:2]}/{hex_part}"

            try:
                data = zf.read(obj_path)
            except KeyError:
                logger.warning("Object not in bundle: %s", digest[:24])
                continue

            if verify_digests:
                actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
                if actual != digest:
                    raise ValueError(
                        f"Digest mismatch: expected {digest}, got {actual}"
                    )

            stored_digest = store.put_bytes(data)
            if verify_digests and stored_digest != digest:
                raise ValueError(
                    f"Store returned different digest: expected {digest}, "
                    f"got {stored_digest}"
                )

            obj_count += 1

        registry.save(record)

    logger.info("Unpacked run %s: %d new objects", run_id, obj_count)

    return UnpackResult(
        run_id=run_id,
        artifact_count=len(artifacts),
        object_count=obj_count,
    )


def list_bundle_contents(bundle_path: str | Path) -> dict[str, Any]:
    """
    List bundle contents without extracting.

    Parameters
    ----------
    bundle_path : str or Path
        Path to bundle file.

    Returns
    -------
    dict
        Bundle contents summary including:

        - ``manifest`` - Full manifest dictionary
        - ``run_id`` - Run identifier
        - ``project`` - Project name
        - ``adapter`` - Adapter name
        - ``artifact_count`` - Number of artifacts
        - ``objects`` - List of object digests
        - ``fingerprint`` - Run fingerprint
        - ``git_commit`` - Git commit SHA
    """
    bundle_path = Path(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        record = json.loads(zf.read("run.json").decode("utf-8"))

        # Discover objects from ZIP contents
        objects: list[str] = []
        for name in zf.namelist():
            if name.startswith("objects/sha256/") and len(name.split("/")) == 4:
                hex_part = name.split("/")[-1]
                if len(hex_part) == 64:
                    objects.append(f"sha256:{hex_part}")

        project = record.get("project", {})
        project_name = (
            project.get("name", "") if isinstance(project, dict) else str(project)
        )

        return {
            "manifest": manifest,
            "run_id": record.get("run_id", ""),
            "project": project_name,
            "adapter": record.get("adapter", ""),
            "artifact_count": len(record.get("artifacts", []) or []),
            "objects": objects,
            "fingerprint": manifest.get("fingerprint"),
            "git_commit": manifest.get("git_commit"),
        }
