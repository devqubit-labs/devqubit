# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Run export API endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devqubit_engine.bundle.pack import BUNDLE_FORMAT_VERSION
from devqubit_ui.dependencies import RegistryDep, StoreDep
from devqubit_ui.services import extract_record_data
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse


logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_TTL_SECONDS = 3600
_DEFAULT_MAX_ENTRIES = 64

_EXPORT_DIR = Path(tempfile.gettempdir()) / "devqubit_exports"

# Allowlist: starts with alnum, then alnum/dash/underscore, max 128 chars.
# Rejects slashes, dots, null bytes, spaces, and any traversal attempt.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _validate_run_id(run_id: str) -> str:
    """Validate run_id against an allowlist pattern.

    Raises HTTPException(400) if the ID contains path separators,
    traversal sequences, or characters outside the allowlist.
    """
    if not _RUN_ID_PATTERN.match(run_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid run_id format",
        )
    return run_id


def _safe_bundle_path(run_id: str) -> Path:
    """
    Build a safe bundle path that never includes user input in the filename.

    Three layers of defense:
    1. Allowlist validation — rejects any run_id with dots, slashes, etc.
    2. Deterministic hash — filename is derived from sha256(run_id), so no
       user-controlled bytes ever reach the filesystem.
    3. Containment check — resolved path must stay inside _EXPORT_DIR.
    """
    _validate_run_id(run_id)

    # Derive filename from hash — completely breaks taint chain.
    # Deterministic: same run_id always maps to same file.
    name = hashlib.sha256(run_id.encode("utf-8")).hexdigest() + ".zip"

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = (_EXPORT_DIR / name).resolve()
    export_root = _EXPORT_DIR.resolve()
    if not bundle_path.parent == export_root:
        # Should never happen with hex digest, but defense in depth.
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    return bundle_path


# ---------------------------------------------------------------------------
# Bundle cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Single entry in the bundle cache."""

    path: Path
    created_at: float


@dataclass
class BundleCache:
    """
    In-process bundle cache with TTL eviction and max-entry cap.

    Attributes
    ----------
    ttl_seconds : int
        Maximum age before an entry is evicted.
    max_entries : int
        Maximum number of bundles kept on disk.
    """

    ttl_seconds: int = _DEFAULT_TTL_SECONDS
    max_entries: int = _DEFAULT_MAX_ENTRIES
    _entries: dict[str, _CacheEntry] = field(default_factory=dict, repr=False)

    def get(self, run_id: str) -> Path | None:
        """Return cached bundle path if present and not expired."""
        entry = self._entries.get(run_id)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self.ttl_seconds:
            self._evict(run_id)
            return None
        if not entry.path.exists():
            self._entries.pop(run_id, None)
            return None
        return entry.path

    def put(self, run_id: str, path: Path) -> None:
        """Store a bundle; evict oldest when limit is exceeded."""
        self._entries[run_id] = _CacheEntry(path=path, created_at=time.monotonic())
        self._enforce_limit()

    def pop(self, run_id: str) -> Path | None:
        """Remove and return a bundle path."""
        entry = self._entries.pop(run_id, None)
        return entry.path if entry else None

    def cleanup(self) -> int:
        """Remove all cached bundles from disk.  Returns files removed."""
        removed = 0
        for entry in self._entries.values():
            try:
                if entry.path.exists():
                    entry.path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("Failed to remove cached bundle: %s", exc)
        self._entries.clear()
        return removed

    def _evict(self, run_id: str) -> None:
        entry = self._entries.pop(run_id, None)
        if entry is not None:
            try:
                if entry.path.exists():
                    entry.path.unlink()
            except OSError as exc:
                logger.debug("Failed to evict bundle for %s: %s", run_id, exc)

    def _enforce_limit(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest_id = min(self._entries, key=lambda k: self._entries[k].created_at)
            self._evict(oldest_id)


bundle_cache = BundleCache()
"""Module-level bundle cache used by export endpoints."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _build_manifest(
    record: dict[str, Any],
    run_id: str,
    written: list[str],
    missing: list[str],
) -> dict[str, Any]:
    """Build a bundle manifest from a run record."""
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
        "created_at": _utc_now_iso(),
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs/{run_id}/export")
async def create_export(
    run_id: str,
    registry: RegistryDep,
    store: StoreDep,
    include_artifacts: bool = True,
) -> JSONResponse:
    """
    Create a portable ZIP bundle for a run.

    The archive contains ``run.json``, a ``manifest.json``, and all
    referenced artifact objects under ``objects/``.
    """
    bundle_path = _safe_bundle_path(run_id)
    logger.info("Creating export bundle for run %s", run_id)

    try:
        run_record = registry.load(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")

    record = extract_record_data(run_record)
    artifacts = record.get("artifacts", []) or []
    if not isinstance(artifacts, list):
        artifacts = []

    digests: list[str] = []
    seen: set[str] = set()
    if include_artifacts:
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

    written: list[str] = []
    missing: list[str] = []

    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("run.json", json.dumps(record, indent=2, default=str))

            for digest in digests:
                hex_part = digest[len("sha256:") :]
                obj_path = f"objects/sha256/{hex_part[:2]}/{hex_part}"
                try:
                    data = store.get_bytes(digest)
                    zf.writestr(obj_path, data)
                    written.append(digest)
                except Exception as exc:
                    logger.warning("Missing object %s: %s", digest[:24], exc)
                    missing.append(digest)

            manifest = _build_manifest(record, run_id, written, missing)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        bundle_cache.put(run_id, bundle_path)

        logger.info(
            "Created export bundle for run %s: %d objects, %d missing",
            run_id,
            len(written),
            len(missing),
        )

        return JSONResponse(
            content={
                "status": "ready",
                "run_id": run_id,
                "artifact_count": len(artifacts),
                "object_count": len(written),
                "missing_objects": missing,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create export bundle: %s", exc)
        if bundle_path.exists():
            bundle_path.unlink()
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")


@router.get("/runs/{run_id}/export/download")
async def download_export(run_id: str, registry: RegistryDep) -> FileResponse:
    """Download a previously created export bundle."""
    bundle_path = bundle_cache.get(run_id)

    if bundle_path is None:
        candidate = _safe_bundle_path(run_id)
        if not candidate.exists():
            raise HTTPException(
                status_code=404,
                detail="Export bundle not found. Create export first.",
            )
        bundle_path = candidate

    try:
        run_record = registry.load(run_id)
        run_name = getattr(run_record, "run_name", None) or run_id[:8]
    except Exception as exc:
        logger.debug("Could not load run name for %s, using short ID: %s", run_id, exc)
        run_name = run_id[:8]

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_name)
    filename = f"{safe_name}_{run_id[:8]}.devqubit.zip"

    return FileResponse(
        path=bundle_path,
        media_type="application/zip",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Run-ID": run_id,
        },
    )


@router.get("/runs/{run_id}/export/info")
async def get_export_info(
    run_id: str,
    registry: RegistryDep,
    store: StoreDep,
) -> JSONResponse:
    """Preview what would be exported without creating the bundle."""
    _validate_run_id(run_id)

    try:
        run_record = registry.load(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")

    record = extract_record_data(run_record)
    artifacts = record.get("artifacts", []) or []

    digests = set()
    for art in artifacts:
        if isinstance(art, dict):
            digest = art.get("digest", "")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                digests.add(digest)

    available = 0
    missing_count = 0
    for digest in digests:
        try:
            if store.exists(digest):
                available += 1
            else:
                missing_count += 1
        except Exception as exc:
            logger.debug("Could not check object %s: %s", digest[:24], exc)
            missing_count += 1

    return JSONResponse(
        content={
            "run_id": run_id,
            "run_name": record.get("run_name"),
            "project": record.get("project"),
            "artifact_count": len(artifacts),
            "object_count": len(digests),
            "available_objects": available,
            "missing_objects": missing_count,
        }
    )


@router.delete("/runs/{run_id}/export")
async def cleanup_export(run_id: str) -> JSONResponse:
    """Remove a previously created export bundle from disk."""
    _validate_run_id(run_id)
    bundle_path = bundle_cache.pop(run_id)
    if bundle_path is not None and bundle_path.exists():
        try:
            bundle_path.unlink()
            logger.info("Cleaned up export bundle for run %s", run_id)
        except Exception as exc:
            logger.warning("Failed to clean up bundle: %s", exc)
    return JSONResponse(content={"status": "cleaned", "run_id": run_id})
