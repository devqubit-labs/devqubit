# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Run export API endpoints."""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devqubit_engine.bundle.pack import BUNDLE_FORMAT_VERSION
from devqubit_ui.dependencies import RegistryDep, StoreDep
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse


logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class ExportResult:
    """Result of export operation."""

    run_id: str
    artifact_count: int
    object_count: int
    missing_objects: list[str]


# In-memory cache for prepared bundles (in production, use Redis or similar)
_bundle_cache: dict[str, Path] = {}


def _utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _build_manifest(
    record: dict[str, Any],
    run_id: str,
    written: list[str],
    missing: list[str],
) -> dict[str, Any]:
    """Build bundle manifest from record."""
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


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert RunRecord to exportable dict."""
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if hasattr(record, "record"):
        return record.record
    if isinstance(record, dict):
        return record
    raise ValueError(f"Cannot convert record to dict: {type(record)}")


@router.post("/runs/{run_id}/export")
async def create_export(
    run_id: str,
    registry: RegistryDep,
    store: StoreDep,
    include_artifacts: bool = True,
):
    """
    Create a run export bundle.

    Prepares a portable ZIP archive containing the run record
    and all referenced artifact objects.
    """
    logger.info("Creating export bundle for run %s", run_id)

    # Load run record
    try:
        run_record = registry.load(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")

    record = _record_to_dict(run_record)
    artifacts = record.get("artifacts", []) or []
    if not isinstance(artifacts, list):
        artifacts = []

    # Collect unique digests
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

    # Create temporary bundle file
    temp_dir = Path(tempfile.gettempdir()) / "devqubit_exports"
    temp_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = temp_dir / f"{run_id}.zip"

    written: list[str] = []
    missing: list[str] = []

    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write run record
            zf.writestr("run.json", json.dumps(record, indent=2, default=str))

            # Write artifact objects
            for digest in digests:
                hex_part = digest[len("sha256:") :]
                obj_path = f"objects/sha256/{hex_part[:2]}/{hex_part}"

                try:
                    data = store.get_bytes(digest)
                    zf.writestr(obj_path, data)
                    written.append(digest)
                except Exception as e:
                    logger.warning("Missing object %s: %s", digest[:24], e)
                    missing.append(digest)

            # Build and write manifest
            manifest = _build_manifest(record, run_id, written, missing)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Cache the bundle path
        _bundle_cache[run_id] = bundle_path

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

    except Exception as e:
        logger.error("Failed to create export bundle: %s", e)
        if bundle_path.exists():
            bundle_path.unlink()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/runs/{run_id}/export/download")
async def download_export(run_id: str, registry: RegistryDep):
    """
    Download a previously created export bundle.
    """
    # Check if bundle exists in cache
    bundle_path = _bundle_cache.get(run_id)

    if bundle_path is None or not bundle_path.exists():
        # Try to find existing bundle or create new one
        temp_dir = Path(tempfile.gettempdir()) / "devqubit_exports"
        bundle_path = temp_dir / f"{run_id}.zip"

        if not bundle_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Export bundle not found. Please create export first.",
            )

    # Get run name for filename
    try:
        run_record = registry.load(run_id)
        run_name = getattr(run_record, "run_name", None) or run_id[:8]
    except Exception:
        run_name = run_id[:8]

    # Sanitize filename
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
async def get_export_info(run_id: str, registry: RegistryDep, store: StoreDep):
    """
    Get information about what would be exported without creating the bundle.
    """
    try:
        run_record = registry.load(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")

    record = _record_to_dict(run_record)
    artifacts = record.get("artifacts", []) or []

    # Count unique objects
    digests = set()
    for art in artifacts:
        if isinstance(art, dict):
            digest = art.get("digest", "")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                digests.add(digest)

    # Check which objects exist
    available = 0
    missing = 0
    for digest in digests:
        try:
            if store.exists(digest):
                available += 1
            else:
                missing += 1
        except Exception:
            missing += 1

    return JSONResponse(
        content={
            "run_id": run_id,
            "run_name": record.get("run_name"),
            "project": record.get("project"),
            "artifact_count": len(artifacts),
            "object_count": len(digests),
            "available_objects": available,
            "missing_objects": missing,
        }
    )


@router.delete("/runs/{run_id}/export")
async def cleanup_export(run_id: str):
    """
    Clean up a previously created export bundle.
    """
    bundle_path = _bundle_cache.pop(run_id, None)

    if bundle_path is not None and bundle_path.exists():
        try:
            bundle_path.unlink()
            logger.info("Cleaned up export bundle for run %s", run_id)
        except Exception as e:
            logger.warning("Failed to clean up bundle: %s", e)

    return JSONResponse(content={"status": "cleaned", "run_id": run_id})
