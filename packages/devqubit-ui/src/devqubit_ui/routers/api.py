# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""JSON API router for React frontend."""

from __future__ import annotations

import json
import logging
from typing import Any

from devqubit_ui.dependencies import RegistryDep, StoreDep
from devqubit_ui.services import (
    ArtifactService,
    DiffService,
    GroupService,
    ProjectService,
    RunService,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Capabilities
# =============================================================================


@router.get("/v1/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Get server capabilities."""
    return {
        "mode": "local",
        "version": "0.1.10",
        "features": {
            "auth": False,
            "workspaces": False,
            "rbac": False,
            "service_accounts": False,
        },
    }


# =============================================================================
# Runs
# =============================================================================


@router.get("/runs")
async def list_runs(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
    q: str = Query("", description="Search query"),
):
    """List runs with optional filtering."""
    service = RunService(registry)

    if q:
        runs_data = service.search_runs(q, limit=limit)
    else:
        runs_data = service.list_runs(
            project=project or None,
            status=status or None,
            limit=limit,
        )

    return JSONResponse(content={"runs": runs_data, "count": len(runs_data)})


@router.get("/runs/{run_id}")
async def get_run(run_id: str, registry: RegistryDep):
    """Get run details."""
    service = RunService(registry)

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    return JSONResponse(content={"run": _record_to_dict(record)})


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, registry: RegistryDep):
    """Delete a run."""
    service = RunService(registry)
    deleted = service.delete_run(run_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")

    return JSONResponse(content={"status": "deleted", "run_id": run_id})


# =============================================================================
# Artifacts
# =============================================================================


@router.get("/runs/{run_id}/artifacts/{idx}")
async def get_artifact(
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
):
    """Get artifact metadata and preview."""
    service = ArtifactService(registry, store)

    try:
        _, artifact = service.get_artifact_metadata(run_id, idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    content_result = service.get_artifact_content(run_id, idx)

    response: dict[str, Any] = {
        "artifact": {
            "kind": artifact.kind,
            "role": artifact.role,
            "media_type": artifact.media_type,
            "digest": artifact.digest,
        },
        "size": content_result.size,
        "preview_available": content_result.preview_available,
        "error": content_result.error,
    }

    if content_result.preview_available and content_result.data:
        if content_result.is_text:
            try:
                content = content_result.data.decode("utf-8")
                response["content"] = content
                if content_result.is_json:
                    response["content_json"] = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

    return JSONResponse(content=response)


@router.get("/runs/{run_id}/artifacts/{idx}/raw")
async def get_artifact_raw(
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
):
    """Download raw artifact."""
    service = ArtifactService(registry, store)

    try:
        data, media_type, filename = service.get_artifact_raw(run_id, idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =============================================================================
# Projects
# =============================================================================


@router.get("/projects")
async def list_projects(registry: RegistryDep):
    """List all projects with stats."""
    service = ProjectService(registry)
    projects = service.list_projects_with_stats()
    return JSONResponse(content={"projects": projects})


@router.post("/projects/{project}/baseline/{run_id}")
async def set_baseline(
    project: str,
    run_id: str,
    registry: RegistryDep,
    redirect: bool = Query(False),
):
    """Set project baseline."""
    service = RunService(registry)

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    if record.project != project:
        raise HTTPException(
            status_code=400,
            detail=f"Run belongs to '{record.project}', not '{project}'",
        )

    service.set_baseline(project, run_id)
    return JSONResponse(
        content={"status": "ok", "project": project, "baseline_run_id": run_id}
    )


# =============================================================================
# Groups
# =============================================================================


@router.get("/groups")
async def list_groups(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
):
    """List run groups."""
    service = GroupService(registry)
    groups = service.list_groups(project=project or None)

    # Convert to serializable dicts
    groups_data = []
    for g in groups:
        if hasattr(g, "__dict__"):
            groups_data.append(
                {
                    "group_id": getattr(g, "group_id", str(g)),
                    "group_name": getattr(g, "group_name", None),
                    "project": getattr(g, "project", None),
                    "run_count": getattr(g, "run_count", 0),
                }
            )
        elif isinstance(g, dict):
            groups_data.append(g)
        else:
            groups_data.append({"group_id": str(g)})

    return JSONResponse(content={"groups": groups_data})


@router.get("/groups/{group_id}")
async def get_group(group_id: str, registry: RegistryDep):
    """Get group runs."""
    service = GroupService(registry)
    runs = service.get_group_runs(group_id)

    runs_data = []
    for r in runs:
        if hasattr(r, "run_id"):
            runs_data.append(
                {
                    "run_id": r.run_id,
                    "run_name": getattr(r, "run_name", None),
                    "project": getattr(r, "project", None),
                    "status": getattr(r, "status", "UNKNOWN"),
                    "created_at": str(getattr(r, "created_at", "")),
                }
            )
        elif isinstance(r, dict):
            runs_data.append(r)

    return JSONResponse(content={"group_id": group_id, "runs": runs_data})


# =============================================================================
# Diff
# =============================================================================


@router.get("/diff")
async def get_diff(
    registry: RegistryDep,
    store: StoreDep,
    a: str = Query(..., description="Run A ID"),
    b: str = Query(..., description="Run B ID"),
):
    """Compare two runs."""
    diff_service = DiffService(registry, store)

    try:
        record_a, record_b, report = diff_service.compare_runs(a, b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return JSONResponse(
        content={
            "run_a": _record_to_summary(record_a),
            "run_b": _record_to_summary(record_b),
            "report": report,
        }
    )


# =============================================================================
# Helpers
# =============================================================================


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert RunRecord to full dict."""
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "ended_at": record.ended_at,
        "fingerprints": record.fingerprints,
        "group_id": record.group_id,
        "group_name": record.group_name,
        "backend": record.record.get("backend", {}),
        "data": record.record.get("data", {}),
        "artifacts": [
            {
                "kind": a.kind,
                "role": a.role,
                "media_type": a.media_type,
                "digest": a.digest,
            }
            for a in (record.artifacts or [])
        ],
        "errors": record.record.get("info", {}).get("errors", []),
    }


def _record_to_summary(record: Any) -> dict[str, Any]:
    """Convert RunRecord to summary dict."""
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
    }
