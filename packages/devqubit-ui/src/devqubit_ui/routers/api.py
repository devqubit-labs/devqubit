# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""JSON API router for the React frontend."""

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
    serialize_record,
    serialize_record_summary,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """
    Server capabilities.

    Returns the running mode and available feature flags.  The hub will
    extend this with auth, workspaces, RBAC, etc.
    """
    return {
        "mode": "local",
        "version": "0.1.11",
        "features": {
            "auth": False,
            "workspaces": False,
            "rbac": False,
            "service_accounts": False,
        },
    }


@router.get("/runs")
async def list_runs(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: str = Query("", description="Search query"),
) -> JSONResponse:
    """
    List or search runs.

    Returns a consistent paginated envelope regardless of whether a
    full-text search (``q``) or a plain listing is used.
    """
    service = RunService(registry)

    if q:
        runs = service.search_runs(q, limit=limit)
        return JSONResponse(
            content={
                "runs": runs,
                "count": len(runs),
                "total": len(runs),
                "offset": 0,
                "has_more": False,
            }
        )

    runs = service.list_runs(
        project=project or None,
        status=status or None,
        limit=limit,
        offset=offset,
    )
    total = service.count_runs(
        project=project or None,
        status=status or None,
    )

    return JSONResponse(
        content={
            "runs": runs,
            "count": len(runs),
            "total": total,
            "offset": offset,
            "has_more": offset + len(runs) < total,
        }
    )


@router.get("/runs/{run_id}")
async def get_run(run_id: str, registry: RegistryDep) -> JSONResponse:
    """Get full run details."""
    service = RunService(registry)
    try:
        record = service.get_run(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(content={"run": serialize_record(record)})


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, registry: RegistryDep) -> JSONResponse:
    """Delete a run."""
    service = RunService(registry)
    if not service.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(content={"status": "deleted", "run_id": run_id})


@router.get("/runs/{run_id}/metric_series")
async def get_metric_series(run_id: str, registry: RegistryDep) -> JSONResponse:
    """
    Get metric time-series data for a run.

    Returns all metric keys with their step/value history, suitable for
    plotting. Only available when metrics were logged with ``step``.
    """
    if not hasattr(registry, "load_metric_series"):
        raise HTTPException(
            status_code=501,
            detail="Backend does not support metric series",
        )

    try:
        series = registry.load_metric_series(run_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Run not found")
    except Exception:
        logger.exception("Failed to load metric series for run %s", run_id)
        raise HTTPException(status_code=500, detail="Failed to load metric series")

    return JSONResponse(content={"run_id": run_id, "series": series})


@router.get("/runs/{run_id}/artifacts/{idx}")
async def get_artifact(
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
) -> JSONResponse:
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
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.debug(
                    "Artifact %d preview decode failed for run %s: %s", idx, run_id, exc
                )

    return JSONResponse(content=response)


@router.get("/runs/{run_id}/artifacts/{idx}/raw")
async def get_artifact_raw(
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
) -> Response:
    """Download raw artifact."""
    service = ArtifactService(registry, store)
    try:
        data, media_type, filename = service.get_artifact_raw(run_id, idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/projects")
async def list_projects(registry: RegistryDep) -> JSONResponse:
    """List all projects with stats."""
    service = ProjectService(registry)
    return JSONResponse(content={"projects": service.list_projects_with_stats()})


@router.post("/projects/{project}/baseline/{run_id}")
async def set_baseline(
    project: str,
    run_id: str,
    registry: RegistryDep,
    redirect: bool = Query(False),
) -> JSONResponse:
    """Set project baseline."""
    service = RunService(registry)
    try:
        record = service.get_run(run_id)
    except (KeyError, FileNotFoundError):
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


@router.get("/groups")
async def list_groups(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
) -> JSONResponse:
    """List run groups."""
    service = GroupService(registry)
    groups = service.list_groups(project=project or None)

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
async def get_group(group_id: str, registry: RegistryDep) -> JSONResponse:
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


@router.get("/diff")
async def get_diff(
    registry: RegistryDep,
    store: StoreDep,
    a: str = Query(..., description="Run A ID"),
    b: str = Query(..., description="Run B ID"),
) -> JSONResponse:
    """Compare two runs."""
    diff_service = DiffService(registry, store)
    try:
        record_a, record_b, report = diff_service.compare_runs(a, b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(
        content={
            "run_a": serialize_record_summary(record_a),
            "run_b": serialize_record_summary(record_b),
            "report": report,
        }
    )
