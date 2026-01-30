# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
API router - REST endpoints for the React frontend.

Provides JSON API endpoints for all UI functionality.

Routes
------
GET /api/v1/capabilities
    Get server capabilities and mode.
GET /api/runs
    List runs with optional filters.
GET /api/runs/{run_id}
    Get run details.
DELETE /api/runs/{run_id}
    Delete a run.
GET /api/runs/{run_id}/artifacts/{index}
    Get artifact metadata and content preview.
GET /api/runs/{run_id}/artifacts/{index}/raw
    Download artifact content.
POST /api/projects/{project}/baseline/{run_id}
    Set a run as the baseline for a project.
GET /api/projects
    List all projects with statistics.
GET /api/groups
    List all groups.
GET /api/groups/{group_id}
    Get group details with runs.
GET /api/diff
    Get diff report between two runs.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_ui.dependencies import RegistryDep
from devqubit_ui.services import DiffService, GroupService, ProjectService, RunService
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger(__name__)
router = APIRouter()


# =========================================================================
# Capabilities
# =========================================================================


@router.get("/v1/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """
    Get server capabilities and mode.

    Returns
    -------
    dict
        Capabilities object with mode and features.
    """
    return {
        "mode": "local",
        "version": "0.1.9",
        "features": {
            "auth": False,
            "workspaces": False,
            "rbac": False,
            "service_accounts": False,
        },
    }


# =========================================================================
# Runs
# =========================================================================


@router.get("/runs")
async def list_runs(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    q: str = Query("", description="Search query"),
) -> JSONResponse:
    """
    List runs as JSON.

    Parameters
    ----------
    registry : RegistryDep
        Injected registry dependency.
    project : str, optional
        Filter by project name.
    status : str, optional
        Filter by run status.
    limit : int, default=50
        Maximum number of runs to return.
    q : str, optional
        Search query.

    Returns
    -------
    JSONResponse
        List of runs as JSON array.
    """
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
async def get_run(
    run_id: str,
    registry: RegistryDep,
) -> JSONResponse:
    """
    Get run details as JSON.

    Parameters
    ----------
    run_id : str
        The run ID.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        Complete run record as JSON.

    Raises
    ------
    HTTPException
        404 if run not found.
    """
    service = RunService(registry)

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    return JSONResponse(content={"run": _record_to_full_dict(record)})


@router.delete("/runs/{run_id}")
async def delete_run(
    run_id: str,
    registry: RegistryDep,
) -> JSONResponse:
    """
    Delete a run.

    Parameters
    ----------
    run_id : str
        The run ID to delete.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        Deletion confirmation.

    Raises
    ------
    HTTPException
        404 if run not found.
    """
    service = RunService(registry)

    deleted = service.delete_run(run_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")

    logger.info("Deleted run: %s", run_id)

    return JSONResponse(content={"status": "ok", "deleted": run_id})


# =========================================================================
# Artifacts
# =========================================================================


@router.get("/runs/{run_id}/artifacts/{index}")
async def get_artifact(
    run_id: str,
    index: int,
    registry: RegistryDep,
) -> JSONResponse:
    """
    Get artifact metadata and content preview.

    Parameters
    ----------
    run_id : str
        The run ID.
    index : int
        Artifact index.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        Artifact metadata and optional preview content.
    """
    service = RunService(registry)
    max_preview_size = 10 * 1024 * 1024  # 10MB

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    if index < 0 or index >= len(record.artifacts):
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = record.artifacts[index]

    # Get artifact content
    content = None
    content_json = None
    error = None
    preview_available = True
    size = 0

    try:
        data = service.get_artifact_content(run_id, index)
        size = len(data) if data else 0

        if size > max_preview_size:
            preview_available = False
        elif data:
            # Try to decode as text
            try:
                text = data.decode("utf-8")
                # Check if it's JSON
                if artifact.media_type in ("application/json", "text/json"):
                    import json

                    content_json = json.loads(text)
                else:
                    content = text
            except (UnicodeDecodeError, Exception):
                # Binary content - no preview
                pass
    except Exception as e:
        error = str(e)

    return JSONResponse(
        content={
            "artifact": {
                "kind": artifact.kind,
                "role": artifact.role,
                "media_type": artifact.media_type,
                "digest": artifact.digest,
            },
            "size": size,
            "content": content,
            "content_json": content_json,
            "preview_available": preview_available,
            "error": error,
        }
    )


@router.get("/runs/{run_id}/artifacts/{index}/raw")
async def download_artifact(
    run_id: str,
    index: int,
    registry: RegistryDep,
) -> Response:
    """
    Download artifact content.

    Parameters
    ----------
    run_id : str
        The run ID.
    index : int
        Artifact index.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    Response
        Raw artifact content with appropriate media type.
    """
    service = RunService(registry)

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    if index < 0 or index >= len(record.artifacts):
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = record.artifacts[index]
    data = service.get_artifact_content(run_id, index)

    if data is None:
        raise HTTPException(status_code=404, detail="Artifact content not found")

    return Response(
        content=data,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.kind}_{index}"',
        },
    )


# =========================================================================
# Projects
# =========================================================================


@router.post("/projects/{project}/baseline/{run_id}")
async def set_baseline(
    project: str,
    run_id: str,
    registry: RegistryDep,
) -> JSONResponse:
    """
    Set a run as the baseline for a project.

    Parameters
    ----------
    project : str
        The project name.
    run_id : str
        The run ID to set as baseline.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        Confirmation with project and baseline run ID.
    """
    service = RunService(registry)

    try:
        record = service.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")

    if record.project != project:
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} belongs to project '{record.project}', not '{project}'",
        )

    service.set_baseline(project, run_id)

    return JSONResponse(
        content={
            "status": "ok",
            "project": project,
            "baseline_run_id": run_id,
        }
    )


@router.get("/projects")
async def list_projects(
    registry: RegistryDep,
) -> JSONResponse:
    """
    List all projects with statistics as JSON.

    Parameters
    ----------
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        List of projects with run counts and baseline info.
    """
    service = ProjectService(registry)
    project_stats = service.list_projects_with_stats()

    return JSONResponse(content={"projects": project_stats})


# =========================================================================
# Groups
# =========================================================================


@router.get("/groups")
async def list_groups(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
) -> JSONResponse:
    """
    List all groups as JSON.

    Parameters
    ----------
    registry : RegistryDep
        Injected registry dependency.
    project : str, optional
        Filter by project name.

    Returns
    -------
    JSONResponse
        List of groups.
    """
    service = GroupService(registry)
    groups = service.list_groups(project=project or None)

    return JSONResponse(content={"groups": groups})


@router.get("/groups/{group_id}")
async def get_group(
    group_id: str,
    registry: RegistryDep,
) -> JSONResponse:
    """
    Get group details with runs.

    Parameters
    ----------
    group_id : str
        The group ID.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    JSONResponse
        Group details with list of runs.
    """
    service = GroupService(registry)
    runs = service.get_group_runs(group_id)

    if not runs:
        raise HTTPException(status_code=404, detail="Group not found")

    return JSONResponse(
        content={
            "group_id": group_id,
            "runs": runs,
        }
    )


# =========================================================================
# Diff
# =========================================================================


@router.get("/diff")
async def get_diff(
    registry: RegistryDep,
    a: str = Query(..., description="Run A (baseline) ID"),
    b: str = Query(..., description="Run B (candidate) ID"),
) -> JSONResponse:
    """
    Get diff report between two runs.

    Parameters
    ----------
    registry : RegistryDep
        Injected registry dependency.
    a : str
        Run A (baseline) ID.
    b : str
        Run B (candidate) ID.

    Returns
    -------
    JSONResponse
        Diff report with run summaries.
    """
    run_service = RunService(registry)
    diff_service = DiffService(registry)

    try:
        run_a = run_service.get_run(a)
        run_b = run_service.get_run(b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Run not found: {e}")

    report = diff_service.compare_runs(a, b)

    return JSONResponse(
        content={
            "run_a": _record_to_summary_dict(run_a),
            "run_b": _record_to_summary_dict(run_b),
            "report": report,
        }
    )


# =========================================================================
# Helpers
# =========================================================================


def _record_to_summary_dict(record: Any) -> dict[str, Any]:
    """Convert RunRecord to summary dictionary."""
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "fingerprints": record.fingerprints,
    }


def _record_to_full_dict(record: Any) -> dict[str, Any]:
    """Convert RunRecord to complete JSON-serializable dictionary."""
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "fingerprints": record.fingerprints,
        "data": record.record.get("data", {}),
        "backend": record.record.get("backend", {}),
        "group_id": record.record.get("group_id"),
        "group_name": record.record.get("group_name"),
        "errors": record.record.get("errors", []),
        "artifacts": [
            {
                "kind": a.kind,
                "role": a.role,
                "media_type": a.media_type,
                "digest": a.digest,
            }
            for a in record.artifacts
        ],
    }
