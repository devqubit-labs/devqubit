# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
API router - REST endpoints for programmatic access.

Provides JSON API endpoints for actions that can be performed
programmatically. These endpoints are designed for CLI/SDK integration.

Routes
------
GET /api/v1/capabilities
    Get server capabilities and mode.
POST /api/projects/{project}/baseline/{run_id}
    Set a run as the baseline for a project.
GET /api/runs
    List runs (JSON response).
GET /api/runs/{run_id}
    Get run details (JSON response).
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_ui.dependencies import RegistryDep
from devqubit_ui.services import RunService
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/v1/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """
    Get server capabilities and mode.

    Returns information about the server's operating mode and available
    features. This endpoint is used by the UI to adapt its interface
    and by clients to discover server capabilities.

    Returns
    -------
    dict
        Capabilities object with mode and features.
    """
    return {
        "mode": "local",
        "version": "0.1.3",
        "features": {
            "auth": False,
            "workspaces": False,
            "rbac": False,
            "service_accounts": False,
        },
    }


@router.post("/projects/{project}/baseline/{run_id}")
async def set_baseline(
    project: str,
    run_id: str,
    registry: RegistryDep,
    redirect: bool = Query(True, description="Redirect to run detail after setting"),
):
    """
    Set a run as the baseline for a project.

    The baseline run is used as the reference point for comparisons.
    Each project can have one baseline at a time.

    Parameters
    ----------
    project : str
        The project name.
    run_id : str
        The run ID to set as baseline.
    registry : RegistryDep
        Injected registry dependency.
    redirect : bool, default=True
        If True, redirect to run detail page (for browser forms).
        If False, return JSON response (for API clients).

    Returns
    -------
    RedirectResponse or JSONResponse
        Redirect to run detail or JSON confirmation.

    Raises
    ------
    HTTPException
        404 if run not found.
        400 if run doesn't belong to the specified project.

    Examples
    --------
    From browser form (redirects):
        POST /api/projects/my-project/baseline/abc123

    From API client (returns JSON):
        POST /api/projects/my-project/baseline/abc123?redirect=false
    """
    service = RunService(registry)

    # Verify run exists and belongs to project
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

    if redirect:
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    return JSONResponse(
        content={
            "status": "ok",
            "project": project,
            "baseline_run_id": run_id,
        }
    )


@router.get("/runs")
async def api_runs_list(
    registry: RegistryDep,
    project: str = Query("", description="Filter by project"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    q: str = Query("", description="Search query"),
):
    """
    List runs as JSON.

    Provides the same filtering as the HTML endpoint but returns
    JSON for programmatic access.

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

    Examples
    --------
    Get all runs:
        GET /api/runs

    Filter by project:
        GET /api/runs?project=vqe&limit=10
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
async def api_run_detail(
    run_id: str,
    registry: RegistryDep,
):
    """
    Get run details as JSON.

    Returns complete run record including parameters, metrics,
    tags, and artifact metadata.

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


def _record_to_full_dict(record: Any) -> dict[str, Any]:
    """
    Convert RunRecord to complete JSON-serializable dictionary.

    Parameters
    ----------
    record : RunRecord
        The run record object.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary representation.
    """
    return {
        "run_id": record.run_id,
        "project": record.project,
        "adapter": record.adapter,
        "status": record.status,
        "created_at": str(record.created_at) if record.created_at else None,
        "fingerprints": record.fingerprints,
        "data": record.record.get("data", {}),
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
