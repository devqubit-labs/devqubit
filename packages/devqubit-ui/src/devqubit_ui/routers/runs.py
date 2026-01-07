# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Runs router - listing, detail, and search views.

Provides endpoints for browsing runs, viewing run details, and searching
across runs. Supports HTMX partial rendering for dynamic filtering.

Routes
------
GET /
    Redirect to runs list.
GET /runs
    List runs with optional filtering.
GET /runs/{run_id}
    Show run details.
GET /search
    Search page with query builder documentation.
"""

from __future__ import annotations

import logging

from devqubit_ui.app import templates
from devqubit_ui.dependencies import IsHtmxDep, RegistryDep
from devqubit_ui.services import RunService
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def index() -> RedirectResponse:
    """
    Redirect root to runs list.

    Returns
    -------
    RedirectResponse
        Redirect to /runs endpoint.
    """
    return RedirectResponse(url="/runs", status_code=302)


@router.get("/runs")
async def runs_list(
    request: Request,
    registry: RegistryDep,
    is_htmx: IsHtmxDep,
    project: str = Query("", description="Filter by project name"),
    status: str = Query("", description="Filter by run status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum runs to return"),
    q: str = Query("", description="Search query"),
):
    """
    List runs with optional filtering.

    Supports filtering by project, status, and full-text search query.
    When called via HTMX, returns only the table fragment for dynamic updates.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    registry : RegistryDep
        Injected registry dependency.
    is_htmx : IsHtmxDep
        True if request is from HTMX.
    project : str, optional
        Filter runs by project name.
    status : str, optional
        Filter runs by status (FINISHED, FAILED, RUNNING).
    limit : int, default=50
        Maximum number of runs to return (1-500).
    q : str, optional
        Search query using devqubit query syntax.

    Returns
    -------
    TemplateResponse
        Full page or HTMX fragment depending on request type.

    Examples
    --------
    List all runs:
        GET /runs

    Filter by project:
        GET /runs?project=my-project

    Search with query:
        GET /runs?q=metric.fidelity > 0.9
    """
    service = RunService(registry)
    projects = service.list_projects()

    if q:
        runs_data = service.search_runs(q, limit=limit)
    else:
        runs_data = service.list_runs(
            project=project or None,
            status=status or None,
            limit=limit,
        )

    context = {
        "request": request,
        "runs": runs_data,
        "projects": projects,
        "filters": {"project": project, "status": status, "limit": limit, "q": q},
    }

    # Return fragment for HTMX requests, full page otherwise
    template = "_runs_table.html" if is_htmx else "runs.html"
    return templates.TemplateResponse(request, template, context)


@router.get("/runs/{run_id}")
async def run_detail(
    request: Request,
    run_id: str,
    registry: RegistryDep,
):
    """
    Show detailed view of a single run.

    Displays run metadata, parameters, metrics, tags, artifacts,
    and any errors that occurred during execution.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    run_id : str
        The unique run identifier.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    TemplateResponse
        Run detail page.

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

    baseline = service.get_baseline(record.project)
    is_baseline = baseline and baseline.get("run_id") == run_id

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": record,
            "record": record.record,
            "artifacts": record.artifacts,
            "is_baseline": is_baseline,
            "baseline": baseline,
        },
    )


@router.get("/search")
async def search(request: Request):
    """
    Display search page with query syntax documentation.

    Provides interactive search interface and documentation for
    the devqubit query language.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    TemplateResponse
        Search page with query builder.
    """
    return templates.TemplateResponse(request, "search.html", {})
