# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Groups router - run group management.

Provides endpoints for viewing and managing run groups. Groups allow
organizing related runs together (e.g., hyperparameter sweeps, A/B tests).

Routes
------
GET /groups
    List all run groups with optional filtering.
GET /groups/{group_id}
    Show runs belonging to a specific group.
"""

from __future__ import annotations

import logging

from devqubit_ui.app import templates
from devqubit_ui.dependencies import IsHtmxDep, RegistryDep
from devqubit_ui.services import GroupService, RunService
from fastapi import APIRouter, Query, Request


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/groups")
async def groups_list(
    request: Request,
    registry: RegistryDep,
    is_htmx: IsHtmxDep,
    project: str = Query("", description="Filter groups by project"),
):
    """
    List run groups with optional project filtering.

    Groups are automatically created when runs have a ``group_id`` set.
    This endpoint lists all groups with their metadata and run counts.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    registry : RegistryDep
        Injected registry dependency.
    is_htmx : IsHtmxDep
        True if request is from HTMX.
    project : str, optional
        Filter groups to those containing runs from this project.

    Returns
    -------
    TemplateResponse
        Groups list page or HTMX fragment.

    Examples
    --------
    List all groups:
        GET /groups

    Filter by project:
        GET /groups?project=vqe-experiments
    """
    group_service = GroupService(registry)
    run_service = RunService(registry)

    groups = group_service.list_groups(project=project or None)
    projects = run_service.list_projects()

    context = {
        "request": request,
        "groups": groups,
        "projects": projects,
        "filters": {"project": project},
    }

    template = "_groups_table.html" if is_htmx else "groups.html"
    return templates.TemplateResponse(request, template, context)


@router.get("/groups/{group_id}")
async def group_detail(
    request: Request,
    group_id: str,
    registry: RegistryDep,
):
    """
    Show all runs belonging to a specific group.

    Displays the group metadata and a table of all runs that
    are part of this group.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    group_id : str
        The unique group identifier.
    registry : RegistryDep
        Injected registry dependency.

    Returns
    -------
    TemplateResponse
        Group detail page with runs table.

    Notes
    -----
    Groups are identified by UUID. The group may have an optional
    human-readable name (``group_name``) set by the user.
    """
    service = GroupService(registry)
    runs = service.get_group_runs(group_id)

    return templates.TemplateResponse(
        request,
        "group_detail.html",
        {"group_id": group_id, "runs": runs},
    )
