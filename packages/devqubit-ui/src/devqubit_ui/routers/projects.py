# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Projects router - project listing and statistics.

Provides endpoints for viewing all projects, their run counts,
and baseline configurations.

Routes
------
GET /projects
    List all projects with statistics.
"""

from __future__ import annotations

import logging

from devqubit_ui.app import templates
from devqubit_ui.dependencies import IsHtmxDep, RegistryDep
from devqubit_ui.services import ProjectService
from fastapi import APIRouter, Request


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/projects")
async def projects_list(
    request: Request,
    registry: RegistryDep,
    is_htmx: IsHtmxDep,
):
    """
    List all projects with run counts and baseline info.

    Displays a table of all projects in the workspace, including:

    - Project name
    - Total run count
    - Current baseline run (if set)

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    registry : RegistryDep
        Injected registry dependency.
    is_htmx : IsHtmxDep
        True if request is from HTMX.

    Returns
    -------
    TemplateResponse
        Projects list page or HTMX fragment.

    Notes
    -----
    Project statistics are computed on each request. For workspaces
    with many projects, consider caching or pagination in future versions.
    """
    service = ProjectService(registry)
    project_stats = service.list_projects_with_stats()

    context = {
        "request": request,
        "projects": project_stats,
    }

    template = "_projects_table.html" if is_htmx else "projects.html"
    return templates.TemplateResponse(request, template, context)
