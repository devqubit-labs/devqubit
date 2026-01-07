# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifacts router - viewing and downloading run artifacts.

Provides endpoints for viewing artifact content and downloading
raw artifact data. Supports text, JSON, and binary artifacts with
size limits to prevent memory issues.

Routes
------
GET /runs/{run_id}/artifacts/{idx}
    View artifact content with formatting.
GET /runs/{run_id}/artifacts/{idx}/raw
    Download raw artifact bytes.
"""

from __future__ import annotations

import json
import logging

from devqubit_ui.app import templates
from devqubit_ui.dependencies import RegistryDep, StoreDep
from devqubit_ui.services import MAX_ARTIFACT_PREVIEW_SIZE, ArtifactService
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/runs/{run_id}/artifacts/{idx}")
async def artifact_view(
    request: Request,
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
):
    """
    View artifact content with appropriate formatting.

    Displays artifact metadata and content. Text and JSON artifacts
    are rendered inline with syntax highlighting. Binary artifacts
    and large artifacts (> 5MB) show size and download link only.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    run_id : str
        The run ID containing the artifact.
    idx : int
        Zero-based index of the artifact in the run's artifact list.
    registry : RegistryDep
        Injected registry dependency.
    store : StoreDep
        Injected object store dependency.

    Returns
    -------
    TemplateResponse
        Artifact view page.

    Raises
    ------
    HTTPException
        404 if run or artifact not found.

    Notes
    -----
    For artifacts larger than MAX_ARTIFACT_PREVIEW_SIZE (5MB), content
    is not loaded into memory. Users must download to view.
    """
    service = ArtifactService(registry, store)

    try:
        _, artifact = service.get_artifact_metadata(run_id, idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Get content with size safety
    content_result = service.get_artifact_content(run_id, idx)

    content = None
    content_json = None

    if content_result.preview_available and content_result.data:
        if content_result.is_text:
            try:
                content = content_result.data.decode("utf-8")
                if content_result.is_json:
                    content_json = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                content = f"<binary data: {content_result.size} bytes>"

    return templates.TemplateResponse(
        request,
        "artifact.html",
        {
            "run_id": run_id,
            "artifact": artifact,
            "idx": idx,
            "content": content,
            "content_json": content_json,
            "size": content_result.size,
            "preview_available": content_result.preview_available,
            "max_preview_size": MAX_ARTIFACT_PREVIEW_SIZE,
            "error": content_result.error,
        },
    )


@router.get("/runs/{run_id}/artifacts/{idx}/raw")
async def artifact_raw(
    run_id: str,
    idx: int,
    registry: RegistryDep,
    store: StoreDep,
):
    """
    Download raw artifact data.

    Returns the artifact bytes with appropriate Content-Type
    and Content-Disposition headers for download.

    Parameters
    ----------
    run_id : str
        The run ID containing the artifact.
    idx : int
        Zero-based index of the artifact.
    registry : RegistryDep
        Injected registry dependency.
    store : StoreDep
        Injected object store dependency.

    Returns
    -------
    Response
        Raw artifact bytes with download headers.

    Raises
    ------
    HTTPException
        404 if run or artifact not found.
        500 if artifact data cannot be loaded.

    Notes
    -----
    The filename is derived from the artifact kind, with path
    separators replaced by underscores.
    """
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
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
