# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Diff router - run comparison functionality.

Provides endpoints for comparing two runs side-by-side, showing
differences in parameters, metrics, and result distributions.

Routes
------
GET /diff
    Compare two runs or show run selection form.
"""

from __future__ import annotations

import logging

from devqubit_ui.app import templates
from devqubit_ui.dependencies import RegistryDep, StoreDep
from devqubit_ui.services import DiffService, RunService
from fastapi import APIRouter, HTTPException, Query, Request


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/diff")
async def diff_view(
    request: Request,
    registry: RegistryDep,
    store: StoreDep,
    a: str = Query("", description="Run A (baseline) ID"),
    b: str = Query("", description="Run B (candidate) ID"),
):
    """
    Compare two runs or display run selection form.

    When both run IDs are provided, computes and displays a detailed
    comparison including:

    - Metadata comparison (project, backend)
    - Fingerprint comparison
    - Program comparison (exact/structural match)
    - Device calibration drift
    - Parameter differences
    - Metric differences
    - Circuit differences
    - Total Variation Distance (TVD) for result distributions

    When run IDs are not provided, shows a form to select runs.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    registry : RegistryDep
        Injected registry dependency.
    store : StoreDep
        Injected object store dependency.
    a : str, optional
        Run ID for the baseline (Run A).
    b : str, optional
        Run ID for the candidate (Run B).

    Returns
    -------
    TemplateResponse
        Either the comparison results or the run selection form.

    Raises
    ------
    HTTPException
        404 if either run is not found.

    Examples
    --------
    Show selection form:
        GET /diff

    Compare two runs:
        GET /diff?a=abc123&b=def456
    """
    if not a or not b:
        # Show run selection form
        run_service = RunService(registry)
        runs = run_service.list_runs(limit=100)
        return templates.TemplateResponse(
            request,
            "diff_select.html",
            {"runs": runs},
        )

    # Compare runs
    diff_service = DiffService(registry, store)

    try:
        record_a, record_b, report = diff_service.compare_runs(a, b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return templates.TemplateResponse(
        request,
        "diff.html",
        {
            "run_a": record_a,
            "run_b": record_b,
            "report": report,
        },
    )
