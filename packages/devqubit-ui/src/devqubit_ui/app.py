# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
DevQubit UI Application Factory.

Creates and configures the FastAPI application with API routers
and static file serving for the React frontend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from devqubit_ui.plugins import load_ui_plugins
from devqubit_ui.routers import api
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


logger = logging.getLogger(__name__)


def create_app(
    registry_factory: Callable | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    registry_factory : Callable, optional
        Factory function that returns a RunRegistry instance.
        If not provided, uses default from devqubit.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    app = FastAPI(
        title="devqubit UI",
        description="Experiment tracking UI for quantum computing",
        version="0.1.9",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store registry factory in app state
    if registry_factory is None:
        from devqubit import get_registry

        registry_factory = get_registry
    app.state.registry_factory = registry_factory

    # Load plugins
    load_ui_plugins(app)

    # Include API router
    app.include_router(api.router, prefix="/api", tags=["api"])

    # Serve React frontend
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        # Mount assets directory for JS/CSS bundles
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{full_path:path}", response_class=HTMLResponse)
        async def serve_spa(request: Request, full_path: str):
            """Serve React SPA for all non-API routes."""
            if full_path.startswith("api/"):
                return HTMLResponse(status_code=404, content="Not found")
            return FileResponse(static_dir / "index.html")

        logger.info("Serving React frontend from %s", static_dir)
    else:
        logger.warning(
            "Static frontend not found at %s. "
            "Run 'npm run build' in frontend/ and copy dist/ to static/",
            static_dir,
        )

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    debug: bool = False,
) -> None:
    """
    Run the development server.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to bind to.
    debug : bool
        Enable debug mode with auto-reload.
    """
    import uvicorn

    uvicorn.run(
        "devqubit_ui.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=debug,
    )
