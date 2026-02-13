# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
DevQubit UI Application Factory.

Creates and configures the FastAPI application with API routers
and static file serving for the React frontend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from devqubit_engine.config import Config
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
from devqubit_ui.plugins import load_ui_plugins
from devqubit_ui.routers import api
from devqubit_ui.routers import export as export_router
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info(
        "devqubit UI started - workspace: %s",
        getattr(app.state, "workspace", "unknown"),
    )
    yield
    from devqubit_ui.routers.export import bundle_cache

    removed = bundle_cache.cleanup()
    if removed:
        logger.info("Cleaned up %d cached export bundle(s)", removed)
    logger.info("devqubit UI shutting down")


def create_app(
    workspace: str | Path | None = None,
    config: Config | None = None,
    registry: RegistryProtocol | None = None,
    store: ObjectStoreProtocol | None = None,
) -> FastAPI:
    """
    Create the devqubit UI FastAPI application.

    Parameters
    ----------
    workspace : str or Path, optional
        Workspace directory containing devqubit data.
    config : Config, optional
        Pre-configured Config object.
    registry : RegistryProtocol, optional
        Pre-configured registry instance.
    store : ObjectStoreProtocol, optional
        Pre-configured object store instance.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    try:
        _version = get_version("devqubit-ui")
    except PackageNotFoundError:
        _version = "0.0.0"

    app = FastAPI(
        title="devqubit UI",
        description="Experiment tracking UI for quantum computing",
        version=_version,
        lifespan=lifespan,
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize configuration
    if config is None:
        if workspace:
            ws_path = Path(workspace).expanduser()
        else:
            ws_path = Path(os.environ.get("DEVQUBIT_HOME", "~/.devqubit")).expanduser()
        config = Config(root_dir=ws_path)

    # Initialize storage backends
    if registry is None:
        registry = create_registry(config=config)
    if store is None:
        store = create_store(config=config)

    # Store dependencies in app.state
    app.state.config = config
    app.state.registry = registry
    app.state.store = store
    app.state.workspace = str(config.root_dir)

    # Load plugins
    load_ui_plugins(app)

    # Include API router
    app.include_router(api.router, prefix="/api", tags=["api"])
    app.include_router(export_router.router, prefix="/api", tags=["export"])

    # Serve React frontend
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{full_path:path}", response_class=HTMLResponse, response_model=None)
        async def serve_spa(
            request: Request,
            full_path: str,
        ) -> HTMLResponse | FileResponse:
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

    logger.info("devqubit UI initialized - workspace: %s", config.root_dir)

    return app


def _run_in_thread(
    app: FastAPI,
    host: str,
    port: int,
    log_level: str,
    access_log: bool = False,
) -> None:
    """Run uvicorn server in a background thread (for Jupyter)."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=access_log,
    )
    server = uvicorn.Server(config)

    def run_server() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=run_server, daemon=True, name="devqubit-ui")
    thread.start()
    time.sleep(0.5)

    actual_port = port
    if hasattr(server, "servers") and server.servers:
        for s in server.servers:
            if s.sockets:
                actual_port = s.sockets[0].getsockname()[1]
                break

    print(f"\n   devqubit UI: http://{host}:{actual_port}")
    print(f"   Workspace: {app.state.workspace}")
    print("   Running in background thread (Jupyter mode)")
    print("   Restart kernel to stop\n")


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    workspace: str | None = None,
    config: Config | None = None,
    debug: bool = False,
    reload: bool = False,
) -> None:
    """
    Run the devqubit UI server.

    Parameters
    ----------
    host : str
        Host address to bind to.
    port : int
        Port number.
    workspace : str, optional
        Workspace directory path.
    config : Config, optional
        Pre-configured Config object.
    debug : bool
        Enable debug mode.
    reload : bool
        Enable auto-reload (development only).
    """
    log_level = "debug" if debug else "info"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if reload:
        if workspace:
            os.environ["DEVQUBIT_HOME"] = workspace
        uvicorn.run(
            "devqubit_ui.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            log_level=log_level,
            access_log=debug,
        )
        return

    app = create_app(workspace=workspace, config=config)

    try:
        asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        in_async_context = False

    if in_async_context:
        _run_in_thread(app, host, port, log_level, access_log=debug)
    else:
        print(f"\n   devqubit UI: http://{host}:{port}")
        print(f"   Workspace: {app.state.workspace}")
        print("   Press Ctrl+C to stop\n")
        uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=debug)


if __name__ == "__main__":
    run_server(debug=True)
