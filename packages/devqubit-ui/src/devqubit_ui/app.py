# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit UI FastAPI application factory.

This module provides the app factory and server runner for the devqubit
web interface. Built on FastAPI with Jinja2 templates and HTMX for
progressive enhancement.

Examples
--------
Create app with default configuration:

>>> app = create_app()

Create app with custom workspace:

>>> app = create_app(workspace="/path/to/workspace")

Run development server:

>>> run_server(debug=True, port=8080)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from devqubit_engine.core.config import Config
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.protocols import ObjectStoreProtocol, RegistryProtocol
from devqubit_ui.filters import register_filters
from devqubit_ui.plugins import load_ui_plugins
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


logger = logging.getLogger(__name__)

# Template engine - initialized once, shared across requests
_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    Currently logs initialization; can be extended for connection pooling,
    background tasks, etc.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Yields
    ------
    None
        Control returns to the application during its lifetime.
    """
    logger.info(
        "devqubit UI started - workspace: %s",
        getattr(app.state, "workspace", "unknown"),
    )
    yield
    logger.info("devqubit UI shutting down")


def create_app(
    workspace: str | Path | None = None,
    config: Config | None = None,
    registry: "RegistryProtocol | None" = None,
    store: "ObjectStoreProtocol | None" = None,
) -> FastAPI:
    """
    Create the devqubit UI FastAPI application.

    Factory function that creates and configures a FastAPI application
    with all routes, templates, and dependencies.

    Parameters
    ----------
    workspace : str or Path, optional
        Workspace directory containing devqubit data.
        Defaults to ``~/.devqubit`` or ``DEVQUBIT_HOME`` environment variable.
    config : Config, optional
        Pre-configured Config object. If not provided, one is created
        from the workspace path.
    registry : RegistryProtocol, optional
        Pre-configured registry instance. Useful for testing with mocks.
    store : ObjectStoreProtocol, optional
        Pre-configured object store instance. Useful for testing with mocks.

    Returns
    -------
    FastAPI
        Fully configured FastAPI application ready to serve requests.

    Examples
    --------
    Basic usage with defaults:

    >>> app = create_app()
    >>> # app is now ready for uvicorn

    Testing with mock dependencies:

    >>> from unittest.mock import Mock
    >>> mock_registry = Mock()
    >>> app = create_app(registry=mock_registry)

    Notes
    -----
    The application stores dependencies in ``app.state`` for access in routes:

    - ``app.state.config`` - Configuration object
    - ``app.state.registry`` - Run registry
    - ``app.state.store`` - Object store
    - ``app.state.workspace`` - Workspace path string
    """
    app = FastAPI(
        title="devqubit UI",
        description="Web interface for devqubit experiment tracking",
        version="0.1.4",
        lifespan=lifespan,
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

    # Store dependencies in app.state for route access
    app.state.config = config
    app.state.registry = registry
    app.state.store = store
    app.state.workspace = str(config.root_dir)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register Jinja2 template filters
    register_filters(templates)

    # Register API routers
    _register_routers(app)

    # Load plugins (enterprise extensions, community plugins)
    load_ui_plugins(app)

    logger.info("devqubit UI initialized - workspace: %s", config.root_dir)

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Register all API routers with the application.

    Internal function that imports and includes all route modules.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application to register routes with.
    """
    from devqubit_ui.routers import (
        api,
        artifacts,
        diff,
        groups,
        projects,
        runs,
    )

    # Core page routes
    app.include_router(runs.router, tags=["runs"])
    app.include_router(projects.router, tags=["projects"])
    app.include_router(groups.router, tags=["groups"])
    app.include_router(diff.router, tags=["diff"])
    app.include_router(artifacts.router, tags=["artifacts"])

    # API routes (JSON endpoints)
    app.include_router(api.router, prefix="/api", tags=["api"])


def _run_in_thread(
    app: FastAPI,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """
    Run uvicorn server in a background thread.

    Used when running in Jupyter notebooks or other environments
    with an existing asyncio event loop.
    """
    import threading
    import time

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for server to start and get actual port
    time.sleep(0.5)

    # Try to get the actual bound port
    actual_port = port
    if hasattr(server, "servers") and server.servers:
        for s in server.servers:
            if s.sockets:
                actual_port = s.sockets[0].getsockname()[1]
                break

    print(f"\n  devqubit UI: http://{host}:{actual_port}")
    print(f"  Workspace: {app.state.workspace}")
    print("  Running in background thread (Jupyter mode)")
    print("  Restart kernel to stop\n")


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

    Convenience function to start the server with uvicorn. For production
    deployments, use uvicorn directly or a production ASGI server.

    Parameters
    ----------
    host : str, default="127.0.0.1"
        Host address to bind to. Use "0.0.0.0" for all interfaces.
    port : int, default=8080
        Port number to listen on.
    workspace : str, optional
        Workspace directory path.
    config : Config, optional
        Pre-configured Config object.
    debug : bool, default=False
        Enable debug mode with verbose logging.
    reload : bool, default=False
        Enable auto-reload on code changes (development only).

    Examples
    --------
    Start development server:

    >>> run_server(debug=True, reload=True)

    Start on all interfaces:

    >>> run_server(host="0.0.0.0", port=80)

    Notes
    -----
    For production, use:

    .. code-block:: bash

        uvicorn devqubit_ui.app:create_app --factory --host 0.0.0.0 --port 8080
    """
    import asyncio

    # Configure logging
    log_level = "debug" if debug else "info"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # For reload mode, uvicorn needs factory pattern
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
        )
        return

    app = create_app(workspace=workspace, config=config)

    # Check if we're in a running event loop (e.g., Jupyter)
    try:
        asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        in_async_context = False

    if in_async_context:
        # Run in background thread for Jupyter/async contexts
        _run_in_thread(app, host, port, log_level)
    else:
        # Normal blocking run
        print(f"\n  devqubit UI: http://{host}:{port}")
        print(f"  Workspace: {app.state.workspace}")
        print("  Press Ctrl+C to stop\n")
        uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    run_server(debug=True)
