# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit Web UI.

A modern web interface for browsing experiment runs, viewing artifacts,
and comparing results. Built on FastAPI with a React frontend.

This package provides:

- Run listing and detail views with filtering
- Artifact viewing and download
- Run comparison (diff) functionality
- Project and group management
- REST API for programmatic access

Quick Start
-----------
Start the development server:

>>> from devqubit_ui import run_server
>>> run_server(debug=True)

Or create an app for custom deployment:

>>> from devqubit_ui import create_app
>>> app = create_app()

Architecture
------------
The UI follows a modular architecture:

- ``app.py`` - Application factory and server runner
- ``routers/api.py`` - JSON API endpoints for the React frontend
- ``static/`` - Built React SPA (served automatically)
- ``plugins.py`` - Plugin discovery and loading
- ``dependencies.py`` - FastAPI dependency injection utilities
- ``services.py`` - Service layer for business logic

Notes
-----
This package is installed as a dependency of ``devqubit`` and is not
considered part of the stable public API. The web interface may change
between versions. For programmatic access, prefer the core ``devqubit``
API or the JSON endpoints at ``/api/*``.
"""

__version__ = "0.1.9"


from devqubit_ui.app import create_app


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


__all__ = [
    "run_server",
    "create_app",
]
