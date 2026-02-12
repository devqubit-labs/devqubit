# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Web UI for devqubit experiment tracking.

A local web interface for browsing runs, viewing artifacts, comparing
experiments, and managing baselines.  Built on FastAPI (backend) with a
React SPA (frontend).

Starting the Server
-------------------
>>> from devqubit_ui import run_server
>>> run_server(port=8080)

Or from the CLI::

    devqubit ui --port 8080

Custom Deployment
-----------------
>>> from devqubit_ui import create_app
>>> app = create_app()  # ASGI app for uvicorn / gunicorn

This package is an internal implementation detail of ``devqubit[ui]``.
The web interface and REST API (``/api/*``) may change between versions.
For programmatic access, prefer the :mod:`devqubit` Python API.
"""

from importlib.metadata import version

from devqubit_ui.app import create_app, run_server


__version__ = version("devqubit-ui")


__all__ = [
    "run_server",
    "create_app",
]
