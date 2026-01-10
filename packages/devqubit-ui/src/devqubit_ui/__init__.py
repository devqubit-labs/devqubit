# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit Web UI.

A modern web interface for browsing experiment runs, viewing artifacts,
and comparing results. Built on FastAPI with Jinja2 templates and HTMX
for progressive enhancement.

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
- ``routers/`` - FastAPI route handlers (one per feature area)
- ``templates/`` - Jinja2 templates with HTMX integration
- ``filters.py`` - Custom Jinja2 template filters
- ``plugins.py`` - Plugin discovery and loading
- ``dependencies.py`` - FastAPI dependency injection utilities

Notes
-----
This package is installed as a dependency of ``devqubit`` and is not
considered part of the stable public API. The web interface may change
between versions. For programmatic access, prefer the core ``devqubit``
API or the JSON endpoints at ``/api/*``.
"""

__version__ = "0.1.3"
