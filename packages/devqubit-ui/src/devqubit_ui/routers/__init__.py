# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
FastAPI routers for devqubit UI.

Provides REST API endpoints for the React frontend:

- ``api``: JSON API for runs, projects, groups, diff, artifacts
- ``export``: Run export/bundle endpoints
"""

from devqubit_ui.routers.api import router as api_router
from devqubit_ui.routers.export import router as export_router


__all__ = ["api_router", "export_router"]
