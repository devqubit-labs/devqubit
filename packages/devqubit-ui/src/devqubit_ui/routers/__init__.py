# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
FastAPI routers for devqubit UI.

Each router handles a logical feature area:

- ``runs``: Run listing, detail views, search
- ``artifacts``: Artifact viewing and download
- ``diff``: Run comparison
- ``groups``: Run group management
- ``projects``: Project listing
- ``api``: REST API endpoints (JSON responses)

All page routes support HTMX partial rendering via the ``HX-Request``
header. When present, routes return only the relevant fragment instead
of the full page.
"""
