# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit UI component registration.

This module exists to support plugin discovery via Python entry points.
It serves as a stable, importable target for the ``devqubit.components``
entry point declared in ``pyproject.toml`` (PEP 621).

The presence of that entry point indicates that the UI distribution is
installed and can be loaded when needed.
"""

from __future__ import annotations


def register() -> None:
    """
    Register the devqubit UI component.

    This function is intentionally a no-op. It serves as a stable,
    importable target for entry point discovery. The actual UI
    functionality is provided by the package modules.
    """
    return None
