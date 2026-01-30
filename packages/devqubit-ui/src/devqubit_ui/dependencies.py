# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
FastAPI dependency injection utilities.

Provides dependency functions for injecting common services into route handlers.
"""

from __future__ import annotations

from typing import Annotated, Any

from devqubit_engine.config import Config
from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
from fastapi import Depends, Request


def get_config(request: Request) -> Config:
    """Get the Config instance from application state."""
    return request.app.state.config


def get_registry(request: Request) -> RegistryProtocol:
    """Get the registry instance from application state."""
    return request.app.state.registry_factory()


def get_store(request: Request) -> ObjectStoreProtocol:
    """Get the object store instance from application state."""
    return request.app.state.store


def get_current_user(request: Request) -> Any | None:
    """Get the current authenticated user."""
    return getattr(request.state, "current_user", None)


def get_capabilities(request: Request) -> dict[str, Any]:
    """Get server capabilities from application state."""
    return getattr(
        request.app.state,
        "capabilities",
        {"mode": "local", "features": {}},
    )


# Type aliases for cleaner dependency injection
ConfigDep = Annotated["Config", Depends(get_config)]
RegistryDep = Annotated["RegistryProtocol", Depends(get_registry)]
StoreDep = Annotated["ObjectStoreProtocol", Depends(get_store)]
CurrentUserDep = Annotated[Any | None, Depends(get_current_user)]
CapabilitiesDep = Annotated[dict[str, Any], Depends(get_capabilities)]
