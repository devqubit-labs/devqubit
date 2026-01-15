# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
FastAPI dependency injection utilities.

This module provides dependency functions for injecting common services
into route handlers. Uses FastAPI's ``Depends()`` pattern for clean,
testable code.

Examples
--------
Using dependencies in a route:

>>> from fastapi import Depends
>>> from devqubit_ui.dependencies import get_registry
>>>
>>> @router.get("/runs")
>>> def list_runs(registry = Depends(get_registry)):
...     return registry.list_runs()
"""

from __future__ import annotations

from typing import Annotated, Any

from devqubit_engine.config import Config
from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol
from devqubit_ui.app import templates
from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates


def get_config(request: Request) -> Config:
    """
    Get the Config instance from application state.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    Config
        The application configuration object.
    """
    return request.app.state.config


def get_registry(request: Request) -> RegistryProtocol:
    """
    Get the registry instance from application state.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    RegistryProtocol
        The run registry for querying and managing runs.
    """
    return request.app.state.registry


def get_store(request: Request) -> ObjectStoreProtocol:
    """
    Get the object store instance from application state.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    ObjectStoreProtocol
        The object store for artifact data.
    """
    return request.app.state.store


def get_templates() -> Jinja2Templates:
    """
    Get the Jinja2 templates instance.

    Returns
    -------
    Jinja2Templates
        The configured template engine.
    """
    return templates


def is_htmx_request(request: Request) -> bool:
    """
    Check if the request is from HTMX.

    Detects HTMX requests by checking for the ``HX-Request`` header,
    which HTMX automatically sends with all requests.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    bool
        True if request originated from HTMX, False otherwise.

    Examples
    --------
    >>> @router.get("/runs")
    >>> def list_runs(is_htmx: bool = Depends(is_htmx_request)):
    ...     template = "_runs_table.html" if is_htmx else "runs.html"
    ...     return templates.TemplateResponse(template, {...})
    """
    return request.headers.get("HX-Request") == "true"


def get_htmx_target(request: Request) -> str | None:
    """
    Get the HTMX target element ID.

    Returns the ID of the element that will be swapped with the response.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    str or None
        The target element ID, or None if not an HTMX request.
    """
    return request.headers.get("HX-Target")


def get_htmx_trigger(request: Request) -> str | None:
    """
    Get the HTMX trigger element ID.

    Returns the ID of the element that triggered the request.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    str or None
        The trigger element ID, or None if not an HTMX request.
    """
    return request.headers.get("HX-Trigger")


def get_current_user(request: Request) -> Any | None:
    """
    Get the current authenticated user.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    Any or None
        The current user object, or None if not authenticated
        or running in local mode.
    """
    return getattr(request.state, "current_user", None)


def get_capabilities(request: Request) -> dict[str, Any]:
    """
    Get server capabilities from application state.

    Returns the capabilities dict that describes the server's
    operating mode and available features.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    dict
        Capabilities object with mode and features.
    """
    return getattr(
        request.app.state,
        "capabilities",
        {
            "mode": "local",
            "features": {},
        },
    )


# Type aliases for cleaner dependency injection
ConfigDep = Annotated["Config", Depends(get_config)]
RegistryDep = Annotated["RegistryProtocol", Depends(get_registry)]
StoreDep = Annotated["ObjectStoreProtocol", Depends(get_store)]
TemplatesDep = Annotated["Jinja2Templates", Depends(get_templates)]
IsHtmxDep = Annotated[bool, Depends(is_htmx_request)]
CurrentUserDep = Annotated[Any | None, Depends(get_current_user)]
CapabilitiesDep = Annotated[dict[str, Any], Depends(get_capabilities)]
