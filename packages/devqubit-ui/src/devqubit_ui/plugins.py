# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UI plugin discovery and loading for FastAPI.

Plugins can extend the devqubit UI by registering via the
``devqubit.ui.plugins`` entry point group. Each plugin receives
the FastAPI application instance and can add routes, middleware,
template filters, or other customizations.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from fastapi import FastAPI


logger = logging.getLogger(__name__)


@runtime_checkable
class UIPlugin(Protocol):
    """
    Protocol defining the UI plugin interface.

    Plugins must be callable and accept a FastAPI application instance.
    They can modify the app by adding routes, middleware, event handlers,
    or other customizations.

    Examples
    --------
    Function-based plugin:

    >>> def my_plugin(app: FastAPI) -> None:
    ...     app.include_router(my_router)

    Class-based plugin:

    >>> class MyPlugin:
    ...     def __call__(self, app: FastAPI) -> None:
    ...         app.include_router(self.router)
    """

    def __call__(self, app: "FastAPI") -> None:
        """
        Register plugin components with the FastAPI application.

        Parameters
        ----------
        app : FastAPI
            The FastAPI application to extend.
        """
        ...


def load_ui_plugins(app: "FastAPI") -> None:
    """
    Load and register UI plugins from entry points.

    Discovers plugins registered under the ``devqubit.ui.plugins`` group
    and calls each with the FastAPI app instance. Plugin loading errors
    are logged but do not prevent application startup.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application to extend with plugins.

    Notes
    -----
    Plugins are loaded in an undefined order. If plugin ordering matters,
    plugins should handle their own dependency management.

    Plugin authors should:

    - Keep initialization fast (defer heavy work)
    - Handle their own exceptions gracefully
    - Use prefixed routes to avoid conflicts
    - Document any app.state requirements

    Examples
    --------
    This function is called automatically by ``create_app()``:

    >>> app = create_app()  # Plugins loaded automatically

    For testing, plugins can be loaded manually:

    >>> app = FastAPI()
    >>> load_ui_plugins(app)
    """
    try:
        from importlib.metadata import entry_points

        # Python 3.10+ API
        try:
            eps = entry_points(group="devqubit.ui.plugins")
        except TypeError:
            # Python 3.9 fallback
            eps = entry_points().get("devqubit.ui.plugins", [])

        loaded_count = 0
        for ep in eps:
            try:
                plugin_fn = ep.load()

                # Validate plugin signature
                if not callable(plugin_fn):
                    logger.warning(
                        "UI plugin %s is not callable, skipping",
                        ep.name,
                    )
                    continue

                plugin_fn(app)
                loaded_count += 1
                logger.info("Loaded UI plugin: %s", ep.name)

            except Exception as e:
                logger.warning(
                    "Failed to load UI plugin %s: %s",
                    ep.name,
                    e,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        if loaded_count > 0:
            logger.info("Loaded %d UI plugin(s)", loaded_count)

    except ImportError:
        logger.debug("importlib.metadata not available, skipping plugins")
    except Exception as e:
        logger.debug("Plugin loading skipped: %s", e)
