# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Adapter plugin discovery and loading system.

This module discovers and loads SDK adapters via Python entry points
under the ``devqubit.adapters`` group. Each adapter provides support
for a specific quantum computing SDK (Qiskit, PennyLane, Cirq, etc.).

Entry Point Group
-----------------
``devqubit.adapters``
    Entry points should point to adapter classes implementing
    :class:`AdapterProtocol`.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)

# Entry point group name for adapter discovery
ADAPTER_ENTRY_POINT_GROUP = "devqubit.adapters"


@runtime_checkable
class AdapterProtocol(Protocol):
    """
    Protocol defining the adapter interface.

    All SDK adapters must implement this interface to integrate
    with the devqubit tracking system.

    Attributes
    ----------
    name : str
        Unique adapter identifier (e.g., "qiskit", "pennylane").

    Methods
    -------
    supports_executor(executor)
        Check if this adapter supports the given executor.
    describe_executor(executor)
        Get a description of the executor for the run record.
    wrap_executor(executor, tracker)
        Wrap the executor for automatic tracking.
    """

    name: str

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if this adapter supports the given executor.

        Parameters
        ----------
        executor : Any
            SDK executor instance (backend, device, sampler, etc.).

        Returns
        -------
        bool
            True if this adapter can handle the executor.
        """
        ...

    def describe_executor(self, executor: Any) -> dict[str, Any]:
        """
        Get a description of the executor for the run record.

        Parameters
        ----------
        executor : Any
            SDK executor instance.

        Returns
        -------
        dict
            Executor description with keys like "name", "type", "provider".
        """
        ...

    def wrap_executor(
        self,
        executor: Any,
        tracker: Any,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
        **kwargs: Any,
    ) -> Any:
        """
        Wrap the executor for automatic tracking.

        Parameters
        ----------
        executor : Any
            SDK executor instance to wrap.
        tracker : Any
            Run tracker instance for logging artifacts.
        log_every_n : int, optional
            Logging frequency: 0 = first only, N = every Nth, -1 = all.
            Default is 0.
        log_new_circuits : bool, optional
            Auto-log when circuit structure changes. Default is True.
        stats_update_interval : int, optional
            Update execution stats every N runs. Default is 1000.
        **kwargs : Any
            Additional adapter-specific options.

        Returns
        -------
        Any
            Wrapped executor with the same interface as the original.
        """
        ...


@dataclass(frozen=True)
class AdapterLoadError:
    """
    Captures an adapter load or instantiation failure for diagnostics.

    Parameters
    ----------
    entry_point : str
        Entry point specification that failed (name=value).
    exc_type : str
        Exception type name.
    message : str
        Exception message.
    traceback : str
        Full formatted traceback.
    """

    entry_point: str
    exc_type: str
    message: str
    traceback: str

    def __str__(self) -> str:
        """Format as a concise error string."""
        return f"{self.entry_point}: {self.exc_type}: {self.message}"


# Module-level cache for loaded adapters (protected by _cache_lock).
# Use tuples internally to prevent accidental external mutation of cached state.
_adapters: tuple[AdapterProtocol, ...] | None = None
_adapter_errors: tuple[AdapterLoadError, ...] = ()
_cache_lock = threading.RLock()


def _iter_adapter_entry_points() -> list[EntryPoint]:
    """
    Get entry points for the adapter group, handling API differences.

    Returns
    -------
    list of EntryPoint
        Entry points registered under ``devqubit.adapters``.

    Notes
    -----
    The entry point API differs across Python/importlib_metadata versions.
    The most compatible approach is to call ``entry_points()`` with no args
    and then use ``.select(group=...)`` if available; otherwise treat the
    result as a mapping keyed by group. :contentReference[oaicite:1]{index=1}
    """
    eps = entry_points()
    select = getattr(eps, "select", None)
    if callable(select):
        return list(eps.select(group=ADAPTER_ENTRY_POINT_GROUP))
    return list(eps.get(ADAPTER_ENTRY_POINT_GROUP, []))


def _make_load_error(ep: EntryPoint, exc: Exception) -> AdapterLoadError:
    """Create a structured AdapterLoadError."""
    return AdapterLoadError(
        entry_point=f"{ep.name}={ep.value}",
        exc_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    )


def clear_adapter_cache() -> None:
    """
    Clear cached adapters and load errors.

    Forces the next :func:`load_adapters` call to rediscover adapters
    from entry points. Useful for testing or after installing new
    adapter packages.
    """
    global _adapters, _adapter_errors
    with _cache_lock:
        _adapters = None
        _adapter_errors = ()
    logger.debug("Adapter cache cleared")


def load_adapters(*, force_reload: bool = False) -> list[AdapterProtocol]:
    """
    Load all available adapters from entry points.

    Discovers and instantiates adapters registered under the
    ``devqubit.adapters`` entry point group. Results are cached
    for performance.

    Parameters
    ----------
    force_reload : bool, optional
        If True, discard cache and reload from entry points.
        Default is False.

    Returns
    -------
    list of AdapterProtocol
        List of successfully instantiated adapters.

    Notes
    -----
    Adapter load errors are captured and can be retrieved via
    :func:`adapter_load_errors`. This allows partial functionality
    when some adapters fail to load.
    """
    global _adapters, _adapter_errors

    with _cache_lock:
        if _adapters is not None and not force_reload:
            return list(_adapters)

        logger.debug("Loading adapters from entry points")

        loaded: list[AdapterProtocol] = []
        errors: list[AdapterLoadError] = []
        seen_names: set[str] = set()

        for ep in _iter_adapter_entry_points():
            ep_spec = f"{ep.name}={ep.value}"
            logger.debug("Loading adapter: %s", ep_spec)

            try:
                adapter_cls = ep.load()
                adapter = adapter_cls()

                name = getattr(adapter, "name", None)
                if not name:
                    raise TypeError(
                        f"Adapter class {adapter_cls.__name__} has no 'name' attribute"
                    )

                # Validate adapter conforms to protocol (runtime structural check)
                if not isinstance(adapter, AdapterProtocol):
                    raise TypeError(
                        f"Adapter {name} does not implement AdapterProtocol"
                    )

                # Guard against duplicate adapter names (prevents nondeterministic resolution)
                if name in seen_names:
                    raise ValueError(f"Duplicate adapter name detected: {name!r}")
                seen_names.add(name)

                # Warn if entry point name differs from adapter.name
                if ep.name != name:
                    logger.warning(
                        "Entry point name %r does not match adapter.name %r",
                        ep.name,
                        name,
                    )

                loaded.append(adapter)
                logger.info("Loaded adapter: %s", name)

            except Exception as e:
                err = _make_load_error(ep, e)
                errors.append(err)
                logger.warning(
                    "Failed to load adapter %s: %s", ep_spec, e, exc_info=True
                )

        _adapters = tuple(loaded)
        _adapter_errors = tuple(errors)

        logger.debug(
            "Adapter loading complete: %d loaded, %d errors",
            len(_adapters),
            len(_adapter_errors),
        )

        return list(_adapters)


def adapter_load_errors() -> list[AdapterLoadError]:
    """
    Get errors encountered during adapter loading.

    Returns
    -------
    list of AdapterLoadError
        List of adapter load errors. Empty if no errors occurred
        or if adapters haven't been loaded yet.
    """
    with _cache_lock:
        return list(_adapter_errors)


def list_available_adapters(*, force_reload: bool = False) -> list[str]:
    """
    List names of available adapters.

    Parameters
    ----------
    force_reload : bool, optional
        If True, reload adapters from entry points.
        Default is False.

    Returns
    -------
    list of str
        Names of successfully loaded adapters (sorted alphabetically).
    """
    return sorted(a.name for a in load_adapters(force_reload=force_reload))


def get_adapter_by_name(name: str) -> AdapterProtocol | None:
    """
    Get an adapter by its name.

    Parameters
    ----------
    name : str
        Adapter name to look up (e.g., "qiskit").

    Returns
    -------
    AdapterProtocol or None
        The adapter instance if found, None otherwise.
    """
    for adapter in load_adapters():
        if adapter.name == name:
            return adapter
    return None


def resolve_adapter(executor: Any, *, force_reload: bool = False) -> AdapterProtocol:
    """
    Resolve the adapter that supports a given executor.

    Finds the single adapter capable of handling the executor.
    Raises an error if no adapter matches or if multiple adapters
    claim support (which would indicate a bug in adapter implementations).

    Parameters
    ----------
    executor : Any
        SDK executor instance (e.g., Qiskit backend, PennyLane device).
    force_reload : bool, optional
        If True, reload adapters before resolving.
        Default is False.

    Returns
    -------
    AdapterProtocol
        The adapter that supports the executor.

    Raises
    ------
    ValueError
        If no adapter supports the executor, or if multiple adapters
        claim support.
    """
    executor_type = type(executor).__name__
    executor_module = getattr(executor, "__module__", "")

    logger.debug(
        "Resolving adapter for executor: %s (module=%s)",
        executor_type,
        executor_module,
    )

    # Semantics preserved: if not found and force_reload=False, retry once with force_reload=True.
    attempt_force_reload = force_reload
    adapters: list[AdapterProtocol] = []

    for _ in range(2):
        adapters = load_adapters(force_reload=attempt_force_reload)

        matches: list[AdapterProtocol] = []
        for adapter in adapters:
            try:
                if adapter.supports_executor(executor):
                    matches.append(adapter)
                    logger.debug("Adapter %s supports executor", adapter.name)
            except Exception as e:
                # Don't let one broken adapter prevent resolution
                logger.warning(
                    "Adapter %s raised exception in supports_executor: %s",
                    adapter.name,
                    e,
                    exc_info=True,
                )

        if len(matches) == 1:
            logger.debug("Resolved adapter: %s", matches[0].name)
            return matches[0]

        if len(matches) > 1:
            names = ", ".join(a.name for a in matches)
            raise ValueError(
                f"Multiple adapters match executor type {executor_type}: {names}. "
                "This indicates a bug in adapter implementations - each executor "
                "type should be supported by exactly one adapter."
            )

        if attempt_force_reload:
            break

        logger.debug("No adapter found, retrying with force_reload=True")
        attempt_force_reload = True

    available = ", ".join(a.name for a in adapters) if adapters else "(none installed)"
    errors = adapter_load_errors()

    error_lines = [
        f"No adapter found for executor type {executor_type} (module={executor_module!r}).",
        f"Available adapters: {available}.",
        "",
        "To fix this:",
        "  1. Install the appropriate adapter package:",
        "     pip install devqubit[qiskit ]     # For Qiskit",
        "     pip install devqubit[pennylane]   # For PennyLane",
        "     pip install devqubit[cirq]        # For Cirq",
        "     pip install devqubit[braket]      # For Amazon Braket",
        "     pip install devqubit[cudaq]       # For NVIDIA CUDA-Q",
    ]

    if errors:
        error_lines.append("")
        error_lines.append("Adapter load errors (may indicate missing dependencies):")
        for err in errors[:5]:
            error_lines.append(f"  - {err.entry_point}: {err.exc_type}: {err.message}")
        if len(errors) > 5:
            error_lines.append(f"  ... and {len(errors) - 5} more errors")

    raise ValueError("\n".join(error_lines))
