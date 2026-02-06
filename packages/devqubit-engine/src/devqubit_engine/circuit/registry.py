# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit handler registry.

This module discovers and manages circuit handlers (loaders, serializers,
and summarizers) via entry points. Handlers are registered by adapter
packages and loaded on demand.

Entry Point Groups
------------------
- ``devqubit.circuit.loaders`` - Circuit loading from serialized formats
- ``devqubit.circuit.serializers`` - Circuit serialization to formats
- ``devqubit.circuit.summarizers`` - Circuit summarization functions
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Protocol, runtime_checkable

from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    LoadedCircuit,
)
from devqubit_engine.circuit.summary import CircuitSummary
from devqubit_engine.errors import DevQubitError


logger = logging.getLogger(__name__)


class CircuitError(DevQubitError):
    """Base exception for circuit operations."""


class LoaderError(CircuitError):
    """Raised when circuit loading fails."""


class SerializerError(CircuitError):
    """Raised when circuit serialization fails."""


@runtime_checkable
class CircuitLoaderProtocol(Protocol):
    """
    Protocol for circuit loaders.

    Loaders convert serialized circuit data back into SDK-native circuit
    objects. Each adapter package provides a loader for its SDK.

    Attributes
    ----------
    name : str
        Loader name for identification.
    """

    name: str

    @property
    def sdk(self) -> SDK:
        """Get the SDK this loader handles."""
        ...

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        ...

    def load(self, data: CircuitData) -> LoadedCircuit:
        """Load circuit from serialized data."""
        ...


@runtime_checkable
class CircuitSerializerProtocol(Protocol):
    """
    Protocol for circuit serializers.

    Serializers convert SDK-native circuit objects to serialized formats.
    Each adapter package provides a serializer for its SDK.

    Attributes
    ----------
    name : str
        Serializer name for identification.
    """

    name: str

    @property
    def sdk(self) -> SDK:
        """Get the SDK this serializer handles."""
        ...

    @property
    def supported_formats(self) -> list[CircuitFormat]:
        """Get supported serialization formats."""
        ...

    def can_serialize(self, circuit: Any) -> bool:
        """Check if this serializer can handle a circuit."""
        ...

    def serialize(
        self,
        circuit: Any,
        fmt: CircuitFormat,
        *,
        name: str = "",
        index: int = 0,
    ) -> CircuitData:
        """Serialize circuit to specified format."""
        ...


#: Type alias for circuit summarizer functions.
CircuitSummarizerFunc = Callable[[Any], CircuitSummary]


@dataclass(frozen=True)
class HandlerLoadError:
    """
    Captures a handler load failure for diagnostics.

    Attributes
    ----------
    entry_point : str
        Entry point specification (name=value).
    exc_type : str
        Exception type name.
    message : str
        Error message.
    traceback : str
        Full traceback.
    """

    entry_point: str
    exc_type: str
    message: str
    traceback: str


class _Registry:
    """
    Internal registry for circuit handlers.

    Provides lazy-loading of handlers from entry points with
    caching, error tracking, and thread safety.
    """

    def __init__(self) -> None:
        self._loaders: dict[SDK, CircuitLoaderProtocol] = {}
        self._serializers: dict[SDK, CircuitSerializerProtocol] = {}
        self._summarizers: dict[SDK, CircuitSummarizerFunc] = {}
        self._errors: list[HandlerLoadError] = []
        self._loaded = False
        self._lock = threading.RLock()

    def _get_entry_points(self, group: str) -> list[Any]:
        """Get entry points for a group, handling API differences."""
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+ API
            return list(eps.select(group=group))
        # Python 3.9 fallback: entry_points() returns a dict
        return list(eps.get(group, []))

    def _ensure_loaded(self) -> None:
        """Load handlers from entry points if not already loaded."""
        if self._loaded:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                return

            logger.debug("Loading handlers from entry points")

            self._errors.clear()

            # Load loaders
            for ep in self._get_entry_points("devqubit.circuit.loaders"):
                self._load_handler(ep, self._loaders, "loader")

            # Load serializers
            for ep in self._get_entry_points("devqubit.circuit.serializers"):
                self._load_handler(ep, self._serializers, "serializer")

            # Load summarizers
            for ep in self._get_entry_points("devqubit.circuit.summarizers"):
                try:
                    func = ep.load()
                    sdk = SDK(ep.name)
                    self._summarizers[sdk] = func
                    logger.debug("Loaded summarizer for %s", sdk.value)
                except Exception as e:
                    error = HandlerLoadError(
                        entry_point=f"{ep.name}={ep.value}",
                        exc_type=type(e).__name__,
                        message=str(e),
                        traceback=traceback.format_exc(),
                    )
                    self._errors.append(error)
                    logger.warning(
                        "Failed to load summarizer %s: %s", ep.name, e, exc_info=True
                    )

            self._loaded = True

            logger.debug(
                "Registry loaded: %d loaders, %d serializers, %d summarizers, %d errors",
                len(self._loaders),
                len(self._serializers),
                len(self._summarizers),
                len(self._errors),
            )

    def _load_handler(
        self,
        ep: Any,
        registry: dict[SDK, Any],
        handler_type: str,
    ) -> None:
        """Load a single handler from an entry point."""
        try:
            handler_cls = ep.load()
            handler = handler_cls()

            if not getattr(handler, "name", None):
                raise TypeError(f"{handler_type.title()} missing 'name' attribute")

            handler_sdk = getattr(handler, "sdk", None)
            if not handler_sdk:
                raise TypeError(f"{handler_type.title()} missing 'sdk' attribute")

            # Coerce string to SDK enum if needed
            if isinstance(handler_sdk, str):
                handler_sdk = SDK(handler_sdk)

            if handler_sdk in registry:
                raise ValueError(
                    f"Duplicate {handler_type} for sdk={handler_sdk.value!r}"
                )

            registry[handler_sdk] = handler
            logger.debug(
                "Loaded %s for %s: %s", handler_type, handler_sdk.value, handler.name
            )

        except Exception as e:
            error = HandlerLoadError(
                entry_point=f"{ep.name}={ep.value}",
                exc_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )
            self._errors.append(error)
            logger.warning(
                "Failed to load %s %s: %s", handler_type, ep.name, e, exc_info=True
            )

    def clear(self) -> None:
        """Clear all cached handlers and errors."""
        with self._lock:
            self._loaders.clear()
            self._serializers.clear()
            self._summarizers.clear()
            self._errors.clear()
            self._loaded = False

    def reload(self) -> None:
        """Force reload of all handlers from entry points."""
        self.clear()
        self._ensure_loaded()

    @property
    def errors(self) -> list[HandlerLoadError]:
        """Get errors encountered during handler loading."""
        self._ensure_loaded()
        return list(self._errors)

    def get_loader(self, sdk: SDK) -> CircuitLoaderProtocol:
        """
        Get loader for an SDK.

        Parameters
        ----------
        sdk : SDK
            Target SDK.

        Returns
        -------
        CircuitLoaderProtocol
            Loader instance.

        Raises
        ------
        LoaderError
            If no loader is available for the SDK.
        """
        self._ensure_loaded()

        if loader := self._loaders.get(sdk):
            return loader

        available = ", ".join(sorted(s.value for s in self._loaders)) or "(none)"
        raise LoaderError(
            f"No loader for SDK '{sdk.value}'. Available: {available}. "
            f"Install the adapter package (e.g., pip install devqubit-{sdk.value})."
        )

    def get_serializer(self, sdk: SDK) -> CircuitSerializerProtocol:
        """
        Get serializer for an SDK.

        Parameters
        ----------
        sdk : SDK
            Target SDK.

        Returns
        -------
        CircuitSerializerProtocol
            Serializer instance.

        Raises
        ------
        SerializerError
            If no serializer is available for the SDK.
        """
        self._ensure_loaded()

        if serializer := self._serializers.get(sdk):
            return serializer

        available = ", ".join(sorted(s.value for s in self._serializers)) or "(none)"
        raise SerializerError(
            f"No serializer for SDK '{sdk.value}'. Available: {available}."
        )

    def get_serializer_for_circuit(self, circuit: Any) -> CircuitSerializerProtocol:
        """
        Find serializer that can handle a circuit object.

        Parameters
        ----------
        circuit : Any
            Circuit object to serialize.

        Returns
        -------
        CircuitSerializerProtocol
            Compatible serializer.

        Raises
        ------
        SerializerError
            If no serializer can handle the circuit.
        """
        self._ensure_loaded()

        for serializer in self._serializers.values():
            try:
                if serializer.can_serialize(circuit):
                    return serializer
            except Exception:
                continue

        raise SerializerError(
            f"No serializer for circuit type: {type(circuit).__name__}. "
            f"Ensure the appropriate adapter package is installed."
        )

    def get_summarizer(self, sdk: SDK) -> CircuitSummarizerFunc | None:
        """
        Get summarizer for an SDK.

        Parameters
        ----------
        sdk : SDK
            Target SDK.

        Returns
        -------
        CircuitSummarizerFunc or None
            Summarizer function, or None if not registered.
        """
        self._ensure_loaded()
        return self._summarizers.get(sdk)

    def list_available(self) -> dict[str, list[str]]:
        """
        List all available handlers.

        Returns
        -------
        dict
            Dictionary with 'loaders', 'serializers', and 'summarizers' keys,
            each containing a sorted list of available SDK values.
        """
        self._ensure_loaded()
        return {
            "loaders": sorted(s.value for s in self._loaders),
            "serializers": sorted(s.value for s in self._serializers),
            "summarizers": sorted(s.value for s in self._summarizers),
        }


# Global registry instance
_registry = _Registry()


def get_loader(sdk: SDK) -> CircuitLoaderProtocol:
    """
    Get loader for an SDK.

    Parameters
    ----------
    sdk : SDK
        Target SDK.

    Returns
    -------
    CircuitLoaderProtocol
        Loader instance.

    Raises
    ------
    LoaderError
        If no loader is available for the SDK.
    """
    return _registry.get_loader(sdk)


def get_serializer(sdk: SDK) -> CircuitSerializerProtocol:
    """
    Get serializer for an SDK.

    Parameters
    ----------
    sdk : SDK
        Target SDK.

    Returns
    -------
    CircuitSerializerProtocol
        Serializer instance.

    Raises
    ------
    SerializerError
        If no serializer is available for the SDK.
    """
    return _registry.get_serializer(sdk)


def get_serializer_for_circuit(circuit: Any) -> CircuitSerializerProtocol:
    """
    Find serializer that can handle a circuit object.

    Parameters
    ----------
    circuit : Any
        Circuit object to serialize.

    Returns
    -------
    CircuitSerializerProtocol
        Compatible serializer.

    Raises
    ------
    SerializerError
        If no serializer can handle the circuit.
    """
    return _registry.get_serializer_for_circuit(circuit)


def get_summarizer(sdk: SDK) -> CircuitSummarizerFunc | None:
    """
    Get summarizer for an SDK.

    Parameters
    ----------
    sdk : SDK
        Target SDK.

    Returns
    -------
    CircuitSummarizerFunc or None
        Summarizer function, or None if not registered.
    """
    return _registry.get_summarizer(sdk)


def list_available() -> dict[str, list[str]]:
    """
    List all available handlers.

    Returns
    -------
    dict
        Dictionary with 'loaders', 'serializers', and 'summarizers' keys.
    """
    return _registry.list_available()


def handler_errors() -> list[HandlerLoadError]:
    """
    Get errors encountered during handler loading.

    Useful for diagnostics when handlers fail to load.

    Returns
    -------
    list of HandlerLoadError
        List of load errors.
    """
    return _registry.errors


def clear_cache() -> None:
    """Clear all cached handlers."""
    _registry.clear()
    logger.debug("Handler cache cleared")


def reload_handlers() -> None:
    """Force reload of all handlers from entry points."""
    _registry.reload()
    logger.debug("Handlers reloaded")


def load_circuit(data: CircuitData) -> LoadedCircuit:
    """
    Load circuit from serialized data.

    Convenience function that gets the appropriate loader and loads
    the circuit in one call.

    Parameters
    ----------
    data : CircuitData
        Serialized circuit data.

    Returns
    -------
    LoadedCircuit
        Loaded circuit container.

    Raises
    ------
    LoaderError
        If no loader is available or loading fails.
    """
    return get_loader(data.sdk).load(data)


def serialize_circuit(
    circuit: Any,
    fmt: CircuitFormat | None = None,
    *,
    name: str = "",
    index: int = 0,
) -> CircuitData:
    """
    Serialize circuit to a format.

    Automatically finds the appropriate serializer based on circuit type.

    Parameters
    ----------
    circuit : Any
        SDK-native circuit object.
    fmt : CircuitFormat, optional
        Target format. Uses serializer's default if not specified.
    name : str, optional
        Circuit name for metadata.
    index : int, optional
        Circuit index in batch.

    Returns
    -------
    CircuitData
        Serialized circuit data.

    Raises
    ------
    SerializerError
        If no serializer can handle the circuit or serialization fails.
    """
    serializer = get_serializer_for_circuit(circuit)

    if fmt is None:
        if not serializer.supported_formats:
            raise SerializerError(
                f"Serializer '{serializer.name}' has no supported formats"
            )
        fmt = serializer.supported_formats[0]

    return serializer.serialize(circuit, fmt, name=name, index=index)
