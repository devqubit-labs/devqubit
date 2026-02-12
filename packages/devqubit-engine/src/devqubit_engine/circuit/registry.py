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
from importlib.metadata import EntryPoint, entry_points
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

_GROUP_LOADERS = "devqubit.circuit.loaders"
_GROUP_SERIALIZERS = "devqubit.circuit.serializers"
_GROUP_SUMMARIZERS = "devqubit.circuit.summarizers"


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

    def _record_error(
        self,
        *,
        ep: EntryPoint,
        exc: Exception,
        handler_type: str,
    ) -> None:
        """
        Record handler load error in a consistent way.

        Parameters
        ----------
        ep : EntryPoint
            Entry point that failed to load.
        exc : Exception
            Exception that was raised.
        handler_type : str
            Human-friendly handler type label (e.g. "loader").
        """
        error = HandlerLoadError(
            entry_point=f"{ep.name}={ep.value}",
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        self._errors.append(error)
        logger.warning(
            "Failed to load %s %s: %s", handler_type, ep.name, exc, exc_info=True
        )

    def _get_entry_points_by_group(self) -> dict[str, list[EntryPoint]]:
        """
        Get entry points grouped by relevant groups, handling API differences.

        Returns
        -------
        dict
            Mapping group name -> list of entry points.
        """
        eps = entry_points()

        # Python 3.10+: EntryPoints has .select()
        select = getattr(eps, "select", None)
        if callable(select):
            return {
                _GROUP_LOADERS: list(eps.select(group=_GROUP_LOADERS)),
                _GROUP_SERIALIZERS: list(eps.select(group=_GROUP_SERIALIZERS)),
                _GROUP_SUMMARIZERS: list(eps.select(group=_GROUP_SUMMARIZERS)),
            }

        # Python 3.9 fallback: entry_points() returns dict-like mapping
        # (types vary between implementations, so keep it simple)
        return {
            _GROUP_LOADERS: list(eps.get(_GROUP_LOADERS, [])),
            _GROUP_SERIALIZERS: list(eps.get(_GROUP_SERIALIZERS, [])),
            _GROUP_SUMMARIZERS: list(eps.get(_GROUP_SUMMARIZERS, [])),
        }

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

            groups = self._get_entry_points_by_group()

            # Load loaders
            for ep in groups[_GROUP_LOADERS]:
                self._load_handler(ep, self._loaders, "loader")

            # Load serializers
            for ep in groups[_GROUP_SERIALIZERS]:
                self._load_handler(ep, self._serializers, "serializer")

            # Load summarizers
            for ep in groups[_GROUP_SUMMARIZERS]:
                try:
                    func = ep.load()
                    sdk = SDK(ep.name)
                    self._summarizers[sdk] = func
                    logger.debug("Loaded summarizer for %s", sdk.value)
                except Exception as e:
                    self._record_error(ep=ep, exc=e, handler_type="summarizer")

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
        ep: EntryPoint,
        registry: dict[SDK, Any],
        handler_type: str,
    ) -> None:
        """
        Load a single handler from an entry point.

        Parameters
        ----------
        ep : EntryPoint
            Entry point to load.
        registry : dict
            Target registry dict to populate (keyed by SDK).
        handler_type : str
            Handler type label ("loader" or "serializer").
        """
        try:
            handler_cls = ep.load()
            handler = handler_cls()

            name = getattr(handler, "name", None)
            if not name:
                raise TypeError(f"{handler_type.title()} missing 'name' attribute")

            handler_sdk = getattr(handler, "sdk", None)
            if not handler_sdk:
                raise TypeError(f"{handler_type.title()} missing 'sdk' attribute")

            if isinstance(handler_sdk, str):
                handler_sdk = SDK(handler_sdk)

            if handler_sdk in registry:
                raise ValueError(
                    f"Duplicate {handler_type} for sdk={handler_sdk.value!r}"
                )

            registry[handler_sdk] = handler
            logger.debug("Loaded %s for %s: %s", handler_type, handler_sdk.value, name)

        except Exception as e:
            self._record_error(ep=ep, exc=e, handler_type=handler_type)

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

        loader = self._loaders.get(sdk)
        if loader is not None:
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

        serializer = self._serializers.get(sdk)
        if serializer is not None:
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
            except Exception as e:
                # Do not fail discovery due to a buggy adapter; keep it quiet by default.
                logger.debug(
                    "Serializer '%s' can_serialize() raised: %s",
                    getattr(serializer, "name", "<unknown>"),
                    e,
                    exc_info=True,
                )
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
