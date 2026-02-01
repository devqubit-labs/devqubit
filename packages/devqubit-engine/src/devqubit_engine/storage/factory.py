# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Storage factory for configuration-driven backend selection.

This module provides factory functions for creating storage backends
based on URLs or configuration. It supports:

- Local filesystem storage (``file://``)
- Amazon S3 (``s3://``) - requires ``devqubit-engine[s3]``
- Google Cloud Storage (``gs://``) - requires ``devqubit-engine[gcs]``
- Custom backends via entry points

Examples
--------
>>> from devqubit_engine.storage.factory import create_store, create_registry

>>> # Local storage (default)
>>> store = create_store()  # Uses ~/.devqubit/objects

>>> # Explicit local path
>>> store = create_store("file:///tmp/objects")

>>> # S3 storage
>>> store = create_store("s3://my-bucket/devqubit/objects")

>>> # Create both from config
>>> from devqubit_engine.config import get_config
>>> config = get_config()
>>> store = create_store(config=config)
>>> registry = create_registry(config=config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from devqubit_engine.config import Config, get_config
from devqubit_engine.storage.backends.local import (
    LocalRegistry,
    LocalStore,
    LocalWorkspace,
)
from devqubit_engine.storage.errors import StorageError
from devqubit_engine.storage.types import ObjectStoreProtocol, RegistryProtocol


logger = logging.getLogger(__name__)


class ObjectStoreBackend(Protocol):
    """
    Protocol for object store backend constructors.

    Backend constructors must be callable and return an ObjectStoreProtocol
    instance.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ObjectStoreProtocol: ...


class RegistryBackend(Protocol):
    """
    Protocol for registry backend constructors.

    Backend constructors must be callable and return a RegistryProtocol
    instance.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> RegistryProtocol: ...


def _lazy_import_s3() -> tuple[ObjectStoreBackend, RegistryBackend]:
    """
    Lazy import S3 backends.

    Returns
    -------
    tuple
        (S3Store, S3Registry) constructors.

    Raises
    ------
    ImportError
        If devqubit-engine[s3] is not installed.
    """
    from devqubit_engine.storage.backends.s3 import S3Registry, S3Store

    return S3Store, S3Registry


def _lazy_import_gcs() -> tuple[ObjectStoreBackend, RegistryBackend]:
    """
    Lazy import GCS backends.

    Returns
    -------
    tuple
        (GCSStore, GCSRegistry) constructors.

    Raises
    ------
    ImportError
        If devqubit-engine[gcs] is not installed.
    """
    from devqubit_engine.storage.backends.gcs import GCSRegistry, GCSStore

    return GCSStore, GCSRegistry


class StorageURL:
    """
    Parsed storage URL with scheme, location, and parameters.

    Handles URL parsing for various storage backends:

    - ``file://path`` or bare paths for local storage
    - ``s3://bucket/prefix`` for Amazon S3
    - ``gs://bucket/prefix`` for Google Cloud Storage

    Parameters
    ----------
    url : str
        Storage URL or local path. Bare paths (e.g., ``/tmp/data``)
        are automatically converted to ``file://`` URLs.

    Attributes
    ----------
    scheme : str
        URL scheme (e.g., "file", "s3", "gs").
    netloc : str
        Network location (bucket name for cloud storage).
    path : str
        Path component (filesystem path or object prefix).
    params : dict
        Query parameters from the URL.

    Examples
    --------
    >>> url = StorageURL("s3://my-bucket/prefix?region=us-east-1")
    >>> url.scheme
    's3'
    >>> url.bucket
    'my-bucket'
    >>> url.prefix
    'prefix'
    >>> url.params
    {'region': 'us-east-1'}
    """

    def __init__(self, url: str) -> None:
        import os

        # Treat empty / absolute / ~ paths as file:// paths
        if not url or url.startswith("/") or url.startswith("~"):
            url = f"file://{url}" if url else "file://"

        parsed = urlparse(url)
        self.scheme = parsed.scheme or "file"
        self.netloc = parsed.netloc
        self.path = parsed.path

        # For file:// URLs, handle path correctly
        if self.scheme == "file":
            full_path = parsed.netloc + parsed.path if parsed.netloc else parsed.path
            self.path = os.path.expanduser(full_path) if full_path else ""

        # Parse query parameters (first value only for each key)
        self.params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    @property
    def is_local(self) -> bool:
        """
        Check if this is a local filesystem URL.

        Returns
        -------
        bool
            True if scheme is "file".
        """
        return self.scheme == "file"

    @property
    def bucket(self) -> str:
        """
        Get bucket name for cloud storage.

        Returns
        -------
        str
            Bucket name (netloc component).
        """
        return self.netloc

    @property
    def prefix(self) -> str:
        """
        Get object prefix for cloud storage.

        Returns
        -------
        str
            Object prefix (path without leading slash).
        """
        return self.path.lstrip("/")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StorageURL(scheme={self.scheme!r}, netloc={self.netloc!r}, path={self.path!r})"


def parse_storage_url(url: str) -> StorageURL:
    """
    Parse a storage URL.

    Parameters
    ----------
    url : str
        Storage URL or local path.

    Returns
    -------
    StorageURL
        Parsed URL object.
    """
    return StorageURL(url)


# Backend registries (scheme -> constructor)
_STORE_BACKENDS: dict[str, ObjectStoreBackend] = {"file": LocalStore}
_REGISTRY_BACKENDS: dict[str, RegistryBackend] = {"file": LocalRegistry}


def register_store_backend(scheme: str, backend: ObjectStoreBackend) -> None:
    """
    Register an object store backend for a URL scheme.

    Parameters
    ----------
    scheme : str
        URL scheme (e.g., "file", "s3", "gs", "my-custom-scheme").
    backend : ObjectStoreBackend
        Callable that constructs an object store.

    Examples
    --------
    >>> register_store_backend("myscheme", MyCustomStore)
    >>> store = create_store("myscheme://location")
    """
    _STORE_BACKENDS[scheme] = backend
    logger.debug("Registered store backend: %s", scheme)


def register_registry_backend(scheme: str, backend: RegistryBackend) -> None:
    """
    Register a registry backend for a URL scheme.

    Parameters
    ----------
    scheme : str
        URL scheme (e.g., "file", "s3", "gs", "my-custom-scheme").
    backend : RegistryBackend
        Callable that constructs a registry.

    Examples
    --------
    >>> register_registry_backend("myscheme", MyCustomRegistry)
    >>> registry = create_registry("myscheme://location")
    """
    _REGISTRY_BACKENDS[scheme] = backend
    logger.debug("Registered registry backend: %s", scheme)


def _load_plugins() -> None:
    """
    Load storage backends from entry points.

    Discovers backends registered under the ``devqubit.storage``
    entry point group.

    Entry Point Contract
    --------------------
    Each entry point should return a class (not instance) that can be
    used as a store or registry constructor. The class is checked for
    protocol compatibility via class attribute inspection.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        from importlib_metadata import entry_points  # type: ignore[import-not-found]

    try:
        eps = entry_points(group="devqubit.storage")
    except TypeError:
        # Python < 3.10 compatibility
        eps = entry_points().get("devqubit.storage", [])

    for ep in eps:
        try:
            backend = ep.load()
            scheme = ep.name

            if not callable(backend):
                logger.warning(
                    "Storage plugin %s is not callable, skipping",
                    ep.name,
                )
                continue

            # Check class/type attributes WITHOUT instantiating
            # This avoids breaking plugins that require constructor args
            backend_attrs = set(dir(backend))

            # Check for store-like interface (has put_bytes, get_bytes)
            store_methods = {"put_bytes", "get_bytes", "exists"}
            if store_methods.issubset(backend_attrs):
                _STORE_BACKENDS[scheme] = backend
                logger.debug("Loaded store plugin: %s", scheme)

            # Check for registry-like interface (has save, load)
            registry_methods = {"save", "load", "list_runs"}
            if registry_methods.issubset(backend_attrs):
                _REGISTRY_BACKENDS[scheme] = backend
                logger.debug("Loaded registry plugin: %s", scheme)

        except Exception as e:
            logger.warning("Failed to load storage plugin %s: %s", ep.name, e)


# Load plugins on module import
_load_plugins()


def _get_storage_url(
    url: str | None,
    config_url: str,
    default_path: Path,
) -> StorageURL:
    """
    Resolve storage URL from arguments, config, or default.

    Parameters
    ----------
    url : str or None
        User-provided URL (highest priority).
    config_url : str
        URL from configuration (second priority).
    default_path : Path
        Default path when no URL is provided.

    Returns
    -------
    StorageURL
        Resolved and parsed storage URL.
    """
    if url:
        return StorageURL(url)
    if config_url:
        return StorageURL(config_url)
    return StorageURL(str(default_path))


def create_store(
    url: str | None = None,
    *,
    config: Config | None = None,
    backend_class: ObjectStoreBackend | None = None,
    **kwargs: Any,
) -> ObjectStoreProtocol:
    """
    Create an object store from URL or configuration.

    Parameters
    ----------
    url : str, optional
        Storage URL (overrides config). Supported schemes:

        - ``file://path`` - Local filesystem (default)
        - ``s3://bucket/prefix`` - Amazon S3 (requires ``pip install devqubit-engine[s3]``)
        - ``gs://bucket/prefix`` - Google Cloud Storage (requires ``pip install devqubit-engine[gcs]``)

    config : Config, optional
        Configuration object. Uses global config if not provided.
    backend_class : callable, optional
        Override backend constructor. Must return ObjectStoreProtocol.
    **kwargs
        Additional backend arguments (e.g., region, endpoint_url).

    Returns
    -------
    ObjectStoreProtocol
        Configured object store.

    Raises
    ------
    StorageError
        If the URL scheme is not supported.
    """
    cfg = config or get_config()
    parsed = _get_storage_url(url, cfg.storage_url, cfg.objects_dir)
    all_kwargs = {**parsed.params, **kwargs}

    logger.debug("Creating store: scheme=%s, path=%s", parsed.scheme, parsed.path)

    if backend_class is not None:
        ctor = backend_class
    elif parsed.scheme == "s3":
        try:
            S3Store, _ = _lazy_import_s3()
            ctor = S3Store
        except ImportError as e:
            raise StorageError(
                f"S3 storage requires additional dependencies. "
                f"Install with: pip install devqubit-engine[s3]\n"
                f"Original error: {e}"
            ) from e
    elif parsed.scheme == "gs":
        try:
            GCSStore, _ = _lazy_import_gcs()
            ctor = GCSStore
        except ImportError as e:
            raise StorageError(
                f"GCS storage requires additional dependencies. "
                f"Install with: pip install devqubit-engine[gcs]\n"
                f"Original error: {e}"
            ) from e
    else:
        ctor = _STORE_BACKENDS.get(parsed.scheme)

    if ctor is None:
        available = sorted(set(["file", "s3", "gs"] + list(_STORE_BACKENDS.keys())))
        raise StorageError(
            f"Unsupported storage scheme: {parsed.scheme!r}. "
            f"Available: {', '.join(available)}."
        )

    if parsed.is_local:
        root = Path(parsed.path) if parsed.path else cfg.objects_dir
        return ctor(root, **all_kwargs)

    return ctor(bucket=parsed.bucket, prefix=parsed.prefix, **all_kwargs)


def create_registry(
    url: str | None = None,
    *,
    config: Config | None = None,
    backend_class: RegistryBackend | None = None,
    **kwargs: Any,
) -> RegistryProtocol:
    """
    Create a run registry from URL or configuration.

    Parameters
    ----------
    url : str, optional
        Registry URL (overrides config). Supported schemes:

        - ``file://path`` - Local SQLite (default)
        - ``s3://bucket/prefix`` - Amazon S3 (requires ``pip install devqubit-engine[s3]``)
        - ``gs://bucket/prefix`` - Google Cloud Storage (requires ``pip install devqubit-engine[gcs]``)

    config : Config, optional
        Configuration object. Uses global config if not provided.
    backend_class : callable, optional
        Override backend constructor. Must return RegistryProtocol.
    **kwargs
        Additional backend arguments.

    Returns
    -------
    RegistryProtocol
        Configured registry.

    Raises
    ------
    StorageError
        If the URL scheme is not supported.
    """
    cfg = config or get_config()
    parsed = _get_storage_url(url, cfg.registry_url, cfg.root_dir)
    all_kwargs = {**parsed.params, **kwargs}

    logger.debug("Creating registry: scheme=%s, path=%s", parsed.scheme, parsed.path)

    if backend_class is not None:
        ctor = backend_class
    elif parsed.scheme == "s3":
        try:
            _, S3Registry = _lazy_import_s3()
            ctor = S3Registry
        except ImportError as e:
            raise StorageError(
                f"S3 storage requires additional dependencies. "
                f"Install with: pip install devqubit-engine[s3]\n"
                f"Original error: {e}"
            ) from e
    elif parsed.scheme == "gs":
        try:
            _, GCSRegistry = _lazy_import_gcs()
            ctor = GCSRegistry
        except ImportError as e:
            raise StorageError(
                f"GCS storage requires additional dependencies. "
                f"Install with: pip install devqubit-engine[gcs]\n"
                f"Original error: {e}"
            ) from e
    else:
        ctor = _REGISTRY_BACKENDS.get(parsed.scheme)

    if ctor is None:
        available = sorted(set(["file", "s3", "gs"] + list(_REGISTRY_BACKENDS.keys())))
        raise StorageError(
            f"Unsupported registry scheme: {parsed.scheme!r}. "
            f"Available: {', '.join(available)}."
        )

    if parsed.is_local:
        root = Path(parsed.path) if parsed.path else cfg.root_dir
        return ctor(root, **all_kwargs)

    return ctor(bucket=parsed.bucket, prefix=parsed.prefix, **all_kwargs)


def create_workspace(
    url: str | None = None,
    *,
    config: Config | None = None,
    **kwargs: Any,
) -> LocalWorkspace:
    """
    Create a combined local workspace (store + registry).

    Parameters
    ----------
    url : str, optional
        Workspace URL or path. Only local storage is supported.
        If omitted, uses config root_dir.
    config : Config, optional
        Configuration object. Uses global config if not provided.
    **kwargs
        Additional LocalWorkspace arguments.

    Returns
    -------
    LocalWorkspace
        Workspace with store and registry at the specified location.

    Raises
    ------
    StorageError
        If a remote URL is provided.
    """
    cfg = config or get_config()
    parsed = _get_storage_url(url, "", cfg.root_dir)

    if not parsed.is_local:
        raise StorageError(
            "Remote workspaces not yet supported. "
            "Use create_store() and create_registry() separately for remote storage."
        )

    root = Path(parsed.path) if parsed.path else cfg.root_dir
    logger.debug("Creating workspace at %s", root)
    return LocalWorkspace(root, **kwargs)
