# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Remote storage backends for centralized experiment tracking.

This module provides HTTP-based storage backends that communicate with
a remote server for centralized experiment tracking in team environments.

The backends implement the standard storage protocols:

- :class:`RemoteStore` - Content-addressed artifact storage via HTTP API
- :class:`RemoteRegistry` - Run metadata registry via HTTP API

Configuration
-------------
Remote backends are configured via environment variables or storage URLs:

.. code-block:: bash

    # Environment variables
    export DEVQUBIT_TRACKING_URI=https://tracking.company.com
    export DEVQUBIT_API_TOKEN=dqt_xxxxxxxxxxxx
    export DEVQUBIT_WORKSPACE=my-workspace

Examples
--------
Direct instantiation:

>>> from devqubit_engine.storage.backends.remote import RemoteStore, RemoteRegistry
>>> store = RemoteStore(
...     server_url="https://tracking.company.com",
...     token="dqt_xxxxxxxxxxxx",
...     workspace="my-workspace",
... )
>>> registry = RemoteRegistry(
...     server_url="https://tracking.company.com",
...     token="dqt_xxxxxxxxxxxx",
...     workspace="my-workspace",
... )

Notes
-----
These backends require network connectivity to the remote server and valid
API credentials. For offline or local-only usage, use the local backends.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import requests
from devqubit_engine.storage.errors import (
    ObjectNotFoundError,
    RegistryError,
    RunNotFoundError,
    StorageError,
)
from devqubit_engine.storage.types import (
    ArtifactRef,
    BaselineInfo,
    ObjectStoreProtocol,
    RegistryProtocol,
    RunSummary,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


if TYPE_CHECKING:
    from devqubit_engine.tracking.record import RunRecord


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 0.5  # seconds

# Environment variable names
ENV_TRACKING_URI = "DEVQUBIT_TRACKING_URI"
ENV_API_TOKEN = "DEVQUBIT_API_TOKEN"
ENV_WORKSPACE = "DEVQUBIT_WORKSPACE"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RemoteClientConfig:
    """
    Configuration for Remote API client.

    Parameters
    ----------
    server_url : str
        Base URL of the remote server (e.g., "https://tracking.example.com").
    token : str
        API token for authentication (format: "dqt_xxxxxxxxxxxx").
    workspace : str
        Workspace identifier (UUID or slug).
    timeout : float, optional
        Request timeout in seconds. Default is 30.0.
    retry_attempts : int, optional
        Number of retry attempts for transient failures. Default is 3.
    retry_backoff : float, optional
        Base backoff time between retries in seconds. Default is 0.5.
    verify_ssl : bool, optional
        Whether to verify SSL certificates. Default is True.

    Raises
    ------
    ValueError
        If server_url or token is empty.
    """

    server_url: str
    token: str
    workspace: str
    timeout: float = DEFAULT_TIMEOUT
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        """Validate configuration on creation."""
        if not self.server_url:
            raise ValueError("server_url is required")
        if not self.token:
            raise ValueError(
                "API token is required. Set DEVQUBIT_API_TOKEN environment variable "
                "or pass token parameter."
            )
        if not self.workspace:
            raise ValueError(
                "Workspace is required. Set DEVQUBIT_WORKSPACE environment variable "
                "or include workspace in the storage URL."
            )

    @classmethod
    def from_env(
        cls,
        server_url: str | None = None,
        token: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> RemoteClientConfig:
        """
        Create configuration from environment variables with overrides.

        Parameters
        ----------
        server_url : str, optional
            Override for DEVQUBIT_TRACKING_URI.
        token : str, optional
            Override for DEVQUBIT_API_TOKEN.
        workspace : str, optional
            Override for DEVQUBIT_WORKSPACE.
        **kwargs
            Additional configuration options.

        Returns
        -------
        RemoteClientConfig
            Configuration instance.
        """
        return cls(
            server_url=server_url or os.getenv(ENV_TRACKING_URI, ""),
            token=token or os.getenv(ENV_API_TOKEN, ""),
            workspace=workspace or os.getenv(ENV_WORKSPACE, ""),
            **kwargs,
        )


# =============================================================================
# HTTP Client Base
# =============================================================================


class RemoteClient:
    """
    Base HTTP client for Remote API communication.

    Provides common functionality for making authenticated requests to the
    remote server with automatic retries, timeout handling, and error mapping.

    Parameters
    ----------
    config : RemoteClientConfig
        Client configuration.

    Attributes
    ----------
    config : RemoteClientConfig
        The client configuration.
    session : requests.Session
        Configured HTTP session with retry logic.
    """

    def __init__(self, config: RemoteClientConfig) -> None:
        self.config = config
        self.session = self._create_session()
        self._api_base = f"{config.server_url.rstrip('/')}/api/v1"

        logger.debug(
            "RemoteClient initialized: server=%s, workspace=%s",
            config.server_url,
            config.workspace,
        )

    def _create_session(self) -> requests.Session:
        """
        Create configured HTTP session with retry logic.

        Returns
        -------
        requests.Session
            Session with authentication headers and retry adapter.
        """
        session = requests.Session()

        # Set default headers
        session.headers.update(
            {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "devqubit-engine/1.0",
            }
        )

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.retry_attempts,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        """
        Make an HTTP request to the Remote API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE, etc.).
        path : str
            API path (e.g., "/runs" or "/runs/{run_id}").
        json : Any, optional
            JSON body to send.
        data : bytes, optional
            Raw bytes body to send.
        params : dict, optional
            Query parameters.
        headers : dict, optional
            Additional headers.
        timeout : float, optional
            Request timeout override.

        Returns
        -------
        requests.Response
            HTTP response object.

        Raises
        ------
        StorageError
            If request fails with non-recoverable error.
        """
        url = f"{self._api_base}{path}"
        request_timeout = timeout or self.config.timeout
        request_headers = headers or {}

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                data=data,
                params=params,
                headers=request_headers,
                timeout=request_timeout,
                verify=self.config.verify_ssl,
            )

            logger.debug(
                "Remote API %s %s -> %d",
                method,
                path,
                response.status_code,
            )

            return response

        except requests.exceptions.Timeout as e:
            raise StorageError(
                f"Request timeout after {request_timeout}s: {method} {path}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise StorageError(
                f"Connection error to remote server {self.config.server_url}: {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise StorageError(f"Request failed: {method} {path}: {e}") from e

    def _raise_for_status(
        self,
        response: requests.Response,
        context: str = "",
    ) -> None:
        """
        Raise appropriate exception for error responses.

        Parameters
        ----------
        response : requests.Response
            HTTP response to check.
        context : str, optional
            Context string for error messages.

        Raises
        ------
        ObjectNotFoundError
            If response is 404.
        StorageError
            For other error status codes.
        """
        if response.ok:
            return

        try:
            error_detail = response.json().get("detail", response.text)
        except (ValueError, KeyError):
            error_detail = response.text

        msg = f"{context}: {error_detail}" if context else error_detail

        if response.status_code == 404:
            raise ObjectNotFoundError(msg)
        if response.status_code == 401:
            raise StorageError(f"Authentication failed: {msg}")
        if response.status_code == 403:
            raise StorageError(f"Access denied: {msg}")

        raise StorageError(f"Remote API error ({response.status_code}): {msg}")

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self) -> RemoteClient:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and close session."""
        self.close()


# =============================================================================
# Object Store Implementation
# =============================================================================


class RemoteStore(ObjectStoreProtocol):
    """
    Content-addressed object store backed by Remote API.

    Stores and retrieves binary artifacts via the remote server's artifact
    endpoints. Objects are identified by their SHA-256 content digest,
    enabling deduplication and integrity verification.

    Parameters
    ----------
    server_url : str
        Base URL of the remote server.
    token : str
        API token for authentication.
    workspace : str
        Workspace identifier.
    timeout : float, optional
        Request timeout in seconds. Default is 30.0.
    retry_attempts : int, optional
        Number of retry attempts. Default is 3.
    verify_ssl : bool, optional
        Whether to verify SSL certificates. Default is True.

    Examples
    --------
    >>> store = RemoteStore(
    ...     server_url="https://tracking.example.com",
    ...     token="dqt_xxxxxxxxxxxx",
    ...     workspace="my-workspace",
    ... )
    >>> digest = store.put_bytes(b"hello world")
    >>> store.get_bytes(digest)
    b'hello world'

    Notes
    -----
    The remote server handles actual storage (typically S3 or similar).
    This client only provides the interface for upload/download.
    """

    def __init__(
        self,
        server_url: str,
        token: str,
        workspace: str,
        timeout: float = DEFAULT_TIMEOUT,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        verify_ssl: bool = True,
    ) -> None:
        config = RemoteClientConfig(
            server_url=server_url,
            token=token,
            workspace=workspace,
            timeout=timeout,
            retry_attempts=retry_attempts,
            verify_ssl=verify_ssl,
        )
        self._client = RemoteClient(config)
        self._workspace = workspace

    @classmethod
    def from_env(cls) -> RemoteStore:
        """
        Create RemoteStore from environment variables.

        Environment Variables
        ---------------------
        DEVQUBIT_TRACKING_URI : str
            Remote server URL (required).
        DEVQUBIT_API_TOKEN : str
            API token for authentication (required).
        DEVQUBIT_WORKSPACE : str
            Workspace identifier (required).

        Returns
        -------
        RemoteStore
            Configured store instance.

        Raises
        ------
        StorageError
            If required environment variables are not set.
        """
        server_url = os.getenv(ENV_TRACKING_URI, "")
        token = os.getenv(ENV_API_TOKEN, "")
        workspace = os.getenv(ENV_WORKSPACE, "")

        if not server_url:
            raise StorageError(f"Environment variable {ENV_TRACKING_URI} is required")
        if not token:
            raise StorageError(f"Environment variable {ENV_API_TOKEN} is required")
        if not workspace:
            raise StorageError(f"Environment variable {ENV_WORKSPACE} is required")

        return cls(server_url=server_url, token=token, workspace=workspace)

    def put_bytes(self, data: bytes) -> str:
        """
        Store bytes and return content digest.

        Uploads data to the remote server and returns the SHA-256 digest.
        If the object already exists on the server, the upload may be
        deduplicated.

        Parameters
        ----------
        data : bytes
            Binary data to store.

        Returns
        -------
        str
            Content digest in format ``sha256:<64-hex-chars>``.

        Raises
        ------
        StorageError
            If upload fails.

        Examples
        --------
        >>> digest = store.put_bytes(b"experiment data")
        >>> digest
        'sha256:a1b2c3d4...'
        """
        # Compute digest locally
        hex_digest = hashlib.sha256(data).hexdigest()
        digest = f"sha256:{hex_digest}"

        # Check if already exists (skip upload)
        if self.exists(digest):
            logger.debug("Object already exists, skipping upload: %s", digest[:24])
            return digest

        # Upload to remote server
        response = self._client._request(
            "POST",
            f"/workspaces/{self._workspace}/objects",
            data=data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Content-Digest": digest,
            },
        )
        self._client._raise_for_status(
            response, f"Failed to upload object {digest[:24]}"
        )

        logger.debug("Uploaded object: %s (%d bytes)", digest[:24], len(data))
        return digest

    def get_bytes(self, digest: str) -> bytes:
        """
        Retrieve bytes by content digest.

        Parameters
        ----------
        digest : str
            Content digest in format ``sha256:<64-hex-chars>``.

        Returns
        -------
        bytes
            The stored binary data.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        StorageError
            If retrieval fails.

        Examples
        --------
        >>> data = store.get_bytes("sha256:a1b2c3d4...")
        """
        self._validate_digest(digest)

        response = self._client._request(
            "GET",
            f"/workspaces/{self._workspace}/objects/{digest}",
            headers={"Accept": "application/octet-stream"},
        )
        self._client._raise_for_status(response, f"Object not found: {digest[:24]}")

        return response.content

    def get_bytes_or_none(self, digest: str) -> bytes | None:
        """
        Retrieve bytes or None if not found.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bytes or None
            The stored data, or None if object doesn't exist.
        """
        try:
            return self.get_bytes(digest)
        except ObjectNotFoundError:
            return None

    def exists(self, digest: str) -> bool:
        """
        Check if object exists in the store.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bool
            True if object exists.
        """
        self._validate_digest(digest)

        response = self._client._request(
            "HEAD",
            f"/workspaces/{self._workspace}/objects/{digest}",
        )
        return response.status_code == 200

    def delete(self, digest: str) -> bool:
        """
        Delete object by digest.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        bool
            True if object was deleted, False if it didn't exist.

        Raises
        ------
        StorageError
            If deletion fails for reasons other than non-existence.
        """
        self._validate_digest(digest)

        response = self._client._request(
            "DELETE",
            f"/workspaces/{self._workspace}/objects/{digest}",
        )

        if response.status_code == 404:
            return False
        self._client._raise_for_status(response, f"Failed to delete {digest[:24]}")
        return True

    def list_digests(self, prefix: str | None = None) -> Iterator[str]:
        """
        List stored object digests.

        Parameters
        ----------
        prefix : str, optional
            Filter by digest prefix (e.g., "sha256:ab").

        Yields
        ------
        str
            Content digests.

        Notes
        -----
        Results are paginated internally. Large stores may take
        significant time to fully iterate.
        """
        params: dict[str, Any] = {"limit": 100}
        if prefix:
            params["prefix"] = prefix

        offset = 0
        while True:
            params["offset"] = offset
            response = self._client._request(
                "GET",
                f"/workspaces/{self._workspace}/objects",
                params=params,
            )
            self._client._raise_for_status(response, "Failed to list objects")

            data = response.json()
            digests = data.get("digests", [])

            if not digests:
                break

            yield from digests
            offset += len(digests)

            if len(digests) < params["limit"]:
                break

    def get_size(self, digest: str) -> int:
        """
        Get size of stored object in bytes.

        Parameters
        ----------
        digest : str
            Content digest.

        Returns
        -------
        int
            Size in bytes.

        Raises
        ------
        ObjectNotFoundError
            If object does not exist.
        """
        self._validate_digest(digest)

        response = self._client._request(
            "HEAD",
            f"/workspaces/{self._workspace}/objects/{digest}",
        )
        self._client._raise_for_status(response, f"Object not found: {digest[:24]}")

        content_length = response.headers.get("Content-Length")
        if content_length:
            return int(content_length)

        # Fallback: fetch and measure
        data = self.get_bytes(digest)
        return len(data)

    @staticmethod
    def _validate_digest(digest: str) -> None:
        """
        Validate digest format.

        Parameters
        ----------
        digest : str
            Digest to validate.

        Raises
        ------
        ValueError
            If digest format is invalid.
        """
        if not digest or not digest.startswith("sha256:"):
            raise ValueError(f"Invalid digest format: {digest!r}")
        hex_part = digest[7:]
        if len(hex_part) != 64:
            raise ValueError(f"Invalid digest length: {digest!r}")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> RemoteStore:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and close client."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RemoteStore(server_url={self._client.config.server_url!r}, "
            f"workspace={self._workspace!r})"
        )


# =============================================================================
# Registry Implementation
# =============================================================================


class RemoteRegistry(RegistryProtocol):
    """
    Run metadata registry backed by Remote API.

    Stores and queries run metadata via the remote server. Provides the same
    interface as local registries but with centralized team storage.

    Parameters
    ----------
    server_url : str
        Base URL of the remote server.
    token : str
        API token for authentication.
    workspace : str
        Workspace identifier.
    timeout : float, optional
        Request timeout in seconds. Default is 30.0.
    retry_attempts : int, optional
        Number of retry attempts. Default is 3.
    verify_ssl : bool, optional
        Whether to verify SSL certificates. Default is True.

    Examples
    --------
    >>> registry = RemoteRegistry(
    ...     server_url="https://tracking.example.com",
    ...     token="dqt_xxxxxxxxxxxx",
    ...     workspace="my-workspace",
    ... )
    >>> registry.save(run_record.to_dict())
    >>> loaded = registry.load("RUN123456")

    Notes
    -----
    The remote server handles persistence, access control, and multi-user
    coordination. This client provides the interface for CRUD operations.
    """

    def __init__(
        self,
        server_url: str,
        token: str,
        workspace: str,
        timeout: float = DEFAULT_TIMEOUT,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        verify_ssl: bool = True,
    ) -> None:
        config = RemoteClientConfig(
            server_url=server_url,
            token=token,
            workspace=workspace,
            timeout=timeout,
            retry_attempts=retry_attempts,
            verify_ssl=verify_ssl,
        )
        self._client = RemoteClient(config)
        self._workspace = workspace

    @classmethod
    def from_env(cls) -> RemoteRegistry:
        """
        Create RemoteRegistry from environment variables.

        Environment Variables
        ---------------------
        DEVQUBIT_TRACKING_URI : str
            Remote server URL (required).
        DEVQUBIT_API_TOKEN : str
            API token for authentication (required).
        DEVQUBIT_WORKSPACE : str
            Workspace identifier (required).

        Returns
        -------
        RemoteRegistry
            Configured registry instance.

        Raises
        ------
        RegistryError
            If required environment variables are not set.
        """
        server_url = os.getenv(ENV_TRACKING_URI, "")
        token = os.getenv(ENV_API_TOKEN, "")
        workspace = os.getenv(ENV_WORKSPACE, "")

        if not server_url:
            raise RegistryError(f"Environment variable {ENV_TRACKING_URI} is required")
        if not token:
            raise RegistryError(f"Environment variable {ENV_API_TOKEN} is required")
        if not workspace:
            raise RegistryError(f"Environment variable {ENV_WORKSPACE} is required")

        return cls(server_url=server_url, token=token, workspace=workspace)

    def save(self, record: dict[str, Any]) -> None:
        """
        Save or update a run record.

        If the run already exists, it will be updated. Otherwise, a new
        run is created.

        Parameters
        ----------
        record : dict
            Run record dictionary with required 'run_id' field.

        Raises
        ------
        RegistryError
            If save operation fails.
        ValueError
            If record is missing 'run_id'.
        """
        run_id = record.get("run_id")
        if not run_id:
            raise ValueError("Run record must have 'run_id' field")

        # Check if exists for update vs create
        if self.exists(run_id):
            response = self._client._request(
                "PATCH",
                f"/runs/{run_id}",
                json={"record_json": record, "workspace": self._workspace},
            )
        else:
            # Extract project name for creation
            project_name = record.get("project", {}).get("name", "default")
            response = self._client._request(
                "POST",
                "/runs",
                json={
                    "project": project_name,
                    "workspace_id": self._workspace,
                    "record_json": record,
                },
            )

        self._client._raise_for_status(response, f"Failed to save run {run_id}")
        logger.debug("Saved run: %s", run_id)

    def load(self, run_id: str) -> RunRecord:
        """
        Load a run record by ID.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord
            Run record wrapper.

        Raises
        ------
        RunNotFoundError
            If run does not exist.
        RegistryError
            If load operation fails.
        """
        from devqubit_engine.tracking.record import RunRecord

        response = self._client._request("GET", f"/runs/{run_id}")

        if response.status_code == 404:
            raise RunNotFoundError(f"Run not found: {run_id}")

        self._client._raise_for_status(response, f"Failed to load run {run_id}")

        data = response.json()
        record = data.get("record_json", data)

        # Parse artifacts if present
        artifacts = []
        for art_dict in record.get("artifacts", []):
            try:
                artifacts.append(ArtifactRef.from_dict(art_dict))
            except (ValueError, KeyError) as e:
                logger.warning("Invalid artifact in run %s: %s", run_id, e)

        rec = RunRecord(record=record, artifacts=artifacts)
        rec.mark_finalized()
        return rec

    def load_or_none(self, run_id: str) -> RunRecord | None:
        """
        Load a run record or return None if not found.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        RunRecord or None
            Run record or None if not found.
        """
        try:
            return self.load(run_id)
        except RunNotFoundError:
            return None

    def exists(self, run_id: str) -> bool:
        """
        Check if run exists.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run exists.
        """
        response = self._client._request("HEAD", f"/runs/{run_id}")
        return response.status_code == 200

    def delete(self, run_id: str) -> bool:
        """
        Delete a run record.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            True if run was deleted, False if it didn't exist.

        Raises
        ------
        RegistryError
            If deletion fails.
        """
        response = self._client._request("DELETE", f"/runs/{run_id}")

        if response.status_code == 404:
            return False

        self._client._raise_for_status(response, f"Failed to delete run {run_id}")
        logger.debug("Deleted run: %s", run_id)
        return True

    def save_metric_points(self, points: list[dict[str, Any]]) -> None:
        """Batch-insert metric time-series points via the remote API."""
        if not points:
            return
        # Group by run_id (typically one, but be safe)
        by_run: dict[str, list[dict[str, Any]]] = {}
        for p in points:
            by_run.setdefault(p["run_id"], []).append(p)

        for run_id, batch in by_run.items():
            response = self._client._request(
                "POST",
                f"/runs/{run_id}/metric_points",
                json={"points": batch},
            )
            self._client._raise_for_status(
                response, f"Failed to save metric points for {run_id}"
            )

    def iter_metric_points(
        self,
        run_id: str,
        key: str,
        *,
        start_step: int | None = None,
        end_step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield metric points for a single key from the remote API."""
        params: dict[str, Any] = {"key": key}
        if start_step is not None:
            params["start_step"] = start_step
        if end_step is not None:
            params["end_step"] = end_step

        response = self._client._request(
            "GET",
            f"/runs/{run_id}/metric_points",
            params=params,
        )
        self._client._raise_for_status(
            response, f"Failed to load metric points for {run_id}"
        )
        for pt in response.json().get("points", []):
            yield {
                "step": pt["step"],
                "timestamp": pt["timestamp"],
                "value": pt["value"],
            }

    def load_metric_series(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
        """Load all metric time-series for a run from the remote API."""
        response = self._client._request(
            "GET",
            f"/runs/{run_id}/metric_series",
        )
        self._client._raise_for_status(
            response, f"Failed to load metric series for {run_id}"
        )
        return response.json().get("series", {})

    def list_runs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        project: str | None = None,
        name: str | None = None,
        adapter: str | None = None,
        status: str | None = None,
        backend_name: str | None = None,
        fingerprint: str | None = None,
        git_commit: str | None = None,
        group_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[RunSummary]:
        """
        List runs with optional filtering.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.
        project : str, optional
            Filter by project name.
        name : str, optional
            Filter by run name (exact match).
        adapter : str, optional
            Filter by adapter name.
        status : str, optional
            Filter by run status.
        backend_name : str, optional
            Filter by backend name.
        fingerprint : str, optional
            Filter by run fingerprint.
        git_commit : str, optional
            Filter by git commit SHA.
        group_id : str, optional
            Filter by group ID.
        created_after : str, optional
            ISO 8601 lower bound (exclusive) on ``created_at``.
        created_before : str, optional
            ISO 8601 upper bound (exclusive) on ``created_at``.

        Returns
        -------
        list of RunSummary
            Matching runs, ordered by created_at descending.
        """
        params: dict[str, Any] = {
            "workspace": self._workspace,
            "limit": limit,
            "offset": offset,
        }

        if project:
            params["project"] = project
        if name:
            params["name"] = name
        if adapter:
            params["adapter"] = adapter
        if status:
            params["status"] = status
        if backend_name:
            params["backend_name"] = backend_name
        if fingerprint:
            params["fingerprint"] = fingerprint
        if git_commit:
            params["git_commit"] = git_commit
        if group_id:
            params["group_id"] = group_id
        if created_after:
            params["created_after"] = created_after
        if created_before:
            params["created_before"] = created_before

        response = self._client._request("GET", "/runs", params=params)
        self._client._raise_for_status(response, "Failed to list runs")

        data = response.json()
        runs = data.get("runs", data) if isinstance(data, dict) else data

        return [self._to_run_summary(r) for r in runs]

    def search_runs(
        self,
        query: str,
        *,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> list[RunRecord]:
        """
        Search runs using a query expression.

        Parameters
        ----------
        query : str
            Query expression (e.g., "metric.fidelity > 0.95").
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.
        sort_by : str, optional
            Field to sort by (e.g., "metric.fidelity").
        descending : bool, optional
            Sort in descending order. Default is True.

        Returns
        -------
        list of RunRecord
            Matching run records.
        """
        from devqubit_engine.tracking.record import RunRecord

        params: dict[str, Any] = {
            "workspace": self._workspace,
            "q": query,
            "limit": limit,
            "offset": offset,
            "descending": descending,
        }
        if sort_by:
            params["sort_by"] = sort_by

        response = self._client._request("GET", "/runs", params=params)
        self._client._raise_for_status(response, "Failed to search runs")

        data = response.json()
        runs = data.get("runs", data) if isinstance(data, dict) else data

        results = []
        for run_data in runs:
            record = run_data.get("record_json", run_data)
            artifacts = []
            for art_dict in record.get("artifacts", []):
                try:
                    artifacts.append(ArtifactRef.from_dict(art_dict))
                except (ValueError, KeyError) as e:
                    logger.warning("Invalid artifact in search results: %s", e)
            rec = RunRecord(record=record, artifacts=artifacts)
            rec.mark_finalized()
            results.append(rec)

        return results

    def list_projects(self) -> list[str]:
        """
        List all unique project names.

        Returns
        -------
        list of str
            Sorted list of project names.
        """
        response = self._client._request(
            "GET",
            "/projects",
            params={"workspace": self._workspace},
        )
        self._client._raise_for_status(response, "Failed to list projects")

        data = response.json()
        projects = data.get("projects", data) if isinstance(data, dict) else data

        return sorted(p.get("name", p) if isinstance(p, dict) else p for p in projects)

    def list_groups(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List run groups with optional project filtering.

        Parameters
        ----------
        project : str, optional
            Filter by project name.
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.

        Returns
        -------
        list of dict
            Group summaries with group_id, group_name, and run_count.
        """
        params: dict[str, Any] = {
            "workspace": self._workspace,
            "limit": limit,
            "offset": offset,
        }
        if project:
            params["project"] = project

        response = self._client._request("GET", "/runs/groups", params=params)
        self._client._raise_for_status(response, "Failed to list groups")

        data = response.json()
        return data.get("groups", data) if isinstance(data, dict) else data

    def list_runs_in_group(
        self,
        group_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunSummary]:
        """
        List runs belonging to a specific group.

        Parameters
        ----------
        group_id : str
            Group identifier.
        limit : int, optional
            Maximum number of results. Default is 100.
        offset : int, optional
            Number of results to skip. Default is 0.

        Returns
        -------
        list of RunSummary
            Runs in the group, ordered by created_at descending.
        """
        return self.list_runs(group_id=group_id, limit=limit, offset=offset)

    def count_runs(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
    ) -> int:
        """
        Count runs matching filters.

        Parameters
        ----------
        project : str, optional
            Filter by project name.
        status : str, optional
            Filter by run status.

        Returns
        -------
        int
            Number of matching runs.
        """
        params: dict[str, Any] = {"workspace": self._workspace, "count_only": True}
        if project:
            params["project"] = project
        if status:
            params["status"] = status

        response = self._client._request("GET", "/runs", params=params)
        self._client._raise_for_status(response, "Failed to count runs")

        data = response.json()
        return data.get("count", len(data.get("runs", [])))

    def set_baseline(self, project: str, run_id: str) -> None:
        """
        Set baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.
        run_id : str
            Run identifier to use as baseline.

        Raises
        ------
        RegistryError
            If operation fails.
        """
        response = self._client._request(
            "POST",
            f"/projects/{project}/baseline",
            json={"run_id": run_id, "workspace": self._workspace},
        )
        self._client._raise_for_status(
            response, f"Failed to set baseline for {project}"
        )
        logger.debug("Set baseline: project=%s, run_id=%s", project, run_id)

    def get_baseline(self, project: str) -> BaselineInfo | None:
        """
        Get baseline run for a project.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        BaselineInfo or None
            Baseline info, or None if no baseline set.
        """
        response = self._client._request(
            "GET",
            f"/projects/{project}/baseline",
            params={"workspace": self._workspace},
        )

        if response.status_code == 404:
            return None

        self._client._raise_for_status(
            response, f"Failed to get baseline for {project}"
        )

        data = response.json()
        if not data or not data.get("run_id"):
            return None

        return BaselineInfo(
            project=project,
            run_id=data["run_id"],
            set_at=data.get("set_at", ""),
        )

    def clear_baseline(self, project: str) -> bool:
        """
        Clear baseline for a project.

        Parameters
        ----------
        project : str
            Project name.

        Returns
        -------
        bool
            True if baseline was cleared, False if none existed.
        """
        response = self._client._request(
            "DELETE",
            f"/projects/{project}/baseline",
            params={"workspace": self._workspace},
        )

        if response.status_code == 404:
            return False

        self._client._raise_for_status(
            response, f"Failed to clear baseline for {project}"
        )
        return True

    @staticmethod
    def _to_run_summary(data: dict[str, Any]) -> RunSummary:
        """
        Convert API response to RunSummary.

        Parameters
        ----------
        data : dict
            Run data from API.

        Returns
        -------
        RunSummary
            Standardized run summary.
        """
        # Handle both direct records and wrapped responses
        record = data.get("record_json", data)
        info = record.get("info", {})
        project = record.get("project", {})

        return RunSummary(
            run_id=record.get("run_id", data.get("run_id", "")),
            run_name=info.get("run_name"),
            project=project.get("name", "") if isinstance(project, dict) else project,
            adapter=record.get("adapter", ""),
            status=info.get("status", ""),
            created_at=record.get("created_at", ""),
            ended_at=info.get("ended_at"),
            group_id=record.get("group_id"),
            group_name=record.get("group_name"),
            parent_run_id=record.get("parent_run_id"),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> RemoteRegistry:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and close client."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RemoteRegistry(server_url={self._client.config.server_url!r}, "
            f"workspace={self._workspace!r})"
        )
