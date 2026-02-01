# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for remote storage backends.

These tests use responses library to mock HTTP requests to the remote API.
They verify the RemoteStore and RemoteRegistry classes implement the storage
protocols correctly and handle various HTTP scenarios.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable
from unittest.mock import patch

import pytest
import responses
from devqubit_engine.storage.backends.remote import (
    DEFAULT_TIMEOUT,
    ENV_API_TOKEN,
    ENV_TRACKING_URI,
    ENV_WORKSPACE,
    RemoteClient,
    RemoteClientConfig,
    RemoteRegistry,
    RemoteStore,
)
from devqubit_engine.storage.errors import (
    ObjectNotFoundError,
    RunNotFoundError,
    StorageError,
)
from devqubit_engine.tracking.record import RunRecord
from responses import matchers


# =============================================================================
# Constants for Tests
# =============================================================================

TEST_SERVER = "https://tracking.example.com"
TEST_TOKEN = "dqt_test_token_12345"
TEST_WORKSPACE = "test-workspace"
TEST_API_BASE = f"{TEST_SERVER}/api/v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def remote_config() -> RemoteClientConfig:
    """Create test remote configuration."""
    return RemoteClientConfig(
        server_url=TEST_SERVER,
        token=TEST_TOKEN,
        workspace=TEST_WORKSPACE,
        timeout=5.0,
        retry_attempts=1,
    )


@pytest.fixture
def remote_client(remote_config: RemoteClientConfig) -> RemoteClient:
    """Create test remote client."""
    client = RemoteClient(remote_config)
    yield client
    client.close()


@pytest.fixture
def remote_store() -> RemoteStore:
    """Create test remote store."""
    store = RemoteStore(
        server_url=TEST_SERVER,
        token=TEST_TOKEN,
        workspace=TEST_WORKSPACE,
        retry_attempts=1,
    )
    yield store
    store.close()


@pytest.fixture
def remote_registry() -> RemoteRegistry:
    """Create test remote registry."""
    registry = RemoteRegistry(
        server_url=TEST_SERVER,
        token=TEST_TOKEN,
        workspace=TEST_WORKSPACE,
        retry_attempts=1,
    )
    yield registry
    registry.close()


@pytest.fixture
def run_factory() -> Callable[..., RunRecord]:
    """Factory for creating test run records."""

    def _make_run(
        run_id: str = "RUN123456",
        project: str = "test_project",
        status: str = "FINISHED",
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        run_name: str | None = None,
        group_id: str | None = None,
        group_name: str | None = None,
    ) -> RunRecord:
        record: dict[str, Any] = {
            "schema": "devqubit.run/1.0",
            "run_id": run_id,
            "created_at": "2024-01-01T00:00:00Z",
            "project": {"name": project},
            "adapter": "manual",
            "info": {"status": status},
            "data": {
                "params": params or {},
                "metrics": metrics or {},
                "tags": tags or {},
            },
            "fingerprints": {
                "run": f"sha256:{hashlib.sha256(run_id.encode()).hexdigest()}",
            },
            "artifacts": [],
        }
        if run_name:
            record["info"]["run_name"] = run_name
        if group_id:
            record["group_id"] = group_id
        if group_name:
            record["group_name"] = group_name

        return RunRecord(record=record, artifacts=[])

    return _make_run


@pytest.fixture
def sample_data() -> bytes:
    """Sample binary data for store tests."""
    return b"test artifact data for remote storage"


@pytest.fixture
def sample_digest(sample_data: bytes) -> str:
    """Digest of sample data."""
    return f"sha256:{hashlib.sha256(sample_data).hexdigest()}"


# =============================================================================
# RemoteClientConfig Tests
# =============================================================================


class TestRemoteClientConfig:
    """Tests for RemoteClientConfig validation and construction."""

    def test_valid_config(self):
        """Valid configuration creates successfully."""
        config = RemoteClientConfig(
            server_url=TEST_SERVER,
            token=TEST_TOKEN,
            workspace=TEST_WORKSPACE,
        )

        assert config.server_url == TEST_SERVER
        assert config.token == TEST_TOKEN
        assert config.workspace == TEST_WORKSPACE
        assert config.timeout == DEFAULT_TIMEOUT

    def test_missing_server_url_raises(self):
        """Empty server_url raises ValueError."""
        with pytest.raises(ValueError, match="server_url is required"):
            RemoteClientConfig(
                server_url="",
                token=TEST_TOKEN,
                workspace=TEST_WORKSPACE,
            )

    def test_missing_token_raises(self):
        """Empty token raises ValueError."""
        with pytest.raises(ValueError, match="API token is required"):
            RemoteClientConfig(
                server_url=TEST_SERVER,
                token="",
                workspace=TEST_WORKSPACE,
            )

    def test_missing_workspace_raises(self):
        """Empty workspace raises ValueError."""
        with pytest.raises(ValueError, match="Workspace is required"):
            RemoteClientConfig(server_url=TEST_SERVER, token=TEST_TOKEN, workspace="")

    def test_from_env(self):
        """from_env reads environment variables."""
        with patch.dict(
            os.environ,
            {
                ENV_TRACKING_URI: TEST_SERVER,
                ENV_API_TOKEN: TEST_TOKEN,
                ENV_WORKSPACE: TEST_WORKSPACE,
            },
        ):
            config = RemoteClientConfig.from_env()

            assert config.server_url == TEST_SERVER
            assert config.token == TEST_TOKEN
            assert config.workspace == TEST_WORKSPACE

    def test_from_env_with_overrides(self):
        """from_env allows parameter overrides."""
        with patch.dict(os.environ, {ENV_TRACKING_URI: "https://wrong.com"}):
            config = RemoteClientConfig.from_env(
                server_url=TEST_SERVER,
                token=TEST_TOKEN,
                workspace=TEST_WORKSPACE,
            )

            assert config.server_url == TEST_SERVER


# =============================================================================
# RemoteClient Tests
# =============================================================================


class TestRemoteClient:
    """Tests for base HTTP client functionality."""

    @responses.activate
    def test_request_success(self, remote_client: RemoteClient):
        """Successful request returns response."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/test",
            json={"status": "ok"},
            status=200,
        )

        response = remote_client._request("GET", "/test")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @responses.activate
    def test_request_includes_auth_header(self, remote_client: RemoteClient):
        """Request includes Bearer token header."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/test",
            json={},
            status=200,
            match=[matchers.header_matcher({"Authorization": f"Bearer {TEST_TOKEN}"})],
        )

        remote_client._request("GET", "/test")

    @responses.activate
    def test_404_raises_object_not_found(self, remote_client: RemoteClient):
        """404 response raises ObjectNotFoundError."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/missing",
            json={"detail": "Not found"},
            status=404,
        )

        response = remote_client._request("GET", "/missing")
        with pytest.raises(ObjectNotFoundError):
            remote_client._raise_for_status(response, "Resource")

    @responses.activate
    def test_401_raises_storage_error(self, remote_client: RemoteClient):
        """401 response raises StorageError."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/protected",
            json={"detail": "Invalid token"},
            status=401,
        )

        response = remote_client._request("GET", "/protected")
        with pytest.raises(StorageError, match="Authentication failed"):
            remote_client._raise_for_status(response)

    @responses.activate
    def test_connection_error_raises_storage_error(self, remote_client: RemoteClient):
        """Connection error raises StorageError."""
        import requests.exceptions

        def raise_connection_error(request):
            raise requests.exceptions.ConnectionError("Connection refused")

        responses.add_callback(
            responses.GET,
            f"{TEST_API_BASE}/test",
            callback=raise_connection_error,
        )

        with pytest.raises(StorageError, match="Connection error"):
            remote_client._request("GET", "/test")


# =============================================================================
# RemoteStore Tests
# =============================================================================


class TestRemoteStore:
    """Tests for content-addressed object storage."""

    @responses.activate
    def test_put_bytes_uploads_data(
        self,
        remote_store: RemoteStore,
        sample_data: bytes,
        sample_digest: str,
    ):
        """put_bytes uploads data and returns digest."""
        # Mock exists check (returns 404 = doesn't exist)
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=404,
        )
        # Mock upload
        responses.add(
            responses.POST,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects",
            json={"digest": sample_digest},
            status=201,
        )

        result = remote_store.put_bytes(sample_data)

        assert result == sample_digest

    @responses.activate
    def test_put_bytes_skips_existing(
        self,
        remote_store: RemoteStore,
        sample_data: bytes,
        sample_digest: str,
    ):
        """put_bytes skips upload for existing objects."""
        # Mock exists check (returns 200 = exists)
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=200,
        )

        result = remote_store.put_bytes(sample_data)

        assert result == sample_digest
        assert len(responses.calls) == 1  # Only HEAD, no POST

    @responses.activate
    def test_get_bytes_retrieves_data(
        self,
        remote_store: RemoteStore,
        sample_data: bytes,
        sample_digest: str,
    ):
        """get_bytes retrieves stored data."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            body=sample_data,
            status=200,
            content_type="application/octet-stream",
        )

        result = remote_store.get_bytes(sample_digest)

        assert result == sample_data

    @responses.activate
    def test_get_bytes_not_found(self, remote_store: RemoteStore, sample_digest: str):
        """get_bytes raises ObjectNotFoundError for missing object."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            json={"detail": "Not found"},
            status=404,
        )

        with pytest.raises(ObjectNotFoundError):
            remote_store.get_bytes(sample_digest)

    @responses.activate
    def test_get_bytes_or_none_returns_none(
        self,
        remote_store: RemoteStore,
        sample_digest: str,
    ):
        """get_bytes_or_none returns None for missing object."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=404,
        )

        result = remote_store.get_bytes_or_none(sample_digest)

        assert result is None

    @responses.activate
    def test_exists_true(self, remote_store: RemoteStore, sample_digest: str):
        """exists returns True for existing object."""
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=200,
        )

        assert remote_store.exists(sample_digest) is True

    @responses.activate
    def test_exists_false(self, remote_store: RemoteStore, sample_digest: str):
        """exists returns False for missing object."""
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=404,
        )

        assert remote_store.exists(sample_digest) is False

    @responses.activate
    def test_delete_existing(self, remote_store: RemoteStore, sample_digest: str):
        """delete returns True for existing object."""
        responses.add(
            responses.DELETE,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=204,
        )

        assert remote_store.delete(sample_digest) is True

    @responses.activate
    def test_delete_missing(self, remote_store: RemoteStore, sample_digest: str):
        """delete returns False for missing object."""
        responses.add(
            responses.DELETE,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=404,
        )

        assert remote_store.delete(sample_digest) is False

    @responses.activate
    def test_get_size(self, remote_store: RemoteStore, sample_digest: str):
        """get_size returns Content-Length header value."""
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{sample_digest}",
            status=200,
            headers={"Content-Length": "1234"},
        )

        assert remote_store.get_size(sample_digest) == 1234

    @responses.activate
    def test_list_digests(self, remote_store: RemoteStore):
        """list_digests paginates through results."""
        _digest = "sha256:" + "a" * 64

        # First page
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects",
            json={"digests": [_digest]},
            status=200,
        )
        # Second page (empty = end)
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects",
            json={"digests": []},
            status=200,
        )

        digests = list(remote_store.list_digests())

        assert digests == [_digest]

    def test_validate_digest_valid(self, remote_store: RemoteStore):
        """Valid digest passes validation."""
        remote_store._validate_digest("sha256:" + "a" * 64)

    def test_validate_digest_invalid_prefix(self, remote_store: RemoteStore):
        """Invalid prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid digest format"):
            remote_store._validate_digest("md5:" + "a" * 64)

    def test_validate_digest_invalid_length(self, remote_store: RemoteStore):
        """Invalid length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid digest length"):
            remote_store._validate_digest("sha256:" + "a" * 32)


# =============================================================================
# RemoteRegistry Tests
# =============================================================================


class TestRemoteRegistry:
    """Tests for run metadata registry."""

    @responses.activate
    def test_save_new_run(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """save creates new run via POST."""
        run = run_factory(run_id="NEW_RUN_1")

        # exists check
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/runs/NEW_RUN_1",
            status=404,
        )
        # create
        responses.add(
            responses.POST,
            f"{TEST_API_BASE}/runs",
            json={"run_id": "NEW_RUN_1"},
            status=201,
        )

        remote_registry.save(run.to_dict())

        assert len(responses.calls) == 2

    @responses.activate
    def test_save_existing_run(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """save updates existing run via PATCH."""
        run = run_factory(run_id="EXIST_RUN")

        # exists check
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/runs/EXIST_RUN",
            status=200,
        )
        # update
        responses.add(
            responses.PATCH,
            f"{TEST_API_BASE}/runs/EXIST_RUN",
            json={"run_id": "EXIST_RUN"},
            status=200,
        )

        remote_registry.save(run.to_dict())

    def test_save_missing_run_id_raises(self, remote_registry: RemoteRegistry):
        """save raises ValueError for missing run_id."""
        with pytest.raises(ValueError, match="run_id"):
            remote_registry.save({"project": {"name": "test"}})

    @responses.activate
    def test_load_existing_run(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """load retrieves run record."""
        original = run_factory(run_id="LOAD_RUN", params={"shots": 1000})

        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/LOAD_RUN",
            json={"record_json": original.to_dict()},
            status=200,
        )

        loaded = remote_registry.load("LOAD_RUN")

        assert loaded.run_id == "LOAD_RUN"
        assert loaded.params == {"shots": 1000}

    @responses.activate
    def test_load_missing_raises(self, remote_registry: RemoteRegistry):
        """load raises RunNotFoundError for missing run."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/MISSING",
            json={"detail": "Not found"},
            status=404,
        )

        with pytest.raises(RunNotFoundError):
            remote_registry.load("MISSING")

    @responses.activate
    def test_load_or_none_returns_none(self, remote_registry: RemoteRegistry):
        """load_or_none returns None for missing run."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/MISSING",
            status=404,
        )

        assert remote_registry.load_or_none("MISSING") is None

    @responses.activate
    def test_exists_true(self, remote_registry: RemoteRegistry):
        """exists returns True for existing run."""
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/runs/EXISTS_1",
            status=200,
        )

        assert remote_registry.exists("EXISTS_1") is True

    @responses.activate
    def test_exists_false(self, remote_registry: RemoteRegistry):
        """exists returns False for missing run."""
        responses.add(
            responses.HEAD,
            f"{TEST_API_BASE}/runs/MISSING",
            status=404,
        )

        assert remote_registry.exists("MISSING") is False

    @responses.activate
    def test_delete_existing(self, remote_registry: RemoteRegistry):
        """delete returns True for existing run."""
        responses.add(
            responses.DELETE,
            f"{TEST_API_BASE}/runs/TO_DELETE",
            status=204,
        )

        assert remote_registry.delete("TO_DELETE") is True

    @responses.activate
    def test_delete_missing(self, remote_registry: RemoteRegistry):
        """delete returns False for missing run."""
        responses.add(
            responses.DELETE,
            f"{TEST_API_BASE}/runs/MISSING",
            status=404,
        )

        assert remote_registry.delete("MISSING") is False

    @responses.activate
    def test_list_runs(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """list_runs returns RunSummary list."""
        run1 = run_factory(run_id="LIST_1", project="proj_a")
        run2 = run_factory(run_id="LIST_2", project="proj_b")

        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs",
            json={"runs": [run1.to_dict(), run2.to_dict()]},
            status=200,
        )

        runs = remote_registry.list_runs(limit=10)

        assert len(runs) == 2
        assert runs[0]["run_id"] == "LIST_1"

    @responses.activate
    def test_list_runs_with_project_filter(self, remote_registry: RemoteRegistry):
        """list_runs passes project filter to API."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs",
            json={"runs": []},
            status=200,
            match=[
                matchers.query_param_matcher(
                    {
                        "workspace": TEST_WORKSPACE,
                        "project": "my_proj",
                        "limit": "10",
                        "offset": "0",
                    },
                    strict_match=False,
                )
            ],
        )

        remote_registry.list_runs(project="my_proj", limit=10)

    @responses.activate
    def test_search_runs(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """search_runs returns RunRecord list."""
        run = run_factory(run_id="SEARCH_1", metrics={"fidelity": 0.95})

        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs",
            json={"runs": [{"record_json": run.to_dict()}]},
            status=200,
        )

        results = remote_registry.search_runs("metric.fidelity > 0.9")

        assert len(results) == 1
        assert results[0].metrics["fidelity"] == 0.95

    @responses.activate
    def test_list_projects(self, remote_registry: RemoteRegistry):
        """list_projects returns sorted project names."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/projects",
            json={"projects": [{"name": "zebra"}, {"name": "alpha"}]},
            status=200,
        )

        projects = remote_registry.list_projects()

        assert projects == ["alpha", "zebra"]

    @responses.activate
    def test_list_groups(self, remote_registry: RemoteRegistry):
        """list_groups returns group summaries."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/groups",
            json={
                "groups": [
                    {"group_id": "grp_1", "group_name": "Sweep 1", "run_count": 5},
                ]
            },
            status=200,
        )

        groups = remote_registry.list_groups()

        assert len(groups) == 1
        assert groups[0]["run_count"] == 5

    @responses.activate
    def test_count_runs(self, remote_registry: RemoteRegistry):
        """count_runs returns count from API."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs",
            json={"count": 42},
            status=200,
        )

        assert remote_registry.count_runs() == 42

    @responses.activate
    def test_set_baseline(self, remote_registry: RemoteRegistry):
        """set_baseline POSTs to baseline endpoint."""
        responses.add(
            responses.POST,
            f"{TEST_API_BASE}/projects/my_proj/baseline",
            json={"run_id": "BASELINE_1"},
            status=200,
        )

        remote_registry.set_baseline("my_proj", "BASELINE_1")

    @responses.activate
    def test_get_baseline(self, remote_registry: RemoteRegistry):
        """get_baseline returns BaselineInfo."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/projects/my_proj/baseline",
            json={"run_id": "BASELINE_1", "set_at": "2024-01-01T00:00:00Z"},
            status=200,
        )

        baseline = remote_registry.get_baseline("my_proj")

        assert baseline is not None
        assert baseline["run_id"] == "BASELINE_1"

    @responses.activate
    def test_get_baseline_none(self, remote_registry: RemoteRegistry):
        """get_baseline returns None when not set."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/projects/my_proj/baseline",
            status=404,
        )

        assert remote_registry.get_baseline("my_proj") is None

    @responses.activate
    def test_clear_baseline(self, remote_registry: RemoteRegistry):
        """clear_baseline DELETEs baseline."""
        responses.add(
            responses.DELETE,
            f"{TEST_API_BASE}/projects/my_proj/baseline",
            status=204,
        )

        assert remote_registry.clear_baseline("my_proj") is True


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestRemoteStoreAndRegistryIntegration:
    """Tests verifying store and registry work together."""

    @responses.activate
    def test_save_and_load_roundtrip(
        self,
        remote_registry: RemoteRegistry,
        run_factory: Callable[..., RunRecord],
    ):
        """Run can be saved and loaded with same data."""
        run = run_factory(
            run_id="ROUNDTRIP",
            project="test",
            params={"shots": 1000},
            metrics={"fidelity": 0.95},
            run_name="my-run",
        )

        # Save: exists check + create
        responses.add(responses.HEAD, f"{TEST_API_BASE}/runs/ROUNDTRIP", status=404)
        responses.add(
            responses.POST,
            f"{TEST_API_BASE}/runs",
            json={"run_id": "ROUNDTRIP"},
            status=201,
        )

        remote_registry.save(run.to_dict())

        # Load
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/ROUNDTRIP",
            json={"record_json": run.to_dict()},
            status=200,
        )

        loaded = remote_registry.load("ROUNDTRIP")

        assert loaded.run_id == "ROUNDTRIP"
        assert loaded.params == {"shots": 1000}
        assert loaded.metrics == {"fidelity": 0.95}

    @responses.activate
    def test_artifacts_parsed_on_load(self, remote_registry: RemoteRegistry):
        """Artifacts in record are parsed to ArtifactRef."""
        record = {
            "run_id": "WITH_ARTIFACTS",
            "project": {"name": "test"},
            "info": {"status": "FINISHED"},
            "data": {"params": {}, "metrics": {}, "tags": {}},
            "artifacts": [
                {
                    "kind": "result.json",
                    "digest": "sha256:" + "a" * 64,
                    "media_type": "application/json",
                    "role": "results",
                }
            ],
        }

        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/WITH_ARTIFACTS",
            json={"record_json": record},
            status=200,
        )

        loaded = remote_registry.load("WITH_ARTIFACTS")

        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].kind == "result.json"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error scenarios."""

    @responses.activate
    def test_server_error_raises_storage_error(self, remote_store: RemoteStore):
        """5xx responses raise StorageError."""
        digest = "sha256:" + "a" * 64
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/workspaces/{TEST_WORKSPACE}/objects/{digest}",
            json={"detail": "Internal error"},
            status=500,
        )

        with pytest.raises(StorageError, match="Remote API error"):
            remote_store.get_bytes(digest)

    @responses.activate
    def test_forbidden_raises_storage_error(self, remote_registry: RemoteRegistry):
        """403 response raises StorageError."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/FORBIDDEN",
            json={"detail": "No access"},
            status=403,
        )

        with pytest.raises(StorageError, match="Access denied"):
            remote_registry.load("FORBIDDEN")

    @responses.activate
    def test_invalid_json_response(self, remote_registry: RemoteRegistry):
        """Invalid JSON response handled gracefully."""
        responses.add(
            responses.GET,
            f"{TEST_API_BASE}/runs/BAD_JSON",
            body="not json",
            status=400,
        )

        with pytest.raises(StorageError):
            remote_registry.load("BAD_JSON")
