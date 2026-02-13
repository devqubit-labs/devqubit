# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Pytest fixtures for devqubit UI tests."""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_registry() -> Mock:
    """Create a mock registry."""
    registry = Mock()
    registry.list_runs.return_value = []
    registry.list_projects.return_value = []
    registry.list_groups.return_value = []
    registry.count_runs.return_value = 0
    registry.get_baseline.return_value = None
    registry.delete.return_value = True
    registry.search_runs.return_value = []
    return registry


@pytest.fixture
def mock_store() -> Mock:
    """Create a mock object store."""
    store = Mock()
    store.get_bytes.return_value = b'{"test": "data"}'
    return store


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock configuration."""
    config = Mock()
    config.root_dir = "/tmp/devqubit-test"
    return config


@pytest.fixture
def app(mock_registry: Mock, mock_store: Mock, mock_config: Mock) -> Any:
    """Create test FastAPI application."""
    try:
        from devqubit_ui.app import create_app

        return create_app(
            config=mock_config,
            registry=mock_registry,
            store=mock_store,
        )
    except ImportError:
        pytest.skip("devqubit-ui not installed")


@pytest.fixture
def client(app: Any) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_run_record(mock_registry, mock_store):
    """Configure mocks for a run with one artifact."""
    record = Mock()
    record.run_name = "test-run"
    record.record = {
        "run_name": "test-run",
        "project": "proj",
        "adapter": "qiskit",
        "artifacts": [
            {
                "kind": "result.json",
                "digest": "sha256:" + "ab" * 32,
                "role": "result",
                "media_type": "application/json",
            }
        ],
        "fingerprints": {"run": "fp1"},
    }
    mock_registry.load.return_value = record
    mock_store.get_bytes.return_value = b'{"counts": {"00": 500}}'
    mock_store.exists.return_value = True
    return record
