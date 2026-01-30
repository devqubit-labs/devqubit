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
