# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Pytest configuration and shared fixtures for devqubit UI tests."""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_registry() -> Mock:
    """
    Create a mock registry for testing.

    Returns
    -------
    Mock
        Mock registry with common methods stubbed.
    """
    registry = Mock()
    registry.list_runs.return_value = []
    registry.list_projects.return_value = []
    registry.list_groups.return_value = []
    registry.count_runs.return_value = 0
    registry.get_baseline.return_value = None
    return registry


@pytest.fixture
def mock_store() -> Mock:
    """
    Create a mock object store for testing.

    Returns
    -------
    Mock
        Mock store with common methods stubbed.
    """
    store = Mock()
    store.get_bytes.return_value = b'{"test": "data"}'
    return store


@pytest.fixture
def mock_config() -> Mock:
    """
    Create a mock configuration for testing.

    Returns
    -------
    Mock
        Mock config with root_dir set.
    """
    config = Mock()
    config.root_dir = "/tmp/devqubit-test"
    return config


@pytest.fixture
def app(mock_registry: Mock, mock_store: Mock, mock_config: Mock) -> Any:
    """
    Create a test FastAPI application with mocked dependencies.

    Parameters
    ----------
    mock_registry : Mock
        Mocked registry.
    mock_store : Mock
        Mocked object store.
    mock_config : Mock
        Mocked configuration.

    Returns
    -------
    FastAPI
        Configured FastAPI application for testing.
    """
    # Import here to avoid issues if devqubit is not installed
    try:
        from devqubit_ui.app import create_app

        return create_app(
            config=mock_config,
            registry=mock_registry,
            store=mock_store,
        )
    except ImportError:
        pytest.skip("devqubit not installed")


@pytest.fixture
def client(app: Any) -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application.

    Yields
    ------
    TestClient
        HTTP test client.
    """
    with TestClient(app) as client:
        yield client
