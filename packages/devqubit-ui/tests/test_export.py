# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for export API endpoints."""

from __future__ import annotations

import zipfile
from io import BytesIO
from unittest.mock import Mock

import pytest


@pytest.fixture()
def _mock_run_record(mock_registry, mock_store):
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


class TestCreateExport:
    """POST /api/runs/{run_id}/export."""

    def test_creates_bundle(self, client, _mock_run_record):
        data = client.post("/api/runs/run-1/export").json()
        assert data["status"] == "ready"
        assert data["run_id"] == "run-1"
        assert data["object_count"] == 1
        assert data["missing_objects"] == []

    def test_run_not_found(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("gone")
        assert client.post("/api/runs/missing/export").status_code == 404


class TestDownloadExport:
    """GET /api/runs/{run_id}/export/download."""

    def test_download_after_create(self, client, _mock_run_record):
        client.post("/api/runs/run-dl/export")
        resp = client.get("/api/runs/run-dl/export/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        zf = zipfile.ZipFile(BytesIO(resp.content))
        names = zf.namelist()
        assert "run.json" in names
        assert "manifest.json" in names
        # artifact object should be packed under objects/
        assert any(n.startswith("objects/sha256/") for n in names)

    def test_download_without_create(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("nope")
        assert client.get("/api/runs/no-bundle/export/download").status_code == 404


class TestExportInfo:
    """GET /api/runs/{run_id}/export/info."""

    def test_info_returns_counts(self, client, _mock_run_record):
        data = client.get("/api/runs/run-info/export/info").json()
        assert data["run_id"] == "run-info"
        assert data["artifact_count"] == 1
        assert data["available_objects"] == 1
        assert data["missing_objects"] == 0

    def test_run_not_found(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("gone")
        assert client.get("/api/runs/missing/export/info").status_code == 404


class TestCleanupExport:
    """DELETE /api/runs/{run_id}/export."""

    def test_cleanup_after_create(self, client, _mock_run_record):
        client.post("/api/runs/run-clean/export")
        resp = client.delete("/api/runs/run-clean/export")
        assert resp.json()["status"] == "cleaned"

        # second download should fail — bundle removed
        assert client.get("/api/runs/run-clean/export/download").status_code == 404

    def test_cleanup_noop_when_missing(self, client):
        assert (
            client.delete("/api/runs/no-such-run/export").json()["status"] == "cleaned"
        )


class TestPathTraversal:
    """Verify run_id sanitization rejects path traversal attempts."""

    # IDs that reach the handler but must be rejected by our allowlist.
    MALICIOUS_IDS = [
        ".hidden",  # dotfile — starts with '.'
        "..secret",  # starts with dots
        "run id with spaces",  # whitespace in the middle
        "%2e%2e",  # percent-encoded dots (arrives as literal string)
    ]

    @pytest.mark.parametrize("malicious_id", MALICIOUS_IDS)
    def test_create_rejects_traversal(self, client, malicious_id):
        resp = client.post(f"/api/runs/{malicious_id}/export")
        assert (
            resp.status_code == 400
        ), f"Expected 400 for run_id={malicious_id!r}, got {resp.status_code}"

    @pytest.mark.parametrize("malicious_id", MALICIOUS_IDS)
    def test_download_rejects_traversal(self, client, malicious_id):
        resp = client.get(f"/api/runs/{malicious_id}/export/download")
        assert (
            resp.status_code == 400
        ), f"Expected 400 for run_id={malicious_id!r}, got {resp.status_code}"

    @pytest.mark.parametrize("malicious_id", MALICIOUS_IDS)
    def test_info_rejects_traversal(self, client, malicious_id):
        resp = client.get(f"/api/runs/{malicious_id}/export/info")
        assert (
            resp.status_code == 400
        ), f"Expected 400 for run_id={malicious_id!r}, got {resp.status_code}"

    @pytest.mark.parametrize("malicious_id", MALICIOUS_IDS)
    def test_cleanup_rejects_traversal(self, client, malicious_id):
        resp = client.delete(f"/api/runs/{malicious_id}/export")
        assert (
            resp.status_code == 400
        ), f"Expected 400 for run_id={malicious_id!r}, got {resp.status_code}"

    def test_valid_ids_accepted(self, client, _mock_run_record):
        """Sanity: typical run IDs still work fine."""
        for valid_id in [
            "abc123",
            "01HWXYZ",
            "550e8400-e29b-41d4-a716-446655440000",
            "run_name-with-dashes_and_underscores",
        ]:
            resp = client.post(f"/api/runs/{valid_id}/export")
            assert (
                resp.status_code == 200
            ), f"Valid run_id={valid_id!r} rejected with {resp.status_code}"
