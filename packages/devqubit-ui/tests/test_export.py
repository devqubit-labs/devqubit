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
