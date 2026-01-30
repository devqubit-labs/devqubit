# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit UI API router."""

from __future__ import annotations

from unittest.mock import Mock


class TestCapabilities:
    """Tests for capabilities endpoint."""

    def test_get_capabilities(self, client):
        """Test capabilities endpoint returns expected structure."""
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "local"
        assert "version" in data
        assert "features" in data


class TestRunsApi:
    """Tests for runs API endpoints."""

    def test_list_runs_empty(self, client):
        """Test runs list with no runs."""
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["count"] == 0

    def test_list_runs_with_data(self, client, mock_registry):
        """Test runs list returns data."""
        mock_registry.list_runs.return_value = [
            {
                "run_id": "test-123",
                "run_name": "test",
                "project": "proj",
                "status": "FINISHED",
                "created_at": "2025-01-01T00:00:00Z",
                "fingerprints": {},
            }
        ]

        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    def test_list_runs_filter_by_project(self, client, mock_registry):
        """Test filtering runs by project."""
        client.get("/api/runs?project=my-project")
        mock_registry.list_runs.assert_called_with(limit=50, project="my-project")

    def test_list_runs_with_search(self, client, mock_registry):
        """Test runs search query."""
        mock_registry.search_runs.return_value = []
        client.get("/api/runs?q=metric.fidelity>0.9")
        mock_registry.search_runs.assert_called()

    def test_get_run_not_found(self, client, mock_registry):
        """Test get run returns 404 for missing run."""
        mock_registry.load.side_effect = KeyError("not found")

        response = client.get("/api/runs/nonexistent")
        assert response.status_code == 404

    def test_get_run_success(self, client, mock_registry):
        """Test get run returns run details."""
        mock_run = Mock()
        mock_run.run_id = "test-123"
        mock_run.run_name = "test run"
        mock_run.project = "proj"
        mock_run.adapter = "qiskit"
        mock_run.status = "FINISHED"
        mock_run.created_at = "2025-01-01T00:00:00Z"
        mock_run.ended_at = None
        mock_run.fingerprints = {"run": "abc123"}
        mock_run.group_id = None
        mock_run.group_name = None
        mock_run.artifacts = []
        mock_run.record = {"backend": {}, "data": {}, "info": {}}
        mock_registry.load.return_value = mock_run

        response = client.get("/api/runs/test-123")
        assert response.status_code == 200
        data = response.json()
        assert data["run"]["run_id"] == "test-123"
        assert data["run"]["project"] == "proj"

    def test_delete_run_success(self, client, mock_registry):
        """Test successful run deletion."""
        mock_registry.delete.return_value = True

        response = client.delete("/api/runs/test-123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    def test_delete_run_not_found(self, client, mock_registry):
        """Test deletion of non-existent run."""
        mock_registry.delete.return_value = False

        response = client.delete("/api/runs/nonexistent")
        assert response.status_code == 404


class TestArtifactsApi:
    """Tests for artifacts API endpoints."""

    def test_get_artifact_not_found_run(self, client, mock_registry):
        """Test artifact returns 404 for missing run."""
        mock_registry.load.side_effect = KeyError("not found")

        response = client.get("/api/runs/nonexistent/artifacts/0")
        assert response.status_code == 404

    def test_get_artifact_not_found_index(self, client, mock_registry):
        """Test artifact returns 404 for invalid index."""
        mock_run = Mock()
        mock_run.artifacts = []
        mock_registry.load.return_value = mock_run

        response = client.get("/api/runs/test-123/artifacts/99")
        assert response.status_code == 404

    def test_get_artifact_raw_not_found(self, client, mock_registry):
        """Test raw artifact returns 404 for missing run."""
        mock_registry.load.side_effect = KeyError("not found")

        response = client.get("/api/runs/nonexistent/artifacts/0/raw")
        assert response.status_code == 404


class TestProjectsApi:
    """Tests for projects API endpoints."""

    def test_list_projects_empty(self, client):
        """Test projects list with no projects."""
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["projects"] == []

    def test_set_baseline_success(self, client, mock_registry):
        """Test setting project baseline."""
        mock_run = Mock()
        mock_run.project = "test-project"
        mock_run.run_id = "test-123"
        mock_registry.load.return_value = mock_run

        response = client.post(
            "/api/projects/test-project/baseline/test-123?redirect=false"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["baseline_run_id"] == "test-123"

    def test_set_baseline_wrong_project(self, client, mock_registry):
        """Test baseline fails for run from different project."""
        mock_run = Mock()
        mock_run.project = "other-project"
        mock_registry.load.return_value = mock_run

        response = client.post(
            "/api/projects/test-project/baseline/test-123?redirect=false"
        )
        assert response.status_code == 400


class TestGroupsApi:
    """Tests for groups API endpoints."""

    def test_list_groups_empty(self, client):
        """Test groups list with no groups."""
        response = client.get("/api/groups")
        assert response.status_code == 200
        data = response.json()
        assert data["groups"] == []

    def test_list_groups_filter_by_project(self, client, mock_registry):
        """Test filtering groups by project."""
        client.get("/api/groups?project=my-project")
        mock_registry.list_groups.assert_called_with(project="my-project")


class TestDiffApi:
    """Tests for diff API endpoint."""

    def test_diff_missing_run(self, client, mock_registry):
        """Test diff with missing run returns 404."""
        mock_registry.load.side_effect = KeyError("not found")

        response = client.get("/api/diff?a=bad-id&b=other-id")
        assert response.status_code == 404


class TestSpaRouting:
    """Tests for SPA routing."""

    def test_root_serves_spa(self, client):
        """Test root path serves index.html."""
        response = client.get("/")
        assert response.status_code == 200
        # SPA index.html should contain React app mount point
        assert "root" in response.text or "<!DOCTYPE" in response.text

    def test_frontend_routes_serve_spa(self, client):
        """Test frontend routes serve index.html (not 404)."""
        for path in ["/runs", "/projects", "/groups", "/diff", "/search"]:
            response = client.get(path)
            # Should serve SPA, not 404
            assert response.status_code == 200
