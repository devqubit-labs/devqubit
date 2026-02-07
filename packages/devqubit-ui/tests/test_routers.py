# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit UI API router."""

from __future__ import annotations

from unittest.mock import Mock


class TestCapabilities:
    """Capabilities endpoint."""

    def test_get_capabilities(self, client):
        response = client.get("/api/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "local"
        assert "version" in data
        assert "features" in data


class TestRunsApi:
    """Runs API endpoints."""

    def test_list_runs_empty(self, client, mock_registry):
        mock_registry.count_runs.return_value = 0

        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["count"] == 0
        assert data["total"] == 0
        assert data["offset"] == 0
        assert data["has_more"] is False

    def test_list_runs_with_data(self, client, mock_registry):
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
        mock_registry.count_runs.return_value = 1

        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["total"] == 1
        assert data["has_more"] is False

    def test_list_runs_pagination(self, client, mock_registry):
        mock_registry.list_runs.return_value = [
            {
                "run_id": f"r-{i}",
                "run_name": f"run-{i}",
                "project": "p",
                "status": "FINISHED",
                "created_at": "2025-01-01T00:00:00Z",
            }
            for i in range(10)
        ]
        mock_registry.count_runs.return_value = 25

        data = client.get("/api/runs?limit=10&offset=0").json()
        assert data["count"] == 10
        assert data["total"] == 25
        assert data["offset"] == 0
        assert data["has_more"] is True

    def test_list_runs_filter_by_project(self, client, mock_registry):
        mock_registry.count_runs.return_value = 0
        client.get("/api/runs?project=my-project")
        mock_registry.list_runs.assert_called_with(
            limit=50, offset=0, project="my-project"
        )

    def test_list_runs_with_search(self, client, mock_registry):
        mock_registry.search_runs.return_value = []
        data = client.get("/api/runs?q=metric.fidelity>0.9").json()
        mock_registry.search_runs.assert_called()
        # search also returns the paginated envelope
        assert "total" in data
        assert "has_more" in data

    def test_get_run_not_found(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("not found")
        assert client.get("/api/runs/nonexistent").status_code == 404

    def test_get_run_file_not_found(self, client, mock_registry):
        """FileNotFoundError from storage also yields 404."""
        mock_registry.load.side_effect = FileNotFoundError("gone")
        assert client.get("/api/runs/nonexistent").status_code == 404

    def test_get_run_success(self, client, mock_registry):
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

        data = client.get("/api/runs/test-123").json()
        assert data["run"]["run_id"] == "test-123"
        assert data["run"]["project"] == "proj"

    def test_delete_run_success(self, client, mock_registry):
        mock_registry.delete.return_value = True
        data = client.delete("/api/runs/test-123").json()
        assert data["status"] == "deleted"

    def test_delete_run_not_found(self, client, mock_registry):
        mock_registry.delete.return_value = False
        assert client.delete("/api/runs/nonexistent").status_code == 404


class TestArtifactsApi:
    """Artifacts API endpoints."""

    def test_get_artifact_not_found_run(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("not found")
        assert client.get("/api/runs/nonexistent/artifacts/0").status_code == 404

    def test_get_artifact_not_found_index(self, client, mock_registry):
        mock_run = Mock()
        mock_run.artifacts = []
        mock_registry.load.return_value = mock_run
        assert client.get("/api/runs/test-123/artifacts/99").status_code == 404

    def test_get_artifact_raw_not_found(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("not found")
        assert client.get("/api/runs/nonexistent/artifacts/0/raw").status_code == 404


class TestProjectsApi:
    """Projects API endpoints."""

    def test_list_projects_empty(self, client):
        data = client.get("/api/projects").json()
        assert data["projects"] == []

    def test_set_baseline_success(self, client, mock_registry):
        mock_run = Mock()
        mock_run.project = "test-project"
        mock_run.run_id = "test-123"
        mock_registry.load.return_value = mock_run

        data = client.post(
            "/api/projects/test-project/baseline/test-123?redirect=false"
        ).json()
        assert data["status"] == "ok"
        assert data["baseline_run_id"] == "test-123"

    def test_set_baseline_wrong_project(self, client, mock_registry):
        mock_run = Mock()
        mock_run.project = "other-project"
        mock_registry.load.return_value = mock_run

        resp = client.post(
            "/api/projects/test-project/baseline/test-123?redirect=false"
        )
        assert resp.status_code == 400


class TestGroupsApi:
    """Groups API endpoints."""

    def test_list_groups_empty(self, client):
        assert client.get("/api/groups").json()["groups"] == []

    def test_list_groups_filter_by_project(self, client, mock_registry):
        client.get("/api/groups?project=my-project")
        mock_registry.list_groups.assert_called_with(project="my-project")


class TestDiffApi:
    """Diff API endpoint."""

    def test_diff_missing_run(self, client, mock_registry):
        mock_registry.load.side_effect = KeyError("not found")
        assert client.get("/api/diff?a=bad-id&b=other-id").status_code == 404


class TestSpaRouting:
    """SPA routing."""

    def test_root_serves_spa(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "root" in response.text or "<!DOCTYPE" in response.text

    def test_frontend_routes_serve_spa(self, client):
        for path in ["/runs", "/projects", "/groups", "/diff", "/search"]:
            assert client.get(path).status_code == 200
