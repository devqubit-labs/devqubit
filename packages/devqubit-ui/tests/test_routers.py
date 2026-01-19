# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit UI routers."""

from __future__ import annotations

from unittest.mock import Mock


class TestRunsRouter:
    """Tests for the runs router."""

    def test_index_redirects_to_runs(self, client):
        """Test that / redirects to /runs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/runs"

    def test_runs_list_empty(self, client):
        """Test runs list with no runs."""
        response = client.get("/runs")
        assert response.status_code == 200
        assert "Runs" in response.text
        assert "No runs found" in response.text

    def test_runs_list_with_runs(self, client, mock_registry):
        """Test runs list with some runs."""
        mock_registry.list_runs.return_value = [
            {
                "run_id": "test-run-123",
                "project": "test-project",
                "status": "FINISHED",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ]
        mock_registry.list_projects.return_value = ["test-project"]

        response = client.get("/runs")
        assert response.status_code == 200
        assert "test-run-123" in response.text or "test-run-12..." in response.text

    def test_runs_list_htmx_returns_fragment(self, client):
        """Test that HTMX requests return only the table fragment."""
        response = client.get("/runs", headers={"HX-Request": "true"})
        assert response.status_code == 200
        # Fragment should not contain full HTML structure
        assert "<!DOCTYPE" not in response.text

    def test_runs_filter_by_project(self, client, mock_registry):
        """Test filtering runs by project."""
        mock_registry.list_projects.return_value = ["proj-a", "proj-b"]

        response = client.get("/runs?project=proj-a")
        assert response.status_code == 200
        mock_registry.list_runs.assert_called_with(limit=50, project="proj-a")

    def test_runs_search_query(self, client, mock_registry):
        """Test searching runs with query."""
        mock_registry.search_runs.return_value = []

        response = client.get("/runs?q=metric.fidelity%20%3E%200.9")
        assert response.status_code == 200
        mock_registry.search_runs.assert_called()

    def test_run_detail_not_found(self, client, mock_registry):
        """Test run detail for non-existent run."""
        mock_registry.load.side_effect = Exception("Not found")

        response = client.get("/runs/nonexistent-id")
        assert response.status_code == 404

    def test_search_page(self, client):
        """Test search page loads."""
        response = client.get("/search")
        assert response.status_code == 200
        assert "Query Syntax" in response.text


class TestProjectsRouter:
    """Tests for the projects router."""

    def test_projects_list_empty(self, client):
        """Test projects list with no projects."""
        response = client.get("/projects")
        assert response.status_code == 200
        assert "Projects" in response.text

    def test_projects_list_with_projects(self, client, mock_registry):
        """Test projects list with some projects."""
        mock_registry.list_projects.return_value = ["project-a", "project-b"]
        mock_registry.count_runs.return_value = 5
        mock_registry.get_baseline.return_value = None

        response = client.get("/projects")
        assert response.status_code == 200
        assert "project-a" in response.text


class TestGroupsRouter:
    """Tests for the groups router."""

    def test_groups_list_empty(self, client):
        """Test groups list with no groups."""
        response = client.get("/groups")
        assert response.status_code == 200
        assert "Run Groups" in response.text

    def test_groups_filter_by_project(self, client, mock_registry):
        """Test filtering groups by project."""
        mock_registry.list_projects.return_value = ["proj-a"]

        response = client.get("/groups?project=proj-a")
        assert response.status_code == 200
        mock_registry.list_groups.assert_called_with(project="proj-a")


class TestDiffRouter:
    """Tests for the diff router."""

    def test_diff_select_page(self, client, mock_registry):
        """Test diff selection page loads."""
        mock_registry.list_runs.return_value = []

        response = client.get("/diff")
        assert response.status_code == 200
        assert "Compare Runs" in response.text

    def test_diff_missing_run_a(self, client, mock_registry):
        """Test diff with missing run A."""
        mock_registry.load.side_effect = Exception("Not found")

        response = client.get("/diff?a=bad-id&b=other-id")
        assert response.status_code == 404


class TestApiRouter:
    """Tests for the API router."""

    def test_api_runs_list(self, client, mock_registry):
        """Test API runs list returns JSON."""
        mock_registry.list_runs.return_value = []

        response = client.get("/api/runs")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert "runs" in data
        assert "count" in data

    def test_api_set_baseline_redirect(self, client, mock_registry):
        """Test set baseline with redirect."""
        mock_run = Mock()
        mock_run.project = "test-project"
        mock_run.run_id = "test-run-123"
        mock_registry.load.return_value = mock_run

        response = client.post(
            "/api/projects/test-project/baseline/test-run-123",
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "/runs/test-run-123" in response.headers["location"]

    def test_api_set_baseline_json(self, client, mock_registry):
        """Test set baseline with JSON response."""
        mock_run = Mock()
        mock_run.project = "test-project"
        mock_registry.load.return_value = mock_run

        response = client.post(
            "/api/projects/test-project/baseline/test-run-123?redirect=false"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["baseline_run_id"] == "test-run-123"

    def test_api_set_baseline_wrong_project(self, client, mock_registry):
        """Test set baseline with run from different project."""
        mock_run = Mock()
        mock_run.project = "other-project"
        mock_registry.load.return_value = mock_run

        response = client.post(
            "/api/projects/test-project/baseline/test-run-123?redirect=false"
        )
        assert response.status_code == 400
