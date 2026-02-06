# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for run lifecycle contract (status + ended_at).

The frontend Duration column and auto-poll logic depend on specific fields
being present in API responses. These tests guard that contract.
"""

from __future__ import annotations

from unittest.mock import Mock


class TestRunLifecycleContract:
    """Ensure list and detail responses carry the fields the UI needs."""

    def test_list_runs_includes_ended_at(self, client, mock_registry):
        """ended_at must be in list response — Duration column reads it."""
        mock_registry.list_runs.return_value = [
            {
                "run_id": "r-1",
                "run_name": "finished run",
                "project": "proj",
                "adapter": "qiskit",
                "status": "FINISHED",
                "created_at": "2025-01-01T00:00:00Z",
                "ended_at": "2025-01-01T00:05:00Z",
            }
        ]

        data = client.get("/api/runs").json()
        run = data["runs"][0]
        assert "ended_at" in run
        assert run["ended_at"] == "2025-01-01T00:05:00Z"

    def test_list_runs_running_has_null_ended_at(self, client, mock_registry):
        """RUNNING run must have ended_at=null — UI uses this to show live timer."""
        mock_registry.list_runs.return_value = [
            {
                "run_id": "r-2",
                "run_name": "active run",
                "project": "proj",
                "adapter": "qiskit",
                "status": "RUNNING",
                "created_at": "2025-01-01T00:00:00Z",
                "ended_at": None,
            }
        ]

        data = client.get("/api/runs").json()
        run = data["runs"][0]
        assert run["status"] == "RUNNING"
        assert run["ended_at"] is None

    def test_get_run_detail_includes_ended_at(self, client, mock_registry):
        """Detail endpoint must include ended_at — detail page Duration reads it."""
        mock_run = Mock()
        mock_run.run_id = "r-3"
        mock_run.run_name = "done"
        mock_run.project = "proj"
        mock_run.adapter = "qiskit"
        mock_run.status = "FINISHED"
        mock_run.created_at = "2025-01-01T00:00:00Z"
        mock_run.ended_at = "2025-01-01T00:10:00Z"
        mock_run.fingerprints = {}
        mock_run.group_id = None
        mock_run.group_name = None
        mock_run.artifacts = []
        mock_run.record = {"backend": {}, "data": {}, "info": {}}
        mock_registry.load.return_value = mock_run

        data = client.get("/api/runs/r-3").json()
        assert data["run"]["ended_at"] == "2025-01-01T00:10:00Z"

    def test_get_run_detail_running_has_null_ended_at(self, client, mock_registry):
        """RUNNING detail must have ended_at=null — UI switches to live timer."""
        mock_run = Mock()
        mock_run.run_id = "r-4"
        mock_run.run_name = "in-progress"
        mock_run.project = "proj"
        mock_run.adapter = "qiskit"
        mock_run.status = "RUNNING"
        mock_run.created_at = "2025-06-01T12:00:00Z"
        mock_run.ended_at = None
        mock_run.fingerprints = {}
        mock_run.group_id = None
        mock_run.group_name = None
        mock_run.artifacts = []
        mock_run.record = {"backend": {}, "data": {}, "info": {}}
        mock_registry.load.return_value = mock_run

        data = client.get("/api/runs/r-4").json()
        assert data["run"]["status"] == "RUNNING"
        assert data["run"]["ended_at"] is None
