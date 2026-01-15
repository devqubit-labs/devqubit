# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
CLI integration tests for devqubit.

Tests all user-facing CLI commands end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from devqubit_engine.storage.backends.local import LocalRegistry
from devqubit_engine.tracking.record import RunRecord


# =============================================================================
# RUNS COMMANDS
# =============================================================================


class TestList:
    """Tests for `devqubit list`."""

    def test_empty_workspace(self, invoke: Callable) -> None:
        result = invoke("list")
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_shows_runs(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("list")
        assert result.exit_code == 0
        assert "sample_project" in result.output

    def test_filter_by_project(self, invoke: Callable, make_run: Callable) -> None:
        make_run(project="alpha")
        make_run(project="beta")
        result = invoke("list", "--project", "alpha")
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" not in result.output

    def test_filter_by_status(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="finished_run", status="FINISHED")
        make_run(run_id="failed_run", status="FAILED")
        result = invoke("list", "--status", "FAILED")
        assert result.exit_code == 0
        assert "failed_run" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("list", "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_limit(self, invoke: Callable, make_run: Callable) -> None:
        for i in range(5):
            make_run(run_id=f"run_{i}")
        result = invoke("list", "--limit", "2")
        assert result.exit_code == 0


class TestSearch:
    """Tests for `devqubit search`."""

    def test_by_metric(self, invoke: Callable, make_run: Callable) -> None:
        make_run(metrics={"fidelity": 0.99})
        make_run(metrics={"fidelity": 0.50})
        result = invoke("search", "metric.fidelity > 0.9")
        assert result.exit_code == 0

    def test_invalid_query(self, invoke: Callable) -> None:
        result = invoke("search", "invalid!!!")
        assert result.exit_code != 0
        assert "Invalid" in result.output


class TestShow:
    """Tests for `devqubit show`."""

    def test_shows_details(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("show", sample_run.run_id)
        assert result.exit_code == 0
        assert sample_run.run_id in result.output
        assert "sample_project" in result.output

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("show", "nonexistent")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("show", sample_run.run_id, "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["run_id"] == sample_run.run_id


class TestDelete:
    """Tests for `devqubit delete`."""

    def test_delete_with_yes(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("delete", sample_run.run_id, "--yes")
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Verify deleted
        result = invoke("show", sample_run.run_id)
        assert result.exit_code != 0

    def test_abort_without_confirm(
        self, invoke: Callable, sample_run: RunRecord
    ) -> None:
        result = invoke("delete", sample_run.run_id, input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("delete", "nonexistent", "--yes")
        assert result.exit_code != 0


class TestProjects:
    """Tests for `devqubit projects`."""

    def test_empty(self, invoke: Callable) -> None:
        result = invoke("projects")
        assert result.exit_code == 0
        assert "No projects" in result.output

    def test_lists_projects(self, invoke: Callable, make_run: Callable) -> None:
        make_run(project="proj_a")
        make_run(project="proj_b")
        result = invoke("projects")
        assert result.exit_code == 0
        assert "proj_a" in result.output
        assert "proj_b" in result.output


class TestGroups:
    """Tests for `devqubit groups` subcommands."""

    def test_list_empty(self, invoke: Callable) -> None:
        result = invoke("groups", "list")
        assert result.exit_code == 0
        assert "No groups" in result.output

    def test_list_with_groups(self, invoke: Callable, make_run: Callable) -> None:
        make_run(group_id="sweep_001")
        make_run(group_id="sweep_001")
        result = invoke("groups", "list")
        assert result.exit_code == 0
        assert "sweep_001" in result.output

    def test_show_group(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="g1_run1", group_id="grp1")
        make_run(run_id="g1_run2", group_id="grp1")
        result = invoke("groups", "show", "grp1")
        assert result.exit_code == 0


# =============================================================================
# ARTIFACTS COMMANDS
# =============================================================================


class TestArtifactsList:
    """Tests for `devqubit artifacts list`."""

    def test_list_artifacts(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "list", sample_run.run_id)
        assert result.exit_code == 0
        assert "counts" in result.output.lower() or "results" in result.output.lower()

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("artifacts", "list", "nonexistent")
        assert result.exit_code != 0

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "list", sample_run.run_id, "--format", "json")
        assert result.exit_code == 0
        json.loads(result.output)


class TestArtifactsShow:
    """Tests for `devqubit artifacts show`."""

    def test_by_index(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "show", sample_run.run_id, "0")
        assert result.exit_code == 0

    def test_raw_output(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "show", sample_run.run_id, "0", "--raw")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "counts" in data


class TestArtifactsCounts:
    """Tests for `devqubit artifacts counts`."""

    def test_shows_counts(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "counts", sample_run.run_id)
        assert result.exit_code == 0
        assert "00" in result.output
        assert "11" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "counts", sample_run.run_id, "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "counts" in data


# =============================================================================
# TAG COMMANDS
# =============================================================================


class TestTagAdd:
    """Tests for `devqubit tag add`."""

    def test_add_tag(
        self, invoke: Callable, make_run: Callable, registry: LocalRegistry
    ) -> None:
        run = make_run(run_id="tag_test")
        result = invoke("tag", "add", run.run_id, "env=prod")
        assert result.exit_code == 0
        assert "Added" in result.output
        # Verify
        updated = registry.load(run.run_id)
        assert updated.record["data"]["tags"]["env"] == "prod"

    def test_add_key_only_tag(
        self, invoke: Callable, make_run: Callable, registry: LocalRegistry
    ) -> None:
        run = make_run(run_id="tag_key_only")
        result = invoke("tag", "add", run.run_id, "important")
        assert result.exit_code == 0


class TestTagRemove:
    """Tests for `devqubit tag remove`."""

    def test_remove_tag(
        self, invoke: Callable, sample_run: RunRecord, registry: LocalRegistry
    ) -> None:
        result = invoke("tag", "remove", sample_run.run_id, "experiment")
        assert result.exit_code == 0
        updated = registry.load(sample_run.run_id)
        assert "experiment" not in updated.record["data"]["tags"]


class TestTagList:
    """Tests for `devqubit tag list`."""

    def test_list_tags(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("tag", "list", sample_run.run_id)
        assert result.exit_code == 0
        assert "experiment" in result.output


# =============================================================================
# BUNDLE COMMANDS
# =============================================================================


class TestPack:
    """Tests for `devqubit pack`."""

    def test_pack_run(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "bundle.zip"
        result = invoke("pack", sample_run.run_id, "--out", str(output))
        assert result.exit_code == 0
        assert output.exists()

    def test_default_filename(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = invoke("pack", sample_run.run_id)
        assert result.exit_code == 0

    def test_not_found(self, invoke: Callable, tmp_path: Path) -> None:
        result = invoke("pack", "nonexistent", "--out", str(tmp_path / "x.zip"))
        assert result.exit_code != 0


class TestUnpack:
    """Tests for `devqubit unpack`."""

    def test_unpack_bundle(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
    ) -> None:
        bundle = tmp_path / "bundle.zip"
        invoke("pack", sample_run.run_id, "--out", str(bundle))

        dest = tmp_path / "new_workspace"
        result = invoke("unpack", str(bundle), "--to", str(dest))
        assert result.exit_code == 0
        assert "Unpacked" in result.output


class TestInfo:
    """Tests for `devqubit info`."""

    def test_bundle_info(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
    ) -> None:
        bundle = tmp_path / "bundle.zip"
        invoke("pack", sample_run.run_id, "--out", str(bundle))

        result = invoke("info", str(bundle))
        assert result.exit_code == 0
        assert sample_run.run_id in result.output

    def test_json_format(
        self, invoke: Callable, sample_run: RunRecord, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "bundle.zip"
        invoke("pack", sample_run.run_id, "--out", str(bundle))

        result = invoke("info", str(bundle), "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "run_id" in data


# =============================================================================
# COMPARE COMMANDS
# =============================================================================


class TestDiff:
    """Tests for `devqubit diff`."""

    def test_diff_runs(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_a", counts={"00": 500, "11": 500})
        run_b = make_run(run_id="diff_b", counts={"00": 480, "11": 520})
        result = invoke("diff", run_a.run_id, run_b.run_id)
        assert result.exit_code == 0

    def test_json_format(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_json_a")
        run_b = make_run(run_id="diff_json_b")
        result = invoke("diff", run_a.run_id, run_b.run_id, "--format", "json")
        assert result.exit_code == 0
        json.loads(result.output)

    def test_summary_format(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_sum_a")
        run_b = make_run(run_id="diff_sum_b")
        result = invoke("diff", run_a.run_id, run_b.run_id, "--format", "summary")
        assert result.exit_code == 0


class TestVerify:
    """Tests for `devqubit verify`."""

    def test_verify_against_baseline(
        self,
        invoke: Callable,
        make_run: Callable,
    ) -> None:
        baseline = make_run(
            run_id="baseline",
            counts={"00": 500, "11": 500},
        )
        candidate = make_run(
            run_id="candidate",
            counts={"00": 495, "11": 505},
        )
        result = invoke(
            "verify",
            candidate.run_id,
            "--baseline",
            baseline.run_id,
        )
        assert "PASS" in result.output or "FAIL" in result.output

    def test_allow_missing(self, invoke: Callable, make_run: Callable) -> None:
        candidate = make_run(project="no_baseline_proj")
        result = invoke(
            "verify",
            candidate.run_id,
            "--project",
            "no_baseline_proj",
            "--allow-missing",
        )
        assert result.exit_code == 0


class TestReplay:
    """Tests for `devqubit replay`."""

    def test_list_backends(self, invoke: Callable) -> None:
        result = invoke("replay", "--list-backends")
        assert result.exit_code == 0

    def test_requires_experimental(
        self,
        invoke: Callable,
        sample_run: RunRecord,
    ) -> None:
        result = invoke("replay", sample_run.run_id)
        assert result.exit_code != 0
        assert "experimental" in result.output.lower()


# =============================================================================
# ADMIN COMMANDS
# =============================================================================


class TestStorageGc:
    """Tests for `devqubit storage gc`."""

    def test_dry_run(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "gc", "--dry-run")
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "gc", "--dry-run", "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "objects_total" in data


class TestStoragePrune:
    """Tests for `devqubit storage prune`."""

    def test_dry_run(self, invoke: Callable, make_run: Callable) -> None:
        make_run(status="FAILED")
        result = invoke("storage", "prune", "--status", "FAILED", "--dry-run")
        assert result.exit_code == 0
        assert "Dry run" in result.output


class TestStorageHealth:
    """Tests for `devqubit storage health`."""

    def test_health_check(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "health")
        assert result.exit_code == 0
        assert "runs" in result.output.lower()


class TestBaseline:
    """Tests for `devqubit baseline` subcommands."""

    def test_set_baseline(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        registry: LocalRegistry,
    ) -> None:
        result = invoke("baseline", "set", "sample_project", sample_run.run_id)
        assert result.exit_code == 0
        baseline = registry.get_baseline("sample_project")
        assert baseline["run_id"] == sample_run.run_id

    def test_get_baseline(self, invoke: Callable, sample_run: RunRecord) -> None:
        invoke("baseline", "set", "sample_project", sample_run.run_id)
        result = invoke("baseline", "get", "sample_project")
        assert result.exit_code == 0
        assert sample_run.run_id in result.output

    def test_get_not_set(self, invoke: Callable) -> None:
        result = invoke("baseline", "get", "no_baseline_proj")
        assert result.exit_code == 0
        assert "No baseline" in result.output

    def test_clear_baseline(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        registry: LocalRegistry,
    ) -> None:
        invoke("baseline", "set", "sample_project", sample_run.run_id)
        result = invoke("baseline", "clear", "sample_project", "--yes")
        assert result.exit_code == 0
        assert registry.get_baseline("sample_project") is None

    def test_list_baselines(self, invoke: Callable, make_run: Callable) -> None:
        run = make_run(project="proj_x")
        invoke("baseline", "set", "proj_x", run.run_id)
        result = invoke("baseline", "list")
        assert result.exit_code == 0
        assert "proj_x" in result.output


class TestConfig:
    """Tests for `devqubit config`."""

    def test_shows_config(self, invoke: Callable, workspace: Path) -> None:
        result = invoke("config")
        assert result.exit_code == 0
        assert "Home" in result.output or str(workspace) in result.output


# =============================================================================
# GLOBAL OPTIONS
# =============================================================================


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_quiet_flag(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("--quiet", "list")
        assert result.exit_code == 0

    def test_help(self, cli_runner: Any) -> None:
        from devqubit_engine.cli import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "devqubit" in result.output.lower()
