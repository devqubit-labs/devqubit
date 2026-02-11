# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""CLI integration tests for devqubit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from click.testing import Result
from devqubit_engine.storage.backends.local import LocalRegistry
from devqubit_engine.tracking.record import RunRecord


# =============================================================================
# TEST HELPERS
# =============================================================================


def parse_json(result: Result) -> Any:
    """Parse JSON from command output, with helpful error on failure."""
    try:
        return json.loads(result.output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON output: {e}\nOutput was:\n{result.output[:500]}")


def assert_ok(result: Result) -> None:
    """Assert command succeeded (exit_code == 0)."""
    assert result.exit_code == 0, f"Command failed:\n{result.output}"


def assert_err(result: Result) -> None:
    """Assert command failed (exit_code != 0)."""
    assert result.exit_code != 0, f"Expected error but got success:\n{result.output}"


# =============================================================================
# RUNS COMMANDS
# =============================================================================


class TestList:
    """Tests for `devqubit list`."""

    def test_empty_workspace(self, invoke: Callable) -> None:
        result = invoke("list")
        assert_ok(result)
        assert "No runs found" in result.output

    def test_shows_runs(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("list")
        assert_ok(result)
        assert "sample_project" in result.output
        assert "Recent Runs" in result.output

    def test_filter_by_project(self, invoke: Callable, make_run: Callable) -> None:
        make_run(project="alpha")
        make_run(project="beta")
        result = invoke("list", "--project", "alpha")
        assert_ok(result)
        assert "alpha" in result.output
        assert "beta" not in result.output

    def test_filter_by_status(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="finished_run", status="FINISHED")
        make_run(run_id="failed_run", status="FAILED")
        result = invoke("list", "--status", "FAILED")
        assert_ok(result)
        assert "failed_run" in result.output

    def test_filter_by_tag(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="tagged_run", tags={"env": "prod"})
        make_run(run_id="untagged_run", tags={})
        result = invoke("list", "--tag", "env=prod")
        assert_ok(result)
        assert "tagged_run" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("list", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(r.get("run_id") == sample_run.run_id for r in data)

    def test_limit(self, invoke: Callable, make_run: Callable) -> None:
        for i in range(5):
            make_run(run_id=f"run_{i}")
        result = invoke("list", "--limit", "2", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert len(data) == 2


class TestSearch:
    """Tests for `devqubit search`."""

    def test_by_metric(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="high_fidelity", metrics={"fidelity": 0.99})
        make_run(run_id="low_fidelity", metrics={"fidelity": 0.50})
        result = invoke("search", "metric.fidelity > 0.9")
        assert_ok(result)
        # Run ID is truncated in table output
        assert "high_fidelit" in result.output

    def test_by_params(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="many_shots", params={"shots": 10000})
        make_run(run_id="few_shots", params={"shots": 100})
        result = invoke("search", "params.shots >= 1000")
        assert_ok(result)
        assert "many_shots" in result.output

    def test_invalid_query(self, invoke: Callable) -> None:
        result = invoke("search", "invalid!!!")
        assert_err(result)
        assert "Invalid" in result.output

    def test_json_format(self, invoke: Callable, make_run: Callable) -> None:
        make_run(metrics={"fidelity": 0.99})
        result = invoke("search", "metric.fidelity > 0.5", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)


class TestShow:
    """Tests for `devqubit show`."""

    def test_shows_details(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("show", sample_run.run_id)
        assert_ok(result)
        assert sample_run.run_id in result.output
        assert "sample_project" in result.output
        assert "FINISHED" in result.output
        assert "fidelity" in result.output

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("show", "nonexistent")
        assert_err(result)
        assert "not found" in result.output.lower()

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("show", sample_run.run_id, "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert data["run_id"] == sample_run.run_id
        assert "project" in data
        assert "artifacts" in data


class TestDelete:
    """Tests for `devqubit delete`."""

    def test_delete_with_yes(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("delete", sample_run.run_id, "--yes")
        assert_ok(result)
        assert "Deleted" in result.output
        # Verify deleted
        result = invoke("show", sample_run.run_id)
        assert_err(result)

    def test_abort_without_confirm(
        self, invoke: Callable, sample_run: RunRecord
    ) -> None:
        result = invoke("delete", sample_run.run_id, input="n\n")
        assert_ok(result)
        assert "Cancelled" in result.output
        # Verify NOT deleted
        result = invoke("show", sample_run.run_id)
        assert_ok(result)

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("delete", "nonexistent", "--yes")
        assert_err(result)


class TestProjects:
    """Tests for `devqubit projects`."""

    def test_empty(self, invoke: Callable) -> None:
        result = invoke("projects")
        assert_ok(result)
        assert "No projects" in result.output

    def test_lists_projects(self, invoke: Callable, make_run: Callable) -> None:
        make_run(project="proj_a")
        make_run(project="proj_b")
        result = invoke("projects")
        assert_ok(result)
        assert "proj_a" in result.output
        assert "proj_b" in result.output

    def test_json_format(self, invoke: Callable, make_run: Callable) -> None:
        make_run(project="json_proj")
        result = invoke("projects", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)
        assert any(p["name"] == "json_proj" for p in data)
        # Check structure
        proj = next(p for p in data if p["name"] == "json_proj")
        assert "run_count" in proj
        assert "baseline_run_id" in proj


class TestGroups:
    """Tests for `devqubit groups` subcommands."""

    def test_list_empty(self, invoke: Callable) -> None:
        result = invoke("groups", "list")
        assert_ok(result)
        assert "No groups" in result.output

    def test_list_with_groups(self, invoke: Callable, make_run: Callable) -> None:
        make_run(group_id="sweep_001", group_name="Parameter Sweep")
        make_run(group_id="sweep_001")
        result = invoke("groups", "list")
        assert_ok(result)
        assert "sweep_001" in result.output

    def test_list_json_format(self, invoke: Callable, make_run: Callable) -> None:
        make_run(group_id="grp_json")
        result = invoke("groups", "list", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)

    def test_show_group(self, invoke: Callable, make_run: Callable) -> None:
        make_run(run_id="g1_run1", group_id="grp1")
        make_run(run_id="g1_run2", group_id="grp1")
        result = invoke("groups", "show", "grp1")
        assert_ok(result)

    def test_show_empty_group(self, invoke: Callable) -> None:
        result = invoke("groups", "show", "nonexistent_group")
        assert_ok(result)
        assert "No runs found" in result.output


# =============================================================================
# ARTIFACTS COMMANDS
# =============================================================================


class TestArtifactsList:
    """Tests for `devqubit artifacts list`."""

    def test_list_artifacts(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "list", sample_run.run_id)
        assert_ok(result)
        assert "results" in result.output.lower() or "counts" in result.output.lower()

    def test_not_found(self, invoke: Callable) -> None:
        result = invoke("artifacts", "list", "nonexistent")
        assert_err(result)

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "list", sample_run.run_id, "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)


class TestArtifactsShow:
    """Tests for `devqubit artifacts show`."""

    def test_by_index(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "show", sample_run.run_id, "0")
        assert_ok(result)

    def test_raw_output(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "show", sample_run.run_id, "0", "--raw")
        assert_ok(result)
        data = parse_json(result)
        assert "counts" in data

    def test_invalid_index(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "show", sample_run.run_id, "999")
        assert_err(result)


class TestArtifactsCounts:
    """Tests for `devqubit artifacts counts`."""

    def test_shows_counts(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "counts", sample_run.run_id)
        assert_ok(result)
        assert "00" in result.output
        assert "11" in result.output
        assert "Total shots" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("artifacts", "counts", sample_run.run_id, "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "counts" in data
        assert data["counts"]["00"] == 500


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
        assert_ok(result)
        assert "Added" in result.output
        # Verify
        updated = registry.load(run.run_id)
        assert updated.record["data"]["tags"]["env"] == "prod"

    def test_add_key_only_tag(
        self, invoke: Callable, make_run: Callable, registry: LocalRegistry
    ) -> None:
        run = make_run(run_id="tag_key_only")
        result = invoke("tag", "add", run.run_id, "important")
        assert_ok(result)
        updated = registry.load(run.run_id)
        assert "important" in updated.record["data"]["tags"]

    def test_add_multiple_tags(
        self, invoke: Callable, make_run: Callable, registry: LocalRegistry
    ) -> None:
        run = make_run(run_id="multi_tag")
        result = invoke("tag", "add", run.run_id, "a=1", "b=2")
        assert_ok(result)
        updated = registry.load(run.run_id)
        assert updated.record["data"]["tags"]["a"] == "1"
        assert updated.record["data"]["tags"]["b"] == "2"


class TestTagRemove:
    """Tests for `devqubit tag remove`."""

    def test_remove_tag(
        self, invoke: Callable, sample_run: RunRecord, registry: LocalRegistry
    ) -> None:
        result = invoke("tag", "remove", sample_run.run_id, "experiment")
        assert_ok(result)
        updated = registry.load(sample_run.run_id)
        assert "experiment" not in updated.record["data"]["tags"]


class TestTagList:
    """Tests for `devqubit tag list`."""

    def test_list_tags(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("tag", "list", sample_run.run_id)
        assert_ok(result)
        assert "experiment" in result.output
        assert "bell" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("tag", "list", sample_run.run_id, "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "experiment" in data


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
        assert_ok(result)
        assert output.exists()
        assert "Packed" in result.output

    def test_pack_json_format(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "bundle.zip"
        result = invoke(
            "pack", sample_run.run_id, "--out", str(output), "--format", "json"
        )
        assert_ok(result)
        data = parse_json(result)
        assert "run_id" in data
        assert "artifact_count" in data

    def test_not_found(self, invoke: Callable, tmp_path: Path) -> None:
        result = invoke("pack", "nonexistent", "--out", str(tmp_path / "x.zip"))
        assert_err(result)


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
        assert_ok(result)
        assert "Unpacked" in result.output

    def test_unpack_json_format(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        tmp_path: Path,
    ) -> None:
        bundle = tmp_path / "bundle.zip"
        invoke("pack", sample_run.run_id, "--out", str(bundle))

        dest = tmp_path / "new_workspace"
        result = invoke("unpack", str(bundle), "--to", str(dest), "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "run_id" in data


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
        assert_ok(result)
        assert sample_run.run_id in result.output
        assert "sample_project" in result.output

    def test_json_format(
        self, invoke: Callable, sample_run: RunRecord, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "bundle.zip"
        invoke("pack", sample_run.run_id, "--out", str(bundle))

        result = invoke("info", str(bundle), "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "run_id" in data
        assert "artifact_count" in data

    def test_invalid_bundle(self, invoke: Callable, tmp_path: Path) -> None:
        fake = tmp_path / "fake.zip"
        fake.write_text("not a zip")
        result = invoke("info", str(fake))
        assert_err(result)


# =============================================================================
# COMPARE COMMANDS
# =============================================================================


class TestDiff:
    """Tests for `devqubit diff`."""

    def test_diff_runs(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_a", counts={"00": 500, "11": 500})
        run_b = make_run(run_id="diff_b", counts={"00": 480, "11": 520})
        result = invoke("diff", run_a.run_id, run_b.run_id)
        assert_ok(result)

    def test_json_format(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_json_a")
        run_b = make_run(run_id="diff_json_b")
        result = invoke("diff", run_a.run_id, run_b.run_id, "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "run_a" in data
        assert "run_b" in data
        assert "identical" in data

    def test_summary_format(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_sum_a")
        run_b = make_run(run_id="diff_sum_b")
        result = invoke("diff", run_a.run_id, run_b.run_id, "--format", "summary")
        assert_ok(result)

    def test_not_found(self, invoke: Callable, make_run: Callable) -> None:
        run_a = make_run(run_id="diff_exists")
        result = invoke("diff", run_a.run_id, "nonexistent")
        assert_err(result)


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
        # Should have some verdict
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
        assert_ok(result)

    def test_json_format(self, invoke: Callable, make_run: Callable) -> None:
        baseline = make_run(run_id="verify_json_base")
        candidate = make_run(run_id="verify_json_cand")
        result = invoke(
            "verify",
            candidate.run_id,
            "--baseline",
            baseline.run_id,
            "--format",
            "json",
        )
        # Exit code depends on pass/fail, but output should be JSON
        data = parse_json(result)
        assert "ok" in data
        assert "failures" in data


# =============================================================================
# ADMIN COMMANDS
# =============================================================================


class TestStorageGc:
    """Tests for `devqubit storage gc`."""

    def test_dry_run(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "gc", "--dry-run")
        assert_ok(result)
        assert "Dry run" in result.output

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "gc", "--dry-run", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "objects_total" in data
        assert "objects_orphaned" in data


class TestStoragePrune:
    """Tests for `devqubit storage prune`."""

    def test_dry_run(self, invoke: Callable, make_run: Callable) -> None:
        make_run(status="FAILED")
        result = invoke("storage", "prune", "--status", "FAILED", "--dry-run")
        assert_ok(result)
        assert "Dry run" in result.output


class TestStorageHealth:
    """Tests for `devqubit storage health`."""

    def test_health_check(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "health")
        assert_ok(result)
        assert "runs" in result.output.lower() or "objects" in result.output.lower()

    def test_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("storage", "health", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "total_runs" in data or "runs" in data


class TestBaseline:
    """Tests for `devqubit baseline` subcommands."""

    def test_set_baseline(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        registry: LocalRegistry,
    ) -> None:
        result = invoke("baseline", "set", "sample_project", sample_run.run_id)
        assert_ok(result)
        baseline = registry.get_baseline("sample_project")
        assert baseline["run_id"] == sample_run.run_id

    def test_get_baseline(self, invoke: Callable, sample_run: RunRecord) -> None:
        invoke("baseline", "set", "sample_project", sample_run.run_id)
        result = invoke("baseline", "get", "sample_project")
        assert_ok(result)
        assert sample_run.run_id in result.output

    def test_get_not_set(self, invoke: Callable) -> None:
        result = invoke("baseline", "get", "no_baseline_proj")
        assert_ok(result)
        assert "No baseline" in result.output

    def test_get_json_format(self, invoke: Callable, sample_run: RunRecord) -> None:
        invoke("baseline", "set", "sample_project", sample_run.run_id)
        result = invoke("baseline", "get", "sample_project", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert data["run_id"] == sample_run.run_id

    def test_clear_baseline(
        self,
        invoke: Callable,
        sample_run: RunRecord,
        registry: LocalRegistry,
    ) -> None:
        invoke("baseline", "set", "sample_project", sample_run.run_id)
        result = invoke("baseline", "clear", "sample_project", "--yes")
        assert_ok(result)
        assert registry.get_baseline("sample_project") is None

    def test_list_baselines(self, invoke: Callable, make_run: Callable) -> None:
        run = make_run(project="proj_x")
        invoke("baseline", "set", "proj_x", run.run_id)
        result = invoke("baseline", "list")
        assert_ok(result)
        assert "proj_x" in result.output

    def test_list_json_format(self, invoke: Callable, make_run: Callable) -> None:
        run = make_run(project="proj_json")
        invoke("baseline", "set", "proj_json", run.run_id)
        result = invoke("baseline", "list", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert isinstance(data, list)


class TestConfig:
    """Tests for `devqubit config`."""

    def test_shows_config(self, invoke: Callable, workspace: Path) -> None:
        result = invoke("config")
        assert_ok(result)
        assert "Home" in result.output or str(workspace) in result.output

    def test_json_format(self, invoke: Callable) -> None:
        result = invoke("config", "--format", "json")
        assert_ok(result)
        data = parse_json(result)
        assert "root_dir" in data or "home" in data


# =============================================================================
# GLOBAL OPTIONS
# =============================================================================


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_quiet_flag(self, invoke: Callable, sample_run: RunRecord) -> None:
        result = invoke("--quiet", "list")
        assert_ok(result)

    def test_help(self, cli_runner: Any) -> None:
        from devqubit_engine.cli import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert_ok(result)
        assert "devqubit" in result.output.lower()
        assert "list" in result.output
        assert "diff" in result.output

    def test_version(self, cli_runner: Any) -> None:
        from devqubit_engine.cli import cli

        result = cli_runner.invoke(cli, ["--version"])
        assert_ok(result)

    def test_command_help(self, invoke: Callable) -> None:
        result = invoke("list", "--help")
        assert_ok(result)
        assert "--project" in result.output
        assert "--format" in result.output
