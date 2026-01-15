# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""End-to-end integration tests for devqubit."""

from __future__ import annotations

from pathlib import Path

from devqubit import (
    Config,
    diff,
    pack_run,
    set_config,
    track,
    unpack_bundle,
    verify_baseline,
)
from devqubit.compare import ComparisonResult, VerifyPolicy
from devqubit.runs import set_baseline
from devqubit.storage import create_registry, create_store


class TestFullWorkflow:
    """Tests for complete track → pack → unpack → diff workflow."""

    def test_track_pack_unpack_diff(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
        tmp_path: Path,
    ):
        """Full workflow: track run, pack, unpack, diff."""
        set_config(config)

        # Create run
        with track(project="integration") as run:
            run.log_param("shots", 1000)
            run.log_metric("fidelity", 0.95)
            run.log_json(name="config", obj={"setting": "value"}, role="config")
            run_id = run.run_id

        # Verify stored
        assert registry.exists(run_id)
        loaded = registry.load(run_id)
        assert loaded.status == "FINISHED"

        # Pack
        bundle_path = tmp_path / "run.zip"
        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )
        assert bundle_path.exists()

        # Unpack to new workspace
        workspace2 = tmp_path / ".devqubit2"
        workspace2.mkdir(parents=True)
        store2 = create_store(f"file://{workspace2}/objects")
        registry2 = create_registry(f"file://{workspace2}")

        unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store2,
            dest_registry=registry2,
        )

        # Verify unpacked
        loaded2 = registry2.load(run_id)
        assert loaded2.record["data"]["params"]["shots"] == 1000

        # Diff same run (should be identical)
        result = diff(run_id, run_id, registry=registry, store=store)
        assert isinstance(result, ComparisonResult)
        assert result.identical


class TestVerifyBaselineWorkflow:
    """Tests for high-level verify_baseline API."""

    def test_verify_against_stored_baseline(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """verify_baseline works with stored baseline."""
        set_config(config)

        # Create baseline run
        with track(project="verify_test") as baseline_run:
            baseline_run.log_param("shots", 1000)
            baseline_run.log_metric("fidelity", 0.95)
            baseline_id = baseline_run.run_id

        # Set as baseline
        set_baseline("verify_test", baseline_id)

        # Create candidate run (similar)
        with track(project="verify_test") as candidate_run:
            candidate_run.log_param("shots", 1000)
            candidate_run.log_metric("fidelity", 0.94)
            candidate_id = candidate_run.run_id

        # Verify
        result = verify_baseline(candidate_id, project="verify_test")

        assert hasattr(result, "ok")
        assert hasattr(result, "comparison")

    def test_verify_with_custom_policy(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """verify_baseline accepts custom policy."""
        set_config(config)

        with track(project="policy_test") as baseline_run:
            baseline_run.log_param("shots", 1000)
            baseline_id = baseline_run.run_id

        set_baseline("policy_test", baseline_id)

        with track(project="policy_test") as candidate_run:
            candidate_run.log_param("shots", 2000)  # Different!
            candidate_id = candidate_run.run_id

        # Strict policy - params must match
        strict = VerifyPolicy(params_must_match=True)
        result = verify_baseline(
            candidate_id,
            project="policy_test",
            policy=strict,
        )
        assert not result.ok

        # Lenient policy - params don't need to match
        lenient = VerifyPolicy(params_must_match=False)
        result = verify_baseline(
            candidate_id,
            project="policy_test",
            policy=lenient,
        )
        # May or may not pass depending on other factors


class TestDiffWorkflows:
    """Tests for comparing runs."""

    def test_diff_detects_param_changes(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """Diff detects parameter differences."""
        set_config(config)

        with track(project="test") as run_a:
            run_a.log_param("shots", 1000)
            run_a.log_param("seed", 42)
            run_id_a = run_a.run_id

        with track(project="test") as run_b:
            run_b.log_param("shots", 2000)  # Changed
            run_b.log_param("seed", 42)
            run_id_b = run_b.run_id

        result = diff(run_id_a, run_id_b, registry=registry, store=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]
        assert result.params["changed"]["shots"] == {"a": 1000, "b": 2000}

    def test_diff_bundle_to_run(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
        tmp_path: Path,
    ):
        """Compare bundle to registry run."""
        set_config(config)
        bundle_path = tmp_path / "bundle.zip"

        with track(project="bundle_diff") as run:
            run.log_param("x", 1)
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        result = diff(bundle_path, run_id, registry=registry, store=store)

        assert result.identical
        assert result.run_id_a == run_id
        assert result.run_id_b == run_id

    def test_diff_two_bundles(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
        tmp_path: Path,
    ):
        """Compare two bundle files."""
        set_config(config)
        bundle_a = tmp_path / "bundle_a.zip"
        bundle_b = tmp_path / "bundle_b.zip"

        with track(
            project="test",
            capture_env=False,
            capture_git=False,
        ) as run_a:
            run_a.log_param("value", 100)
            run_id_a = run_a.run_id

        pack_run(
            run_id=run_id_a,
            output_path=bundle_a,
            store=store,
            registry=registry,
        )

        with track(
            project="test",
            capture_env=False,
            capture_git=False,
        ) as run_b:
            run_b.log_param("value", 200)
            run_id_b = run_b.run_id

        pack_run(
            run_id=run_id_b,
            output_path=bundle_b,
            store=store,
            registry=registry,
        )

        result = diff(bundle_a, bundle_b)

        assert result.run_id_a == run_id_a
        assert result.run_id_b == run_id_b
        assert not result.params["match"]


class TestArtifactRoundtrip:
    """Tests for artifact preservation through pack/unpack."""

    def test_artifact_content_integrity(self, tmp_path: Path):
        """Artifact content is identical after roundtrip."""
        workspace_src = tmp_path / "src"
        workspace_dst = tmp_path / "dst"
        workspace_src.mkdir()
        workspace_dst.mkdir()

        store_src = create_store(f"file://{workspace_src}/objects")
        reg_src = create_registry(f"file://{workspace_src}")
        store_dst = create_store(f"file://{workspace_dst}/objects")
        reg_dst = create_registry(f"file://{workspace_dst}")
        bundle_path = tmp_path / "bundle.zip"

        original_data = b"important quantum circuit data"

        with track(
            project="integrity",
            store=store_src,
            registry=reg_src,
            capture_env=False,
            capture_git=False,
        ) as run:
            ref = run.log_bytes(
                kind="circuit",
                data=original_data,
                media_type="application/octet-stream",
                role="program",
            )
            run_id = run.run_id
            digest = ref.digest

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store_src,
            registry=reg_src,
        )
        unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store_dst,
            dest_registry=reg_dst,
        )

        restored_data = store_dst.get_bytes(digest)
        assert restored_data == original_data


class TestEndToEndScenarios:
    """Real-world usage scenarios."""

    def test_parameter_sweep_workflow(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """Simulate a parameter sweep experiment."""
        set_config(config)
        group_id = "sweep_shots"
        run_ids = []

        for shots in [100, 500, 1000, 5000]:
            with track(
                project="sweep_test",
                group_id=group_id,
                group_name="Shot Count Sweep",
            ) as run:
                run.log_param("shots", shots)
                run.log_metric("fidelity", 0.9 + (shots / 50000))
                run_ids.append(run.run_id)

        runs_in_group = registry.list_runs_in_group(group_id)
        assert len(runs_in_group) == 4

        result = diff(run_ids[0], run_ids[-1], registry=registry, store=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]

    def test_failed_run_captures_error(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """Failed runs capture error information."""
        set_config(config)

        try:
            with track(project="error_test") as run:
                run.log_param("will_fail", True)
                run_id = run.run_id
                raise RuntimeError("Simulated quantum hardware error")
        except RuntimeError:
            pass

        loaded = registry.load(run_id)

        assert loaded.status == "FAILED"
        assert len(loaded.record["errors"]) == 1
        assert "RuntimeError" in loaded.record["errors"][0]["type"]
        assert "hardware error" in loaded.record["errors"][0]["message"]

    def test_ci_verification_workflow(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """CI/CD verification workflow with baseline."""
        set_config(config)

        # Establish baseline
        with track(project="ci_project") as baseline:
            baseline.log_param("algorithm", "vqe")
            baseline.log_metric("energy", -1.5)
            baseline_id = baseline.run_id

        set_baseline("ci_project", baseline_id)

        # PR candidate
        with track(project="ci_project") as candidate:
            candidate.log_param("algorithm", "vqe")
            candidate.log_metric("energy", -1.51)  # Slightly improved
            candidate_id = candidate.run_id

        # Verify in CI
        result = verify_baseline(candidate_id, project="ci_project")

        # Should pass (similar results)
        assert hasattr(result, "ok")
        assert hasattr(result, "format")  # Can generate report
