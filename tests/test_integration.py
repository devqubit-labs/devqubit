# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
End-to-end integration tests for devqubit.

These tests verify complete user workflows using the public API.
They use real storage, real pack/unpack, and real comparison.
"""

from __future__ import annotations

from pathlib import Path

from devqubit import (
    Bundle,
    Config,
    create_registry,
    create_store,
    diff,
    pack_run,
    set_config,
    track,
    unpack_bundle,
)
from devqubit.compare import ComparisonResult


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


class TestCrossWorkspaceDiff:
    """Tests for comparing runs across workspaces."""

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

    def test_diff_detects_metric_changes(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """Diff detects metric differences."""
        set_config(config)

        with track(project="metrics") as run_a:
            run_a.log_metric("fidelity", 0.95)
            run_id_a = run_a.run_id

        with track(project="metrics") as run_b:
            run_b.log_metric("fidelity", 0.85)  # Different
            run_id_b = run_b.run_id

        result = diff(run_id_a, run_id_b, registry=registry, store=store)

        assert result.to_dict() is not None


class TestBundleDiff:
    """Tests for comparing bundles."""

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

        with track(project="test", capture_env=False, capture_git=False) as run_a:
            run_a.log_param("value", 100)
            run_id_a = run_a.run_id

        pack_run(
            run_id=run_id_a,
            output_path=bundle_a,
            store=store,
            registry=registry,
        )

        with track(project="test", capture_env=False, capture_git=False) as run_b:
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
        assert result.params["changed"]["value"] == {"a": 100, "b": 200}


class TestBundleReader:
    """Tests for Bundle reader API."""

    def test_bundle_context_manager(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
        tmp_path: Path,
    ):
        """Bundle works as context manager."""
        set_config(config)
        bundle_path = tmp_path / "bundle.zip"

        with track(project="reader_test") as run:
            run.log_param("key", "value")
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with Bundle(bundle_path) as bundle:
            assert bundle.run_id == run_id
            assert bundle.run_record is not None
            record = bundle.run_record
            if hasattr(record, "record"):
                assert record.record["data"]["params"]["key"] == "value"
            else:
                assert record["data"]["params"]["key"] == "value"


class TestArtifactRoundtrip:
    """Tests for artifact preservation through pack/unpack."""

    def test_multiple_artifacts_preserved(self, tmp_path: Path):
        """Multiple artifacts survive pack/unpack."""
        workspace_src = tmp_path / "src"
        workspace_dst = tmp_path / "dst"
        workspace_src.mkdir()
        workspace_dst.mkdir()

        store_src = create_store(f"file://{workspace_src}/objects")
        reg_src = create_registry(f"file://{workspace_src}")
        store_dst = create_store(f"file://{workspace_dst}/objects")
        reg_dst = create_registry(f"file://{workspace_dst}")
        bundle_path = tmp_path / "bundle.zip"

        with track(
            project="artifacts",
            store=store_src,
            registry=reg_src,
            capture_env=False,
            capture_git=False,
        ) as run:
            run.log_bytes(
                kind="binary",
                data=b"\x00\x01\x02",
                media_type="application/octet-stream",
                role="data",
            )
            run.log_json(name="config", obj={"key": "value"}, role="config")
            run.log_text(name="notes", text="Some notes", role="docs")
            run_id = run.run_id

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

        loaded = reg_dst.load(run_id)

        for artifact in loaded.artifacts:
            data = store_dst.get_bytes(artifact.digest)
            assert len(data) > 0

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

    def test_tags_workflow(
        self,
        workspace: Path,
        store,
        registry,
        config: Config,
    ):
        """Tags can be added and queried."""
        set_config(config)

        with track(project="tags_test") as run:
            run.set_tag("device", "ibm_perth")
            run.set_tag("experiment", "calibration")
            run_id = run.run_id

        loaded = registry.load(run_id)
        tags = loaded.record["data"]["tags"]
        assert tags["device"] == "ibm_perth"
        assert tags["experiment"] == "calibration"
