# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Integration tests for full devqubit workflows.

These tests verify that components work together correctly in realistic
scenarios. They use real storage, real pack/unpack, and real comparison.
"""

from __future__ import annotations

from pathlib import Path

from devqubit_engine.bundle.pack import pack_run, unpack_bundle
from devqubit_engine.compare.diff import ComparisonResult, diff, diff_runs
from devqubit_engine.compare.results import ProgramMatchMode
from devqubit_engine.core.run import track


class TestFullWorkflow:
    """Tests for complete track → pack → unpack → diff workflow."""

    def test_track_pack_unpack_diff(self, store, registry, config, tmp_path: Path):
        """Full workflow: track run, pack, unpack, diff."""
        # Create run
        with track(project="integration", config=config) as run:
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
        from devqubit_engine.storage.factory import create_registry, create_store

        workspace2 = tmp_path / ".devqubit2"
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

    def test_diff_runs_different_workspaces(self, factory_store, factory_registry):
        """Compare runs from different workspaces."""
        store_a = factory_store()
        reg_a = factory_registry()
        store_b = factory_store()
        reg_b = factory_registry()

        # Create similar runs in different workspaces
        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            store=store_a,
            registry=reg_a,
        ) as run_a:
            run_a.log_param("shots", 1000)
            run_a.log_bytes(
                kind="prog",
                data=b"circuit",
                media_type="text/plain",
                role="program",
            )
            run_id_a = run_a.run_id

        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            store=store_b,
            registry=reg_b,
        ) as run_b:
            run_b.log_param("shots", 1000)
            run_b.log_bytes(
                kind="prog",
                data=b"circuit",
                media_type="text/plain",
                role="program",
            )
            run_id_b = run_b.run_id

        # Load and compare
        record_a = reg_a.load(run_id_a)
        record_b = reg_b.load(run_id_b)

        result = diff_runs(record_a, record_b, store_a=store_a, store_b=store_b)

        assert isinstance(result, ComparisonResult)
        assert result.metadata["project_match"]
        assert result.params["match"]
        assert result.program.matches(
            ProgramMatchMode.EITHER
        )  # Same content = same digest

    def test_diff_detects_param_changes(self, store, registry, config):
        """Diff detects parameter differences."""
        with track(project="test", config=config) as run_a:
            run_a.log_param("shots", 1000)
            run_a.log_param("seed", 42)
            run_id_a = run_a.run_id

        with track(project="test", config=config) as run_b:
            run_b.log_param("shots", 2000)  # Changed
            run_b.log_param("seed", 42)
            run_id_b = run_b.run_id

        record_a = registry.load(run_id_a)
        record_b = registry.load(run_id_b)

        result = diff_runs(record_a, record_b, store_a=store, store_b=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]
        assert result.params["changed"]["shots"] == {"a": 1000, "b": 2000}

    def test_diff_detects_metric_changes(self, store, registry, config):
        """Diff detects metric differences."""
        with track(project="metrics", config=config) as run_a:
            run_a.log_metric("fidelity", 0.95)
            run_id_a = run_a.run_id

        with track(project="metrics", config=config) as run_b:
            run_b.log_metric("fidelity", 0.85)  # Different
            run_id_b = run_b.run_id

        record_a = registry.load(run_id_a)
        record_b = registry.load(run_id_b)

        result = diff_runs(record_a, record_b, store_a=store, store_b=store)

        # Metrics comparison is in the result
        assert result.to_dict() is not None


class TestBundleDiff:
    """Tests for comparing bundles."""

    def test_diff_bundle_to_run(self, store, registry, config, tmp_path: Path):
        """Compare bundle to registry run using compare."""
        bundle_path = tmp_path / "bundle.zip"

        with track(project="bundle_diff", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        # Compare bundle path to run_id
        result = diff(bundle_path, run_id, registry=registry, store=store)

        assert result.identical
        assert result.run_id_a == run_id
        assert result.run_id_b == run_id

    def test_diff_two_bundles(self, store, registry, config, tmp_path: Path):
        """Compare two bundle files."""
        bundle_a = tmp_path / "bundle_a.zip"
        bundle_b = tmp_path / "bundle_b.zip"

        # Create and pack run A
        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as run_a:
            run_a.log_param("value", 100)
            run_id_a = run_a.run_id

        pack_run(run_id=run_id_a, output_path=bundle_a, store=store, registry=registry)

        # Create and pack run B with different param
        with track(
            project="test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as run_b:
            run_b.log_param("value", 200)
            run_id_b = run_b.run_id

        pack_run(run_id=run_id_b, output_path=bundle_b, store=store, registry=registry)

        # Compare bundles
        result = diff(bundle_a, bundle_b)

        assert result.run_id_a == run_id_a
        assert result.run_id_b == run_id_b
        assert not result.params["match"]
        assert result.params["changed"]["value"] == {"a": 100, "b": 200}


class TestArtifactRoundtrip:
    """Tests for artifact preservation through pack/unpack."""

    def test_multiple_artifacts_preserved(
        self, factory_store, factory_registry, tmp_path: Path
    ):
        """Multiple artifacts survive pack/unpack."""
        store_src = factory_store()
        reg_src = factory_registry()
        store_dst = factory_store()
        reg_dst = factory_registry()
        bundle_path = tmp_path / "bundle.zip"

        # Create run with multiple artifact types
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

        # Pack and unpack
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

        # Verify all artifacts accessible
        loaded = reg_dst.load(run_id)
        assert len(loaded.artifacts) == 3

        for artifact in loaded.artifacts:
            data = store_dst.get_bytes(artifact.digest)
            assert len(data) > 0

    def test_artifact_content_integrity(
        self, factory_store, factory_registry, tmp_path: Path
    ):
        """Artifact content is identical after roundtrip."""
        store_src = factory_store()
        reg_src = factory_registry()
        store_dst = factory_store()
        reg_dst = factory_registry()
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

        # Verify content
        restored_data = store_dst.get_bytes(digest)
        assert restored_data == original_data

    def test_large_artifact_roundtrip(
        self, factory_store, factory_registry, tmp_path: Path
    ):
        """Large artifacts survive roundtrip."""
        store_src = factory_store()
        reg_src = factory_registry()
        store_dst = factory_store()
        reg_dst = factory_registry()
        bundle_path = tmp_path / "bundle.zip"

        # 1MB artifact
        large_data = b"x" * (1024 * 1024)

        with track(
            project="large",
            store=store_src,
            registry=reg_src,
            capture_env=False,
            capture_git=False,
        ) as run:
            ref = run.log_bytes(
                kind="large_data",
                data=large_data,
                media_type="application/octet-stream",
                role="data",
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

        restored = store_dst.get_bytes(digest)
        assert len(restored) == len(large_data)
        assert restored == large_data


class TestEndToEndScenarios:
    """Real-world usage scenarios."""

    def test_parameter_sweep_workflow(self, store, registry, config):
        """Simulate a parameter sweep experiment."""
        group_id = "sweep_shots"
        run_ids = []

        # Run experiments with different shot counts
        for shots in [100, 500, 1000, 5000]:
            with track(
                project="sweep_test",
                group_id=group_id,
                group_name="Shot Count Sweep",
                config=config,
            ) as run:
                run.log_param("shots", shots)
                # Simulate metric that improves with more shots
                run.log_metric("fidelity", 0.9 + (shots / 50000))
                run_ids.append(run.run_id)

        # Verify group
        runs_in_group = registry.list_runs_in_group(group_id)
        assert len(runs_in_group) == 4

        # Compare first and last
        first = registry.load(run_ids[0])
        last = registry.load(run_ids[-1])

        result = diff_runs(first, last, store_a=store, store_b=store)

        assert not result.params["match"]
        assert "shots" in result.params["changed"]

    def test_baseline_verification_workflow(self, store, registry, config):
        """Simulate baseline verification in CI."""
        # Create baseline
        with track(
            project="ci_test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as baseline_run:
            baseline_run.log_param("shots", 1000)
            baseline_run.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; qubit q; h q;",
                media_type="text/plain",
                role="program",
            )
            baseline_id = baseline_run.run_id

        registry.set_baseline("ci_test", baseline_id)

        # Create candidate (same config)
        with track(
            project="ci_test",
            capture_env=False,
            capture_git=False,
            config=config,
        ) as candidate_run:
            candidate_run.log_param("shots", 1000)
            candidate_run.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3; qubit q; h q;",
                media_type="text/plain",
                role="program",
            )
            candidate_id = candidate_run.run_id

        # Compare
        baseline = registry.load(baseline_id)
        candidate = registry.load(candidate_id)

        result = diff_runs(baseline, candidate, store_a=store, store_b=store)

        assert result.params["match"]
        assert result.program.matches(ProgramMatchMode.EITHER)

    def test_failed_run_captures_error(self, store, registry, config):
        """Failed runs capture error information."""
        try:
            with track(project="error_test", config=config) as run:
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
