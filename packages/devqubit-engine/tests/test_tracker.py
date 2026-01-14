# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for run tracking functionality."""

from __future__ import annotations

from devqubit_engine.core.run import track


class TestBasicTracking:
    """Tests for basic parameter, metric, and tag tracking."""

    def test_params_and_metrics(self, store, registry, config):
        """Log params, metrics, and tags."""
        with track(project="test_project", config=config) as run:
            run.log_param("shots", 1000)
            run.log_metric("fidelity", 0.95)
            run.set_tag("backend", "simulator")
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["data"]["params"]["shots"] == 1000
        assert loaded.record["data"]["metrics"]["fidelity"] == 0.95
        assert loaded.record["data"]["tags"]["backend"] == "simulator"

    def test_batch_logging(self, store, registry, config):
        """Log multiple params, metrics, tags at once."""
        with track(project="batch", config=config) as run:
            run.log_params({"a": 1, "b": 2, "c": 3})
            run.log_metrics({"x": 0.5, "y": 0.6})
            run.set_tags({"env": "test", "version": "1.0"})
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["data"]["params"] == {"a": 1, "b": 2, "c": 3}
        assert loaded.record["data"]["metrics"]["x"] == 0.5
        assert loaded.record["data"]["metrics"]["y"] == 0.6
        assert loaded.record["data"]["tags"]["env"] == "test"

    def test_metric_keeps_last(self, store, registry, config):
        """Repeated metric logging keeps last value."""
        with track(project="max_metric", config=config) as run:
            run.log_metric("score", 0.5)
            run.log_metric("score", 0.9)
            run.log_metric("score", 0.7)
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["data"]["metrics"]["score"] == 0.7

    def test_metric_with_step(self, store, registry, config):
        """Metric logging with step creates time series."""
        with track(project="series", config=config) as run:
            run.log_metric("loss", 1.0, step=0)
            run.log_metric("loss", 0.5, step=1)
            run.log_metric("loss", 0.2, step=2)
            run_id = run.run_id

        loaded = registry.load(run_id)

        series = loaded.record["data"]["metric_series"]["loss"]
        assert len(series) == 3
        assert series[0]["step"] == 0
        assert series[0]["value"] == 1.0
        assert series[2]["value"] == 0.2


class TestArtifactLogging:
    """Tests for artifact logging."""

    def test_log_bytes(self, store, registry, config):
        """Log binary artifact."""
        with track(project="artifacts", config=config) as run:
            ref = run.log_bytes(
                kind="test.data",
                data=b"binary content",
                media_type="application/octet-stream",
                role="test",
            )

        assert ref.digest.startswith("sha256:")

        data = store.get_bytes(ref.digest)
        assert data == b"binary content"

    def test_log_json(self, store, registry, config):
        """Log JSON artifact."""
        with track(project="json_test", config=config) as run:
            _ = run.log_json(
                name="config",
                obj={"setting": "value", "count": 42},
                role="config",
            )
            run_id = run.run_id

        loaded = registry.load(run_id)

        artifact = next(a for a in loaded.artifacts if a.kind == "json.config")
        assert artifact.role == "config"
        assert artifact.media_type == "application/json"

    def test_log_text(self, store, registry, config):
        """Log text artifact."""
        with track(project="text_test", config=config) as run:
            ref = run.log_text(
                name="notes",
                text="Experiment notes here",
                role="documentation",
            )

        data = store.get_bytes(ref.digest)
        assert data == b"Experiment notes here"

    def test_log_file(self, store, registry, config, tmp_path):
        """Log file as artifact."""
        # Create a test file
        test_file = tmp_path / "test_input.txt"
        test_file.write_text("file content")

        with track(project="file_test", config=config) as run:
            ref = run.log_file(
                path=test_file,
                kind="input.file",
                role="input",
            )

        data = store.get_bytes(ref.digest)
        assert data == b"file content"


class TestRunLifecycle:
    """Tests for run lifecycle management."""

    def test_successful_run(self, store, registry, config):
        """Successful run has FINISHED status."""
        with track(project="success", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.status == "FINISHED"
        assert "ended_at" in loaded.record["info"]

    def test_failed_run(self, store, registry, config):
        """Failed run captures error and has FAILED status."""
        try:
            with track(project="failing", config=config) as run:
                run.log_param("before_error", True)
                run_id = run.run_id
                raise ValueError("Test error message")
        except ValueError:
            pass

        loaded = registry.load(run_id)

        assert loaded.status == "FAILED"
        assert len(loaded.record["errors"]) == 1
        assert loaded.record["errors"][0]["type"] == "ValueError"
        assert "Test error message" in loaded.record["errors"][0]["message"]

    def test_run_with_name(self, store, registry, config):
        """Run name is stored in info."""
        with track(
            project="named",
            run_name="experiment_v1",
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["info"]["run_name"] == "experiment_v1"


class TestConfigLogging:
    """Tests for compile/execute configuration logging."""

    def test_log_compile_config(self, store, registry, config):
        """Log transpiler configuration."""
        with track(project="compile", config=config) as run:
            run.log_compile(
                {
                    "optimization_level": 3,
                    "seed_transpiler": 42,
                    "routing_method": "sabre",
                }
            )
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["compile"]["optimization_level"] == 3
        assert loaded.record["compile"]["routing_method"] == "sabre"

    def test_log_execute_config(self, store, registry, config):
        """Log execution configuration."""
        with track(project="execute", config=config) as run:
            run.log_execute(
                {
                    "shots": 4000,
                    "resilience_level": 1,
                    "dynamical_decoupling": True,
                }
            )
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.record["execute"]["shots"] == 4000
        assert loaded.record["execute"]["resilience_level"] == 1


class TestFingerprints:
    """Tests for run fingerprinting."""

    def test_fingerprints_computed(self, store, registry, config):
        """Fingerprints are computed on finalization."""
        with track(project="fingerprint", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.fingerprints
        assert "run" in loaded.fingerprints
        assert loaded.run_fingerprint.startswith("sha256:")

    def test_same_content_same_fingerprint(self, store, registry, config):
        """Identical runs have identical fingerprints."""
        fingerprints = []

        for _ in range(2):
            with track(
                project="identical",
                capture_env=False,
                capture_git=False,
                config=config,
            ) as run:
                run.log_param("x", 42)
                run.log_bytes(
                    kind="data.test",
                    data=b"same",
                    media_type="text/plain",
                    role="test",
                )
                run_id = run.run_id

            loaded = registry.load(run_id)
            fingerprints.append(loaded.program_fingerprint)

        assert fingerprints[0] == fingerprints[1]


class TestEnvironmentCapture:
    """Tests for environment and provenance capture."""

    def test_environment_captured_by_default(self, store, registry, config):
        """Environment is captured by default."""
        with track(project="env", config=config) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert "environment" in loaded.record
        assert "python_version" in loaded.record["environment"]

    def test_environment_capture_disabled(self, store, registry, config):
        """Environment capture can be disabled."""
        with track(project="no_env", capture_env=False, config=config) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert "environment" not in loaded.record


class TestGroupTracking:
    """Tests for run grouping functionality."""

    def test_group_id_stored(self, store, registry, config):
        """Group ID is stored in run record."""
        with track(
            project="groups",
            group_id="sweep_20240101",
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.group_id == "sweep_20240101"

    def test_group_name_stored(self, store, registry, config):
        """Group name is stored in run record."""
        with track(
            project="groups",
            group_id="sweep_001",
            group_name="Shot Count Sweep",
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.group_name == "Shot Count Sweep"

    def test_parent_run_id_stored(self, store, registry, config):
        """Parent run ID is stored for lineage."""
        with track(
            project="lineage",
            parent_run_id="PARENT123456",
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.parent_run_id == "PARENT123456"

    def test_grouped_sweep(self, store, registry, config):
        """Multiple runs can share a group."""
        group_id = "parameter_sweep"
        run_ids = []

        for shots in [100, 1000, 10000]:
            with track(
                project="sweep",
                group_id=group_id,
                group_name="Shots Sweep",
                config=config,
            ) as run:
                run.log_param("shots", shots)
                run_ids.append(run.run_id)

        for run_id in run_ids:
            loaded = registry.load(run_id)
            assert loaded.group_id == group_id

        # List runs in group
        runs = registry.list_runs_in_group(group_id)
        assert len(runs) == 3

    def test_group_without_name(self, store, registry, config):
        """Group ID without name is valid."""
        with track(
            project="unnamed_group",
            group_id="anon_sweep",
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.group_id == "anon_sweep"
        assert loaded.group_name is None

    def test_run_lineage_chain(self, store, registry, config):
        """Chain of parent-child runs."""
        # Create parent
        with track(project="lineage", config=config) as parent:
            parent.log_param("generation", 1)
            parent_id = parent.run_id

        # Create child
        with track(
            project="lineage",
            parent_run_id=parent_id,
            config=config,
        ) as child:
            child.log_param("generation", 2)
            child_id = child.run_id

        parent_loaded = registry.load(parent_id)
        child_loaded = registry.load(child_id)

        assert parent_loaded.parent_run_id is None
        assert child_loaded.parent_run_id == parent_id
