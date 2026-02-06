# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for run tracking functionality."""

from __future__ import annotations

from devqubit_engine.tracking.run import track


class TestRunLifecycle:
    """Tests for complete run lifecycle."""

    def test_successful_run(self, store, registry, config):
        """Complete successful run with all data types."""
        with track(
            project="lifecycle", store=store, registry=registry, config=config
        ) as run:
            run.log_param("shots", 1000)
            run.log_metric("fidelity", 0.95)
            run.set_tag("backend", "simulator")
            run.log_bytes(
                kind="circuit.qasm",
                data=b"OPENQASM 3.0;",
                media_type="text/plain",
                role="program",
            )
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.status == "FINISHED"
        assert loaded.ended_at is not None
        assert loaded.params["shots"] == 1000
        assert loaded.metrics["fidelity"] == 0.95
        assert loaded.tags["backend"] == "simulator"
        assert len(loaded.artifacts) >= 1
        assert loaded.run_fingerprint.startswith("sha256:")

    def test_failed_run_captures_error(self, store, registry, config):
        """Failed run captures error and has FAILED status."""
        try:
            with track(
                project="failing", store=store, registry=registry, config=config
            ) as run:
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
        """Run name is stored correctly."""
        with track(
            project="named",
            run_name="experiment_v1",
            store=store,
            registry=registry,
            config=config,
        ) as run:
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.run_name == "experiment_v1"


class TestLogging:
    """Tests for parameter, metric, artifact, and config logging."""

    def test_batch_logging(self, store, registry, config):
        """Log multiple params/metrics/tags at once."""
        with track(
            project="batch", store=store, registry=registry, config=config
        ) as run:
            run.log_params({"a": 1, "b": 2})
            run.log_metrics({"x": 0.5, "y": 0.6})
            run.set_tags({"env": "test"})
            run_id = run.run_id

        loaded = registry.load(run_id)

        assert loaded.params == {"a": 1, "b": 2}
        assert loaded.metrics["x"] == 0.5
        assert loaded.tags["env"] == "test"

    def test_metric_time_series(self, store, registry, config):
        """Metric with step creates time series."""
        with track(
            project="series", store=store, registry=registry, config=config
        ) as run:
            run.log_metric("loss", 1.0, step=0)
            run.log_metric("loss", 0.5, step=1)
            run.log_metric("loss", 0.2, step=2)
            run_id = run.run_id

        loaded = registry.load(run_id)

        series = loaded.record["data"]["metric_series"]["loss"]
        assert len(series) == 3
        assert series[0]["value"] == 1.0
        assert series[2]["value"] == 0.2


class TestArtifactLogging:
    """Tests for artifact logging methods."""

    def test_log_bytes(self, store, registry, config):
        """Log binary artifact."""
        with track(
            project="artifacts", store=store, registry=registry, config=config
        ) as run:
            ref = run.log_bytes(
                kind="test.data",
                data=b"binary content",
                media_type="application/octet-stream",
                role="test",
            )

        assert ref.digest.startswith("sha256:")
        assert store.get_bytes(ref.digest) == b"binary content"

    def test_log_json(self, store, registry, config):
        """Log JSON artifact."""
        with track(
            project="json", store=store, registry=registry, config=config
        ) as run:
            run.log_json(
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
        with track(
            project="text", store=store, registry=registry, config=config
        ) as run:
            ref = run.log_text(
                name="notes",
                text="Experiment notes",
                role="documentation",
            )

        assert store.get_bytes(ref.digest) == b"Experiment notes"

    def test_log_file(self, store, registry, config, tmp_path):
        """Log file as artifact."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("file content")

        with track(
            project="file", store=store, registry=registry, config=config
        ) as run:
            ref = run.log_file(path=test_file, kind="input.file", role="input")

        assert store.get_bytes(ref.digest) == b"file content"


class TestGroupTracking:
    """Tests for run grouping and lineage."""

    def test_grouped_sweep(self, store, registry, config):
        """Multiple runs share a group."""
        group_id = "parameter_sweep"
        run_ids = []

        for shots in [100, 1000, 10000]:
            with track(
                project="sweep",
                group_id=group_id,
                group_name="Shots Sweep",
                store=store,
                registry=registry,
                config=config,
            ) as run:
                run.log_param("shots", shots)
                run_ids.append(run.run_id)

        # All runs have same group
        for run_id in run_ids:
            loaded = registry.load(run_id)
            assert loaded.group_id == group_id
            assert loaded.group_name == "Shots Sweep"

        # Registry can list runs in group
        runs = registry.list_runs_in_group(group_id)
        assert len(runs) == 3

    def test_run_lineage(self, store, registry, config):
        """Parent-child run relationship."""
        with track(
            project="lineage", store=store, registry=registry, config=config
        ) as parent:
            parent.log_param("generation", 1)
            parent_id = parent.run_id

        with track(
            project="lineage",
            parent_run_id=parent_id,
            store=store,
            registry=registry,
            config=config,
        ) as child:
            child.log_param("generation", 2)
            child_id = child.run_id

        parent_loaded = registry.load(parent_id)
        child_loaded = registry.load(child_id)

        assert parent_loaded.parent_run_id is None
        assert child_loaded.parent_run_id == parent_id


class TestFingerprints:
    """Tests for run fingerprinting."""

    def test_same_content_same_fingerprint(self, store, registry, config):
        """Identical runs have identical fingerprints."""
        fingerprints = []

        for _ in range(2):
            with track(
                project="identical",
                capture_env=False,
                capture_git=False,
                store=store,
                registry=registry,
                config=config,
            ) as run:
                run.log_param("x", 42)
                run.log_bytes(
                    kind="data.test",
                    data=b"same content",
                    media_type="text/plain",
                    role="test",
                )
                run_id = run.run_id

            loaded = registry.load(run_id)
            fingerprints.append(loaded.program_fingerprint)

        assert fingerprints[0] == fingerprints[1]
