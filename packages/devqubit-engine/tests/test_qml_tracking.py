# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""End-to-end tests for QML metric tracking backend."""

from __future__ import annotations

import math

import devqubit_engine.tracking.run as run_mod
import pytest
from devqubit_engine.tracking.history import (
    iter_metric_points,
    metric_history_long,
    to_dataframe,
)
from devqubit_engine.tracking.record import RunRecord


track = run_mod.track


class TestSummaryUpdate:
    """Scalar summary must always reflect the latest step-logged value."""

    def test_step_metric_updates_summary(self, store, registry, qml_config):
        """log_metric(step=...) writes to both metric_series AND metrics."""
        with track(
            project="vqe",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            run.log_metric("loss", 1.0, step=0)
            run.log_metric("loss", 0.3, step=1)
            run.log_metric("loss", 0.05, step=2)
            rid = run.run_id

        loaded = registry.load(rid)

        # Summary reflects the last logged value
        assert loaded.metrics["loss"] == pytest.approx(0.05)

        # Full history is intact
        assert len(loaded.metric_series["loss"]) == 3

    def test_summary_and_scalar_coexist(self, store, registry, qml_config):
        """Scalar-only and step-based metrics live side by side."""
        with track(
            project="vqe",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            run.log_metric("final_fidelity", 0.99)  # scalar-only
            run.log_metric("train/loss", 1.0, step=0)  # series
            run.log_metric("train/loss", 0.1, step=1)
            rid = run.run_id

        loaded = registry.load(rid)

        assert loaded.metrics["final_fidelity"] == pytest.approx(0.99)
        assert loaded.metrics["train/loss"] == pytest.approx(0.1)
        assert "train/loss" in loaded.metric_series
        assert "final_fidelity" not in loaded.metric_series


class TestFlush:
    """flush() must persist summary + metric_points mid-run."""

    def test_manual_flush_persists_data(self, store, registry, qml_config):
        """After flush, reloading the run shows partial data."""
        with track(
            project="flush",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            for i in range(100):
                run.log_metric("loss", 1.0 / (i + 1), step=i)

            run.flush()

            # Read back from the DB mid-run
            mid_run = registry.load(run.run_id)
            assert mid_run.status == "RUNNING"
            assert mid_run.metrics["loss"] == pytest.approx(1.0 / 100)

            # metric_points table also has data
            pts = list(registry.iter_metric_points(run.run_id, "loss"))
            assert len(pts) == 100

    def test_flush_on_failed_run(self, store, registry, qml_config):
        """Data flushed before a crash survives in the registry."""
        rid = ""  # assigned inside context; used after pytest.raises catches

        with pytest.raises(RuntimeError):
            with track(
                project="crash",
                store=store,
                registry=registry,
                config=qml_config,
                capture_env=False,
            ) as run:
                rid = run.run_id
                for i in range(200):
                    run.log_metric("energy", -0.5 * i, step=i)
                run.flush()
                raise RuntimeError("simulated crash")

        loaded = registry.load(rid)
        assert loaded.status == "FAILED"
        pts = list(registry.iter_metric_points(rid, "energy"))
        assert len(pts) == 200

    def test_background_flush(self, store, registry, qml_config, monkeypatch):
        """Background worker persists data without user intervention."""
        import time

        # Speed up for testing: 0.5s instead of default 10s
        monkeypatch.setattr(run_mod, "_FLUSH_INTERVAL_SECONDS", 0.5)

        with track(
            project="auto",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            rid = run.run_id
            for i in range(500):
                run.log_metric("loss", 1.0 / (i + 1), step=i)

            # Wait for background worker to fire
            time.sleep(1.0)

            # Data should be in the DB — no manual flush called
            pts = list(registry.iter_metric_points(rid, "loss"))
            assert len(pts) == 500

        # After finalize, everything is still there
        pts_all = list(registry.iter_metric_points(rid, "loss"))
        assert len(pts_all) == 500


class TestHistoryReadPath:
    """History API returns correct data for plotting."""

    def _create_runs(self, store, registry, config, n_runs=3, n_steps=100):
        """Helper: create n_runs with n_steps each."""
        run_ids = []
        for r in range(n_runs):
            with track(
                project="qnn",
                store=store,
                registry=registry,
                config=config,
                capture_env=False,
                run_name=f"run_{r}",
            ) as run:
                for i in range(n_steps):
                    run.log_metric("train/loss", 1.0 / (i + 1 + r), step=i)
                    run.log_metric("val/acc", 0.5 + 0.005 * i, step=i)
                run_ids.append(run.run_id)
        return run_ids

    def test_iter_metric_points_full(self, store, registry, qml_config):
        """iter_metric_points returns all points in step order."""
        ids = self._create_runs(
            store,
            registry,
            qml_config,
            n_runs=1,
            n_steps=50,
        )

        pts = list(iter_metric_points(registry, ids[0], "train/loss"))

        assert len(pts) == 50
        steps = [p["step"] for p in pts]
        assert steps == list(range(50))
        assert all("timestamp" in p for p in pts)

    def test_iter_metric_points_range(self, store, registry, qml_config):
        """Step range filtering works."""
        ids = self._create_runs(
            store,
            registry,
            qml_config,
            n_runs=1,
            n_steps=100,
        )

        pts = list(
            iter_metric_points(
                registry,
                ids[0],
                "train/loss",
                start_step=20,
                end_step=30,
            )
        )

        assert len(pts) == 11
        assert pts[0]["step"] == 20
        assert pts[-1]["step"] == 30

    def test_long_format_multi_run(self, store, registry, qml_config):
        """metric_history_long returns overlay-ready long-format rows."""
        self._create_runs(
            store,
            registry,
            qml_config,
            n_runs=3,
            n_steps=50,
        )

        rows = metric_history_long(
            registry,
            project="qnn",
            keys=["train/loss"],
        )

        assert len(rows) == 3 * 50
        assert all(r["key"] == "train/loss" for r in rows)
        run_ids_seen = {r["run_id"] for r in rows}
        assert len(run_ids_seen) == 3

    def test_long_format_downsampling(self, store, registry, qml_config):
        """max_points caps total returned rows."""
        self._create_runs(
            store,
            registry,
            qml_config,
            n_runs=2,
            n_steps=200,
        )

        rows = metric_history_long(
            registry,
            project="qnn",
            keys=["train/loss"],
            max_points=50,
        )

        assert len(rows) <= 50

    def test_long_format_to_dataframe(self, store, registry, qml_config):
        """to_dataframe converts rows to pandas successfully."""
        pytest.importorskip("pandas")
        self._create_runs(
            store,
            registry,
            qml_config,
            n_runs=2,
            n_steps=30,
        )

        rows = metric_history_long(
            registry,
            project="qnn",
            keys=["val/acc"],
        )
        df = to_dataframe(rows)

        assert len(df) == 2 * 30
        assert set(df.columns) >= {"run_id", "key", "step", "value"}

    def test_long_format_raises_without_selector(self, store, registry, qml_config):
        """metric_history_long raises ValueError without run_ids/project."""
        with pytest.raises(ValueError, match="Provide run_ids"):
            metric_history_long(registry)


class TestRunRecordMetricSeries:
    """RunRecord.metric_series provides clean access to history."""

    def test_metric_series_present(self):
        """Property returns series when data exists."""
        rec = RunRecord(
            record={
                "run_id": "abc",
                "created_at": "2026-01-01T00:00:00Z",
                "data": {
                    "params": {},
                    "metrics": {"loss": 0.1},
                    "tags": {},
                    "metric_series": {
                        "loss": [
                            {
                                "value": 1.0,
                                "step": 0,
                                "timestamp": "2026-01-01T00:00:00Z",
                            },
                            {
                                "value": 0.1,
                                "step": 1,
                                "timestamp": "2026-01-01T00:00:01Z",
                            },
                        ],
                    },
                },
            }
        )

        assert len(rec.metric_series["loss"]) == 2
        assert rec.metric_series["loss"][0]["value"] == 1.0

    def test_metric_series_absent(self):
        """Property returns {} on records without metric_series."""
        rec = RunRecord(
            record={
                "run_id": "old",
                "created_at": "2025-01-01T00:00:00Z",
                "data": {"params": {}, "metrics": {}, "tags": {}},
            }
        )

        assert rec.metric_series == {}

    def test_metric_series_after_finalize(self):
        """Finalized snapshot still has metric_series."""
        rec = RunRecord(
            record={
                "run_id": "snap",
                "created_at": "2026-01-01T00:00:00Z",
                "data": {
                    "params": {},
                    "metrics": {},
                    "tags": {},
                    "metric_series": {
                        "x": [{"value": 42, "step": 0, "timestamp": "t"}]
                    },
                },
            }
        )
        rec.mark_finalized()

        assert rec.metric_series["x"][0]["value"] == 42


class TestMetricPointsTable:
    """LocalRegistry.metric_points table works for large-ish series."""

    def test_large_series_round_trip(self, store, registry, qml_config):
        """10k points per metric survive write → flush → read cycle."""
        n = 10_000
        with track(
            project="scale",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            rid = run.run_id
            for i in range(n):
                run.log_metric("energy", math.sin(i * 0.01), step=i)
            # Single flush for all 10k
            run.flush()

        pts = list(registry.iter_metric_points(rid, "energy"))
        assert len(pts) == n
        assert pts[0]["step"] == 0
        assert pts[-1]["step"] == n - 1

    def test_multiple_keys(self, store, registry, qml_config):
        """Different metric keys are stored and queried independently."""
        with track(
            project="multi",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            rid = run.run_id
            for i in range(50):
                run.log_metric("loss", 1.0 / (i + 1), step=i)
                run.log_metric("fidelity", 0.5 + i * 0.01, step=i)

        loss_pts = list(registry.iter_metric_points(rid, "loss"))
        fid_pts = list(registry.iter_metric_points(rid, "fidelity"))

        assert len(loss_pts) == 50
        assert len(fid_pts) == 50
        assert loss_pts[0]["value"] == pytest.approx(1.0)
        assert fid_pts[-1]["value"] == pytest.approx(0.99)

    def test_delete_run_cleans_metric_points(self, store, registry, qml_config):
        """Deleting a run also removes its metric_points rows."""
        with track(
            project="del",
            store=store,
            registry=registry,
            config=qml_config,
            capture_env=False,
        ) as run:
            rid = run.run_id
            for i in range(20):
                run.log_metric("x", float(i), step=i)

        assert len(list(registry.iter_metric_points(rid, "x"))) == 20

        registry.delete(rid)

        # metric_points rows are gone
        assert len(list(registry.iter_metric_points(rid, "x"))) == 0
