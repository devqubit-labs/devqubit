# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit_engine.dataframe."""

from __future__ import annotations

import pytest
from devqubit_engine.dataframe import _flatten_record, runs_to_dataframe


pd = pytest.importorskip("pandas")


class TestFlattenRecord:
    def test_standard_fields(self, run_factory):
        rec = run_factory(run_id="r-1", project="proj", status="FINISHED")
        row = _flatten_record(rec)
        assert row["run_id"] == "r-1"
        assert row["project"] == "proj"
        assert row["status"] == "FINISHED"
        assert row["backend_name"] == "test_backend"

    def test_params_prefixed(self, run_factory):
        rec = run_factory(params={"shots": 1000, "opt_level": 3})
        row = _flatten_record(rec)
        assert row["param.shots"] == 1000
        assert row["param.opt_level"] == 3

    def test_metrics_prefixed(self, run_factory):
        rec = run_factory(metrics={"fidelity": 0.95, "tvd": 0.02})
        row = _flatten_record(rec)
        assert row["metric.fidelity"] == 0.95
        assert row["metric.tvd"] == 0.02

    def test_tags_prefixed(self, run_factory):
        rec = run_factory(tags={"device": "ibm_kyoto"})
        row = _flatten_record(rec)
        assert row["tag.device"] == "ibm_kyoto"

    def test_running_has_null_ended_at(self, run_factory):
        rec = run_factory(status="RUNNING")
        row = _flatten_record(rec)
        assert row["ended_at"] is None


class TestRunsToDataframe:
    def test_basic(self, registry, run_factory):
        for i, fid in enumerate([0.9, 0.95]):
            rec = run_factory(
                run_id=f"run-{i}",
                project="vqe",
                metrics={"fidelity": fid},
            )
            registry.save(rec.record)

        df = runs_to_dataframe(registry, project="vqe")
        assert len(df) == 2
        assert "metric.fidelity" in df.columns
        assert set(df["run_id"]) == {"run-0", "run-1"}

    def test_empty(self, registry):
        df = runs_to_dataframe(registry)
        assert len(df) == 0
        assert "run_id" in df.columns
        assert "status" in df.columns

    def test_column_order(self, registry, run_factory):
        rec = run_factory(
            run_id="r-order",
            params={"z_param": 1, "a_param": 2},
            metrics={"z_metric": 0.5},
            tags={"a_tag": "x"},
        )
        registry.save(rec.record)

        df = runs_to_dataframe(registry)
        cols = list(df.columns)
        # Standard columns come first
        assert cols[0] == "run_id"
        # Dynamic columns sorted alphabetically
        dynamic = [c for c in cols if "." in c]
        assert dynamic == sorted(dynamic)

    def test_heterogeneous_columns(self, registry, run_factory):
        """Runs with different params produce NaN for missing."""
        r1 = run_factory(run_id="r-het-1", params={"shots": 1000})
        r2 = run_factory(run_id="r-het-2", params={"depth": 5})
        registry.save(r1.record)
        registry.save(r2.record)

        df = runs_to_dataframe(registry)
        assert pd.isna(df.loc[df["run_id"] == "r-het-1", "param.depth"].iloc[0])
        assert pd.isna(df.loc[df["run_id"] == "r-het-2", "param.shots"].iloc[0])

    def test_project_filter(self, registry, run_factory):
        for i, proj in enumerate(["vqe", "bell", "vqe"]):
            rec = run_factory(run_id=f"r-pf-{i}", project=proj)
            registry.save(rec.record)

        df = runs_to_dataframe(registry, project="vqe")
        assert len(df) == 2
        assert all(df["project"] == "vqe")

    def test_status_filter(self, registry, run_factory):
        for i, st in enumerate(["FINISHED", "FAILED", "FINISHED"]):
            rec = run_factory(run_id=f"r-st-{i}", status=st)
            registry.save(rec.record)

        df = runs_to_dataframe(registry, status="FAILED")
        assert len(df) == 1
        assert df.iloc[0]["status"] == "FAILED"

    def test_group_filter(self, registry, run_factory):
        r1 = run_factory(run_id="r-g1", group_id="sweep-1")
        r2 = run_factory(run_id="r-g2", group_id="sweep-2")
        r3 = run_factory(run_id="r-g3", group_id="sweep-1")
        for r in [r1, r2, r3]:
            registry.save(r.record)

        df = runs_to_dataframe(registry, group_id="sweep-1")
        assert set(df["run_id"]) == {"r-g1", "r-g3"}

    def test_limit(self, registry, run_factory):
        for i in range(10):
            rec = run_factory(run_id=f"r-lim-{i:02d}")
            registry.save(rec.record)

        df = runs_to_dataframe(registry, limit=3)
        assert len(df) == 3
