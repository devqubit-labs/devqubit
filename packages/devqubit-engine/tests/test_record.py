# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for RunRecord — properties, finalization, snapshot immutability, threading."""

from __future__ import annotations

import threading

from devqubit_engine.tracking.record import RunRecord


class TestRunRecordProperties:
    """Properties return correct values from the underlying dict."""

    def test_immutable_fields_cached_at_init(self, run_factory):
        """run_id and created_at are cached at construction, never change."""
        rec = run_factory(run_id="CACHED001")

        assert rec.run_id == "CACHED001"
        assert rec.created_at == "2024-01-01T00:00:00Z"

        # Even if someone mutates the raw dict (shouldn't, but test robustness)
        rec.record["run_id"] = "MUTATED"
        assert rec.run_id == "CACHED001"

    def test_nested_project_name(self, run_factory):
        rec = run_factory(project="vqe_bench")
        assert rec.project == "vqe_bench"

    def test_adapter(self, run_factory):
        rec = run_factory(adapter="devqubit-qiskit")
        assert rec.adapter == "devqubit-qiskit"

    def test_status(self, run_factory):
        rec = run_factory(status="FAILED")
        assert rec.status == "FAILED"

    def test_default_status_is_running(self):
        """If info dict has no status key, default is RUNNING."""
        rec = RunRecord(record={"run_id": "X", "created_at": "", "info": {}})
        assert rec.status == "RUNNING"

    def test_backend_name(self, run_factory):
        rec = run_factory(backend_name="ibm_brisbane")
        assert rec.backend_name == "ibm_brisbane"

    def test_params_metrics_tags(self, run_factory):
        rec = run_factory(
            params={"shots": 1024},
            metrics={"fidelity": 0.99},
            tags={"device": "sim"},
        )
        assert rec.params == {"shots": 1024}
        assert rec.metrics == {"fidelity": 0.99}
        assert rec.tags == {"device": "sim"}

    def test_run_name_and_alias(self, run_factory):
        rec = run_factory(run_name="baseline-v3")
        assert rec.run_name == "baseline-v3"
        assert rec.name == "baseline-v3"

    def test_group_fields(self, run_factory):
        rec = run_factory(group_id="sweep-1", group_name="Shots Sweep")
        assert rec.group_id == "sweep-1"
        assert rec.group_name == "Shots Sweep"

    def test_parent_run_id(self, run_factory):
        rec = run_factory(parent_run_id="PARENT001")
        assert rec.parent_run_id == "PARENT001"

    def test_fingerprints_dict(self, run_factory):
        rec = run_factory()
        fps = rec.fingerprints
        assert "run" in fps
        assert fps["run"].startswith("sha256:")

    def test_run_fingerprint_shortcut(self, run_factory):
        rec = run_factory()
        assert rec.run_fingerprint == rec.fingerprints.get("run")

    def test_program_fingerprint_shortcut(self, run_factory):
        rec = run_factory()
        assert rec.program_fingerprint == rec.fingerprints.get("program")

    def test_ended_at_none_when_running(self, run_factory):
        rec = run_factory(status="RUNNING")
        assert rec.ended_at is None

    def test_ended_at_when_set(self, run_factory):
        rec = run_factory(ended_at="2024-01-01T01:00:00Z")
        assert rec.ended_at == "2024-01-01T01:00:00Z"


class TestRunRecordEdgeCases:
    """Handles missing, empty, or malformed record fields gracefully."""

    def test_minimal_record(self):
        """Just run_id and created_at — everything else empty/default."""
        rec = RunRecord(
            record={"run_id": "MIN01", "created_at": "2024-01-01T00:00:00Z"}
        )

        assert rec.run_id == "MIN01"
        assert rec.project == ""
        assert rec.adapter == ""
        assert rec.status == "RUNNING"
        assert rec.params == {}
        assert rec.metrics == {}
        assert rec.tags == {}
        assert rec.fingerprints == {}
        assert rec.backend_name is None
        assert rec.group_id is None

    def test_project_as_string(self):
        """project field as plain string instead of dict."""
        rec = RunRecord(
            record={
                "run_id": "P1",
                "created_at": "",
                "project": "flat_str",
            }
        )
        assert rec.project == "flat_str"

    def test_project_as_empty_string(self):
        rec = RunRecord(
            record={
                "run_id": "P2",
                "created_at": "",
                "project": "",
            }
        )
        assert rec.project == ""

    def test_info_not_a_dict(self):
        """If info is something weird, defaults are returned."""
        rec = RunRecord(
            record={
                "run_id": "X",
                "created_at": "",
                "info": "broken",
            }
        )
        assert rec.status == "RUNNING"
        assert rec.run_name is None
        assert rec.ended_at is None


class TestMarkFinalized:
    """mark_finalized freezes a snapshot for lock-free reads."""

    def test_properties_identical_before_and_after(self, run_factory):
        """All properties return same values after finalization."""
        rec = run_factory(
            run_id="FIN001",
            project="fin_test",
            adapter="qiskit",
            status="FINISHED",
            run_name="v1",
            backend_name="ibm_kyoto",
            params={"shots": 1000},
            metrics={"fidelity": 0.9},
            tags={"env": "ci"},
            group_id="g1",
            group_name="Sweep",
            parent_run_id="P001",
            ended_at="2024-01-01T01:00:00Z",
        )

        # Capture before values
        before = {
            "project": rec.project,
            "adapter": rec.adapter,
            "status": rec.status,
            "run_name": rec.run_name,
            "backend_name": rec.backend_name,
            "params": rec.params,
            "metrics": rec.metrics,
            "tags": rec.tags,
            "group_id": rec.group_id,
            "group_name": rec.group_name,
            "parent_run_id": rec.parent_run_id,
            "ended_at": rec.ended_at,
            "fingerprints": rec.fingerprints,
        }

        rec.mark_finalized()

        assert rec.project == before["project"]
        assert rec.adapter == before["adapter"]
        assert rec.status == before["status"]
        assert rec.run_name == before["run_name"]
        assert rec.backend_name == before["backend_name"]
        assert rec.params == before["params"]
        assert rec.metrics == before["metrics"]
        assert rec.tags == before["tags"]
        assert rec.group_id == before["group_id"]
        assert rec.group_name == before["group_name"]
        assert rec.parent_run_id == before["parent_run_id"]
        assert rec.ended_at == before["ended_at"]
        assert rec.fingerprints == before["fingerprints"]

    def test_snapshot_is_immutable_to_caller(self, run_factory):
        """Mutating dicts returned by properties doesn't affect the record."""
        rec = run_factory(params={"x": 1}, metrics={"y": 2.0}, tags={"z": "v"})
        rec.mark_finalized()

        # Mutate returned dicts
        rec.params["x"] = 999
        rec.metrics["y"] = 999.0
        rec.tags["z"] = "mutated"
        rec.fingerprints["run"] = "corrupted"

        # Original snapshot untouched
        assert rec.params["x"] == 1
        assert rec.metrics["y"] == 2.0
        assert rec.tags["z"] == "v"
        assert rec.fingerprints["run"].startswith("sha256:")

    def test_to_dict_works_after_finalization(self, run_factory):
        """to_dict still reads from the underlying record, not snapshot."""
        rec = run_factory(run_id="DICT01", params={"a": 1})
        rec.mark_finalized()

        d = rec.to_dict()
        assert d["run_id"] == "DICT01"
        assert d["data"]["params"]["a"] == 1

    def test_cached_run_id_and_created_at_bypass_lock(self, run_factory):
        """run_id and created_at never touch the lock, even before finalization."""
        rec = run_factory(run_id="LOCK01")
        # These always use cached values
        assert rec.run_id == "LOCK01"
        assert rec.created_at == "2024-01-01T00:00:00Z"

        rec.mark_finalized()
        assert rec.run_id == "LOCK01"
        assert rec.created_at == "2024-01-01T00:00:00Z"


class TestRunRecordThreadSafety:
    """Concurrent reads must be consistent, especially after finalization."""

    def test_concurrent_reads_after_finalization(self, run_factory):
        """Many threads reading finalized record see consistent data."""
        rec = run_factory(
            run_id="THR001",
            project="threaded",
            params={"shots": 8192},
            metrics={"fidelity": 0.95},
            tags={"env": "prod"},
        )
        rec.mark_finalized()

        errors = []

        def _read():
            try:
                for _ in range(100):
                    assert rec.run_id == "THR001"
                    assert rec.project == "threaded"
                    assert rec.params == {"shots": 8192}
                    assert rec.metrics == {"fidelity": 0.95}
                    assert rec.tags == {"env": "prod"}
            except AssertionError as e:
                errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety violation: {errors}"

    def test_concurrent_reads_during_active_run(self, run_factory):
        """Reads during an active (non-finalized) run don't crash."""
        rec = run_factory(run_id="ACTIVE01", params={"x": 42})
        errors = []

        def _read():
            try:
                for _ in range(50):
                    _ = rec.params
                    _ = rec.status
                    _ = rec.fingerprints
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
