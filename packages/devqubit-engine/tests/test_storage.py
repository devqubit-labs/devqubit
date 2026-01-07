# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for storage implementations."""

from __future__ import annotations

from pathlib import Path

from devqubit_engine.storage.factory import create_registry, create_store


class TestObjectStore:
    """Tests for LocalStore object storage."""

    def test_roundtrip(self, store):
        """Put and get returns same data."""
        digest = store.put_bytes(b"hello world")
        assert digest.startswith("sha256:")
        data = store.get_bytes(digest)
        assert data == b"hello world"

    def test_idempotent(self, store):
        """Storing same content returns same digest."""
        d1 = store.put_bytes(b"same content")
        d2 = store.put_bytes(b"same content")
        assert d1 == d2

    def test_exists(self, store):
        """exists() returns correct status."""
        digest = store.put_bytes(b"test")
        assert store.exists(digest)
        assert not store.exists("sha256:" + "0" * 64)

    def test_delete(self, store):
        """delete() removes object."""
        digest = store.put_bytes(b"to delete")
        assert store.delete(digest)
        assert not store.exists(digest)


class TestRegistry:
    """Tests for LocalRegistry run storage."""

    def test_save_and_load(self, registry, run_factory):
        """Save and load roundtrips correctly."""
        run = run_factory(run_id="RUN0000001")
        registry.save(run.to_dict())
        loaded = registry.load("RUN0000001")
        assert loaded.run_id == "RUN0000001"

    def test_list_runs(self, registry, run_factory):
        """list_runs returns runs in order."""
        for i in range(5):
            run = run_factory(run_id=f"RUN00000{i:02d}")
            record = run.to_dict()
            record["created_at"] = f"2025-01-{i+1:02d}T00:00:00Z"
            registry.save(record)

        runs = registry.list_runs(limit=3)
        assert len(runs) == 3

    def test_exists(self, registry, run_factory):
        """exists() returns correct status."""
        registry.save(run_factory(run_id="EXISTS1234").to_dict())
        assert registry.exists("EXISTS1234")
        assert not registry.exists("NOTEXIST12")

    def test_delete(self, registry, run_factory):
        """delete() removes run."""
        registry.save(run_factory(run_id="TODELETE12").to_dict())
        assert registry.delete("TODELETE12")
        assert not registry.exists("TODELETE12")


class TestStorageFactory:
    """Tests for storage factory functions."""

    def test_create_store_local(self, tmp_path: Path):
        """create_store creates local store from file:// URI."""
        store = create_store(f"file://{tmp_path}/objects")
        digest = store.put_bytes(b"test")
        assert store.exists(digest)

    def test_create_registry_local(self, tmp_path: Path, run_factory):
        """create_registry creates local registry from file:// URI."""
        registry = create_registry(f"file://{tmp_path}")
        registry.save(run_factory(run_id="FACTORY123").to_dict())
        assert registry.exists("FACTORY123")


class TestRegistrySearch:
    """Tests for search_runs functionality."""

    def test_search_by_params(self, registry, run_factory):
        """Search runs by parameter value."""
        for i, shots in enumerate([100, 1000, 1000]):
            record = run_factory(
                run_id=f"SEARCH{i:04d}",
                params={"shots": shots},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("params.shots = 1000")
        assert len(results) == 2

    def test_search_by_metric(self, registry, run_factory):
        """Search runs by metric comparison."""
        for i, fidelity in enumerate([0.8, 0.9, 0.95]):
            record = run_factory(
                run_id=f"METRIC{i:04d}",
                metrics={"fidelity": fidelity},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("metric.fidelity > 0.85")
        assert len(results) == 2

    def test_search_by_tag(self, registry, run_factory):
        """Search runs by tag value."""
        for i, device in enumerate(["ibm_kyoto", "ibm_osaka", "google_sycamore"]):
            record = run_factory(
                run_id=f"TAG{i:04d}",
                tags={"device": device},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("tags.device ~ ibm")
        assert len(results) == 2

    def test_search_with_sort(self, registry, run_factory):
        """Search with sorting."""
        for i, fidelity in enumerate([0.7, 0.9, 0.8]):
            record = run_factory(
                run_id=f"SORT{i:04d}",
                metrics={"fidelity": fidelity},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs(
            "metric.fidelity > 0",
            sort_by="metric.fidelity",
            descending=True,
        )

        assert results[0].record["data"]["metrics"]["fidelity"] == 0.9

    def test_search_with_limit(self, registry, run_factory):
        """Search respects limit."""
        for i in range(10):
            record = run_factory(
                run_id=f"LIMIT{i:04d}",
                params={"x": 1},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("params.x = 1", limit=5)
        assert len(results) == 5

    def test_search_multiple_conditions(self, registry, run_factory):
        """Search with AND conditions."""
        test_cases = [(1000, 0.9), (1000, 0.8), (2000, 0.9)]
        for shots, fidelity in test_cases:
            record = run_factory(
                run_id=f"MULTI{shots}{int(fidelity*100)}",
                params={"shots": shots},
                metrics={"fidelity": fidelity},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("params.shots = 1000 and metric.fidelity > 0.85")
        assert len(results) == 1

    def test_search_by_status(self, registry, run_factory):
        """Search by run status."""
        record = run_factory(run_id="FINISHED01", status="FINISHED").to_dict()
        registry.save(record)

        record = run_factory(run_id="FAILED0001", status="FAILED").to_dict()
        registry.save(record)

        results = registry.search_runs("status = FINISHED")
        assert len(results) == 1
        assert results[0].run_id == "FINISHED01"


class TestRegistryGroups:
    """Tests for run groups functionality."""

    def test_list_groups(self, registry, run_factory):
        """List run groups."""
        for i in range(3):
            record = run_factory(
                run_id=f"GROUP_A_{i}",
                group_id="sweep_001",
                group_name="Parameter Sweep",
            ).to_dict()
            registry.save(record)

        for i in range(2):
            record = run_factory(
                run_id=f"GROUP_B_{i}",
                group_id="sweep_002",
            ).to_dict()
            registry.save(record)

        groups = registry.list_groups()
        assert len(groups) == 2

        group_001 = next(g for g in groups if g["group_id"] == "sweep_001")
        assert group_001["run_count"] == 3
        assert group_001["group_name"] == "Parameter Sweep"

    def test_list_runs_in_group(self, registry, run_factory):
        """List runs within a group."""
        for i in range(5):
            record = run_factory(
                run_id=f"INGROUP_{i}",
                group_id="my_group",
            ).to_dict()
            registry.save(record)

        registry.save(run_factory(run_id="OUTSIDE").to_dict())

        runs = registry.list_runs_in_group("my_group")
        assert len(runs) == 5

    def test_list_groups_by_project(self, registry, run_factory):
        """Filter groups by project."""
        record = run_factory(
            run_id="PROJ_A",
            project="project_a",
            group_id="group_a",
        ).to_dict()
        registry.save(record)

        record = run_factory(
            run_id="PROJ_B",
            project="project_b",
            group_id="group_b",
        ).to_dict()
        registry.save(record)

        groups = registry.list_groups(project="project_a")
        assert len(groups) == 1
        assert groups[0]["group_id"] == "group_a"

    def test_empty_groups(self, registry, run_factory):
        """No groups when no grouped runs."""
        registry.save(run_factory(run_id="NOGR001").to_dict())
        registry.save(run_factory(run_id="NOGR002").to_dict())

        groups = registry.list_groups()
        assert len(groups) == 0

    def test_list_runs_with_group_filter(self, registry, run_factory):
        """list_runs can filter by group_id."""
        for i in range(3):
            record = run_factory(
                run_id=f"GRP1_{i}",
                group_id="group_1",
            ).to_dict()
            registry.save(record)

        for i in range(2):
            record = run_factory(
                run_id=f"GRP2_{i}",
                group_id="group_2",
            ).to_dict()
            registry.save(record)

        runs = registry.list_runs(group_id="group_1")
        assert len(runs) == 3
