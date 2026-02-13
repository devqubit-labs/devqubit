# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for storage backends and factory."""

from __future__ import annotations

from pathlib import Path

from devqubit_engine.storage.factory import create_registry, create_store


class TestObjectStore:
    """Tests for content-addressed object storage."""

    def test_put_get_roundtrip(self, store):
        """Store and retrieve data correctly."""
        digest = store.put_bytes(b"hello world")

        assert digest.startswith("sha256:")
        assert len(digest) == 71  # sha256: + 64 hex chars
        assert store.get_bytes(digest) == b"hello world"

    def test_content_addressed_deduplication(self, store):
        """Same content always returns same digest."""
        d1 = store.put_bytes(b"identical content")
        d2 = store.put_bytes(b"identical content")

        assert d1 == d2

    def test_exists(self, store):
        """exists() reflects actual storage state."""
        digest = store.put_bytes(b"test data")

        result = store.exists(digest)
        assert result
        result = store.exists("sha256:" + "0" * 64)
        assert not result

    def test_delete(self, store):
        """delete() removes object from store."""
        digest = store.put_bytes(b"to delete")

        result = store.delete(digest)
        assert result
        result = store.exists(digest)
        assert not result
        result = store.delete(digest)  # Second delete returns False
        assert not result

    def test_get_size(self, store):
        """get_size() returns correct byte count."""
        data = b"x" * 1000
        digest = store.put_bytes(data)

        result = store.get_size(digest)
        assert result == 1000


class TestRegistry:
    """Tests for run metadata registry."""

    def test_save_and_load(self, registry, run_factory):
        """Save and load roundtrips correctly."""
        run = run_factory(run_id="RUN0000001", project="test", params={"shots": 1000})
        registry.save(run.to_dict())

        loaded = registry.load("RUN0000001")

        assert loaded.run_id == "RUN0000001"
        assert loaded.project == "test"
        assert loaded.params == {"shots": 1000}

    def test_exists(self, registry, run_factory):
        """exists() reflects actual registry state."""
        registry.save(run_factory(run_id="EXISTS001").to_dict())

        assert registry.exists("EXISTS001")
        assert not registry.exists("NOTEXIST1")

    def test_delete(self, registry, run_factory):
        """delete() removes run from registry."""
        registry.save(run_factory(run_id="TODELETE1").to_dict())

        assert registry.delete("TODELETE1")
        assert not registry.exists("TODELETE1")

    def test_list_runs(self, registry, run_factory):
        """list_runs returns runs with pagination."""
        for i in range(5):
            record = run_factory(run_id=f"LIST{i:05d}").to_dict()
            record["created_at"] = f"2025-01-{i+1:02d}T00:00:00Z"
            registry.save(record)

        runs = registry.list_runs(limit=3)

        assert len(runs) == 3

    def test_list_runs_by_project(self, registry, run_factory):
        """list_runs filters by project."""
        registry.save(run_factory(run_id="PROJ_A_1", project="project_a").to_dict())
        registry.save(run_factory(run_id="PROJ_A_2", project="project_a").to_dict())
        registry.save(run_factory(run_id="PROJ_B_1", project="project_b").to_dict())

        runs = registry.list_runs(project="project_a")

        assert len(runs) == 2
        assert all(r["project"] == "project_a" for r in runs)

    def test_list_runs_by_name(self, registry, run_factory):
        """list_runs filters by run name."""
        registry.save(
            run_factory(
                run_id="NAME_A_1",
                project="proj",
                run_name="baseline-v1",
            ).to_dict()
        )
        registry.save(
            run_factory(
                run_id="NAME_A_2",
                project="proj",
                run_name="baseline-v2",
            ).to_dict()
        )
        registry.save(
            run_factory(
                run_id="NAME_A_3",
                project="proj",
                run_name="experiment-1",
            ).to_dict()
        )

        runs = registry.list_runs(project="proj", name="baseline-v1")

        assert len(runs) == 1
        assert runs[0]["run_id"] == "NAME_A_1"

    def test_list_runs_returns_run_name(self, registry, run_factory):
        """list_runs includes run_name in results."""
        registry.save(
            run_factory(
                run_id="RNAME001",
                run_name="my-experiment",
            ).to_dict()
        )

        runs = registry.list_runs()
        run = next(r for r in runs if r["run_id"] == "RNAME001")

        assert run.get("run_name") == "my-experiment"

    def test_list_runs_in_group_returns_run_name(self, registry, run_factory):
        """list_runs_in_group includes run_name in results."""
        registry.save(
            run_factory(
                run_id="GRPNAME1",
                run_name="grouped-run",
                group_id="test_group",
            ).to_dict()
        )

        runs = registry.list_runs_in_group("test_group")

        assert len(runs) == 1
        assert runs[0].get("run_name") == "grouped-run"


class TestRegistrySearch:
    """Tests for search_runs query functionality."""

    def test_search_by_param(self, registry, run_factory):
        """Search runs by parameter value."""
        for i, shots in enumerate([100, 1000, 1000]):
            record = run_factory(
                run_id=f"PARAM{i:04d}",
                params={"shots": shots},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("params.shots = 1000")

        assert len(results) == 2

    def test_search_by_metric_comparison(self, registry, run_factory):
        """Search runs by metric comparison."""
        for i, fidelity in enumerate([0.8, 0.9, 0.95]):
            record = run_factory(
                run_id=f"METRIC{i:04d}",
                metrics={"fidelity": fidelity},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("metric.fidelity > 0.85")

        assert len(results) == 2

    def test_search_by_tag_pattern(self, registry, run_factory):
        """Search runs by tag pattern matching."""
        devices = ["ibm_kyoto", "ibm_osaka", "google_sycamore"]
        for i, device in enumerate(devices):
            record = run_factory(
                run_id=f"TAG{i:05d}",
                tags={"device": device},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs("tags.device ~ ibm")

        assert len(results) == 2

    def test_search_with_sort(self, registry, run_factory):
        """Search with sorting by metric."""
        for i, fidelity in enumerate([0.7, 0.9, 0.8]):
            record = run_factory(
                run_id=f"SORT{i:05d}",
                metrics={"fidelity": fidelity},
            ).to_dict()
            registry.save(record)

        results = registry.search_runs(
            "metric.fidelity > 0",
            sort_by="metric.fidelity",
            descending=True,
        )

        assert results[0].metrics["fidelity"] == 0.9

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
        registry.save(run_factory(run_id="FINISHED1", status="FINISHED").to_dict())
        registry.save(run_factory(run_id="FAILED001", status="FAILED").to_dict())

        results = registry.search_runs("status = FINISHED")

        assert len(results) == 1
        assert results[0].run_id == "FINISHED1"


class TestRegistryGroups:
    """Tests for run groups functionality."""

    def test_list_groups(self, registry, run_factory):
        """List run groups with counts."""
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
        """List runs within a specific group."""
        for i in range(5):
            record = run_factory(
                run_id=f"INGROUP_{i}",
                group_id="my_group",
            ).to_dict()
            registry.save(record)

        registry.save(run_factory(run_id="OUTSIDE01").to_dict())

        runs = registry.list_runs_in_group("my_group")

        assert len(runs) == 5

    def test_list_groups_by_project(self, registry, run_factory):
        """Filter groups by project."""
        registry.save(
            run_factory(
                run_id="PROJ_A",
                project="project_a",
                group_id="group_a",
            ).to_dict()
        )
        registry.save(
            run_factory(
                run_id="PROJ_B",
                project="project_b",
                group_id="group_b",
            ).to_dict()
        )

        groups = registry.list_groups(project="project_a")

        assert len(groups) == 1
        assert groups[0]["group_id"] == "group_a"

    def test_no_groups_when_ungrouped(self, registry, run_factory):
        """No groups returned when runs have no group_id."""
        registry.save(run_factory(run_id="UNGROUPED1").to_dict())
        registry.save(run_factory(run_id="UNGROUPED2").to_dict())

        groups = registry.list_groups()

        assert len(groups) == 0


class TestStorageFactory:
    """Tests for storage factory functions."""

    def test_create_store_from_file_uri(self, tmp_path: Path):
        """create_store creates LocalStore from file:// URI."""
        store = create_store(f"file://{tmp_path}/objects")

        digest = store.put_bytes(b"factory test")

        assert store.exists(digest)

    def test_create_registry_from_file_uri(self, tmp_path: Path, run_factory):
        """create_registry creates LocalRegistry from file:// URI."""
        registry = create_registry(f"file://{tmp_path}")

        registry.save(run_factory(run_id="FACTORY01").to_dict())

        assert registry.exists("FACTORY01")
