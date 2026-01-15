# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for garbage collection and storage hygiene."""

from __future__ import annotations

from devqubit_engine.storage.gc import (
    check_workspace_health,
    gc_run,
    prune_runs,
)
from devqubit_engine.tracking.run import track


class TestGarbageCollection:
    """Tests for object garbage collection."""

    def test_gc_no_orphans(self, store, registry, config):
        """GC with no orphans reports nothing to delete."""
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"referenced content",
                media_type="text/plain",
                role="test",
            )

        stats = gc_run(store, registry, dry_run=True)

        assert stats.referenced_objects >= 1
        assert stats.unreferenced_objects == 0
        assert stats.bytes_reclaimable == 0

    def test_gc_finds_orphans(self, store, registry, config):
        """GC finds orphaned objects not referenced by any run."""
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"referenced",
                media_type="text/plain",
                role="test",
            )

        # Add orphan directly to store (no run references it)
        orphan_digest = store.put_bytes(b"orphaned content")

        stats = gc_run(store, registry, dry_run=True)

        assert stats.unreferenced_objects >= 1
        assert stats.bytes_reclaimable > 0
        # Dry run doesn't delete
        assert store.exists(orphan_digest)

    def test_gc_deletes_orphans(self, store, registry, config):
        """GC deletes orphaned objects when not dry run."""
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"keep me",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        orphan_digest = store.put_bytes(b"delete me")

        stats = gc_run(store, registry, dry_run=False)

        assert stats.objects_deleted >= 1
        assert stats.bytes_reclaimed > 0
        assert not store.exists(orphan_digest)

        # Referenced object still exists
        loaded = registry.load(run_id)
        assert store.exists(loaded.artifacts[0].digest)


class TestPruneRuns:
    """Tests for pruning old or failed runs."""

    def test_prune_by_status(self, store, registry, config):
        """Prune runs by status."""
        # Create failed runs
        for _ in range(3):
            try:
                with track(project="prune_test", config=config) as run:
                    raise ValueError("intentional failure")
            except ValueError:
                pass

        # Create successful run
        with track(project="prune_test", config=config) as run:
            run.log_param("x", 1)

        stats = prune_runs(
            registry,
            status="FAILED",
            keep_latest=0,
            dry_run=True,
        )

        assert stats.runs_scanned >= 3
        assert stats.runs_pruned == 3

    def test_prune_keeps_latest(self, store, registry, config):
        """Prune respects keep_latest count."""
        # Create 5 failed runs
        for _ in range(5):
            try:
                with track(project="prune_keep", config=config):
                    raise ValueError("fail")
            except ValueError:
                pass

        stats = prune_runs(
            registry,
            status="FAILED",
            keep_latest=2,
            dry_run=True,
        )

        # Should keep 2 latest, prune 3
        assert stats.runs_pruned == 3

    def test_prune_dry_run_preserves(self, store, registry, config):
        """Dry run doesn't actually delete."""
        try:
            with track(project="prune_dry", config=config) as run:
                run_id = run.run_id
                raise ValueError("fail")
        except ValueError:
            pass

        prune_runs(
            registry,
            status="FAILED",
            keep_latest=0,
            dry_run=True,
        )

        assert registry.exists(run_id)

    def test_prune_actually_deletes(self, store, registry, config):
        """Non-dry run deletes runs."""
        try:
            with track(project="prune_real", config=config) as run:
                run_id = run.run_id
                raise ValueError("fail")
        except ValueError:
            pass

        prune_runs(
            registry,
            status="FAILED",
            keep_latest=0,
            dry_run=False,
        )

        assert not registry.exists(run_id)


class TestWorkspaceHealth:
    """Tests for workspace health diagnostics."""

    def test_healthy_workspace(self, store, registry, config):
        """Healthy workspace reports no issues."""
        with track(project="health", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"content",
                media_type="text/plain",
                role="test",
            )

        health = check_workspace_health(store, registry)

        assert health["total_runs"] == 1
        assert health["orphaned_objects"] == 0
        assert health["missing_objects"] == 0

    def test_detects_orphaned_objects(self, store, registry, config):
        """Health check detects orphaned objects."""
        with track(project="health", config=config):
            pass

        # Add orphan
        store.put_bytes(b"orphan")

        health = check_workspace_health(store, registry)

        assert health["orphaned_objects"] == 1

    def test_detects_missing_objects(self, store, registry, config):
        """Health check detects missing referenced objects."""
        with track(project="health", config=config) as run:
            ref = run.log_bytes(
                kind="test.data",
                data=b"will be deleted",
                media_type="text/plain",
                role="test",
            )
            digest = ref.digest

        # Delete object directly (simulates corruption)
        store.delete(digest)

        health = check_workspace_health(store, registry)

        assert health["missing_objects"] == 1
