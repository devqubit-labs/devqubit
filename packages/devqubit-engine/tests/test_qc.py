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
        """GC with no orphans deletes nothing."""
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"test content",
                media_type="text/plain",
                role="test",
            )

        stats = gc_run(store, registry, dry_run=True)

        # GCStats has: referenced_objects, unreferenced_objects, bytes_reclaimable
        assert stats.referenced_objects >= 1
        assert stats.unreferenced_objects == 0
        assert stats.bytes_reclaimable == 0

    def test_gc_finds_orphans(self, store, registry, config):
        """GC finds orphaned objects."""
        # Create run with artifact
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"referenced",
                media_type="text/plain",
                role="test",
            )

        # Add orphaned object directly to store
        orphan_digest = store.put_bytes(b"orphaned content")

        stats = gc_run(store, registry, dry_run=True)

        assert stats.unreferenced_objects >= 1
        assert stats.bytes_reclaimable > 0
        # Orphan not deleted in dry run
        assert store.exists(orphan_digest)

    def test_gc_deletes_orphans(self, store, registry, config):
        """GC deletes orphaned objects when not dry run."""
        # Create referenced object
        with track(project="gc_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"keep me",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        # Add orphan
        orphan_digest = store.put_bytes(b"delete me")

        # Run GC
        stats = gc_run(store, registry, dry_run=False)

        assert stats.objects_deleted >= 1
        assert stats.bytes_reclaimed > 0
        assert not store.exists(orphan_digest)

        # Referenced object still exists
        loaded = registry.load(run_id)
        ref_digest = loaded.artifacts[0].digest
        assert store.exists(ref_digest)


class TestPruneRuns:
    """Tests for pruning old runs."""

    def test_prune_by_status(self, store, registry, config):
        """Prune runs by status."""
        # Create some failed runs
        for _ in range(3):
            try:
                with track(project="prune_test", config=config) as run:
                    _ = run.run_id
                    raise ValueError("fail")
            except ValueError:
                pass

        # Create successful run
        with track(project="prune_test", config=config) as run:
            run.log_param("x", 1)

        # PruneStats is a dataclass with: runs_scanned, runs_pruned, artifacts_orphaned
        stats = prune_runs(
            registry,
            status="FAILED",
            keep_latest=0,
            dry_run=True,
        )

        assert stats.runs_scanned >= 3
        assert stats.runs_pruned == 3  # Would delete all failed runs

    def test_prune_keeps_latest(self, store, registry, config):
        """Prune respects keep_latest."""
        # Create 5 failed runs
        for _ in range(5):
            try:
                with track(project="prune_test", config=config) as run:
                    _ = run.run_id
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

    def test_prune_dry_run_no_delete(self, store, registry, config):
        """Dry run doesn't delete."""
        try:
            with track(project="prune_test", config=config) as run:
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
            with track(project="prune_test", config=config) as run:
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
    """Tests for workspace health check."""

    def test_healthy_workspace(self, store, registry, config):
        """Check healthy workspace."""
        with track(project="health_test", config=config) as run:
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

    def test_detects_orphans(self, store, registry, config):
        """Health check detects orphaned objects."""
        with track(project="health_test", config=config) as run:
            _ = run.run_id

        # Add orphan
        store.put_bytes(b"orphan")

        health = check_workspace_health(store, registry)

        assert health["orphaned_objects"] == 1

    def test_detects_missing(self, store, registry, config):
        """Health check detects missing objects."""
        with track(project="health_test", config=config) as run:
            ref = run.log_bytes(
                kind="test.data",
                data=b"content",
                media_type="text/plain",
                role="test",
            )
            digest = ref.digest

        # Delete the object directly
        store.delete(digest)

        health = check_workspace_health(store, registry)

        assert health["missing_objects"] == 1
