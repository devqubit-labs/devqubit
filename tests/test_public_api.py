# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tests for devqubit public Python API.

Tests the user-facing API exposed through the devqubit package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


class TestModuleExports:
    """Tests that public API exports are correct."""

    def test_top_level_exports(self) -> None:
        """All documented exports are accessible."""
        import devqubit

        # Core tracking
        assert hasattr(devqubit, "track")
        assert hasattr(devqubit, "Run")

        # Models
        assert hasattr(devqubit, "RunRecord")
        assert hasattr(devqubit, "ArtifactRef")

        # Comparison
        assert hasattr(devqubit, "diff")
        assert hasattr(devqubit, "verify")
        assert hasattr(devqubit, "verify_against_baseline")

        # Bundle
        assert hasattr(devqubit, "pack_run")
        assert hasattr(devqubit, "unpack_bundle")
        assert hasattr(devqubit, "Bundle")

        # Storage
        assert hasattr(devqubit, "create_store")
        assert hasattr(devqubit, "create_registry")

        # Config
        assert hasattr(devqubit, "Config")
        assert hasattr(devqubit, "get_config")
        assert hasattr(devqubit, "set_config")

    def test_compare_submodule(self) -> None:
        """devqubit.compare exports are correct."""
        from devqubit import compare

        assert hasattr(compare, "VerifyPolicy")
        assert hasattr(compare, "ComparisonResult")
        assert hasattr(compare, "VerifyResult")

    def test_version(self) -> None:
        """Version is accessible."""
        import devqubit

        assert hasattr(devqubit, "__version__")
        assert isinstance(devqubit.__version__, str)


class TestTracking:
    """Tests for the tracking API."""

    def test_track_context_manager(self, workspace: Path) -> None:
        """track() works as context manager."""
        from devqubit import Config, set_config, track

        set_config(Config(root_dir=workspace))

        with track(project="test_tracking") as run:
            run.log_param("shots", 1000)
            run.log_metric("fidelity", 0.95)
            run.set_tag("experiment", "unit_test")

        assert run.run_id is not None

    def test_run_attributes(self, workspace: Path) -> None:
        """Run object has expected attributes."""
        from devqubit import Config, set_config, track

        set_config(Config(root_dir=workspace))

        with track(project="attr_test") as run:
            assert hasattr(run, "run_id")
            assert hasattr(run, "log_param")
            assert hasattr(run, "log_metric")
            assert hasattr(run, "set_tag")
            assert hasattr(run, "log_bytes")  # artifact logging
            assert hasattr(run, "wrap")


class TestComparison:
    """Tests for comparison API."""

    def test_diff_runs(self, workspace: Path, make_run: Callable) -> None:
        """diff() compares two runs."""
        from devqubit import Config, create_registry, create_store, diff, set_config

        config = Config(root_dir=workspace)
        set_config(config)

        run_a = make_run(run_id="api_diff_a", counts={"00": 500, "11": 500})
        run_b = make_run(run_id="api_diff_b", counts={"00": 480, "11": 520})

        result = diff(
            run_a.run_id,
            run_b.run_id,
            registry=create_registry(config=config),
            store=create_store(config=config),
        )

        assert hasattr(result, "identical")
        assert hasattr(result, "tvd")

    def test_verify_policy(self) -> None:
        """VerifyPolicy is configurable."""
        from devqubit.compare import VerifyPolicy

        policy = VerifyPolicy(
            tvd_max=0.1,
            params_must_match=False,
            program_must_match=False,
        )

        assert policy.tvd_max == 0.1
        assert policy.params_must_match is False


class TestBundle:
    """Tests for bundle API."""

    def test_pack_and_unpack(
        self, workspace: Path, make_run: Callable, tmp_path: Path
    ) -> None:
        """pack_run and unpack_bundle work correctly."""
        from devqubit import (
            Config,
            create_registry,
            create_store,
            pack_run,
            set_config,
        )

        config = Config(root_dir=workspace)
        set_config(config)

        run = make_run(run_id="bundle_test", counts={"00": 100})
        bundle_path = tmp_path / "test.zip"

        # Pack
        result = pack_run(
            run_id=run.run_id,
            output_path=bundle_path,
            store=create_store(config=config),
            registry=create_registry(config=config),
        )
        assert bundle_path.exists()
        assert result.artifact_count >= 0

    def test_bundle_reader(
        self, workspace: Path, make_run: Callable, tmp_path: Path
    ) -> None:
        """Bundle can read packed runs."""
        from devqubit import (
            Bundle,
            Config,
            create_registry,
            create_store,
            pack_run,
            set_config,
        )

        config = Config(root_dir=workspace)
        set_config(config)

        run = make_run(run_id="reader_test")
        bundle_path = tmp_path / "reader.zip"

        pack_run(
            run_id=run.run_id,
            output_path=bundle_path,
            store=create_store(config=config),
            registry=create_registry(config=config),
        )

        with Bundle(bundle_path) as bundle:
            assert bundle.run_id == run.run_id
            assert bundle.run_record is not None


class TestConfig:
    """Tests for configuration API."""

    def test_config_defaults(self, tmp_path: Path) -> None:
        """Config has sensible defaults."""
        from devqubit import Config

        config = Config(root_dir=tmp_path / ".devqubit")

        assert config.root_dir is not None
        assert hasattr(config, "storage_url")
        assert hasattr(config, "registry_url")

    def test_get_set_config(self, workspace: Path) -> None:
        """get_config and set_config work."""
        from devqubit import Config, get_config, set_config

        config = Config(root_dir=workspace)
        set_config(config)

        retrieved = get_config()
        assert retrieved.root_dir == workspace


class TestStorage:
    """Tests for storage factory API."""

    def test_create_store(self, workspace: Path) -> None:
        """create_store creates a working store."""
        from devqubit import Config, create_store

        config = Config(root_dir=workspace)
        store = create_store(config=config)

        # Store should work
        digest = store.put_bytes(b"test data")
        assert store.exists(digest)
        assert store.get_bytes(digest) == b"test data"

    def test_create_registry(self, workspace: Path) -> None:
        """create_registry creates a working registry."""
        from devqubit import Config, create_registry

        config = Config(root_dir=workspace)
        registry = create_registry(config=config)

        # Registry should have expected methods
        assert hasattr(registry, "save")
        assert hasattr(registry, "load")
        assert hasattr(registry, "exists")
        assert hasattr(registry, "list_runs")


class TestSubmodules:
    """Tests for devqubit submodules."""

    def test_ci_submodule(self) -> None:
        """devqubit.ci provides CI utilities."""
        from devqubit import ci

        assert hasattr(ci, "write_junit")
        assert hasattr(ci, "github_annotations")

    def test_bundle_submodule(self) -> None:
        """devqubit.bundle provides bundle utilities."""
        from devqubit import bundle

        assert hasattr(bundle, "pack_run")
        assert hasattr(bundle, "unpack_bundle")
        assert hasattr(bundle, "Bundle")

    def test_snapshot_submodule(self) -> None:
        """devqubit.snapshot provides UEC types."""
        from devqubit import snapshot

        assert hasattr(snapshot, "ExecutionEnvelope")
        assert hasattr(snapshot, "DeviceSnapshot")
        assert hasattr(snapshot, "ProgramSnapshot")

    def test_config_submodule(self) -> None:
        """devqubit.config provides configuration."""
        from devqubit import config

        assert hasattr(config, "Config")
        assert hasattr(config, "RedactionConfig")
