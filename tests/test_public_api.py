# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit public Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


class TestTopLevelExports:
    """Tests that public API exports are correct."""

    def test_core_tracking(self) -> None:
        """Core tracking API is accessible."""
        import devqubit

        assert hasattr(devqubit, "track")
        assert hasattr(devqubit, "Run")
        assert hasattr(devqubit, "wrap_backend")

    def test_config(self) -> None:
        """Config API is accessible."""
        import devqubit

        assert hasattr(devqubit, "Config")
        assert hasattr(devqubit, "get_config")
        assert hasattr(devqubit, "set_config")

    def test_version(self) -> None:
        """Version is accessible."""
        import devqubit

        assert hasattr(devqubit, "__version__")
        assert isinstance(devqubit.__version__, str)

    def test_low_level_not_in_top_level(self) -> None:
        """Low-level APIs are in submodules, not top-level."""
        import devqubit

        # These should NOT be in top-level __all__
        assert "RunRecord" not in devqubit.__all__
        assert "ArtifactRef" not in devqubit.__all__
        assert "create_store" not in devqubit.__all__
        assert "create_registry" not in devqubit.__all__
        # Moved to submodules
        assert "diff" not in devqubit.__all__
        assert "verify_baseline" not in devqubit.__all__
        assert "pack_run" not in devqubit.__all__
        assert "unpack_bundle" not in devqubit.__all__
        assert "Bundle" not in devqubit.__all__


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
            assert hasattr(run, "log_bytes")
            assert hasattr(run, "wrap")


class TestComparison:
    """Tests for comparison API."""

    def test_diff_runs(self, workspace: Path, make_run: Callable) -> None:
        """diff() compares two runs."""
        from devqubit import Config, set_config
        from devqubit.compare import diff
        from devqubit.storage import create_registry, create_store

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
        from devqubit import Config, set_config
        from devqubit.bundle import pack_run
        from devqubit.storage import create_registry, create_store

        config = Config(root_dir=workspace)
        set_config(config)

        run = make_run(run_id="bundle_test", counts={"00": 100})
        bundle_path = tmp_path / "test.zip"

        result = pack_run(
            run.run_id,
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
        from devqubit import Config, set_config
        from devqubit.bundle import Bundle, pack_run
        from devqubit.storage import create_registry, create_store

        config = Config(root_dir=workspace)
        set_config(config)

        run = make_run(run_id="reader_test")
        bundle_path = tmp_path / "reader.zip"

        pack_run(
            run.run_id,
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


# =============================================================================
# SUBMODULES
# =============================================================================


class TestRunsSubmodule:
    """Tests for devqubit.runs submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import runs

        # Run loading
        assert hasattr(runs, "load_run")
        assert hasattr(runs, "load_run_or_none")
        assert hasattr(runs, "run_exists")
        # Listing/searching
        assert hasattr(runs, "list_runs")
        assert hasattr(runs, "search_runs")
        assert hasattr(runs, "count_runs")
        # Projects/groups
        assert hasattr(runs, "list_projects")
        assert hasattr(runs, "list_groups")
        # Baselines
        assert hasattr(runs, "get_baseline")
        assert hasattr(runs, "set_baseline")
        assert hasattr(runs, "clear_baseline")

    def test_list_runs(self, workspace: Path, make_run: Callable) -> None:
        """list_runs returns runs."""
        from devqubit import Config, set_config
        from devqubit.runs import list_runs

        set_config(Config(root_dir=workspace))

        make_run(project="test_proj")
        make_run(project="test_proj")
        make_run(project="other_proj")

        all_runs = list_runs()
        assert len(all_runs) >= 3

        filtered = list_runs(project="test_proj")
        assert len(filtered) == 2

    def test_baseline_management(self, workspace: Path, make_run: Callable) -> None:
        """Baseline get/set/clear work."""
        from devqubit import Config, set_config
        from devqubit.runs import clear_baseline, get_baseline, set_baseline

        set_config(Config(root_dir=workspace))

        run = make_run(project="baseline_test")

        # Initially no baseline
        assert get_baseline("baseline_test") is None

        # Set baseline
        set_baseline("baseline_test", run.run_id)
        baseline = get_baseline("baseline_test")
        assert baseline is not None
        assert baseline["run_id"] == run.run_id

        # Clear baseline
        assert clear_baseline("baseline_test") is True
        assert get_baseline("baseline_test") is None


class TestCompareSubmodule:
    """Tests for devqubit.compare submodule."""

    def test_core_functions(self) -> None:
        """Core functions are available."""
        from devqubit import compare

        assert hasattr(compare, "diff")
        assert hasattr(compare, "verify_baseline")

    def test_result_types(self) -> None:
        """Result types are available."""
        from devqubit import compare

        assert hasattr(compare, "ComparisonResult")
        assert hasattr(compare, "VerifyResult")
        assert hasattr(compare, "VerifyPolicy")

    def test_program_match_mode(self) -> None:
        """ProgramMatchMode is available."""
        from devqubit.compare import ProgramMatchMode

        assert hasattr(ProgramMatchMode, "EXACT")
        assert hasattr(ProgramMatchMode, "STRUCTURAL")

    def test_verdict_types(self) -> None:
        """Verdict types are available."""
        from devqubit import compare

        assert hasattr(compare, "Verdict")
        assert hasattr(compare, "VerdictCategory")

    def test_drift_types(self) -> None:
        """Drift analysis types are available."""
        from devqubit import compare

        assert hasattr(compare, "DriftResult")
        assert hasattr(compare, "DriftThresholds")

    def test_format_options(self) -> None:
        """FormatOptions is available."""
        from devqubit import compare

        assert hasattr(compare, "FormatOptions")


class TestErrorsSubmodule:
    """Tests for devqubit.errors submodule."""

    def test_exports(self) -> None:
        """All expected exceptions are available."""
        from devqubit import errors

        # Base
        assert hasattr(errors, "DevQubitError")
        # Storage
        assert hasattr(errors, "StorageError")
        assert hasattr(errors, "ObjectNotFoundError")
        assert hasattr(errors, "RunNotFoundError")
        # Query
        assert hasattr(errors, "QueryParseError")
        # Envelope
        assert hasattr(errors, "MissingEnvelopeError")
        assert hasattr(errors, "EnvelopeValidationError")

    def test_catch_run_not_found(self, workspace: Path) -> None:
        """RunNotFoundError can be caught."""
        import pytest

        from devqubit import Config, set_config
        from devqubit.errors import RunNotFoundError
        from devqubit.runs import load_run

        set_config(Config(root_dir=workspace))

        with pytest.raises(RunNotFoundError):
            load_run("nonexistent_run_id")


class TestAdaptersSubmodule:
    """Tests for devqubit.adapters submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import adapters

        assert hasattr(adapters, "AdapterProtocol")
        assert hasattr(adapters, "list_available_adapters")
        assert hasattr(adapters, "adapter_load_errors")
        assert hasattr(adapters, "get_adapter_by_name")

    def test_list_adapters(self) -> None:
        """list_available_adapters returns a list."""
        from devqubit.adapters import list_available_adapters

        result = list_available_adapters()
        assert isinstance(result, list)


class TestStorageSubmodule:
    """Tests for devqubit.storage submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import storage

        assert hasattr(storage, "create_store")
        assert hasattr(storage, "create_registry")
        assert hasattr(storage, "ObjectStoreProtocol")
        assert hasattr(storage, "RegistryProtocol")
        assert hasattr(storage, "ArtifactRef")
        assert hasattr(storage, "RunSummary")
        assert hasattr(storage, "BaselineInfo")

    def test_create_store(self, workspace: Path) -> None:
        """create_store creates a working store."""
        from devqubit import Config
        from devqubit.storage import create_store

        config = Config(root_dir=workspace)
        store = create_store(config=config)

        digest = store.put_bytes(b"test data")
        assert store.exists(digest)
        assert store.get_bytes(digest) == b"test data"

    def test_create_registry(self, workspace: Path) -> None:
        """create_registry creates a working registry."""
        from devqubit import Config
        from devqubit.storage import create_registry

        config = Config(root_dir=workspace)
        registry = create_registry(config=config)

        assert hasattr(registry, "save")
        assert hasattr(registry, "load")
        assert hasattr(registry, "exists")
        assert hasattr(registry, "list_runs")


class TestBundleSubmodule:
    """Tests for devqubit.bundle submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import bundle

        assert hasattr(bundle, "pack_run")
        assert hasattr(bundle, "unpack_bundle")
        assert hasattr(bundle, "Bundle")
        assert hasattr(bundle, "list_bundle_contents")


class TestCiSubmodule:
    """Tests for devqubit.ci submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import ci

        assert hasattr(ci, "write_junit")
        assert hasattr(ci, "result_to_junit")
        assert hasattr(ci, "github_annotations")


class TestConfigSubmodule:
    """Tests for devqubit.config submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import config

        assert hasattr(config, "Config")
        assert hasattr(config, "RedactionConfig")
        assert hasattr(config, "get_config")
        assert hasattr(config, "set_config")
        assert hasattr(config, "reset_config")
        assert hasattr(config, "load_config")


class TestUECSubmodule:
    """Tests for devqubit.uec submodule."""

    def test_exports(self) -> None:
        """All expected exports are available."""
        from devqubit import uec

        assert hasattr(uec, "ExecutionEnvelope")
        assert hasattr(uec, "DeviceSnapshot")
        assert hasattr(uec, "ProgramSnapshot")
        assert hasattr(uec, "ExecutionSnapshot")
        assert hasattr(uec, "ResultSnapshot")
        assert hasattr(uec, "ValidationResult")
