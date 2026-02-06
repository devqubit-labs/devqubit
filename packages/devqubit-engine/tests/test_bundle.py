# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for bundle pack/unpack operations."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from devqubit_engine.bundle.pack import pack_run, unpack_bundle
from devqubit_engine.bundle.reader import Bundle, is_bundle_path
from devqubit_engine.tracking.run import track


class TestBundleRoundtrip:
    """Tests for full pack/unpack roundtrip between workspaces."""

    def test_roundtrip_preserves_data(
        self,
        factory_store,
        factory_registry,
        tmp_path: Path,
    ):
        """Full roundtrip preserves params, metrics, and artifacts."""
        store_a = factory_store()
        reg_a = factory_registry()
        store_b = factory_store()
        reg_b = factory_registry()
        bundle_path = tmp_path / "bundle.zip"

        # Create run with various data types
        with track(
            project="roundtrip",
            store=store_a,
            registry=reg_a,
            capture_env=False,
            capture_git=False,
        ) as run:
            run.log_param("shots", 1000)
            run.log_param("seed", 42)
            run.log_metric("fidelity", 0.95)
            run.log_bytes(
                kind="test.artifact",
                data=b"artifact content",
                media_type="application/octet-stream",
                role="test",
            )
            run_id = run.run_id

        # Pack from workspace A
        pack_result = pack_run(
            run_id,
            output_path=bundle_path,
            store=store_a,
            registry=reg_a,
        )

        assert pack_result.run_id == run_id
        assert bundle_path.exists()
        assert pack_result.artifact_count >= 1

        # Unpack to workspace B
        unpack_result = unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store_b,
            dest_registry=reg_b,
        )

        assert unpack_result.run_id == run_id

        # Verify data integrity
        loaded = reg_b.load(run_id)
        assert loaded.params["shots"] == 1000
        assert loaded.params["seed"] == 42
        assert loaded.metrics["fidelity"] == 0.95
        assert len(loaded.artifacts) >= 1

        # Verify artifact content is accessible
        artifact = loaded.artifacts[0]
        data = store_b.get_bytes(artifact.digest)
        assert data == b"artifact content"


class TestPackRun:
    """Tests for packing runs into bundles."""

    def test_pack_creates_valid_bundle(self, store, registry, config, tmp_path: Path):
        """Pack creates bundle with required structure."""
        bundle_path = tmp_path / "test.zip"

        with track(
            project="pack_test", store=store, registry=registry, config=config
        ) as run:
            run.log_param("x", 42)
            run_id = run.run_id

        pack_run(
            run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        # Verify bundle structure
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "run.json" in names

        # Verify it's detected as bundle
        assert is_bundle_path(bundle_path)

    def test_pack_strict_fails_on_missing_objects(
        self,
        store,
        registry,
        config,
        tmp_path: Path,
    ):
        """Strict mode fails if referenced objects are missing."""
        bundle_path = tmp_path / "test.zip"

        with track(
            project="pack_test", store=store, registry=registry, config=config
        ) as run:
            run_id = run.run_id

        # Add fake artifact reference
        run_record = registry.load(run_id)
        record = run_record.to_dict()
        record["artifacts"] = [
            {
                "digest": "sha256:deadbeef" + "0" * 56,
                "role": "test",
                "kind": "fake.artifact",
                "media_type": "application/octet-stream",
            }
        ]
        registry.save(record)

        with pytest.raises(FileNotFoundError, match="Missing"):
            pack_run(
                run_id,
                output_path=bundle_path,
                store=store,
                registry=registry,
                strict=True,
            )


class TestUnpackBundle:
    """Tests for unpacking bundles."""

    def test_unpack_fails_if_exists(self, store, registry, config, tmp_path: Path):
        """Unpack fails if run exists and overwrite=False."""
        bundle_path = tmp_path / "test.zip"

        with track(
            project="test", store=store, registry=registry, config=config
        ) as run:
            run_id = run.run_id

        pack_run(
            run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with pytest.raises(FileExistsError):
            unpack_bundle(
                bundle_path=bundle_path,
                dest_store=store,
                dest_registry=registry,
                overwrite=False,
            )

    def test_unpack_with_overwrite(self, store, registry, config, tmp_path: Path):
        """Unpack succeeds with overwrite=True."""
        bundle_path = tmp_path / "test.zip"

        with track(
            project="test", store=store, registry=registry, config=config
        ) as run:
            run_id = run.run_id

        pack_run(
            run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        # Should not raise
        result = unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store,
            dest_registry=registry,
            overwrite=True,
        )
        assert result.run_id == run_id


class TestBundleReader:
    """Tests for Bundle reader class."""

    def test_read_bundle_contents(self, store, registry, config, tmp_path: Path):
        """Bundle reader provides access to manifest, record, and objects."""
        bundle_path = tmp_path / "test.zip"

        with track(
            project="reader_test", store=store, registry=registry, config=config
        ) as run:
            run.log_param("key", "value")
            run.log_bytes(
                kind="test.data",
                data=b"test content",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        pack_run(
            run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with Bundle(bundle_path) as b:
            # Manifest access
            assert b.run_id == run_id
            assert b.manifest["run_id"] == run_id

            # Run record access
            assert b.run_record["data"]["params"]["key"] == "value"

            # Object store access
            artifacts = b.run_record.get("artifacts", [])
            assert len(artifacts) >= 1

            digest = artifacts[0]["digest"]
            data = b.store.get_bytes(digest)
            assert data == b"test content"

            # List objects
            objects = b.list_objects()
            assert len(objects) >= 1
            assert all(o.startswith("sha256:") for o in objects)


class TestBundleDetection:
    """Tests for bundle path validation."""

    def test_valid_bundle_detected(self, tmp_path: Path):
        """Valid bundle with required files is detected."""
        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("run.json", "{}")

        assert is_bundle_path(bundle_path)

    def test_invalid_paths_not_detected(self, tmp_path: Path):
        """Invalid paths are not detected as bundles."""
        # Nonexistent
        assert not is_bundle_path(tmp_path / "nonexistent.zip")

        # Missing manifest
        no_manifest = tmp_path / "no_manifest.zip"
        with zipfile.ZipFile(no_manifest, "w") as zf:
            zf.writestr("run.json", "{}")
        assert not is_bundle_path(no_manifest)

        # Not a zip
        not_zip = tmp_path / "not_a.zip"
        not_zip.write_text("not a zip file")
        assert not is_bundle_path(not_zip)
