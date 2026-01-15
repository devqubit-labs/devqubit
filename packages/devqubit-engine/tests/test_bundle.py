# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for bundle pack/unpack operations and bundle reading."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from devqubit_engine.bundle.pack import list_bundle_contents, pack_run, unpack_bundle
from devqubit_engine.bundle.reader import Bundle, is_bundle_path
from devqubit_engine.tracking.run import track


class TestIsBundle:
    """Tests for bundle path validation."""

    def test_valid_bundle(self, tmp_path: Path):
        """Valid bundle with required files is detected."""
        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("run.json", "{}")

        assert is_bundle_path(bundle_path)

    def test_nonexistent_path(self, tmp_path: Path):
        """Nonexistent path is not a bundle."""
        assert not is_bundle_path(tmp_path / "nonexistent.zip")

    def test_missing_manifest(self, tmp_path: Path):
        """Zip without manifest.json is not a bundle."""
        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("run.json", "{}")

        assert not is_bundle_path(bundle_path)

    def test_not_a_zip(self, tmp_path: Path):
        """Non-zip file is not a bundle."""
        path = tmp_path / "test.zip"
        path.write_text("not a zip")

        assert not is_bundle_path(path)


class TestPackRun:
    """Tests for packing runs into bundles."""

    def test_pack_creates_bundle(self, store, registry, config, tmp_path: Path):
        """Pack creates valid bundle with manifest and run record."""
        bundle_path = tmp_path / "test.zip"

        with track(project="pack_test", config=config) as run:
            run.log_param("x", 42)
            run_id = run.run_id

        result = pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        assert result.run_id == run_id
        assert result.bundle_path == bundle_path
        assert bundle_path.exists()

        # Verify bundle structure
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "run.json" in names

    def test_pack_includes_artifacts(self, store, registry, config, tmp_path: Path):
        """Pack includes artifact objects in bundle."""
        bundle_path = tmp_path / "test.zip"

        with track(project="pack_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"test artifact content",
                media_type="application/octet-stream",
                role="test",
            )
            run_id = run.run_id

        result = pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        assert result.object_count >= 1
        assert result.artifact_count >= 1
        assert not result.missing_objects

    def test_pack_strict_fails_on_missing(
        self, store, registry, config, tmp_path: Path
    ):
        """Strict mode fails if objects are missing."""
        bundle_path = tmp_path / "test.zip"

        with track(project="pack_test", config=config) as run:
            run_id = run.run_id

        # Manually add a fake artifact reference
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
                run_id=run_id,
                output_path=bundle_path,
                store=store,
                registry=registry,
                strict=True,
            )


class TestUnpackBundle:
    """Tests for unpacking bundles."""

    def test_unpack_restores_run(self, factory_store, factory_registry, tmp_path: Path):
        """Unpack restores run record and objects."""
        store_a = factory_store()
        reg_a = factory_registry()
        store_b = factory_store()
        reg_b = factory_registry()
        bundle_path = tmp_path / "test.zip"

        # Create and pack in workspace A
        with track(
            project="unpack_test",
            store=store_a,
            registry=reg_a,
            capture_env=False,
            capture_git=False,
        ) as run:
            run.log_param("value", 123)
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store_a,
            registry=reg_a,
        )

        # Unpack to workspace B
        result = unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store_b,
            dest_registry=reg_b,
        )

        assert result.run_id == run_id
        loaded = reg_b.load(run_id)
        assert loaded.record["data"]["params"]["value"] == 123

    def test_unpack_fails_if_exists(self, store, registry, config, tmp_path: Path):
        """Unpack fails if run exists and overwrite=False."""
        bundle_path = tmp_path / "test.zip"

        with track(project="test", config=config) as run:
            run_id = run.run_id

        pack_run(
            run_id=run_id,
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

    def test_unpack_overwrite(self, store, registry, config, tmp_path: Path):
        """Unpack succeeds with overwrite=True."""
        bundle_path = tmp_path / "test.zip"

        with track(project="test", config=config) as run:
            run_id = run.run_id

        pack_run(
            run_id=run_id,
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


class TestRoundtrip:
    """Tests for full pack/unpack roundtrip."""

    def test_roundtrip_preserves_data(
        self, factory_store, factory_registry, tmp_path: Path
    ):
        """Full roundtrip preserves params and artifacts."""
        store_a = factory_store()
        reg_a = factory_registry()
        store_b = factory_store()
        reg_b = factory_registry()
        bundle_path = tmp_path / "bundle.zip"

        # Create run with params and artifact
        with track(
            project="roundtrip",
            store=store_a,
            registry=reg_a,
            capture_env=False,
            capture_git=False,
        ) as run:
            run.log_param("shots", 1000)
            run.log_param("seed", 42)
            run.log_bytes(
                kind="test.artifact",
                data=b"artifact data",
                media_type="application/octet-stream",
                role="test",
            )
            run_id = run.run_id

        # Pack and unpack
        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store_a,
            registry=reg_a,
        )
        unpack_bundle(
            bundle_path=bundle_path,
            dest_store=store_b,
            dest_registry=reg_b,
        )

        # Verify
        loaded = reg_b.load(run_id)
        assert loaded.record["data"]["params"]["shots"] == 1000
        assert loaded.record["data"]["params"]["seed"] == 42
        assert len(loaded.artifacts) >= 1

        # Verify artifact content is accessible
        artifact = loaded.artifacts[0]
        data = store_b.get_bytes(artifact.digest)
        assert data == b"artifact data"


class TestBundleReader:
    """Tests for Bundle reader class."""

    def test_read_manifest_and_record(self, store, registry, config, tmp_path: Path):
        """Bundle reader provides access to manifest and run record."""
        bundle_path = tmp_path / "test.zip"

        with track(project="reader_test", config=config) as run:
            run.log_param("x", 1)
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with Bundle(bundle_path) as b:
            assert b.run_id == run_id
            assert b.manifest["run_id"] == run_id
            assert b.run_record["data"]["params"]["x"] == 1

    def test_bundle_store_access(self, store, registry, config, tmp_path: Path):
        """Bundle store provides read access to objects."""
        bundle_path = tmp_path / "test.zip"

        with track(project="store_test", config=config) as run:
            run.log_bytes(
                kind="test.data",
                data=b"test content",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with Bundle(bundle_path) as b:
            artifacts = b.run_record.get("artifacts", [])
            assert len(artifacts) >= 1

            digest = artifacts[0]["digest"]
            data = b.store.get_bytes(digest)
            assert data == b"test content"

    def test_list_objects(self, store, registry, config, tmp_path: Path):
        """Bundle lists contained objects."""
        bundle_path = tmp_path / "test.zip"

        with track(project="list_test", config=config) as run:
            run.log_bytes(
                kind="abcdef",
                data=b"aaa",
                media_type="text/plain",
                role="test",
            )
            run.log_bytes(
                kind="fedcba",
                data=b"bbb",
                media_type="text/plain",
                role="test",
            )
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        with Bundle(bundle_path) as b:
            objects = b.list_objects()
            assert len(objects) >= 2
            assert all(o.startswith("sha256:") for o in objects)


class TestListBundleContents:
    """Tests for listing bundle contents without extraction."""

    def test_list_contents(self, store, registry, config, tmp_path: Path):
        """list_bundle_contents returns summary without extracting."""
        bundle_path = tmp_path / "test.zip"

        with track(project="list_contents", config=config) as run:
            run.log_param("key", "value")
            run_id = run.run_id

        pack_run(
            run_id=run_id,
            output_path=bundle_path,
            store=store,
            registry=registry,
        )

        contents = list_bundle_contents(bundle_path)

        assert contents["run_id"] == run_id
        assert "manifest" in contents
        assert contents["manifest"]["run_id"] == run_id
