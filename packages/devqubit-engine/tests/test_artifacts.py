# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for artifact utilities."""

from __future__ import annotations

import json

from devqubit_engine.storage.artifacts.io import (
    load_artifact_bytes,
    load_artifact_json,
    load_artifact_text,
)
from devqubit_engine.storage.artifacts.lookup import (
    find_all_artifacts,
    find_artifact,
    get_artifact,
    get_artifact_bytes,
    list_artifacts,
)


class TestArtifactLookup:
    """Tests for finding artifacts in run records."""

    def test_find_by_role(self, run_factory, artifact_factory):
        """Find artifact by role."""
        program = artifact_factory(b"circuit", kind="circuit.qasm", role="program")
        results = artifact_factory(b"counts", kind="result.json", role="results")
        run = run_factory(artifacts=[program, results])

        found = find_artifact(run, role="program")

        assert found is not None
        assert found.role == "program"
        assert found.kind == "circuit.qasm"

    def test_find_by_kind_contains(self, run_factory, artifact_factory):
        """Find artifact by kind substring (case-insensitive)."""
        qasm = artifact_factory(b"qasm", kind="source.openqasm3", role="program")
        qpy = artifact_factory(b"qpy", kind="qiskit.qpy.circuits", role="program")
        run = run_factory(artifacts=[qasm, qpy])

        found = find_artifact(run, kind_contains="QASM")  # Case insensitive

        assert found is not None
        assert "qasm" in found.kind.lower()

    def test_find_by_role_and_kind(self, run_factory, artifact_factory):
        """Find artifact by both role and kind."""
        program_qasm = artifact_factory(b"qasm", kind="source.qasm", role="program")
        results_json = artifact_factory(b"json", kind="result.json", role="results")
        run = run_factory(artifacts=[program_qasm, results_json])

        found = find_artifact(run, role="program", kind_contains="qasm")

        assert found is not None
        assert found.role == "program"
        assert "qasm" in found.kind

    def test_find_returns_none_when_not_found(self, run_factory, artifact_factory):
        """find_artifact returns None when no match."""
        artifact = artifact_factory(b"data", kind="something.else", role="test")
        run = run_factory(artifacts=[artifact])

        assert find_artifact(run, role="program") is None
        assert find_artifact(run, kind_contains="notexist") is None

    def test_find_all_artifacts(self, run_factory, artifact_factory):
        """find_all_artifacts returns all matching."""
        a1 = artifact_factory(b"p1", kind="program.qasm", role="program")
        a2 = artifact_factory(b"p2", kind="program.qpy", role="program")
        a3 = artifact_factory(b"r1", kind="result.json", role="results")
        run = run_factory(artifacts=[a1, a2, a3])

        found = find_all_artifacts(run, role="program")

        assert len(found) == 2


class TestArtifactSelector:
    """Tests for get_artifact selector patterns."""

    def test_get_by_index(self, run_factory, artifact_factory):
        """Get artifact by index."""
        a0 = artifact_factory(b"first", kind="first.txt", role="test")
        a1 = artifact_factory(b"second", kind="second.txt", role="test")
        run = run_factory(artifacts=[a0, a1])

        assert get_artifact(run, 0) == a0
        assert get_artifact(run, 1) == a1
        assert get_artifact(run, 99) is None

    def test_get_by_digest_prefix(self, run_factory, artifact_factory):
        """Get artifact by digest prefix."""
        art = artifact_factory(b"data", kind="test.txt", role="test")
        run = run_factory(artifacts=[art])

        # Full digest works
        found = get_artifact(run, art.digest)
        assert found == art

        # Prefix works
        prefix = art.digest[:20]
        found = get_artifact(run, prefix)
        assert found == art

    def test_get_by_role_kind_pattern(self, run_factory, artifact_factory):
        """Get artifact by 'role:kind' pattern."""
        art = artifact_factory(b"qasm", kind="source.openqasm3", role="program")
        run = run_factory(artifacts=[art])

        found = get_artifact(run, "program:openqasm")

        assert found == art

    def test_get_by_kind_substring(self, run_factory, artifact_factory):
        """Get artifact by kind substring."""
        art = artifact_factory(b"counts", kind="result.counts.json", role="results")
        run = run_factory(artifacts=[art])

        found = get_artifact(run, "counts")

        assert found == art


class TestArtifactLoading:
    """Tests for loading artifact content."""

    def test_load_bytes(self, store, artifact_factory, run_factory):
        """load_artifact_bytes returns raw bytes."""
        art = artifact_factory(b"raw bytes content", kind="test.bin", role="test")
        _ = run_factory(artifacts=[art])

        data = load_artifact_bytes(art, store)

        assert data == b"raw bytes content"

    def test_load_text(self, store, artifact_factory, run_factory):
        """load_artifact_text decodes as UTF-8."""
        art = artifact_factory(
            "OPENQASM 3.0;".encode(),
            kind="circuit.qasm",
            role="program",
        )

        text = load_artifact_text(art, store)

        assert text == "OPENQASM 3.0;"

    def test_load_json(self, store, artifact_factory):
        """load_artifact_json parses JSON content."""
        payload = {"counts": {"00": 500, "11": 500}, "shots": 1000}
        art = artifact_factory(
            json.dumps(payload).encode(),
            kind="result.json",
            role="results",
        )

        parsed = load_artifact_json(art, store)

        assert parsed == payload

    def test_get_artifact_bytes_by_selector(self, store, artifact_factory, run_factory):
        """get_artifact_bytes combines lookup and load."""
        art = artifact_factory(b"content", kind="test.data", role="test")
        run = run_factory(artifacts=[art])

        data = get_artifact_bytes(run, "test.data", store)

        assert data == b"content"


class TestArtifactListing:
    """Tests for list_artifacts with extended info."""

    def test_list_with_size(self, store, run_factory, artifact_factory):
        """list_artifacts includes size when store provided."""
        art = artifact_factory(b"x" * 100, kind="test.bin", role="test")
        run = run_factory(artifacts=[art])

        infos = list_artifacts(run, store=store)

        assert len(infos) == 1
        assert infos[0].size == 100
        assert infos[0].kind == "test.bin"

    def test_list_filtered_by_role(self, run_factory, artifact_factory):
        """list_artifacts filters by role."""
        program = artifact_factory(b"p", kind="circuit.qasm", role="program")
        results = artifact_factory(b"r", kind="result.json", role="results")
        run = run_factory(artifacts=[program, results])

        infos = list_artifacts(run, role="program")

        assert len(infos) == 1
        assert infos[0].role == "program"
