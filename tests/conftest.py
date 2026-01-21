# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Test fixtures for devqubit CLI tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from click.testing import CliRunner, Result
from devqubit_engine.cli import cli
from devqubit_engine.config import Config
from devqubit_engine.storage.backends.local import LocalRegistry, LocalStore
from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.tracking.record import RunRecord
from devqubit_engine.utils.common import utc_now_iso


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Isolated temporary workspace."""
    ws = tmp_path / ".devqubit"
    ws.mkdir(parents=True)
    (ws / "objects").mkdir()
    return ws


@pytest.fixture
def config(workspace: Path) -> Config:
    """Config pointing to temp workspace."""
    return Config(root_dir=workspace)


@pytest.fixture
def store(workspace: Path) -> LocalStore:
    """Local object store."""
    return LocalStore(workspace / "objects")


@pytest.fixture
def registry(workspace: Path) -> LocalRegistry:
    """Local registry."""
    return LocalRegistry(workspace)


@pytest.fixture
def invoke(cli_runner: CliRunner, workspace: Path) -> Callable[..., Result]:
    """
    Invoke CLI commands in isolated workspace.

    Returns Result with exit_code, output, etc.
    Exceptions are caught so tests can verify error handling.

    Usage:
        result = invoke("list")
        result = invoke("show", "run123", "--format", "json")
        result = invoke("delete", "run123", input="y\\n")
    """

    def _invoke(
        *args: str,
        input: str | None = None,
        catch_exceptions: bool = True,
    ) -> Result:
        return cli_runner.invoke(
            cli,
            ["--root", str(workspace), *args],
            catch_exceptions=catch_exceptions,
            input=input,
        )

    return _invoke


@pytest.fixture
def make_run(registry: LocalRegistry, store: LocalStore) -> Callable[..., RunRecord]:
    """
    Factory for creating persisted run records.

    Creates runs with realistic structure and saves to registry.
    """
    counter = [0]

    def _create(
        run_id: str | None = None,
        project: str = "test_project",
        adapter: str = "manual",
        backend: str = "aer_simulator",
        status: str = "FINISHED",
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        counts: dict[str, int] | None = None,
        group_id: str | None = None,
        group_name: str | None = None,
    ) -> RunRecord:
        counter[0] += 1
        if run_id is None:
            run_id = f"run_{counter[0]:08d}"

        artifacts: list[ArtifactRef] = []

        # Add counts artifact if provided
        if counts:
            data = json.dumps({"counts": counts}).encode()
            digest = store.put_bytes(data)
            artifacts.append(
                ArtifactRef(
                    kind="result.counts.json",
                    digest=digest,
                    media_type="application/json",
                    role="results",
                )
            )

        now = utc_now_iso()
        record: dict[str, Any] = {
            "schema": "devqubit.run/0.1",
            "run_id": run_id,
            "created_at": now,
            "project": {"name": project},
            "adapter": adapter,
            "info": {
                "status": status,
                "ended_at": now if status == "FINISHED" else None,
            },
            "data": {
                "params": params or {},
                "metrics": metrics or {},
                "tags": tags or {},
            },
            "backend": {"name": backend, "type": "simulator", "provider": "aer"},
            "fingerprints": {
                "run": f"sha256:{hashlib.sha256(run_id.encode()).hexdigest()}",
                "program": f"sha256:{hashlib.sha256(b'program').hexdigest()}",
            },
            "artifacts": [a.to_dict() for a in artifacts],
        }

        if group_id:
            record["group_id"] = group_id
        if group_name:
            record["group_name"] = group_name

        registry.save(record)
        return RunRecord(record=record, artifacts=artifacts)

    return _create


@pytest.fixture
def sample_run(make_run: Callable[..., RunRecord]) -> RunRecord:
    """Single sample run with counts and tags."""
    return make_run(
        run_id="sample_run_001",
        project="sample_project",
        params={"shots": 1000, "optimization_level": 2},
        metrics={"fidelity": 0.95, "success_rate": 0.98},
        tags={"experiment": "bell", "validated": "true"},
        counts={"00": 500, "11": 500},
    )


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    """Ensure tests don't use real ~/.devqubit."""
    monkeypatch.setenv("DEVQUBIT_HOME", str(workspace))
