# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for devqubit_engine tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.core.config import Config
from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.local import LocalRegistry, LocalStore
from devqubit_engine.uec.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.result import CountsFormat
from devqubit_engine.uec.types import ArtifactRef


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory structure."""
    ws = tmp_path / ".devqubit"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "objects").mkdir(exist_ok=True)
    (ws / "runs").mkdir(exist_ok=True)
    return ws


@pytest.fixture
def config(workspace: Path) -> Config:
    """Create a Config object using the temp workspace as root_dir."""
    return Config(root_dir=workspace)


@pytest.fixture
def store(workspace: Path) -> LocalStore:
    """Create a local object store."""
    return LocalStore(workspace / "objects")


@pytest.fixture
def registry(workspace: Path) -> LocalRegistry:
    """Create a local registry."""
    return LocalRegistry(workspace)


@pytest.fixture
def factory_store(tmp_path: Path) -> Callable[[], LocalStore]:
    """Factory that creates new stores (for cross-workspace tests)."""
    counter = [0]

    def _create() -> LocalStore:
        counter[0] += 1
        path = tmp_path / f"workspace_{counter[0]}" / "objects"
        path.mkdir(parents=True, exist_ok=True)
        return LocalStore(path)

    return _create


@pytest.fixture
def factory_registry(tmp_path: Path) -> Callable[[], LocalRegistry]:
    """Factory that creates new registries (for cross-workspace tests)."""
    counter = [0]

    def _create() -> LocalRegistry:
        counter[0] += 1
        path = tmp_path / f"workspace_{counter[0]}"
        path.mkdir(parents=True, exist_ok=True)
        return LocalRegistry(path)

    return _create


# =============================================================================
# Run Record Fixture
# =============================================================================


@pytest.fixture
def run_factory() -> Callable[..., RunRecord]:
    """Factory fixture for creating customizable run records."""

    def _make_run_record(
        run_id: str = "TEST123456",
        project: str = "test_project",
        adapter: str = "test_adapter",
        backend_name: str = "test_backend",
        backend_type: str = "simulator",
        provider: str = "test_provider",
        status: str = "FINISHED",
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        group_id: str | None = None,
        group_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> RunRecord:
        record: dict[str, Any] = {
            "schema": "devqubit.run/0.1",
            "run_id": run_id,
            "created_at": "2024-01-01T00:00:00Z",
            "project": {"name": project},
            "adapter": adapter,
            "info": {"status": status},
            "data": {
                "params": params or {},
                "metrics": metrics or {},
                "tags": tags or {},
            },
            "backend": {
                "name": backend_name,
                "type": backend_type,
                "provider": provider,
            },
            "fingerprints": {
                "run": f"sha256:{hashlib.sha256(run_id.encode()).hexdigest()}",
                "program": f"sha256:{hashlib.sha256(b'program').hexdigest()}",
            },
        }

        if group_id:
            record["group_id"] = group_id
        if group_name:
            record["group_name"] = group_name
        if parent_run_id:
            record["parent_run_id"] = parent_run_id

        return RunRecord(record=record, artifacts=artifacts or [])

    return _make_run_record


# =============================================================================
# Artifact Fixtures
# =============================================================================


@pytest.fixture
def artifact_factory(store: LocalStore) -> Callable[..., ArtifactRef]:
    """Factory for creating artifacts stored in the test store."""

    def _create(
        data: bytes,
        kind: str,
        role: str = "test",
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        digest = store.put_bytes(data)
        return ArtifactRef(
            kind=kind,
            digest=digest,
            media_type=media_type,
            role=role,
        )

    return _create


@pytest.fixture
def bell_state_counts() -> dict[str, int]:
    """Ideal Bell state measurement counts (1000 shots)."""
    return {"00": 500, "11": 500}


@pytest.fixture
def counts_artifact_factory(
    store: LocalStore,
) -> Callable[[dict[str, int]], ArtifactRef]:
    """Factory for creating counts artifacts."""

    def _create(counts: dict[str, int]) -> ArtifactRef:
        data = json.dumps({"counts": counts}).encode("utf-8")
        digest = store.put_bytes(data)
        return ArtifactRef(
            kind="result.counts.json",
            digest=digest,
            media_type="application/json",
            role="results",
        )

    return _create


# =============================================================================
# Calibration Fixtures
# =============================================================================


@pytest.fixture
def calibration_factory() -> Callable[..., DeviceCalibration]:
    """Factory for creating device calibrations."""

    def _create(
        num_qubits: int = 5,
        calibration_time: str = "2024-01-01T10:00:00Z",
        t1_base: float = 100.0,
        t2_base: float = 80.0,
    ) -> DeviceCalibration:
        qubits = [
            QubitCalibration(
                qubit=i,
                t1_us=t1_base + (i * 5),
                t2_us=t2_base + (i * 3),
                readout_error=0.01 + (i * 0.002),
                gate_error_1q=0.001,
            )
            for i in range(num_qubits)
        ]
        gates = [
            GateCalibration(gate="cx", qubits=(i, i + 1), error=0.01 + (i * 0.002))
            for i in range(num_qubits - 1)
        ]
        cal = DeviceCalibration(
            calibration_time=calibration_time,
            qubits=qubits,
            gates=gates,
        )
        cal.compute_medians()
        return cal

    return _create


@pytest.fixture
def snapshot_factory() -> Callable[..., DeviceSnapshot]:
    """Factory for creating device snapshots."""

    def _create(
        backend_name: str = "test_backend",
        backend_type: str = "simulator",
        provider: str = "test_provider",
        calibration: DeviceCalibration | None = None,
        num_qubits: int | None = None,
    ) -> DeviceSnapshot:
        return DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name=backend_name,
            backend_type=backend_type,
            provider=provider,
            num_qubits=num_qubits,
            calibration=calibration,
        )

    return _create


@pytest.fixture
def minimal_snapshot(snapshot_factory) -> DeviceSnapshot:
    """A minimal device snapshot without calibration."""
    return snapshot_factory()


@pytest.fixture
def calibrated_snapshot(snapshot_factory, calibration_factory) -> DeviceSnapshot:
    """A device snapshot with full calibration data."""
    return snapshot_factory(
        backend_name="ibm_test",
        calibration=calibration_factory(num_qubits=5),
    )


# =============================================================================
# UEC Result Fixtures
# =============================================================================


@pytest.fixture
def qiskit_counts_format() -> CountsFormat:
    """Standard Qiskit counts format (canonical bit order)."""
    return CountsFormat(
        source_sdk="qiskit",
        source_key_format="qiskit_little_endian",
        bit_order="cbit0_right",
        transformed=False,
    )


@pytest.fixture
def braket_counts_format() -> CountsFormat:
    """Braket counts format (transformed to canonical)."""
    return CountsFormat(
        source_sdk="braket",
        source_key_format="braket_big_endian",
        bit_order="cbit0_right",
        transformed=True,
    )


@pytest.fixture
def minimal_producer() -> ProducerInfo:
    """Minimal valid ProducerInfo for testing."""
    return ProducerInfo.create(
        adapter="devqubit-test",
        adapter_version="0.1.0",
        sdk="test-sdk",
        sdk_version="1.0.0",
        frontends=["test-sdk"],
    )


# =============================================================================
# Circuit Fixtures
# =============================================================================


@pytest.fixture
def bell_qasm2() -> str:
    """Bell state circuit in OpenQASM 2.0."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""


@pytest.fixture
def ghz_qasm2() -> str:
    """3-qubit GHZ state in OpenQASM 2.0."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
"""


@pytest.fixture
def circuit_data_factory() -> Callable[..., CircuitData]:
    """Factory for creating CircuitData objects."""

    def _create(
        data: str | bytes,
        format: CircuitFormat = CircuitFormat.OPENQASM2,
        sdk: SDK = SDK.QISKIT,
        name: str | None = None,
    ) -> CircuitData:
        return CircuitData(data=data, format=format, sdk=sdk, name=name)

    return _create
