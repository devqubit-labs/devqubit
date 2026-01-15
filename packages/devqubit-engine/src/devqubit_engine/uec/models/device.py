# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot for capturing quantum backend state.

This module defines DeviceSnapshot for point-in-time capture of
backend configuration and calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.uec.models.calibration import DeviceCalibration


@dataclass
class FrontendConfig:
    """
    Frontend/primitive configuration for multi-layer SDK stacks.

    Captures configuration from high-level abstractions (e.g., PennyLane
    devices, Qiskit Runtime primitives) that sit above the physical backend.

    Parameters
    ----------
    name : str
        Frontend identifier (e.g., "SamplerV2", "braket.aws.qubit").
    sdk : str
        SDK/framework name (e.g., "qiskit_runtime", "pennylane").
    sdk_version : str, optional
        SDK version string.
    config : dict, optional
        Frontend-specific configuration options.
    """

    name: str
    sdk: str
    sdk_version: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "sdk": self.sdk}
        if self.sdk_version:
            d["sdk_version"] = self.sdk_version
        if self.config:
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrontendConfig:
        return cls(
            name=str(d.get("name", "")),
            sdk=str(d.get("sdk", "")),
            sdk_version=d.get("sdk_version"),
            config=d.get("config", {}),
        )


@dataclass
class DeviceSnapshot:
    """
    Point-in-time snapshot of a quantum backend and its calibration state.

    Captures the complete state of a quantum device at execution time
    for reproducibility and drift detection.

    Parameters
    ----------
    captured_at : str
        Snapshot capture timestamp (ISO 8601).
    backend_name : str
        Backend identifier (e.g., "ibm_brisbane", "Aspen-M-3").
    backend_type : str
        Backend type: "simulator", "hardware", or custom.
    provider : str
        Physical provider identifier (e.g., "ibm_quantum", "aws_braket").
    backend_id : str, optional
        Stable unique identifier (ARN, resource name).
    num_qubits : int, optional
        Number of qubits on the backend.
    connectivity : list of tuple, optional
        Edge list of connected qubit pairs.
    native_gates : list of str, optional
        Native gate names supported by the backend.
    calibration : DeviceCalibration, optional
        Calibration data bundle.
    frontend : FrontendConfig, optional
        Frontend configuration for multi-layer stacks.
    sdk_versions : dict, optional
        SDK version strings for all involved layers.
    raw_properties_ref : ArtifactRef, optional
        Reference to raw backend properties artifact.
    """

    captured_at: str
    backend_name: str
    backend_type: str
    provider: str

    backend_id: str | None = None
    num_qubits: int | None = None
    connectivity: list[tuple[int, int]] | None = None
    native_gates: list[str] | None = None

    calibration: DeviceCalibration | None = None
    frontend: FrontendConfig | None = None

    sdk_versions: dict[str, str] = field(default_factory=dict)
    raw_properties_ref: ArtifactRef | None = None

    schema_version: str = "devqubit.device_snapshot/1.0"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "captured_at": self.captured_at,
            "backend_name": self.backend_name,
            "backend_type": self.backend_type,
            "provider": self.provider,
        }

        if self.backend_id:
            d["backend_id"] = self.backend_id
        if self.num_qubits is not None:
            d["num_qubits"] = int(self.num_qubits)
        if self.connectivity is not None:
            d["connectivity"] = [list(edge) for edge in self.connectivity]
        if self.native_gates:
            d["native_gates"] = self.native_gates
        if self.calibration:
            d["calibration"] = self.calibration.to_dict()
        if self.frontend:
            d["frontend"] = self.frontend.to_dict()
        if self.sdk_versions:
            d["sdk_versions"] = self.sdk_versions
        if self.raw_properties_ref:
            d["raw_properties_ref"] = self.raw_properties_ref.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceSnapshot:
        calibration = None
        if isinstance(d.get("calibration"), dict):
            calibration = DeviceCalibration.from_dict(d["calibration"])

        frontend = None
        if isinstance(d.get("frontend"), dict):
            frontend = FrontendConfig.from_dict(d["frontend"])

        raw_properties_ref = None
        if isinstance(d.get("raw_properties_ref"), dict):
            raw_properties_ref = ArtifactRef.from_dict(d["raw_properties_ref"])

        connectivity = None
        if isinstance(d.get("connectivity"), list):
            connectivity = [tuple(edge) for edge in d["connectivity"]]

        return cls(
            captured_at=str(d.get("captured_at", "")),
            backend_name=str(d.get("backend_name", "")),
            backend_type=str(d.get("backend_type", "")),
            provider=str(d.get("provider", "")),
            backend_id=d.get("backend_id"),
            num_qubits=d.get("num_qubits"),
            connectivity=connectivity,
            native_gates=d.get("native_gates"),
            calibration=calibration,
            frontend=frontend,
            sdk_versions=d.get("sdk_versions", {}),
            raw_properties_ref=raw_properties_ref,
            schema_version=d.get("schema", "devqubit.device_snapshot/1.0"),
        )

    def get_calibration_summary(self) -> dict[str, Any] | None:
        """Get calibration summary metrics."""
        if not self.calibration:
            return None

        self.calibration.compute_medians()

        return {
            "calibration_time": self.calibration.calibration_time,
            "median_t1_us": self.calibration.median_t1_us,
            "median_t2_us": self.calibration.median_t2_us,
            "median_readout_error": self.calibration.median_readout_error,
            "median_2q_error": self.calibration.median_2q_error,
            "num_qubits": len(self.calibration.qubits),
            "num_gates": len(self.calibration.gates),
        }
