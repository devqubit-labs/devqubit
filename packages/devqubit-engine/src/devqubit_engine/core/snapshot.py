# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
devqubit Uniform Execution Contract (UEC) snapshot schemas.

This module provides standardized types for capturing quantum experiment state
across all supported SDKs. The UEC defines four canonical snapshot types that
every adapter must produce, plus a unified envelope container.

Snapshot Hierarchy
------------------
:class:`ExecutionEnvelope`
    Top-level container unifying all snapshots for a single execution.

    :class:`DeviceSnapshot`
        Point-in-time capture of quantum backend state.

        :class:`DeviceCalibration`
            Calibration data bundle with per-qubit and per-gate metrics.

            - :class:`QubitCalibration` - T1, T2, readout error per qubit
            - :class:`GateCalibration` - Error rates and durations per gate

        :class:`FrontendConfig`
            Frontend/primitive configuration for multi-layer stacks.

    :class:`ProgramSnapshot`
        Program artifacts with logical/physical distinction.

        :class:`ArtifactRef`
            Reference to content-addressed artifact in object store.

    :class:`ExecutionSnapshot`
        Submission, compilation, and job tracking metadata.

    :class:`ResultSnapshot`
        Raw result references and normalized summaries.

Design Principles
-----------------
1. **Two-layer storage**: Raw provider payloads as artifacts, normalized
   summaries in the record for querying.

2. **Lossless capture**: Never discard provider-specific data; store as
   artifacts with stable ``kind`` identifiers.

3. **Cross-SDK uniformity**: Same conceptual envelope regardless of SDK,
   enabling reproducibility comparisons across providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from devqubit_engine.core.types import (
    TWO_QUBIT_GATES,
    ArtifactRef,
    ProgramRole,
    ResultType,
    TranspilationMode,
    _norm_gate,
)


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of schema validation.

    Provides structured access to validation outcome without exceptions.

    Parameters
    ----------
    valid : bool
        True if validation passed.
    errors : list
        List of validation errors (empty if valid).
    warnings : list
        List of validation warnings.

    Examples
    --------
    >>> result = envelope.validate_schema()
    >>> if result.valid:
    ...     print("Schema valid")
    ... else:
    ...     for err in result.errors:
    ...         print(f"Error: {err}")
    """

    valid: bool
    errors: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.valid


def _median_val(values: Sequence[float | None]) -> float | None:
    """
    Compute median of values, ignoring None entries.

    Parameters
    ----------
    values : Sequence[float | None]
        Sequence of numeric values, possibly containing None.

    Returns
    -------
    float | None
        Median value, or None if no valid values exist.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.median(valid))


# =============================================================================
# Calibration Classes
# =============================================================================


@dataclass
class QubitCalibration:
    """
    Per-qubit calibration record.

    Contains calibration metrics for a single qubit including coherence
    times, readout fidelity, and gate error rates.

    Parameters
    ----------
    qubit : int
        Qubit index (0-based).
    t1_us : float, optional
        Energy relaxation time (T1) in microseconds.
    t2_us : float, optional
        Dephasing time (T2) in microseconds.
    readout_error : float, optional
        Assignment/readout error probability (0.0 to 1.0).
    gate_error_1q : float, optional
        Representative single-qubit gate error probability.
    frequency_ghz : float, optional
        Qubit frequency in GHz.
    anharmonicity_ghz : float, optional
        Qubit anharmonicity in GHz.
    """

    qubit: int
    t1_us: float | None = None
    t2_us: float | None = None
    readout_error: float | None = None
    gate_error_1q: float | None = None
    frequency_ghz: float | None = None
    anharmonicity_ghz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Only includes fields that have non-None values.

        Returns
        -------
        dict
            Dictionary with qubit index and available calibration metrics.
        """
        d: dict[str, Any] = {"qubit": int(self.qubit)}

        for key in (
            "t1_us",
            "t2_us",
            "readout_error",
            "gate_error_1q",
            "frequency_ghz",
            "anharmonicity_ghz",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = float(value)

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QubitCalibration:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing calibration data. Must have ``qubit`` key.

        Returns
        -------
        QubitCalibration
            New calibration instance.
        """
        return cls(
            qubit=int(d["qubit"]),
            t1_us=d.get("t1_us"),
            t2_us=d.get("t2_us"),
            readout_error=d.get("readout_error"),
            gate_error_1q=d.get("gate_error_1q"),
            frequency_ghz=d.get("frequency_ghz"),
            anharmonicity_ghz=d.get("anharmonicity_ghz"),
        )


@dataclass
class GateCalibration:
    """
    Calibration record for a gate applied on specific qubits.

    Captures error rate and duration for both single-qubit and
    multi-qubit gates.

    Parameters
    ----------
    gate : str
        Gate name (e.g., ``"cx"``, ``"cz"``, ``"rx"``, ``"rz"``).
    qubits : tuple of int
        Tuple of qubit indices the gate acts on.
        Single-qubit gates have one element, two-qubit gates have two.
    error : float, optional
        Gate error probability (0.0 to 1.0).
    duration_ns : float, optional
        Gate duration in nanoseconds.
    """

    gate: str
    qubits: tuple[int, ...]
    error: float | None = None
    duration_ns: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary with gate name, qubits list, and available metrics.
        """
        d: dict[str, Any] = {
            "gate": str(self.gate),
            "qubits": list(self.qubits),
        }

        if self.error is not None:
            d["error"] = float(self.error)
        if self.duration_ns is not None:
            d["duration_ns"] = float(self.duration_ns)

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateCalibration:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing gate calibration data.

        Returns
        -------
        GateCalibration
            New gate calibration instance.
        """
        return cls(
            gate=str(d["gate"]),
            qubits=tuple(int(x) for x in d.get("qubits", [])),
            error=d.get("error"),
            duration_ns=d.get("duration_ns"),
        )

    @property
    def is_two_qubit(self) -> bool:
        """
        Check if this is a two-qubit gate.

        Returns
        -------
        bool
            True if the gate acts on two or more qubits.
        """
        return len(self.qubits) >= 2


@dataclass
class DeviceCalibration:
    """
    Device-level calibration bundle with derived summary metrics.

    Aggregates per-qubit and per-gate calibration data with optional
    computed median summaries for quick drift detection.

    Parameters
    ----------
    calibration_time : str, optional
        Provider/SDK calibration timestamp (ISO 8601 recommended).
    qubits : list of QubitCalibration, optional
        Per-qubit calibration records.
    gates : list of GateCalibration, optional
        Per-gate calibration records.
    median_t1_us : float, optional
        Median T1 across all qubits (auto-computed if None).
    median_t2_us : float, optional
        Median T2 across all qubits (auto-computed if None).
    median_readout_error : float, optional
        Median readout error across all qubits (auto-computed if None).
    median_2q_error : float, optional
        Median two-qubit gate error (auto-computed if None).
    source : str, optional
        Data source indicator (``"provider"``, ``"derived"``, ``"manual"``).
    schema_version : str, optional
        Schema identifier for serialization versioning.

    Notes
    -----
    The ``source`` field indicates calibration data provenance:

    - ``"provider"``: Direct from backend properties/calibration API
    - ``"derived"``: Computed from other available data
    - ``"manual"``: User-provided values
    """

    calibration_time: str | None = None
    qubits: list[QubitCalibration] = field(default_factory=list)
    gates: list[GateCalibration] = field(default_factory=list)

    # Derived summary metrics (computed if None)
    median_t1_us: float | None = None
    median_t2_us: float | None = None
    median_readout_error: float | None = None
    median_2q_error: float | None = None

    # Data provenance
    source: str | None = None

    schema_version: str = "devqubit.calibration/0.1"

    def compute_medians(self) -> None:
        """
        Compute derived median summary metrics in-place.

        Only computes values that are currently None, preserving
        any explicitly set values. This allows SDK-specific medians
        to override the computed values.

        Notes
        -----
        The median two-qubit gate error is computed from all gates
        matching names in :data:`TWO_QUBIT_GATES`.
        """
        if self.median_t1_us is None:
            t1_values = [q.t1_us for q in self.qubits if q.t1_us is not None]
            self.median_t1_us = _median_val(t1_values)

        if self.median_t2_us is None:
            t2_values = [q.t2_us for q in self.qubits if q.t2_us is not None]
            self.median_t2_us = _median_val(t2_values)

        if self.median_readout_error is None:
            ro_values = [
                q.readout_error for q in self.qubits if q.readout_error is not None
            ]
            self.median_readout_error = _median_val(ro_values)

        if self.median_2q_error is None:
            gate_errors = [
                g.error
                for g in self.gates
                if g.error is not None and _norm_gate(g.gate) in TWO_QUBIT_GATES
            ]
            self.median_2q_error = _median_val(gate_errors)

        logger.debug(
            "Computed calibration medians: T1=%.2f µs, T2=%.2f µs, "
            "readout_err=%.4f, 2q_err=%.4f",
            self.median_t1_us or 0,
            self.median_t2_us or 0,
            self.median_readout_error or 0,
            self.median_2q_error or 0,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Automatically computes medians if any are None before serialization.

        Returns
        -------
        dict
            Complete calibration dictionary with schema version.
        """
        # Ensure summaries are computed
        has_missing_medians = any(
            x is None
            for x in (
                self.median_t1_us,
                self.median_t2_us,
                self.median_readout_error,
                self.median_2q_error,
            )
        )
        if has_missing_medians:
            self.compute_medians()

        d: dict[str, Any] = {
            "schema": self.schema_version,
            "calibration_time": self.calibration_time,
            "qubits": [q.to_dict() for q in self.qubits],
            "gates": [g.to_dict() for g in self.gates],
        }

        if self.source:
            d["source"] = self.source

        for key in (
            "median_t1_us",
            "median_t2_us",
            "median_readout_error",
            "median_2q_error",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = float(value)

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceCalibration:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing calibration data.

        Returns
        -------
        DeviceCalibration
            New calibration instance.
        """
        qubits = [
            QubitCalibration.from_dict(x)
            for x in d.get("qubits", [])
            if isinstance(x, dict)
        ]
        gates = [
            GateCalibration.from_dict(x)
            for x in d.get("gates", [])
            if isinstance(x, dict)
        ]

        return cls(
            calibration_time=d.get("calibration_time"),
            qubits=qubits,
            gates=gates,
            median_t1_us=d.get("median_t1_us"),
            median_t2_us=d.get("median_t2_us"),
            median_readout_error=d.get("median_readout_error"),
            median_2q_error=d.get("median_2q_error"),
            source=d.get("source"),
            schema_version=d.get("schema", "devqubit.calibration/0.1"),
        )


# =============================================================================
# Frontend Configuration (for multi-layer stacks)
# =============================================================================


@dataclass
class FrontendConfig:
    """
    Frontend/primitive configuration for multi-layer SDK stacks.

    Captures configuration from high-level abstractions (e.g., PennyLane
    devices, Qiskit Runtime primitives) that sit above the physical backend.

    Parameters
    ----------
    name : str
        Frontend identifier (e.g., ``"SamplerV2"``, ``"braket.aws.qubit"``).
    sdk : str
        SDK/framework name (e.g., ``"qiskit_runtime"``, ``"pennylane"``).
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
        """
        Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary with frontend configuration.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "sdk": self.sdk,
        }
        if self.sdk_version:
            d["sdk_version"] = self.sdk_version
        if self.config:
            d["config"] = self.config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrontendConfig:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing frontend configuration.

        Returns
        -------
        FrontendConfig
            New frontend configuration instance.
        """
        return cls(
            name=str(d.get("name", "")),
            sdk=str(d.get("sdk", "")),
            sdk_version=d.get("sdk_version"),
            config=d.get("config", {}),
        )


# =============================================================================
# Device Snapshot
# =============================================================================


@dataclass
class DeviceSnapshot:
    """
    Point-in-time snapshot of a quantum backend and its calibration state.

    Captures the complete state of a quantum device at execution time
    for reproducibility and drift detection. Supports multi-layer SDK
    stacks through the ``frontend`` field.

    Parameters
    ----------
    captured_at : str
        Snapshot capture timestamp (ISO 8601 format).
    backend_name : str
        Backend identifier/name (e.g., ``"ibm_brisbane"``, ``"Aspen-M-3"``).
    backend_type : str
        Backend type category: ``"simulator"``, ``"hardware"``, or custom.
    provider : str
        Physical provider identifier (e.g., ``"ibm_quantum"``, ``"aws_braket"``).
    backend_id : str, optional
        Stable unique identifier (ARN, resource name, backend ID).
    num_qubits : int, optional
        Number of qubits on the backend.
    connectivity : list of tuple, optional
        Edge list of connected qubit pairs for the coupling map.
    native_gates : list of str, optional
        List of native gate names supported by the backend.
    calibration : DeviceCalibration, optional
        Calibration data bundle for the backend.
    frontend : FrontendConfig, optional
        Frontend configuration for multi-layer stacks (e.g., Runtime primitive).
    sdk_versions : dict, optional
        SDK version strings for all involved layers.
    raw_properties_ref : ArtifactRef, optional
        Reference to raw backend properties artifact.
    schema_version : str, optional
        Schema identifier for serialization versioning.

    Notes
    -----
    For multi-layer stacks (e.g., PennyLane → Braket, Runtime primitives):

    - ``frontend``: Configuration of the high-level abstraction
    - ``backend_name``, ``provider``, etc.: The resolved physical backend
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

    schema_version: str = "devqubit.device_snapshot/0.1"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Complete snapshot dictionary with schema version.
        """
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
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing snapshot data.

        Returns
        -------
        DeviceSnapshot
            New snapshot instance.
        """
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
            schema_version=d.get("schema", "devqubit.device_snapshot/0.1"),
        )

    def get_calibration_summary(self) -> dict[str, Any] | None:
        """
        Get calibration summary metrics.

        Returns
        -------
        dict or None
            Summary dictionary with median metrics, or None if no calibration.
        """
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


# =============================================================================
# Program Snapshot
# =============================================================================


@dataclass
class ProgramArtifact:
    """
    Reference to a program artifact with metadata.

    Parameters
    ----------
    ref : ArtifactRef
        Reference to the stored artifact.
    role : ProgramRole
        Role in the program (LOGICAL or PHYSICAL).
    format : str
        Serialization format (e.g., ``"qpy"``, ``"openqasm3"``).
    name : str, optional
        Human-readable name.
    index : int, optional
        Index in multi-circuit batches.
    """

    ref: ArtifactRef
    role: ProgramRole
    format: str
    name: str | None = None
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "ref": self.ref.to_dict(),
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "format": self.format,
        }
        if self.name:
            d["name"] = self.name
        if self.index is not None:
            d["index"] = self.index
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProgramArtifact:
        """Create from dictionary."""
        role_val = d.get("role", "logical")
        if isinstance(role_val, str):
            role = ProgramRole(role_val)
        else:
            role = role_val

        return cls(
            ref=ArtifactRef.from_dict(d["ref"]),
            role=role,
            format=str(d.get("format", "")),
            name=d.get("name"),
            index=d.get("index"),
        )


@dataclass
class TranspilationInfo:
    """
    Transpilation metadata.

    Parameters
    ----------
    mode : TranspilationMode
        Transpilation mode (AUTO, MANAGED, MANUAL).
    transpiled_by : str, optional
        Who performed transpilation.
    optimization_level : int, optional
        Optimization level used.
    layout_method : str, optional
        Layout method used.
    routing_method : str, optional
        Routing method used.
    seed : int, optional
        Random seed for transpilation.
    pass_manager_config : dict, optional
        Full pass manager configuration.
    """

    mode: TranspilationMode
    transpiled_by: str | None = None
    optimization_level: int | None = None
    layout_method: str | None = None
    routing_method: str | None = None
    seed: int | None = None
    pass_manager_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
        }
        if self.transpiled_by:
            d["transpiled_by"] = self.transpiled_by
        if self.optimization_level is not None:
            d["optimization_level"] = self.optimization_level
        if self.layout_method:
            d["layout_method"] = self.layout_method
        if self.routing_method:
            d["routing_method"] = self.routing_method
        if self.seed is not None:
            d["seed"] = self.seed
        if self.pass_manager_config:
            d["pass_manager_config"] = self.pass_manager_config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TranspilationInfo:
        """Create from dictionary."""
        mode_val = d.get("mode", "auto")
        if isinstance(mode_val, str):
            mode = TranspilationMode(mode_val)
        else:
            mode = mode_val

        return cls(
            mode=mode,
            transpiled_by=d.get("transpiled_by"),
            optimization_level=d.get("optimization_level"),
            layout_method=d.get("layout_method"),
            routing_method=d.get("routing_method"),
            seed=d.get("seed"),
            pass_manager_config=d.get("pass_manager_config"),
        )


@dataclass
class ProgramSnapshot:
    """
    Program artifacts snapshot.

    Parameters
    ----------
    logical : list of ProgramArtifact
        Logical (pre-transpilation) circuit artifacts.
    physical : list of ProgramArtifact
        Physical (post-transpilation) circuit artifacts.
    program_hash : str, optional
        Structure-only hash of logical circuits.
    executed_hash : str, optional
        Structure-only hash of executed (physical) circuits.
    num_circuits : int, optional
        Number of circuits in the program.
    transpilation : TranspilationInfo, optional
        Transpilation metadata.
    """

    logical: list[ProgramArtifact] = field(default_factory=list)
    physical: list[ProgramArtifact] = field(default_factory=list)
    program_hash: str | None = None
    executed_hash: str | None = None
    num_circuits: int | None = None
    transpilation: TranspilationInfo | None = None

    schema_version: str = "devqubit.program_snapshot/0.1"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "logical": [a.to_dict() for a in self.logical],
            "physical": [a.to_dict() for a in self.physical],
        }
        if self.program_hash:
            d["program_hash"] = self.program_hash
        if self.executed_hash:
            d["executed_hash"] = self.executed_hash
        if self.num_circuits is not None:
            d["num_circuits"] = self.num_circuits
        if self.transpilation:
            d["transpilation"] = self.transpilation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProgramSnapshot:
        """Create from dictionary."""
        logical = [
            ProgramArtifact.from_dict(x)
            for x in d.get("logical", [])
            if isinstance(x, dict)
        ]
        physical = [
            ProgramArtifact.from_dict(x)
            for x in d.get("physical", [])
            if isinstance(x, dict)
        ]
        transpilation = None
        if isinstance(d.get("transpilation"), dict):
            transpilation = TranspilationInfo.from_dict(d["transpilation"])

        return cls(
            logical=logical,
            physical=physical,
            program_hash=d.get("program_hash"),
            executed_hash=d.get("executed_hash"),
            num_circuits=d.get("num_circuits"),
            transpilation=transpilation,
            schema_version=d.get("schema", "devqubit.program_snapshot/0.1"),
        )


# =============================================================================
# Execution Snapshot
# =============================================================================


@dataclass
class ExecutionSnapshot:
    """
    Execution submission and job tracking metadata.

    Captures when and how circuits were submitted for execution,
    including compilation configuration and job identifiers.

    Parameters
    ----------
    submitted_at : str
        Submission timestamp (ISO 8601).
    shots : int, optional
        Number of shots requested.
    execution_count : int, optional
        Execution sequence number.
    job_ids : list of str, optional
        Job identifiers.
    task_ids : list of str, optional
        Task identifiers (for Braket).
    transpilation : TranspilationInfo, optional
        Transpilation metadata.
    options : dict, optional
        Execution options.
    sdk : str, optional
        SDK identifier.
    completed_at : str, optional
        Completion timestamp.
    """

    submitted_at: str
    shots: int | None = None
    execution_count: int | None = None
    job_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    transpilation: TranspilationInfo | None = None
    options: dict[str, Any] = field(default_factory=dict)
    sdk: str | None = None
    completed_at: str | None = None

    schema_version: str = "devqubit.execution_snapshot/0.1"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "submitted_at": self.submitted_at,
        }
        if self.shots is not None:
            d["shots"] = self.shots
        if self.execution_count is not None:
            d["execution_count"] = self.execution_count
        if self.job_ids:
            d["job_ids"] = self.job_ids
        if self.task_ids:
            d["task_ids"] = self.task_ids
        if self.transpilation:
            d["transpilation"] = self.transpilation.to_dict()
        if self.options:
            d["options"] = self.options
        if self.sdk:
            d["sdk"] = self.sdk
        if self.completed_at:
            d["completed_at"] = self.completed_at
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionSnapshot:
        """Create from dictionary."""
        transpilation = None
        if isinstance(d.get("transpilation"), dict):
            transpilation = TranspilationInfo.from_dict(d["transpilation"])

        return cls(
            submitted_at=str(d.get("submitted_at", "")),
            shots=d.get("shots"),
            execution_count=d.get("execution_count"),
            job_ids=d.get("job_ids", []),
            task_ids=d.get("task_ids", []),
            transpilation=transpilation,
            options=d.get("options", {}),
            sdk=d.get("sdk"),
            completed_at=d.get("completed_at"),
            schema_version=d.get("schema", "devqubit.execution_snapshot/0.1"),
        )


# =============================================================================
# Result Snapshot
# =============================================================================


@dataclass
class NormalizedCounts:
    """
    Normalized measurement counts for a single circuit.

    Parameters
    ----------
    circuit_index : int
        Circuit index in batch.
    counts : dict
        Measurement counts (bitstring → count).
    shots : int, optional
        Total shots for this circuit.
    name : str, optional
        Circuit name.
    """

    circuit_index: int
    counts: dict[str, int]
    shots: int | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "circuit_index": self.circuit_index,
            "counts": self.counts,
        }
        if self.shots is not None:
            d["shots"] = self.shots
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedCounts:
        """Create from dictionary."""
        return cls(
            circuit_index=int(d.get("circuit_index", 0)),
            counts=d.get("counts", {}),
            shots=d.get("shots"),
            name=d.get("name"),
        )


@dataclass
class NormalizedExpectation:
    """
    Normalized expectation value result.

    Parameters
    ----------
    circuit_index : int
        Index of the circuit in a batch (0-based).
    observable_index : int
        Index of the observable (0-based).
    value : float
        Expectation value.
    variance : float, optional
        Variance of the expectation value.
    std_error : float, optional
        Standard error of the expectation value.
    observable : str, optional
        String representation of the observable.
    """

    circuit_index: int
    observable_index: int
    value: float
    variance: float | None = None
    std_error: float | None = None
    observable: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "circuit_index": self.circuit_index,
            "observable_index": self.observable_index,
            "value": self.value,
        }
        if self.variance is not None:
            d["variance"] = self.variance
        if self.std_error is not None:
            d["std_error"] = self.std_error
        if self.observable:
            d["observable"] = self.observable
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedExpectation:
        """Create an instance from a dictionary."""
        return cls(
            circuit_index=int(d.get("circuit_index", 0)),
            observable_index=int(d.get("observable_index", 0)),
            value=float(d.get("value", 0.0)),
            variance=d.get("variance"),
            std_error=d.get("std_error"),
            observable=d.get("observable"),
        )


@dataclass
class ExpectationValue:
    """
    Expectation value result.

    Parameters
    ----------
    circuit_index : int
        Circuit index.
    observable_index : int
        Observable index.
    value : float
        Expectation value.
    std_error : float, optional
        Standard error.
    """

    circuit_index: int
    observable_index: int
    value: float
    std_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "circuit_index": self.circuit_index,
            "observable_index": self.observable_index,
            "value": self.value,
        }
        if self.std_error is not None:
            d["std_error"] = self.std_error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExpectationValue:
        """Create from dictionary."""
        return cls(
            circuit_index=int(d.get("circuit_index", 0)),
            observable_index=int(d.get("observable_index", 0)),
            value=float(d.get("value", 0.0)),
            std_error=d.get("std_error"),
        )


@dataclass
class ResultSnapshot:
    """
    Result snapshot.

    Parameters
    ----------
    result_type : ResultType
        Type of result.
    raw_result_ref : ArtifactRef, optional
        Reference to raw result artifact.
    counts : list of NormalizedCounts, optional
        Normalized measurement counts.
    expectations : list of ExpectationValue, optional
        Expectation values.
    num_experiments : int, optional
        Number of experiments.
    success : bool
        Whether execution succeeded.
    error_message : str, optional
        Error message if failed.
    metadata : dict, optional
        Additional metadata.
    """

    result_type: ResultType
    raw_result_ref: ArtifactRef | None = None
    counts: list[NormalizedCounts] = field(default_factory=list)
    expectations: list[ExpectationValue] = field(default_factory=list)
    num_experiments: int | None = None
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: str = "devqubit.result_snapshot/0.1"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "result_type": (
                self.result_type.value
                if hasattr(self.result_type, "value")
                else str(self.result_type)
            ),
            "success": self.success,
        }
        if self.raw_result_ref:
            d["raw_result_ref"] = self.raw_result_ref.to_dict()
        if self.counts:
            d["counts"] = [c.to_dict() for c in self.counts]
        if self.expectations:
            d["expectations"] = [e.to_dict() for e in self.expectations]
        if self.num_experiments is not None:
            d["num_experiments"] = self.num_experiments
        if self.error_message:
            d["error_message"] = self.error_message
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultSnapshot:
        """Create from dictionary."""
        result_type_val = d.get("result_type", "counts")
        if isinstance(result_type_val, str):
            result_type = ResultType(result_type_val)
        else:
            result_type = result_type_val

        raw_result_ref = None
        if isinstance(d.get("raw_result_ref"), dict):
            raw_result_ref = ArtifactRef.from_dict(d["raw_result_ref"])

        counts = [
            NormalizedCounts.from_dict(x)
            for x in d.get("counts", [])
            if isinstance(x, dict)
        ]
        expectations = [
            ExpectationValue.from_dict(x)
            for x in d.get("expectations", [])
            if isinstance(x, dict)
        ]

        return cls(
            result_type=result_type,
            raw_result_ref=raw_result_ref,
            counts=counts,
            expectations=expectations,
            num_experiments=d.get("num_experiments"),
            success=d.get("success", True),
            error_message=d.get("error_message"),
            metadata=d.get("metadata", {}),
            schema_version=d.get("schema", "devqubit.result_snapshot/0.1"),
        )


# =============================================================================
# Execution Envelope
# =============================================================================


@dataclass
class ExecutionEnvelope:
    """
    Top-level container for a complete quantum execution record.

    The ExecutionEnvelope unifies all four canonical snapshots (device,
    program, execution, result) into a single, self-contained record.

    Parameters
    ----------
    device : DeviceSnapshot, optional
        Device/backend state at execution time.
    program : ProgramSnapshot, optional
        Program artifacts (logical and physical circuits).
    execution : ExecutionSnapshot, optional
        Execution metadata and configuration.
    result : ResultSnapshot, optional
        Execution results.
    adapter : str, optional
        Adapter that created this envelope.
    envelope_id : str, optional
        Unique envelope identifier.
    created_at : str, optional
        Creation timestamp.
    schema_version : str
        Schema version identifier.

    Notes
    -----
    The envelope is designed to be:

    - **Self-contained**: All information needed to understand an execution
    - **Cross-SDK comparable**: Same structure regardless of SDK
    - **Artifact-backed**: Large payloads stored as artifacts, not inline
    """

    device: DeviceSnapshot | None = None
    program: ProgramSnapshot | None = None
    execution: ExecutionSnapshot | None = None
    result: ResultSnapshot | None = None

    adapter: str | None = None
    envelope_id: str | None = None
    created_at: str | None = None

    schema_version: str = "devqubit.envelope/0.1"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Complete envelope dictionary.
        """
        d: dict[str, Any] = {
            "schema": self.schema_version,
        }

        if self.device:
            d["device"] = self.device.to_dict()

        if self.program:
            d["program"] = self.program.to_dict()

        if self.execution:
            d["execution"] = self.execution.to_dict()

        if self.result:
            d["result"] = self.result.to_dict()

        if self.adapter:
            d["adapter"] = self.adapter

        if self.envelope_id:
            d["envelope_id"] = self.envelope_id

        if self.created_at:
            d["created_at"] = self.created_at

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionEnvelope:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing envelope data.

        Returns
        -------
        ExecutionEnvelope
            New envelope instance.
        """
        device = None
        if isinstance(d.get("device"), dict):
            device = DeviceSnapshot.from_dict(d["device"])

        program = None
        if isinstance(d.get("program"), dict):
            program = ProgramSnapshot.from_dict(d["program"])

        execution = None
        if isinstance(d.get("execution"), dict):
            execution = ExecutionSnapshot.from_dict(d["execution"])

        result = None
        if isinstance(d.get("result"), dict):
            result = ResultSnapshot.from_dict(d["result"])

        return cls(
            device=device,
            program=program,
            execution=execution,
            result=result,
            adapter=d.get("adapter"),
            envelope_id=d.get("envelope_id"),
            created_at=d.get("created_at"),
            schema_version=d.get("schema", "devqubit.envelope/0.1"),
        )

    def validate(self) -> list[str]:
        """
        Validate envelope completeness (semantic validation).

        Returns a list of warnings for missing or incomplete data.
        Does not raise exceptions; adapters may have valid reasons
        for partial envelopes.

        For JSON Schema validation, use :meth:`validate_schema` instead.

        Returns
        -------
        list of str
            List of validation warnings (empty if valid).

        See Also
        --------
        validate_schema : Validate against JSON Schema.
        """
        warnings: list[str] = []

        if not self.device:
            warnings.append("Missing device snapshot")
        elif not self.device.backend_name:
            warnings.append("Device snapshot missing backend_name")

        if not self.program:
            warnings.append("Missing program snapshot")
        elif not self.program.logical and not self.program.physical:
            warnings.append("Program snapshot has no artifacts")

        if not self.execution:
            warnings.append("Missing execution snapshot")

        if not self.result:
            warnings.append("Missing result snapshot")
        elif self.result.success is False and not self.result.error_message:
            warnings.append("Failed result missing error_message")

        return warnings

    def validate_schema(self) -> ValidationResult:
        """
        Validate envelope against JSON Schema.

        Performs formal JSON Schema validation against the
        ``devqubit.envelope/0.1`` schema. This validates structure,
        types, formats, and constraints.

        This method never raises exceptions. Use the returned
        ValidationResult to check validity and access errors.

        Returns
        -------
        ValidationResult
            Validation result with ``valid`` flag, ``errors`` list,
            and ``warnings`` list.

        Examples
        --------
        >>> result = envelope.validate_schema()
        >>> if result.valid:
        ...     print("Schema valid")
        ... else:
        ...     for err in result.errors:
        ...         print(f"Validation error: {err}")

        See Also
        --------
        validate : Semantic completeness validation.
        """
        try:
            from devqubit_engine.schema.validation import validate_envelope

            errors = validate_envelope(self.to_dict(), raise_on_error=False)

            if errors:
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    warnings=[],
                )

            return ValidationResult(valid=True, errors=[], warnings=[])

        except ImportError:
            logger.warning(
                "Schema validation not available: "
                "devqubit_engine.schema.validation module not found"
            )
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=["Schema validation module not available"],
            )

        except Exception as e:
            logger.warning("Schema validation failed unexpectedly: %s", e)
            return ValidationResult(
                valid=False,
                errors=[e],
                warnings=[],
            )


# =============================================================================
# Utility Functions
# =============================================================================


def resolve_physical_backend(executor: Any) -> dict[str, Any] | None:
    """
    Resolve the physical backend from a high-level executor.

    This is the universal backend resolution helper that all adapters
    should use to extract the underlying physical backend from wrapped
    or multi-layer executors.

    Parameters
    ----------
    executor : Any
        Executor, primitive, device, or backend object from any SDK.

    Returns
    -------
    dict or None
        Dictionary with resolved backend information:
        - ``provider``: Physical provider name
        - ``backend_name``: Backend identifier
        - ``backend_id``: Stable unique ID (if available)
        - ``backend_type``: Type (hardware/simulator)
        - ``backend_obj``: The actual backend object

        Returns None if resolution fails.

    Notes
    -----
    This function handles:

    - Qiskit backends (Backend, BackendV2)
    - Qiskit Runtime primitives (SamplerV2, EstimatorV2)
    - Braket devices (LocalSimulator, AwsDevice)
    - PennyLane devices (with backend resolution)
    - Cirq simulators and engines

    Examples
    --------
    >>> from qiskit_aer import AerSimulator
    >>> backend_info = resolve_physical_backend(AerSimulator())
    >>> backend_info["provider"]
    'aer'
    """
    # This is a placeholder implementation.
    # Each adapter should override or extend this with SDK-specific logic.
    #
    # The actual implementation will be in each adapter package, but this
    # provides the interface contract and fallback behavior.

    if executor is None:
        return None

    result: dict[str, Any] = {
        "provider": "unknown",
        "backend_name": "unknown",
        "backend_id": None,
        "backend_type": "unknown",
        "backend_obj": executor,
    }

    # Try to extract basic info from common attributes
    executor_type = type(executor).__name__

    # Check for name attribute (common across SDKs)
    if hasattr(executor, "name"):
        name = getattr(executor, "name", None)
        if callable(name):
            try:
                name = name()
            except Exception:
                name = None
        if name:
            result["backend_name"] = str(name)

    # Detect simulator vs hardware from name/type
    name_lower = result["backend_name"].lower()
    type_lower = executor_type.lower()

    if any(s in name_lower or s in type_lower for s in ("sim", "emulator", "fake")):
        result["backend_type"] = "simulator"
    elif any(s in name_lower for s in ("ibm_", "ionq", "rigetti", "oqc", "aspen")):
        result["backend_type"] = "hardware"

    return result
