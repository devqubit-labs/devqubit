# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Program snapshot for capturing circuit artifacts.

This module defines ProgramSnapshot for storing logical and physical
circuit representations along with transpilation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from devqubit_engine.storage.types import ArtifactRef


class ProgramRole(str, Enum):
    """
    Role of a program artifact in the execution pipeline.

    Attributes
    ----------
    LOGICAL
        User-provided circuit before any transpilation.
    PHYSICAL
        Circuit after transpilation, conforming to backend ISA.
    """

    LOGICAL = "logical"
    PHYSICAL = "physical"


class TranspilationMode(str, Enum):
    """
    Transpilation handling mode for circuit submission.

    Attributes
    ----------
    AUTO
        Adapter transpiles if needed (checks ISA compatibility).
    MANUAL
        User handles transpilation; adapter logs as-is.
    MANAGED
        Provider/runtime handles transpilation server-side.
    """

    AUTO = "auto"
    MANUAL = "manual"
    MANAGED = "managed"


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
        mode_val = d.get("mode", "auto")
        mode = TranspilationMode(mode_val) if isinstance(mode_val, str) else mode_val
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
        Serialization format (e.g., "qpy", "openqasm3").
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
        role_val = d.get("role", "logical")
        role = ProgramRole(role_val) if isinstance(role_val, str) else role_val
        return cls(
            ref=ArtifactRef.from_dict(d["ref"]),
            role=role,
            format=str(d.get("format", "")),
            name=d.get("name"),
            index=d.get("index"),
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
    structural_hash : str, optional
        Structural hash of logical circuits. Deterministic hash of circuit
        structure (gate sequence, qubits, controls) that IGNORES parameter
        values. Used for "same circuit template?" comparison.
    parametric_hash : str, optional
        Structural + parameters hash. Deterministic hash of structure PLUS
        bound parameter values for this execution. Changes when any parameter
        value changes. Used for "same circuit with same params?" comparison.
    executed_structural_hash : str, optional
        Structural hash of executed (physical) circuits after transpilation.
    executed_parametric_hash : str, optional
        Structural + parameters hash of executed circuits.
    num_circuits : int, optional
        Number of circuits in the program.
    transpilation : TranspilationInfo, optional
        Transpilation metadata.

    Notes
    -----
    Hash semantics (contract for adapters):

    - ``structural_hash``: Structure only. Same value means same circuit template,
      even if parameter values differ (e.g., VQE iterations).
    - ``parametric_hash``: Structure + bound params. Same value means identical
      circuit for this specific execution.

    For adapter runs, ``structural_hash`` and ``parametric_hash`` are REQUIRED.
    For manual runs, they are optional (compare will report "hash unavailable").

    If circuit has no parameters: ``parametric_hash == structural_hash`` (still required).
    """

    logical: list[ProgramArtifact] = field(default_factory=list)
    physical: list[ProgramArtifact] = field(default_factory=list)
    structural_hash: str | None = None
    parametric_hash: str | None = None
    executed_structural_hash: str | None = None
    executed_parametric_hash: str | None = None
    num_circuits: int | None = None
    transpilation: TranspilationInfo | None = None

    schema_version: str = "devqubit.program_snapshot/1.0"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "logical": [a.to_dict() for a in self.logical],
            "physical": [a.to_dict() for a in self.physical],
        }
        if self.structural_hash:
            d["structural_hash"] = self.structural_hash
        if self.parametric_hash:
            d["parametric_hash"] = self.parametric_hash
        if self.executed_structural_hash:
            d["executed_structural_hash"] = self.executed_structural_hash
        if self.executed_parametric_hash:
            d["executed_parametric_hash"] = self.executed_parametric_hash
        if self.num_circuits is not None:
            d["num_circuits"] = self.num_circuits
        if self.transpilation:
            d["transpilation"] = self.transpilation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProgramSnapshot:
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
            structural_hash=d.get("structural_hash"),
            parametric_hash=d.get("parametric_hash"),
            executed_structural_hash=d.get("executed_structural_hash"),
            executed_parametric_hash=d.get("executed_parametric_hash"),
            num_circuits=d.get("num_circuits"),
            transpilation=transpilation,
            schema_version=d.get("schema", "devqubit.program_snapshot/1.0"),
        )
