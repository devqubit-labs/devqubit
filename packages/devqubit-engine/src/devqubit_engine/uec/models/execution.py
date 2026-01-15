# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Execution metadata models.

This module defines:
- ProducerInfo: SDK stack information for envelope producer
- ExecutionSnapshot: Job submission and tracking metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.uec.models.program import TranspilationInfo


logger = logging.getLogger(__name__)


def _get_engine_version() -> str:
    """Get devqubit-engine version from package metadata."""
    try:
        from importlib.metadata import version

        return version("devqubit-engine")
    except Exception:
        return "unknown"


@dataclass
class ProducerInfo:
    """
    SDK stack information for the envelope producer.

    Captures the complete toolchain that produced an execution envelope,
    from the high-level frontend down to the physical backend SDK.

    Parameters
    ----------
    name : str
        Producer name. Always "devqubit" for devqubit-engine.
    engine_version : str
        devqubit-engine version string.
    adapter : str
        Adapter identifier (e.g., "devqubit-qiskit", "devqubit-braket").
    adapter_version : str
        Adapter version string.
    sdk : str
        Primary/lowest SDK name (e.g., "qiskit", "braket-sdk", "cirq").
    sdk_version : str
        Primary SDK version string.
    frontends : list of str
        SDK stack from highest to lowest layer.

    Examples
    --------
    >>> producer = ProducerInfo.create(
    ...     adapter="devqubit-qiskit",
    ...     adapter_version="0.3.0",
    ...     sdk="qiskit",
    ...     sdk_version="1.3.0",
    ...     frontends=["qiskit"],
    ... )
    """

    name: str
    engine_version: str
    adapter: str
    adapter_version: str
    sdk: str
    sdk_version: str
    frontends: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frontends:
            raise ValueError(
                "frontends must be a non-empty list. "
                "For simple setups, use [sdk_name]."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine_version": self.engine_version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "sdk": self.sdk,
            "sdk_version": self.sdk_version,
            "frontends": self.frontends,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProducerInfo:
        return cls(
            name=str(d.get("name", "devqubit")),
            engine_version=str(d.get("engine_version", "unknown")),
            adapter=str(d.get("adapter", "")),
            adapter_version=str(d.get("adapter_version", "")),
            sdk=str(d.get("sdk", "")),
            sdk_version=str(d.get("sdk_version", "")),
            frontends=d.get("frontends", ["unknown"]),
        )

    @classmethod
    def create(
        cls,
        *,
        adapter: str,
        adapter_version: str,
        sdk: str,
        sdk_version: str,
        frontends: list[str],
    ) -> ProducerInfo:
        """Create ProducerInfo with auto-detected engine version."""
        return cls(
            name="devqubit",
            engine_version=_get_engine_version(),
            adapter=adapter,
            adapter_version=adapter_version,
            sdk=sdk,
            sdk_version=sdk_version,
            frontends=frontends,
        )


@dataclass
class ExecutionSnapshot:
    """
    Execution submission and job tracking metadata.

    Parameters
    ----------
    submitted_at : str
        Submission timestamp (ISO 8601).
    shots : int, optional
        Number of shots requested.
    execution_count : int, optional
        Execution sequence number.
    job_ids : list of str
        Job identifiers.
    task_ids : list of str
        Task identifiers (for Braket).
    transpilation : TranspilationInfo, optional
        Transpilation metadata.
    options : dict
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

    schema_version: str = "devqubit.execution_snapshot/1.0"

    def to_dict(self) -> dict[str, Any]:
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
            schema_version=d.get("schema", "devqubit.execution_snapshot/1.0"),
        )
