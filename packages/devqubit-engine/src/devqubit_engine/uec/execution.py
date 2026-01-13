# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Execution snapshot for job submission metadata.

This module defines ExecutionSnapshot for capturing when and how
circuits were submitted for execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.uec.program import TranspilationInfo


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

    schema_version: str = "devqubit.execution_snapshot/0.1"

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
            schema_version=d.get("schema", "devqubit.execution_snapshot/0.1"),
        )
