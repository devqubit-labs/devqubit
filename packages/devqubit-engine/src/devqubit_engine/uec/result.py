# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result snapshot for capturing execution results.

This module defines ResultSnapshot and normalized result types
for measurement counts and expectation values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.uec.types import ArtifactRef, ResultType


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
        return cls(
            circuit_index=int(d.get("circuit_index", 0)),
            counts=d.get("counts", {}),
            shots=d.get("shots"),
            name=d.get("name"),
        )


@dataclass
class NormalizedExpectation:
    """
    Normalized expectation value result with full metadata.

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
    Simple expectation value result.

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
    counts : list of NormalizedCounts
        Normalized measurement counts.
    expectations : list of ExpectationValue
        Expectation values.
    num_experiments : int, optional
        Number of experiments.
    success : bool
        Whether execution succeeded.
    error_message : str, optional
        Error message if failed.
    metadata : dict
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
        result_type_val = d.get("result_type", "counts")
        result_type = (
            ResultType(result_type_val)
            if isinstance(result_type_val, str)
            else result_type_val
        )

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
