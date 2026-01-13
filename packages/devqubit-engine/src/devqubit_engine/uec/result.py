# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Result snapshot for capturing execution results.

This module defines ResultSnapshot and normalized result types
for measurement counts, quasi-probabilities, and expectation values.

Canonical Bit Order
-------------------
UEC standard: ``cbit0_right`` (little-endian string, LSB on right)
- Qiskit: native little-endian → no transformation needed
- Braket: big-endian → adapter must reverse bitstrings
- Cirq: big-endian integers → adapter must reverse bitstrings
"""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.uec.types import ArtifactRef


# =============================================================================
# Counts Format Metadata
# =============================================================================


@dataclass
class CountsFormat:
    """
    Metadata describing the format of measurement counts.

    Required when counts are present. Describes bit ordering convention
    and source SDK to enable cross-SDK comparison.

    Parameters
    ----------
    source_sdk : str
        SDK that produced the raw counts (qiskit, braket, cirq, pennylane).
    source_key_format : str
        Original format identifier describing how keys were encoded.
        Examples: "qiskit_little_endian", "qiskit_register_spaced",
        "braket_big_endian", "cirq_big_endian_int", "hex", "0b_prefixed".
    bit_order : str
        Bit ordering convention for the counts keys.
        Canonical is "cbit0_right" (LSB on right, like Qiskit).
    transformed : bool
        Whether the adapter transformed keys to canonical format.
    num_clbits : int, optional
        Number of classical bits (for padding/validation).
    registers : list, optional
        Register layout metadata for multi-register circuits.
    """

    source_sdk: str
    source_key_format: str
    bit_order: str = "cbit0_right"
    transformed: bool = False
    num_clbits: int | None = None
    registers: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "source_sdk": self.source_sdk,
            "source_key_format": self.source_key_format,
            "bit_order": self.bit_order,
            "transformed": self.transformed,
        }
        if self.num_clbits is not None:
            d["num_clbits"] = self.num_clbits
        if self.registers:
            d["registers"] = self.registers
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CountsFormat:
        """Create from dictionary."""
        return cls(
            source_sdk=str(d.get("source_sdk", "")),
            source_key_format=str(d.get("source_key_format", "")),
            bit_order=str(d.get("bit_order", "cbit0_right")),
            transformed=bool(d.get("transformed", False)),
            num_clbits=d.get("num_clbits"),
            registers=d.get("registers"),
        )


# =============================================================================
# Quasi-Probability Distribution
# =============================================================================


@dataclass
class QuasiProbability:
    """
    Quasi-probability distribution from error-mitigated execution.

    IBM Runtime Sampler returns quasi-distributions that may contain
    negative probabilities due to error mitigation techniques.

    Parameters
    ----------
    distribution : dict
        Bitstring to quasi-probability mapping. Values may be negative.
    precision : float, optional
        Rounding precision applied by the runtime.
    sum_probs : float, optional
        Sum of all probabilities (ideally 1.0).
    min_prob : float, optional
        Minimum probability value (may be negative).
    max_prob : float, optional
        Maximum probability value.
    """

    distribution: dict[str, float]
    precision: float | None = None
    sum_probs: float | None = None
    min_prob: float | None = None
    max_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {"distribution": self.distribution}
        if self.precision is not None:
            d["precision"] = self.precision
        if self.sum_probs is not None:
            d["sum_probs"] = self.sum_probs
        if self.min_prob is not None:
            d["min_prob"] = self.min_prob
        if self.max_prob is not None:
            d["max_prob"] = self.max_prob
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuasiProbability:
        """Create from dictionary."""
        return cls(
            distribution=d.get("distribution", {}),
            precision=d.get("precision"),
            sum_probs=d.get("sum_probs"),
            min_prob=d.get("min_prob"),
            max_prob=d.get("max_prob"),
        )

    @classmethod
    def from_quasi_dist(
        cls,
        quasi_dist: dict[int | str, float],
        num_clbits: int | None = None,
        precision: float | None = None,
    ) -> QuasiProbability:
        """
        Create from IBM Runtime quasi-distribution.

        Handles integer keys (common in SamplerResult) by converting
        to bitstrings.

        Parameters
        ----------
        quasi_dist : dict
            Quasi-distribution from IBM Runtime (int or str keys).
        num_clbits : int, optional
            Number of classical bits for bitstring padding.
        precision : float, optional
            Precision value from runtime.

        Returns
        -------
        QuasiProbability
            Structured quasi-probability with computed stats.
        """
        distribution: dict[str, float] = {}
        for key, prob in quasi_dist.items():
            if isinstance(key, int):
                if num_clbits:
                    bitstring = format(key, f"0{num_clbits}b")
                else:
                    bitstring = bin(key)[2:]
            else:
                bitstring = str(key)
            distribution[bitstring] = float(prob)

        probs = list(distribution.values())
        return cls(
            distribution=distribution,
            precision=precision,
            sum_probs=sum(probs) if probs else None,
            min_prob=min(probs) if probs else None,
            max_prob=max(probs) if probs else None,
        )


# =============================================================================
# Normalized Expectation Value
# =============================================================================


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
        """Convert to JSON-serializable dictionary."""
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
        """Create from dictionary."""
        return cls(
            circuit_index=int(d.get("circuit_index", 0)),
            observable_index=int(d.get("observable_index", 0)),
            value=float(d.get("value", 0.0)),
            variance=d.get("variance"),
            std_error=d.get("std_error"),
            observable=d.get("observable"),
        )


# =============================================================================
# Result Error
# =============================================================================


@dataclass
class ResultError:
    """
    Structured error information for failed executions.

    Parameters
    ----------
    type : str
        Exception class name (e.g., "TimeoutError", "IBMRuntimeError").
    message : str
        Short error message.
    stack_hash : str, optional
        Hash of stack trace for grouping similar errors.
    retryable : bool, optional
        Whether the error is likely transient and retryable.
    details : dict, optional
        Additional error context.
    """

    type: str
    message: str
    stack_hash: str | None = None
    retryable: bool | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "type": self.type,
            "message": self.message,
        }
        if self.stack_hash:
            d["stack_hash"] = self.stack_hash
        if self.retryable is not None:
            d["retryable"] = self.retryable
        if self.details:
            d["details"] = self.details
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultError:
        """Create from dictionary."""
        return cls(
            type=str(d.get("type", "UnknownError")),
            message=str(d.get("message", "")),
            stack_hash=d.get("stack_hash"),
            retryable=d.get("retryable"),
            details=d.get("details"),
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        retryable: bool | None = None,
    ) -> ResultError:
        """
        Create from Python exception.

        Parameters
        ----------
        exc : BaseException
            The exception to convert.
        retryable : bool, optional
            Override retryable detection.

        Returns
        -------
        ResultError
            Structured error with stack hash.
        """
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        stack_str = "".join(tb)
        stack_hash = hashlib.sha256(stack_str.encode()).hexdigest()[:16]

        if retryable is None:
            retryable_types = (
                "TimeoutError",
                "ConnectionError",
                "TransientError",
                "ServiceUnavailable",
                "RateLimitError",
            )
            retryable = type(exc).__name__ in retryable_types

        return cls(
            type=type(exc).__name__,
            message=str(exc)[:500],
            stack_hash=stack_hash,
            retryable=retryable,
        )


# =============================================================================
# Result Item (per-item in batch)
# =============================================================================


@dataclass
class ResultItem:
    """
    Single item in batch execution results.

    Each item in ``result.items[]`` represents results for one circuit
    or parameter set. For single-circuit runs, ``items`` has one element.

    Parameters
    ----------
    item_index : int
        Position in the batch (0-based).
    success : bool
        Whether this item succeeded.
    counts : dict, optional
        Measurement counts with format metadata.
        Structure: {"counts": {...}, "shots": N, "format": CountsFormat}
    quasi_probability : QuasiProbability, optional
        Quasi-probability distribution (IBM Runtime Sampler).
    expectation : NormalizedExpectation, optional
        Expectation value result.
    raw_ref : ArtifactRef, optional
        Reference to raw SDK result artifact.
    error_message : str, optional
        Error message if this item failed.

    Notes
    -----
    Exactly ONE of counts, quasi_probability, or expectation should be set.
    """

    item_index: int
    success: bool
    counts: dict[str, Any] | None = None
    quasi_probability: QuasiProbability | None = None
    expectation: NormalizedExpectation | None = None
    raw_ref: ArtifactRef | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "item_index": self.item_index,
            "success": self.success,
        }
        if self.counts is not None:
            d["counts"] = self.counts
        if self.quasi_probability is not None:
            d["quasi_probability"] = self.quasi_probability.to_dict()
        if self.expectation is not None:
            d["expectation"] = self.expectation.to_dict()
        if self.raw_ref is not None:
            d["raw_ref"] = self.raw_ref.to_dict()
        if self.error_message:
            d["error_message"] = self.error_message
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultItem:
        """Create from dictionary."""
        quasi = None
        if d.get("quasi_probability"):
            quasi = QuasiProbability.from_dict(d["quasi_probability"])

        exp = None
        if d.get("expectation"):
            exp = NormalizedExpectation.from_dict(d["expectation"])

        raw_ref = None
        if d.get("raw_ref"):
            raw_ref = ArtifactRef.from_dict(d["raw_ref"])

        return cls(
            item_index=int(d.get("item_index", 0)),
            success=bool(d.get("success", True)),
            counts=d.get("counts"),
            quasi_probability=quasi,
            expectation=exp,
            raw_ref=raw_ref,
            error_message=d.get("error_message"),
        )

    @classmethod
    def from_counts(
        cls,
        item_index: int,
        counts: dict[str, int],
        shots: int,
        format_info: CountsFormat,
        raw_ref: ArtifactRef | None = None,
    ) -> ResultItem:
        """
        Create ResultItem from measurement counts.

        Parameters
        ----------
        item_index : int
            Position in batch.
        counts : dict
            Bitstring to count mapping.
        shots : int
            Total shots.
        format_info : CountsFormat
            Counts format metadata.
        raw_ref : ArtifactRef, optional
            Reference to raw result.

        Returns
        -------
        ResultItem
            Configured result item with counts.
        """
        return cls(
            item_index=item_index,
            success=True,
            counts={
                "counts": counts,
                "shots": shots,
                "format": format_info.to_dict(),
            },
            raw_ref=raw_ref,
        )


# =============================================================================
# Result Snapshot
# =============================================================================


@dataclass
class ResultSnapshot:
    """
    Result snapshot with success/status/items structure.

    Parameters
    ----------
    success : bool
        Overall execution success.
    status : str
        Normalized status: "completed", "failed", "cancelled", "partial".
    items : list of ResultItem
        Results for each item in batch. Always a list.
    error : ResultError, optional
        Structured error information if failed.
    raw_result_ref : ArtifactRef, optional
        Reference to complete raw result artifact.
    metadata : dict
        Additional metadata.

    Notes
    -----
    Use factory methods for common cases:
    - ``ResultSnapshot.create_success()`` for successful results
    - ``ResultSnapshot.create_failed()`` for exception handling
    - ``ResultSnapshot.create_partial()`` for partial failures
    """

    success: bool
    status: str
    items: list[ResultItem] = field(default_factory=list)
    error: ResultError | None = None
    raw_result_ref: ArtifactRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: str = "devqubit.result_snapshot/1.0"

    def __post_init__(self) -> None:
        """Validate status values."""
        valid_statuses = ("completed", "failed", "cancelled", "partial")
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got: {self.status}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "success": self.success,
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
        }
        if self.error is not None:
            d["error"] = self.error.to_dict()
        if self.raw_result_ref is not None:
            d["raw_result_ref"] = self.raw_result_ref.to_dict()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultSnapshot:
        """Create from dictionary."""
        items = [
            ResultItem.from_dict(x) for x in d.get("items", []) if isinstance(x, dict)
        ]

        error = None
        if d.get("error"):
            error = ResultError.from_dict(d["error"])

        raw_ref = None
        if d.get("raw_result_ref"):
            raw_ref = ArtifactRef.from_dict(d["raw_result_ref"])

        return cls(
            success=bool(d.get("success", False)),
            status=str(d.get("status", "failed")),
            items=items,
            error=error,
            raw_result_ref=raw_ref,
            metadata=d.get("metadata", {}),
            schema_version=d.get("schema", "devqubit.result_snapshot/1.0"),
        )

    @classmethod
    def create_success(
        cls,
        items: list[ResultItem],
        raw_result_ref: ArtifactRef | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResultSnapshot:
        """
        Create successful result snapshot.

        Parameters
        ----------
        items : list of ResultItem
            Results for each batch item.
        raw_result_ref : ArtifactRef, optional
            Reference to raw result.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        ResultSnapshot
            Successful result snapshot.
        """
        return cls(
            success=True,
            status="completed",
            items=items,
            raw_result_ref=raw_result_ref,
            metadata=metadata or {},
        )

    @classmethod
    def create_failed(
        cls,
        exception: BaseException,
        partial_items: list[ResultItem] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResultSnapshot:
        """
        Create failed result snapshot from exception.

        This is the recommended way to handle failures - ensures
        envelope is always created even on exceptions.

        Parameters
        ----------
        exception : BaseException
            The exception that caused failure.
        partial_items : list of ResultItem, optional
            Any partial results obtained before failure.
        metadata : dict, optional
            Additional context.

        Returns
        -------
        ResultSnapshot
            Failed result snapshot with structured error.
        """
        return cls(
            success=False,
            status="failed",
            items=partial_items or [],
            error=ResultError.from_exception(exception),
            metadata=metadata or {},
        )

    @classmethod
    def create_partial(
        cls,
        items: list[ResultItem],
        error: ResultError | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResultSnapshot:
        """
        Create partial result snapshot (some items succeeded).

        Parameters
        ----------
        items : list of ResultItem
            Mix of successful and failed items.
        error : ResultError, optional
            Overall error information.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        ResultSnapshot
            Partial result snapshot.
        """
        return cls(
            success=False,
            status="partial",
            items=items,
            error=error,
            metadata=metadata or {},
        )
