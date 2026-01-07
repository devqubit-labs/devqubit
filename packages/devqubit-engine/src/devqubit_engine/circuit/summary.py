# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit summary extraction and comparison.

This module provides the :class:`CircuitSummary` dataclass for uniform
circuit statistics across SDKs, and functions for summarizing and
comparing circuits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat


logger = logging.getLogger(__name__)


@dataclass
class CircuitSummary:
    """
    Semantic summary of a quantum circuit.

    Provides uniform statistics across all SDKs. Fields are populated
    by SDK-native summarizers registered via entry points.

    Attributes
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    num_clbits : int
        Number of classical bits.
    depth : int
        Circuit depth (longest path from input to output).
    gate_count_1q : int
        Single-qubit gate count.
    gate_count_2q : int
        Two-qubit gate count.
    gate_count_multi : int
        Multi-qubit gate count (3+ qubits).
    gate_count_measure : int
        Measurement operation count.
    gate_count_total : int
        Total gate count.
    gate_types : dict
        Mapping from gate name to count.
    has_parameters : bool
        Whether circuit has parameterized gates.
    parameter_count : int
        Number of parameters.
    is_clifford : bool or None
        True if all gates are Clifford gates, None if unknown.
    source_format : CircuitFormat
        Format the summary was extracted from.
    sdk : SDK
        Associated SDK.
    """

    num_qubits: int = 0
    num_clbits: int = 0
    depth: int = 0
    gate_count_1q: int = 0
    gate_count_2q: int = 0
    gate_count_multi: int = 0
    gate_count_measure: int = 0
    gate_count_total: int = 0
    gate_types: dict[str, int] = field(default_factory=dict)
    has_parameters: bool = False
    parameter_count: int = 0
    is_clifford: bool | None = None
    source_format: CircuitFormat = CircuitFormat.UNKNOWN
    sdk: SDK = SDK.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns
        -------
        dict
            Dictionary representation with enum values as strings.
        """
        return {
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "depth": self.depth,
            "gate_count_1q": self.gate_count_1q,
            "gate_count_2q": self.gate_count_2q,
            "gate_count_multi": self.gate_count_multi,
            "gate_count_measure": self.gate_count_measure,
            "gate_count_total": self.gate_count_total,
            "gate_types": self.gate_types,
            "has_parameters": self.has_parameters,
            "parameter_count": self.parameter_count,
            "is_clifford": self.is_clifford,
            "source_format": self.source_format.value,
            "sdk": self.sdk.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CircuitSummary:
        """
        Create from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with summary fields.

        Returns
        -------
        CircuitSummary
            Reconstructed summary.
        """
        return cls(
            num_qubits=int(d.get("num_qubits", 0)),
            num_clbits=int(d.get("num_clbits", 0)),
            depth=int(d.get("depth", 0)),
            gate_count_1q=int(d.get("gate_count_1q", 0)),
            gate_count_2q=int(d.get("gate_count_2q", 0)),
            gate_count_multi=int(d.get("gate_count_multi", 0)),
            gate_count_measure=int(d.get("gate_count_measure", 0)),
            gate_count_total=int(d.get("gate_count_total", 0)),
            gate_types=dict(d.get("gate_types", {})),
            has_parameters=bool(d.get("has_parameters", False)),
            parameter_count=int(d.get("parameter_count", 0)),
            is_clifford=d.get("is_clifford"),
            source_format=CircuitFormat(d.get("source_format", "unknown")),
            sdk=SDK(d.get("sdk", "unknown")),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CircuitSummary(qubits={self.num_qubits}, depth={self.depth}, "
            f"gates={self.gate_count_total}, sdk={self.sdk.value})"
        )


@dataclass
class CircuitDiff:
    """
    Semantic diff between two circuit summaries.

    Captures differences between circuits for comparison and drift
    detection.

    Attributes
    ----------
    match : bool
        True if summaries are semantically equivalent.
    changes : list of str
        Human-readable list of changes.
    metrics : dict
        Quantitative diff metrics (deltas, percentages).
    summary_a : CircuitSummary
        First (baseline) summary.
    summary_b : CircuitSummary
        Second (comparison) summary.
    """

    match: bool
    changes: list[str]
    metrics: dict[str, Any]
    summary_a: CircuitSummary
    summary_b: CircuitSummary

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "match": self.match,
            "changes": self.changes,
            "metrics": self.metrics,
            "summary_a": self.summary_a.to_dict(),
            "summary_b": self.summary_b.to_dict(),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        status = "match" if self.match else f"{len(self.changes)} changes"
        return f"CircuitDiff({status})"


def summarize_circuit_data(data: CircuitData) -> CircuitSummary:
    """
    Extract summary from CircuitData.

    Loads the circuit using the registered loader and summarizes
    using the registered summarizer for the SDK.

    Parameters
    ----------
    data : CircuitData
        Serialized circuit data.

    Returns
    -------
    CircuitSummary
        Extracted summary.

    Raises
    ------
    ValueError
        If no summarizer is registered for the SDK.
    LoaderError
        If circuit loading fails.
    """
    from devqubit_engine.circuit.registry import get_loader, get_summarizer

    loaded = get_loader(data.sdk).load(data)

    summarizer = get_summarizer(data.sdk)
    if summarizer is None:
        raise ValueError(f"No summarizer registered for {data.sdk.value}")

    summary = summarizer(loaded.circuit)
    summary.source_format = data.format
    summary.sdk = data.sdk

    logger.debug(
        "Summarized circuit: qubits=%d, depth=%d, gates=%d",
        summary.num_qubits,
        summary.depth,
        summary.gate_count_total,
    )

    return summary


def summarize_circuit(circuit: Any, sdk: SDK | None = None) -> CircuitSummary:
    """
    Extract summary from SDK-native circuit.

    Parameters
    ----------
    circuit : Any
        SDK-native circuit object (e.g., QuantumCircuit, Circuit).
    sdk : SDK, optional
        SDK hint. Auto-detected from circuit type if not provided.

    Returns
    -------
    CircuitSummary
        Circuit summary.

    Raises
    ------
    ValueError
        If SDK cannot be detected or no summarizer is registered.
    """
    from devqubit_engine.circuit.registry import get_summarizer

    if sdk is None:
        sdk = _detect_sdk(circuit)

    summarizer = get_summarizer(sdk)
    if summarizer is None:
        raise ValueError(f"No summarizer registered for {sdk.value}")

    return summarizer(circuit)


def _detect_sdk(circuit: Any) -> SDK:
    """
    Detect SDK from circuit object type.

    Parameters
    ----------
    circuit : Any
        Circuit object.

    Returns
    -------
    SDK
        Detected SDK.

    Raises
    ------
    ValueError
        If SDK cannot be detected.
    """
    module = type(circuit).__module__

    sdk_prefixes = [
        ("qiskit", SDK.QISKIT),
        ("braket", SDK.BRAKET),
        ("cirq", SDK.CIRQ),
        ("pennylane", SDK.PENNYLANE),
    ]

    for prefix, sdk in sdk_prefixes:
        if prefix in module:
            return sdk

    raise ValueError(
        f"Cannot detect SDK for circuit type '{type(circuit).__name__}' "
        f"from module '{module}'"
    )


def diff_summaries(
    summary_a: CircuitSummary,
    summary_b: CircuitSummary,
) -> CircuitDiff:
    """
    Compare two circuit summaries.

    Computes semantic differences between two summaries, including
    quantitative metrics and human-readable change descriptions.

    Parameters
    ----------
    summary_a : CircuitSummary
        First (baseline) summary.
    summary_b : CircuitSummary
        Second (comparison) summary.

    Returns
    -------
    CircuitDiff
        Semantic diff with changes and metrics.
    """
    changes: list[str] = []
    metrics: dict[str, Any] = {}

    def _compare(
        metric_key: str,
        label: str,
        val_a: int,
        val_b: int,
        show_pct: bool = True,
    ) -> None:
        """Compare two values and record changes."""
        if val_a == val_b:
            return

        delta = val_b - val_a
        metrics[f"{metric_key}_delta"] = delta

        if show_pct and val_a > 0:
            pct = (delta / val_a) * 100
            metrics[f"{metric_key}_delta_pct"] = pct
            changes.append(f"{label}: {val_a} → {val_b} ({pct:+.1f}%)")
        else:
            changes.append(f"{label}: {val_a} → {val_b}")

    # Compare numeric fields
    _compare(
        "num_qubits",
        "qubits",
        summary_a.num_qubits,
        summary_b.num_qubits,
        show_pct=False,
    )
    _compare(
        "num_clbits",
        "clbits",
        summary_a.num_clbits,
        summary_b.num_clbits,
        show_pct=False,
    )
    _compare("depth", "depth", summary_a.depth, summary_b.depth)
    _compare(
        "gate_count_1q",
        "1Q gates",
        summary_a.gate_count_1q,
        summary_b.gate_count_1q,
    )
    _compare(
        "gate_count_2q",
        "2Q gates",
        summary_a.gate_count_2q,
        summary_b.gate_count_2q,
    )
    _compare(
        "gate_count_total",
        "total gates",
        summary_a.gate_count_total,
        summary_b.gate_count_total,
    )
    _compare(
        "parameter_count",
        "parameters",
        summary_a.parameter_count,
        summary_b.parameter_count,
        show_pct=False,
    )

    # Clifford status change
    if summary_a.is_clifford != summary_b.is_clifford:
        changes.append(
            f"is_clifford: {summary_a.is_clifford} → {summary_b.is_clifford}"
        )
        metrics["is_clifford_changed"] = True
        metrics["is_clifford_a"] = summary_a.is_clifford
        metrics["is_clifford_b"] = summary_b.is_clifford

    # Gate type changes
    types_a = set(summary_a.gate_types.keys())
    types_b = set(summary_b.gate_types.keys())

    if new_gates := types_b - types_a:
        changes.append(f"new gate types: {', '.join(sorted(new_gates))}")
        metrics["new_gate_types"] = sorted(new_gates)

    if removed_gates := types_a - types_b:
        changes.append(f"removed gate types: {', '.join(sorted(removed_gates))}")
        metrics["removed_gate_types"] = sorted(removed_gates)

    logger.debug(
        "Compared summaries: %s",
        "match" if len(changes) == 0 else f"{len(changes)} changes",
    )

    return CircuitDiff(
        match=len(changes) == 0,
        changes=changes,
        metrics=metrics,
        summary_a=summary_a,
        summary_b=summary_b,
    )
