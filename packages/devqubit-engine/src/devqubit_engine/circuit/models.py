# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit data models.

Provides core models for circuit serialization, loading, and gate classification.
The classification framework is SDK-agnostic; actual gate definitions are
provided by adapter packages.

SDK Registration
----------------
Adding a new SDK requires **only** two steps in this module:

1. Add an entry to :class:`SDK` and :class:`CircuitFormat` enums.
2. Add a single :class:`SDKDescriptor` row to :data:`SDK_REGISTRY`.

All downstream modules (extractors, summary, etc.) derive their SDK-specific
lookup tables from ``SDK_REGISTRY`` automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


# =============================================================================
# Enums
# =============================================================================


class SDK(str, Enum):
    """Supported quantum computing SDKs."""

    QISKIT = "qiskit"
    BRAKET = "braket"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    CUDAQ = "cudaq"
    UNKNOWN = "unknown"


class CircuitFormat(str, Enum):
    """Circuit serialization formats."""

    # Native formats
    QPY = "qpy"
    JAQCD = "jaqcd"
    CIRQ_JSON = "cirq_json"
    TAPE_JSON = "tape_json"
    CUDAQ_JSON = "cudaq_json"

    # Interchange
    OPENQASM3 = "openqasm3"
    OPENQASM2 = "openqasm2"

    # Fallback
    TEXT = "text"
    UNKNOWN = "unknown"

    @property
    def is_native(self) -> bool:
        """Check if format is SDK-native (registered in SDK_REGISTRY)."""
        return self in _FORMAT_TO_SDK

    @property
    def sdk(self) -> SDK:
        """Get associated SDK for native formats."""
        return _FORMAT_TO_SDK.get(self, SDK.UNKNOWN)


# =============================================================================
# SDK Registry
# =============================================================================


@dataclass(frozen=True, slots=True)
class SDKDescriptor:
    """
    Metadata for a supported SDK.

    Each descriptor captures everything the engine needs to detect,
    extract, and load circuits for a given SDK.  All per-SDK lookup
    tables in ``extractors``, ``summary``, etc. are derived from
    :data:`SDK_REGISTRY` at import time.

    Parameters
    ----------
    sdk : SDK
        Enum member identifying the SDK.
    native_format : CircuitFormat
        Default native serialization format.
    kind_pattern : str
        Substring to match in artifact ``kind`` strings
        (case-insensitive).  Used for both RunRecord scanning and
        envelope ref resolution.
    adapter_pattern : str
        Substring to match in ``record.adapter`` for SDK detection.
    module_pattern : str
        Substring to match in ``type(circuit).__module__`` for
        auto-detecting SDK from a live circuit object.
    is_binary : bool
        ``True`` if the native format is binary (e.g. QPY);
        ``False`` if it is UTF-8 text (e.g. JSON).
    """

    sdk: SDK
    native_format: CircuitFormat
    kind_pattern: str
    adapter_pattern: str
    module_pattern: str
    is_binary: bool = False


# Central registry of all supported SDKs.
SDK_REGISTRY: tuple[SDKDescriptor, ...] = (
    SDKDescriptor(
        sdk=SDK.QISKIT,
        native_format=CircuitFormat.QPY,
        kind_pattern="qpy",
        adapter_pattern="qiskit",
        module_pattern="qiskit",
        is_binary=True,
    ),
    SDKDescriptor(
        sdk=SDK.BRAKET,
        native_format=CircuitFormat.JAQCD,
        kind_pattern="jaqcd",
        adapter_pattern="braket",
        module_pattern="braket",
    ),
    SDKDescriptor(
        sdk=SDK.CIRQ,
        native_format=CircuitFormat.CIRQ_JSON,
        kind_pattern="cirq",
        adapter_pattern="cirq",
        module_pattern="cirq",
    ),
    SDKDescriptor(
        sdk=SDK.PENNYLANE,
        native_format=CircuitFormat.TAPE_JSON,
        kind_pattern="tape",
        adapter_pattern="pennylane",
        module_pattern="pennylane",
    ),
    SDKDescriptor(
        sdk=SDK.CUDAQ,
        native_format=CircuitFormat.CUDAQ_JSON,
        kind_pattern="cudaq",
        adapter_pattern="cudaq",
        module_pattern="cudaq",
    ),
)


# =============================================================================
# Derived lookup tables
# =============================================================================

#: Native format => SDK mapping (used by ``CircuitFormat.sdk`` property).
_FORMAT_TO_SDK: dict[CircuitFormat, SDK] = {
    d.native_format: d.sdk for d in SDK_REGISTRY
}


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class CircuitData:
    """
    Container for serialized circuit data.

    Attributes
    ----------
    data : bytes or str
        Serialized circuit data.
    format : CircuitFormat
        Serialization format.
    sdk : SDK
        Associated SDK.
    name : str
        Circuit name.
    index : int
        Index in batch.
    metadata : dict
        Additional metadata.
    """

    data: bytes | str
    format: CircuitFormat
    sdk: SDK = SDK.UNKNOWN
    name: str = ""
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        """Check if data is binary."""
        return isinstance(self.data, bytes)

    def as_bytes(self) -> bytes:
        """Get data as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        return self.data.encode("utf-8")

    def as_text(self) -> str:
        """Get data as text."""
        if isinstance(self.data, str):
            return self.data
        return self.data.decode("utf-8")


@dataclass
class LoadedCircuit:
    """
    Container for a loaded circuit object.

    Attributes
    ----------
    circuit : Any
        SDK-native circuit object.
    sdk : SDK
        Associated SDK.
    source_format : CircuitFormat
        Format loaded from.
    name : str
        Circuit name.
    index : int
        Index in batch.
    """

    circuit: Any
    sdk: SDK
    source_format: CircuitFormat
    name: str = ""
    index: int = 0


# =============================================================================
# Gate classification
# =============================================================================


class GateCategory(Enum):
    """Gate categories for classification."""

    SINGLE_QUBIT = auto()
    TWO_QUBIT = auto()
    MULTI_QUBIT = auto()
    MEASURE = auto()
    BARRIER = auto()
    OTHER = auto()


@dataclass(frozen=True, slots=True)
class GateInfo:
    """Gate metadata for classification."""

    category: GateCategory
    is_clifford: bool = False


class GateClassifier:
    """
    Gate classifier with adapter-defined metadata.

    Provides consistent gate classification across SDKs. Each adapter
    defines its own gate mappings and uses this classifier to populate
    CircuitSummary fields.

    Parameters
    ----------
    gates : dict[str, GateInfo]
        Mapping from gate names to GateInfo metadata.
    normalizer : callable, optional
        Function to normalize gate names before lookup.
        Defaults to str.lower.
    """

    def __init__(
        self,
        gates: dict[str, GateInfo],
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._gates = gates
        self._normalize = normalizer or str.lower

    def classify(self, gate_name: str) -> GateInfo:
        """Get gate info for a gate name."""
        normalized = self._normalize(gate_name)
        return self._gates.get(normalized, GateInfo(GateCategory.OTHER))

    def classify_counts(
        self,
        gate_counts: dict[str, int],
    ) -> dict[str, int | bool | None]:
        """
        Classify gate counts and compute summary statistics.

        Parameters
        ----------
        gate_counts : dict[str, int]
            Mapping from gate name to count.

        Returns
        -------
        dict
            Statistics with keys: gate_count_1q, gate_count_2q,
            gate_count_multi, gate_count_measure, is_clifford.
        """
        gate_count_1q = 0
        gate_count_2q = 0
        gate_count_multi = 0
        gate_count_measure = 0
        is_clifford = True
        has_gates = False

        for gate_name, count in gate_counts.items():
            info = self.classify(gate_name)
            has_gates = True

            match info.category:
                case GateCategory.SINGLE_QUBIT:
                    gate_count_1q += count
                    if not info.is_clifford:
                        is_clifford = False
                case GateCategory.TWO_QUBIT:
                    gate_count_2q += count
                    if not info.is_clifford:
                        is_clifford = False
                case GateCategory.MULTI_QUBIT:
                    gate_count_multi += count
                    is_clifford = False
                case GateCategory.MEASURE:
                    gate_count_measure += count
                case GateCategory.BARRIER:
                    pass
                case _:
                    is_clifford = False

        return {
            "gate_count_1q": gate_count_1q,
            "gate_count_2q": gate_count_2q,
            "gate_count_multi": gate_count_multi,
            "gate_count_measure": gate_count_measure,
            "is_clifford": is_clifford if has_gates else None,
        }
