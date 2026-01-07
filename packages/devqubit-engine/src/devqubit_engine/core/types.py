# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Core types for devqubit.

This module defines the foundational data structures, enumerations,
and utilities used throughout the library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Regex pattern for validating SHA-256 digest format
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _norm_gate(name: str) -> str:
    """
    Normalize gate names for consistent matching across SDKs.

    - Lowercase
    - Remove non-alphanumerics
    - Strip common class-name suffixes ("Gate", "PowGate") so Cirq/Qiskit class
      names like "CXPowGate" and "FSimGate" normalize to "cx" and "fsim".
    """
    # Precompiled regex for speed
    _NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

    # Suffixes commonly seen in SDK class names (e.g. "CXPowGate", "FSimGate")
    _GATE_CLASS_SUFFIXES = ("powgate", "gate")

    s = _NON_ALNUM_RE.sub("", str(name).lower())
    for suffix in _GATE_CLASS_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


# Common two-qubit gate names for median error calculation
TWO_QUBIT_GATES = frozenset(
    _norm_gate(x)
    for x in {
        # Common controlled / entangling gates
        "cx",
        "cnot",
        "cz",
        "cy",
        "ecr",
        "rzx",
        # Swap family
        "swap",
        "iswap",
        "pswap",
        "phased_iswap",
        # Interaction / rotation families
        "rxx",
        "ryy",
        "rzz",
        "xx",
        "yy",
        "zz",
        "xy",
        # Ion-trap style
        "ms",
        # FSim family (Cirq/Google)
        "fsim",
        "phased_fsim",
        # Google-specific named aliases
        "sycamore",
        "syc",
        "willow",
        # Braket controlled-phase (2-qubit)
        "cphaseshift",
        "cphaseshift00",
        "cphaseshift01",
        "cphaseshift10",
        "cphaseshift11",
    }
)


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


class ResultType(str, Enum):
    """
    Type of quantum execution result.

    Attributes
    ----------
    COUNTS
        Measurement counts (bitstring histograms).
    QUASI_DIST
        Quasi-probability distributions.
    EXPECTATION
        Expectation values from estimator primitives.
    SAMPLES
        Raw measurement samples/shots.
    STATEVECTOR
        Full statevector (simulator only).
    DENSITY_MATRIX
        Full density matrix (simulator only).
    OTHER
        Other undefined result type.
    """

    COUNTS = "counts"
    QUASI_DIST = "quasi_dist"
    EXPECTATION = "expectation"
    SAMPLES = "samples"
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    OTHER = "other"


@dataclass(frozen=True)
class ArtifactRef:
    """
    Immutable reference to a stored artifact.

    Represents a content-addressed pointer to an artifact stored in
    the object store. The digest provides deduplication and integrity
    verification.

    Parameters
    ----------
    kind : str
        Artifact type identifier (e.g., "qiskit.qpy.circuits",
        "source.openqasm3", "pennylane.tape").
    digest : str
        Content digest in format ``sha256:<64-hex-chars>``.
    media_type : str
        MIME type of the artifact content (e.g., "application/x-qpy",
        "application/json").
    role : str
        Logical role indicating the artifact's purpose. Common values:
        "program", "results", "device_snapshot", "artifact".
    meta : dict, optional
        Additional metadata attached to the artifact reference.

    Raises
    ------
    ValueError
        If any field fails validation (empty, wrong format, etc.).

    Notes
    -----
    This class is frozen (immutable) to ensure artifact references
    remain consistent after creation and can be safely used as dict keys.
    """

    kind: str
    digest: str
    media_type: str
    role: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate artifact reference fields on creation."""
        if not self.kind or len(self.kind) < 3:
            raise ValueError(
                f"Invalid artifact kind: {self.kind!r}. "
                "Kind must be at least 3 characters."
            )

        if not isinstance(self.digest, str) or not _DIGEST_PATTERN.fullmatch(
            self.digest
        ):
            raise ValueError(
                f"Invalid digest format: {self.digest!r}. "
                "Expected 'sha256:<64-hex-chars>'."
            )

        if not self.media_type or len(self.media_type) < 3:
            raise ValueError(
                f"Invalid media_type: {self.media_type!r}. "
                "Media type must be at least 3 characters."
            )

        if not self.role:
            raise ValueError("Artifact role cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization.
            The ``meta`` field is only included if not None.
        """
        d: dict[str, Any] = {
            "kind": self.kind,
            "digest": self.digest,
            "media_type": self.media_type,
            "role": self.role,
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactRef:
        """
        Create an ArtifactRef from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing artifact reference fields.
            Required keys: "kind", "digest", "media_type", "role".
            Optional key: "meta".

        Returns
        -------
        ArtifactRef
            New artifact reference instance.

        Raises
        ------
        KeyError
            If required fields are missing.
        ValueError
            If field validation fails.
        """
        return cls(
            kind=str(d.get("kind", "")),
            digest=str(d.get("digest", "")),
            media_type=str(d.get("media_type", "")),
            role=str(d.get("role", "")),
            meta=d.get("meta", {}),
        )
