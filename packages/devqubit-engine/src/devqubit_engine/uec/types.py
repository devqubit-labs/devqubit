# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Core types for the Uniform Execution Contract (UEC).

This module defines foundational data structures and enumerations
used throughout the UEC snapshot system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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
        MIME type of the artifact content.
    role : str
        Logical role indicating the artifact's purpose.
    meta : dict, optional
        Additional metadata attached to the artifact reference.

    Raises
    ------
    ValueError
        If any field fails validation.
    """

    kind: str
    digest: str
    media_type: str
    role: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
        """Convert to a JSON-serializable dictionary."""
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
        """Create an ArtifactRef from a dictionary."""
        return cls(
            kind=str(d.get("kind", "")),
            digest=str(d.get("digest", "")),
            media_type=str(d.get("media_type", "")),
            role=str(d.get("role", "")),
            meta=d.get("meta", {}),
        )


@dataclass
class ValidationResult:
    """
    Result of schema validation.

    Parameters
    ----------
    valid : bool
        True if validation passed.
    errors : list
        List of validation errors (empty if valid).
    warnings : list
        List of validation warnings.
    """

    valid: bool
    errors: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.valid

    def __iter__(self):
        """Iterate over validation errors."""
        return iter(self.errors)

    def __len__(self) -> int:
        """Return number of validation errors."""
        return len(self.errors)

    @property
    def ok(self) -> bool:
        """
        Alias for valid - returns True if no errors.

        Returns
        -------
        bool
            True if validation passed without errors.
        """
        return self.valid and len(self.errors) == 0

    @property
    def error_count(self) -> int:
        """
        Return the number of validation errors.

        Returns
        -------
        int
            Count of validation errors.
        """
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """
        Return the number of validation warnings.

        Returns
        -------
        int
            Count of validation warnings.
        """
        return len(self.warnings)
