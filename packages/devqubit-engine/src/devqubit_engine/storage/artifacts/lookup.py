# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact browsing and discovery.

Functions for finding, listing, and accessing artifacts from run records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from devqubit_engine.storage.artifacts.io import (
    load_artifact_bytes,
    load_artifact_text,
)


if TYPE_CHECKING:
    from devqubit_engine.core.record import RunRecord
    from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Finding artifacts
# =============================================================================


def find_artifact(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
) -> ArtifactRef | None:
    """
    Find first artifact matching criteria.

    Parameters
    ----------
    record : RunRecord
        Run record containing artifacts.
    role : str, optional
        Required artifact role (e.g., "program", "results", "device_snapshot").
    kind_contains : str, optional
        Substring required in artifact kind (case-insensitive).

    Returns
    -------
    ArtifactRef or None
        First matching artifact reference, or None if not found.

    Examples
    --------
    >>> artifact = find_artifact(record, role="program", kind_contains="qasm")
    >>> if artifact:
    ...     print(f"Found: {artifact.kind}")
    """
    for artifact in record.artifacts:
        if role and artifact.role != role:
            continue
        if kind_contains and kind_contains.lower() not in artifact.kind.lower():
            continue
        logger.debug("Found artifact: role=%s, kind=%s", artifact.role, artifact.kind)
        return artifact
    return None


def find_all_artifacts(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
) -> list[ArtifactRef]:
    """
    Find all artifacts matching criteria.

    Parameters
    ----------
    record : RunRecord
        Run record containing artifacts.
    role : str, optional
        Required artifact role.
    kind_contains : str, optional
        Substring required in artifact kind (case-insensitive).

    Returns
    -------
    list of ArtifactRef
        All matching artifact references.
    """
    results = []
    for artifact in record.artifacts:
        if role and artifact.role != role:
            continue
        if kind_contains and kind_contains.lower() not in artifact.kind.lower():
            continue
        results.append(artifact)
    return results


def get_artifact_digests(
    record: RunRecord,
    role: str,
    *,
    kind_contains: str | None = None,
) -> list[str]:
    """
    Extract sorted artifact digests from a run record.

    Parameters
    ----------
    record : RunRecord
        Run record containing artifacts.
    role : str
        Filter by artifact role (e.g., "program", "results").
    kind_contains : str, optional
        Filter by substring in artifact kind.

    Returns
    -------
    list of str
        Sorted list of artifact digests matching filters.
    """
    artifacts = find_all_artifacts(record, role=role, kind_contains=kind_contains)
    return sorted(a.digest for a in artifacts)


# =============================================================================
# Accessing artifacts by selector
# =============================================================================


def get_artifact(record: RunRecord, selector: str | int) -> ArtifactRef | None:
    """
    Get artifact by index or selector.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Either:
        - int: artifact index
        - str: digest prefix, kind, or "role:kind" pattern

    Returns
    -------
    ArtifactRef or None
        Matching artifact or None if not found.
    """
    if isinstance(selector, int):
        if 0 <= selector < len(record.artifacts):
            return record.artifacts[selector]
        return None

    selector = str(selector)

    # Digest prefix
    if selector.startswith("sha256:"):
        for art in record.artifacts:
            if art.digest.startswith(selector):
                return art
        return None

    # role:kind pattern
    if ":" in selector:
        role, kind = selector.split(":", 1)
        for art in record.artifacts:
            if art.role == role and kind in art.kind:
                return art
        return None

    # Kind substring match
    for art in record.artifacts:
        if selector in art.kind:
            return art

    return None


def get_artifact_bytes(
    record: RunRecord,
    selector: str | int,
    store: ObjectStoreProtocol,
) -> bytes | None:
    """
    Get artifact content bytes by selector.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Artifact selector (see get_artifact).
    store : ObjectStoreProtocol
        Object store.

    Returns
    -------
    bytes or None
        Artifact content or None if not found.
    """
    art = get_artifact(record, selector)
    if not art:
        return None
    return load_artifact_bytes(art, store)


def get_artifact_text(
    record: RunRecord,
    selector: str | int,
    store: ObjectStoreProtocol,
    *,
    encoding: str = "utf-8",
) -> str | None:
    """
    Get artifact content as text by selector.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Artifact selector.
    store : ObjectStoreProtocol
        Object store.
    encoding : str, default="utf-8"
        Text encoding.

    Returns
    -------
    str or None
        Artifact text content or None if not found or decode fails.
    """
    art = get_artifact(record, selector)
    if not art:
        return None
    return load_artifact_text(art, store, encoding=encoding)


# =============================================================================
# Artifact info
# =============================================================================


@dataclass
class ArtifactInfo:
    """
    Extended artifact information for display.

    Attributes
    ----------
    ref : ArtifactRef
        Underlying artifact reference.
    index : int
        Position in artifacts list.
    name : str
        Artifact name from metadata.
    size : int or None
        Size in bytes if available.
    """

    ref: ArtifactRef
    index: int
    name: str
    size: int | None = None

    @property
    def kind(self) -> str:
        return self.ref.kind

    @property
    def digest(self) -> str:
        return self.ref.digest

    @property
    def digest_short(self) -> str:
        return self.ref.digest[:20] + "..."

    @property
    def role(self) -> str:
        return self.ref.role

    @property
    def media_type(self) -> str:
        return self.ref.media_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "media_type": self.media_type,
            "digest": self.digest,
        }
        if self.size is not None:
            d["size_bytes"] = self.size
        if self.ref.meta:
            d["meta"] = self.ref.meta
        return d

    def __repr__(self) -> str:
        size_str = f", {self.size}B" if self.size else ""
        return f"ArtifactInfo({self.index}: {self.role}/{self.kind}{size_str})"


def list_artifacts(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
    store: ObjectStoreProtocol | None = None,
) -> list[ArtifactInfo]:
    """
    List artifacts from a run record with extended info.

    Parameters
    ----------
    record : RunRecord
        Run record to list artifacts from.
    role : str, optional
        Filter by role (e.g., "program", "results", "device_snapshot").
    kind_contains : str, optional
        Filter by kind substring.
    store : ObjectStoreProtocol, optional
        If provided, include size information.

    Returns
    -------
    list of ArtifactInfo
        Artifact information sorted by role then kind.
    """
    results: list[ArtifactInfo] = []

    for i, art in enumerate(record.artifacts):
        if role and art.role != role:
            continue
        if kind_contains and kind_contains.lower() not in art.kind.lower():
            continue

        meta = art.meta or {}
        name = (
            meta.get("name") or meta.get("filename") or meta.get("program_name") or ""
        )

        size = None
        if store:
            try:
                size = store.get_size(art.digest)
            except Exception:
                pass

        results.append(ArtifactInfo(ref=art, index=i, name=name, size=size))

    results.sort(key=lambda a: (a.role, a.kind, a.index))
    logger.debug("Listed %d artifacts (filtered by role=%s)", len(results), role)
    return results
