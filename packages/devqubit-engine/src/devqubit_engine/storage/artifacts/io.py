# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact I/O utilities.

Low-level functions for loading artifact content from object stores.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol

logger = logging.getLogger(__name__)


def load_artifact_bytes(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
) -> bytes | None:
    """
    Load artifact bytes from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference containing digest.
    store : ObjectStoreProtocol
        Object store to retrieve data from.

    Returns
    -------
    bytes or None
        Raw bytes, or None on failure.
    """
    try:
        return store.get_bytes(artifact.digest)
    except Exception as e:
        logger.debug("Failed to load artifact %s: %s", artifact.digest[:16], e)
        return None


def load_artifact_text(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
    *,
    encoding: str = "utf-8",
) -> str | None:
    """
    Load artifact as text from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference.
    store : ObjectStoreProtocol
        Object store.
    encoding : str, default="utf-8"
        Text encoding.

    Returns
    -------
    str or None
        Decoded text, or None on failure.
    """
    data = load_artifact_bytes(artifact, store)
    if data is None:
        return None
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        logger.debug("Failed to decode artifact as %s: %s", encoding, e)
        return None


def load_artifact_json(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
) -> Any | None:
    """
    Load and parse JSON artifact from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference containing digest.
    store : ObjectStoreProtocol
        Object store to retrieve data from.

    Returns
    -------
    Any or None
        Parsed JSON payload, or None on failure.
    """
    text = load_artifact_text(artifact, store)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse artifact as JSON: %s", e)
        return None
