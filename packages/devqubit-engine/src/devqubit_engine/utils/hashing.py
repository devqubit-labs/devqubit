# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Cryptographic hashing utilities.

This module provides functions for computing SHA-256 digests of bytes
and Python objects. All digests are returned in the format ``sha256:<hex>``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 digest of raw bytes.

    Parameters
    ----------
    data : bytes
        Bytes to hash.

    Returns
    -------
    str
        Digest in format ``sha256:<64-hex-chars>``.
    """
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_digest(obj: Any) -> str:
    """
    Compute SHA-256 digest of a canonicalized Python object.

    The object is serialized to JSON with sorted keys and compact
    separators to ensure stable, reproducible hashes across different
    Python sessions and machines.

    Parameters
    ----------
    obj : Any
        Object to hash. Must be JSON-serializable (or have a ``str``
        representation for non-serializable types).

    Returns
    -------
    str
        Digest in format ``sha256:<64-hex-chars>``.

    Notes
    -----
    The canonicalization ensures:

    - Dictionary keys are sorted alphabetically
    - No extra whitespace (compact separators)
    - Non-ASCII characters are preserved (not escaped)
    - Non-serializable objects fall back to their string representation
    """
    # Canonical JSON serialization for stable hashing
    canonical_json = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"sha256:{hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()}"


def verify_digest(data: bytes, expected_digest: str) -> bool:
    """
    Verify that data matches an expected digest.

    Parameters
    ----------
    data : bytes
        Data to verify.
    expected_digest : str
        Expected digest in format ``sha256:<hex>``.

    Returns
    -------
    bool
        True if the computed digest matches the expected digest.
    """
    computed = sha256_bytes(data)
    return computed == expected_digest
