# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Common utilities.

This module provides small, shared utility functions:
- Time utilities (UTC timestamps)
- Run record helpers (manual run detection)
- Cryptographic hashing (SHA-256)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """
    Return current UTC time as an ISO 8601 string.

    Returns
    -------
    str
        ISO 8601 formatted UTC timestamp (e.g., "2024-01-15T10:30:00Z").
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_manual_run_record(record: dict[str, Any]) -> bool:
    """
    Check if run record represents a manual run.

    Parameters
    ----------
    record : dict
        Run record dictionary.

    Returns
    -------
    bool
        True if manual run, False if adapter run.

    Examples
    --------
    >>> is_manual_run_record({"adapter": "manual"})
    True
    >>> is_manual_run_record({"adapter": "devqubit-qiskit"})
    False
    """
    adapter = record.get("adapter")
    if not adapter or adapter == "" or adapter == "manual":
        return True
    return False


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
    separators to ensure stable, reproducible hashes.

    Parameters
    ----------
    obj : Any
        Object to hash. Must be JSON-serializable.

    Returns
    -------
    str
        Digest in format ``sha256:<64-hex-chars>``.
    """
    canonical_json = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"sha256:{hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()}"
