# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Time utilities."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """
    Return current UTC time as an ISO 8601 string.

    The returned timestamp has second precision and uses the "Z" suffix
    to indicate UTC (instead of "+00:00").

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
