# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Jinja2 template filters for the devqubit UI.

This module provides custom Jinja2 filters for formatting data
in templates. Filters are registered with the template engine
during application initialization.

Examples
--------
In templates, use filters with the pipe operator:

.. code-block:: html+jinja

    {{ run.created_at | ago }}
    {{ run.run_id | short_id }}
    {{ artifact.digest | short_digest }}
    {{ data | json_pretty }}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi.templating import Jinja2Templates


def register_filters(templates: "Jinja2Templates") -> None:
    """
    Register all custom template filters.

    Parameters
    ----------
    templates : Jinja2Templates
        The FastAPI Jinja2Templates instance to register filters with.

    Notes
    -----
    Filters are added to ``templates.env.filters`` and become available
    in all templates rendered by this engine.
    """
    templates.env.filters["json_pretty"] = json_pretty
    templates.env.filters["ago"] = time_ago
    templates.env.filters["short_id"] = short_id
    templates.env.filters["short_digest"] = short_digest


def json_pretty(value: Any) -> str:
    """
    Pretty-print JSON data with indentation.

    Parameters
    ----------
    value : Any
        Value to format. Can be a dict, list, or JSON string.

    Returns
    -------
    str
        Formatted JSON string with 2-space indentation.

    Examples
    --------
    >>> json_pretty({"key": "value", "nested": {"a": 1}})
    '{\\n  "key": "value",\\n  "nested": {\\n    "a": 1\\n  }\\n}'

    >>> json_pretty('{"compact": true}')
    '{\\n  "compact": true\\n}'
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def time_ago(value: Any) -> str:
    """
    Convert timestamp to human-readable relative time.

    Parameters
    ----------
    value : Any
        Timestamp value. Accepts ISO format strings, datetime objects,
        or any object with string representation.

    Returns
    -------
    str
        Human-readable relative time (e.g., "5m ago", "2h ago", "3d ago").

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> now = datetime.now(timezone.utc)
    >>> time_ago(now - timedelta(minutes=5))
    '5m ago'

    >>> time_ago(now - timedelta(hours=2))
    '2h ago'

    >>> time_ago(now - timedelta(days=3))
    '3d ago'

    Notes
    -----
    Time ranges:

    - < 60 seconds: "just now"
    - < 60 minutes: "Xm ago"
    - < 24 hours: "Xh ago"
    - >= 24 hours: "Xd ago"
    """
    if not value:
        return ""

    try:
        if isinstance(value, str):
            # Handle ISO format with or without timezone
            value = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            dt = value
        else:
            return str(value)[:19]

        # Ensure timezone-aware comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()

        if seconds < 0:
            return "in the future"
        elif seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"

    except (ValueError, TypeError, AttributeError):
        # Fallback: return truncated string representation
        return str(value)[:19]


def short_id(value: str | None, length: int = 12) -> str:
    """
    Shorten a run ID for display.

    Parameters
    ----------
    value : str or None
        The full run ID string.
    length : int, default=12
        Number of characters to show before truncation.

    Returns
    -------
    str
        Shortened ID with ellipsis, or empty string if None.

    Examples
    --------
    >>> short_id("abc123def456ghi789")
    'abc123def456...'

    >>> short_id(None)
    ''

    >>> short_id("short")
    'short'
    """
    if not value:
        return ""
    if len(value) <= length:
        return value
    return value[:length] + "..."


def short_digest(value: str | None, length: int = 20) -> str:
    """
    Shorten a content digest for display.

    Parameters
    ----------
    value : str or None
        The full digest string (e.g., SHA-256 hash).
    length : int, default=20
        Number of characters to show before truncation.

    Returns
    -------
    str
        Shortened digest with ellipsis, or empty string if None.

    Examples
    --------
    >>> short_digest("sha256:abc123def456ghi789jkl012mno345pqr678")
    'sha256:abc123def456...'

    >>> short_digest(None)
    ''
    """
    if not value:
        return ""
    if len(value) <= length:
        return value
    return value[:length] + "..."
