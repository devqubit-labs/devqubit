# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Simple query language for searching runs.

This module provides a minimal query parser and evaluator for filtering
runs by params, metrics, and tags without requiring a full database.
It supports AND-joined conditions with various comparison operators.

Query Syntax
------------
Queries consist of conditions joined by AND::

    field op value [and field op value ...]

Supported Fields
----------------
- ``params.*`` : Experiment parameters
- ``metric.*`` or ``metrics.*`` : Logged metrics
- ``tag.*`` or ``tags.*`` : String tags
- ``project``, ``adapter``, ``status``, ``backend``, ``fingerprint`` : Top-level fields

Supported Operators
-------------------
- ``=`` : Equals
- ``!=`` : Not equals
- ``>`` : Greater than
- ``>=`` : Greater than or equals
- ``<`` : Less than
- ``<=`` : Less than or equals
- ``~`` : Contains (case-insensitive substring match)
- ``exists`` : Field exists

Examples
--------
>>> from devqubit_engine.query import parse_query, search_records

>>> # Parse a query
>>> query = parse_query("metric.fidelity > 0.95 and params.shots = 1000")

>>> # Search records
>>> results = search_records(
...     records,
...     "metric.fidelity > 0.95",
...     sort_by="metric.fidelity",
...     descending=True,
...     limit=10,
... )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from devqubit_engine.errors import DevQubitError


logger = logging.getLogger(__name__)


class Op(Enum):
    """
    Comparison operators.

    Attributes
    ----------
    EQ : str
        Equals (``=``).
    NE : str
        Not equals (``!=``).
    GT : str
        Greater than (``>``).
    GE : str
        Greater than or equals (``>=``).
    LT : str
        Less than (``<``).
    LE : str
        Less than or equals (``<=``).
    CONTAINS : str
        Substring match for strings (``~``).
    EXISTS : str
        Field exists check.
    """

    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    CONTAINS = "~"
    EXISTS = "exists"


# Map of string operators to Op enum for faster lookup
_OP_MAP: dict[str, Op] = {op.value: op for op in Op}


@dataclass(frozen=True)
class Condition:
    """
    Single query condition.

    Attributes
    ----------
    field : str
        Dot-separated field path (e.g., "params.shots", "metric.fidelity").
    op : Op
        Comparison operator.
    value : Any
        Value to compare against. None for EXISTS op.

    Examples
    --------
    >>> cond = Condition(field="metric.fidelity", op=Op.GT, value=0.95)
    >>> cond = Condition(field="params.custom", op=Op.EXISTS, value=None)
    """

    field: str
    op: Op
    value: Any

    def __repr__(self) -> str:
        """Return string representation."""
        if self.op == Op.EXISTS:
            return f"Condition({self.field} exists)"
        return f"Condition({self.field} {self.op.value} {self.value!r})"


@dataclass
class Query:
    """
    Parsed query consisting of AND-joined conditions.

    For MVP, only AND logic is supported. OR can be added later if needed.

    Attributes
    ----------
    conditions : list of Condition
        All conditions must match (AND logic).

    Examples
    --------
    >>> query = parse_query("params.shots = 1000 and metric.fidelity > 0.9")
    >>> len(query.conditions)
    2
    """

    conditions: list[Condition]

    def __repr__(self) -> str:
        """Return string representation."""
        if not self.conditions:
            return "Query(empty)"
        return f"Query({len(self.conditions)} conditions)"

    def __bool__(self) -> bool:
        """Return True if query has conditions."""
        return len(self.conditions) > 0


# Token patterns for lexer (case-insensitive for keywords)
_TOKEN_PATTERNS = [
    ("AND", r"\band\b"),
    ("OR", r"\bor\b"),  # recognized but not fully supported in MVP
    ("EXISTS", r"\bexists\b"),
    ("OP", r">=|<=|!=|>|<|=|~"),
    ("NUMBER", r"-?\d+\.?\d*"),
    ("STRING", r'"[^"]*"|\'[^\']*\''),
    ("FIELD", r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("WS", r"\s+"),
]

# Compile with IGNORECASE for keyword matching (and, AND, And all work)
_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS),
    re.IGNORECASE,
)


class QueryParseError(DevQubitError, ValueError):
    """
    Raised when query parsing fails.

    This exception is raised when the query string contains
    invalid syntax, unrecognized operators, or unsupported
    constructs.
    """


def _tokenize(query: str) -> list[tuple[str, str]]:
    """
    Tokenize a query string.

    Parameters
    ----------
    query : str
        Query string to tokenize.

    Returns
    -------
    list of tuple
        List of (token_type, value) pairs.

    Raises
    ------
    QueryParseError
        If query contains invalid tokens.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(query):
        match = _TOKEN_RE.match(query, pos)
        if not match:
            # Show context around the error
            context = query[pos : pos + 20]
            if len(query) > pos + 20:
                context += "..."
            raise QueryParseError(f"Invalid token at position {pos}: {context!r}")
        token_type = match.lastgroup
        value = match.group()
        pos = match.end()
        if token_type and token_type != "WS":
            tokens.append((token_type, value))
    return tokens


def _parse_value(token_type: str, token_value: str) -> Any:
    """Parse a token value into Python type."""
    if token_type == "NUMBER":
        if "." in token_value:
            return float(token_value)
        return int(token_value)
    if token_type == "STRING":
        # Strip quotes
        return token_value[1:-1]
    if token_type == "FIELD":
        # Bare word treated as string
        return token_value
    return token_value


def _parse_op(op_str: str) -> Op:
    """
    Parse operator string to Op enum.

    Parameters
    ----------
    op_str : str
        Operator string (e.g., ">=", "~").

    Returns
    -------
    Op
        Parsed operator.

    Raises
    ------
    QueryParseError
        If operator is not recognized.
    """
    op = _OP_MAP.get(op_str)
    if op is None:
        valid_ops = ", ".join(_OP_MAP.keys())
        raise QueryParseError(
            f"Unknown operator: {op_str!r}. Valid operators: {valid_ops}"
        )
    return op


def parse_query(query_str: str) -> Query:
    """
    Parse a query string into a Query object.

    Parameters
    ----------
    query_str : str
        Query string in format::

            field op value [and field op value ...]

        Supported fields:
        - ``params.*`` : experiment parameters
        - ``metric.*`` or ``metrics.*`` : logged metrics
        - ``tags.*`` or ``tag.*`` : string tags
        - ``project``, ``adapter``, ``status``, ``backend``, ``fingerprint`` : top-level

        Supported operators:
        - ``=`` (equals), ``!=`` (not equals)
        - ``>``, ``>=``, ``<``, ``<=`` (numeric comparison)
        - ``~`` (contains, for strings)
        - ``exists`` (field exists)

    Returns
    -------
    Query
        Parsed query.

    Raises
    ------
    QueryParseError
        If query syntax is invalid.

    Examples
    --------
    >>> q = parse_query("metric.fidelity > 0.95")
    >>> q = parse_query("params.shots = 1000 and tags.device ~ ibm")
    >>> q = parse_query("params.custom_field exists")
    >>> q = parse_query("")  # Returns empty query that matches all
    """
    if not query_str or not query_str.strip():
        logger.debug("Empty query, will match all records")
        return Query(conditions=[])

    tokens = _tokenize(query_str)
    conditions: list[Condition] = []
    i = 0

    while i < len(tokens):
        # Expect: FIELD [OP VALUE | EXISTS]
        if tokens[i][0] != "FIELD":
            raise QueryParseError(
                f"Expected field name, got '{tokens[i][1]}' ({tokens[i][0]})"
            )
        field = tokens[i][1]
        i += 1

        if i >= len(tokens):
            raise QueryParseError(f"Expected operator after field '{field}'")

        # Check for EXISTS
        if tokens[i][0] == "EXISTS":
            conditions.append(Condition(field=field, op=Op.EXISTS, value=None))
            i += 1
        elif tokens[i][0] == "OP":
            op_str = tokens[i][1]
            op = _parse_op(op_str)  # Now properly converts to QueryParseError
            i += 1

            if i >= len(tokens):
                raise QueryParseError(f"Expected value after operator '{op_str}'")

            value = _parse_value(tokens[i][0], tokens[i][1])
            conditions.append(Condition(field=field, op=op, value=value))
            i += 1
        else:
            raise QueryParseError(
                f"Expected operator after '{field}', got '{tokens[i][1]}'"
            )

        # Check for AND (explicit connector required between conditions)
        if i < len(tokens):
            if tokens[i][0] == "AND":
                i += 1
            elif tokens[i][0] == "OR":
                raise QueryParseError(
                    "OR not supported. Use separate queries or combine results."
                )
            elif tokens[i][0] == "FIELD":
                # Implicit AND - allow for convenience
                logger.debug("Implicit AND between conditions at position %d", i)

    logger.debug("Parsed query with %d conditions", len(conditions))
    return Query(conditions=conditions)


def _get_nested_value(obj: dict[str, Any], path: str) -> tuple[bool, Any]:
    """
    Get nested value from dict by dot-separated path.

    Parameters
    ----------
    obj : dict
        Dictionary to traverse.
    path : str
        Dot-separated field path.

    Returns
    -------
    tuple
        (found: bool, value: Any)
    """
    parts = path.split(".")
    current: Any = obj

    for part in parts:
        if not isinstance(current, dict):
            return (False, None)
        if part not in current:
            return (False, None)
        current = current[part]

    return (True, current)


def _resolve_field(record: dict[str, Any], field: str) -> tuple[bool, Any]:
    """
    Resolve a field path against a run record.

    Handles aliasing for common patterns:
    - ``metric.*`` → ``data.metrics.*``
    - ``metrics.*`` → ``data.metrics.*``
    - ``params.*`` → ``data.params.*``
    - ``tags.*`` → ``data.tags.*``
    - ``tag.*`` → ``data.tags.*``

    Parameters
    ----------
    record : dict
        Run record dictionary.
    field : str
        Field path.

    Returns
    -------
    tuple
        (found: bool, value: Any)
    """
    # Handle top-level aliases with correct "exists" semantics
    if field == "project":
        if "project" not in record:
            return (False, None)
        proj = record["project"]
        if isinstance(proj, dict):
            if "name" not in proj:
                return (False, None)
            return (True, proj["name"])
        return (True, str(proj) if proj else "")

    if field == "adapter":
        if "adapter" not in record:
            return (False, None)
        return (True, record["adapter"])

    if field == "status":
        info = record.get("info")
        if not isinstance(info, dict) or "status" not in info:
            return (False, None)
        return (True, info["status"])

    if field == "backend":
        backend = record.get("backend")
        if not isinstance(backend, dict) or "name" not in backend:
            return (False, None)
        return (True, backend["name"])

    if field == "fingerprint":
        fps = record.get("fingerprints")
        if not isinstance(fps, dict) or "run" not in fps:
            return (False, None)
        return (True, fps["run"])

    # Handle prefixed fields
    if field.startswith("metric.") or field.startswith("metrics."):
        # Strip prefix and look in data.metrics
        suffix = field.split(".", 1)[1]
        data = record.get("data")
        if not isinstance(data, dict):
            return (False, None)
        return _get_nested_value(data, f"metrics.{suffix}")

    if field.startswith("params."):
        suffix = field.split(".", 1)[1]
        data = record.get("data")
        if not isinstance(data, dict):
            return (False, None)
        return _get_nested_value(data, f"params.{suffix}")

    if field.startswith("tag.") or field.startswith("tags."):
        suffix = field.split(".", 1)[1]
        data = record.get("data")
        if not isinstance(data, dict):
            return (False, None)
        return _get_nested_value(data, f"tags.{suffix}")

    # Direct path lookup
    return _get_nested_value(record, field)


def _coerce_numeric_pair(left: Any, right: Any) -> tuple[Any, Any]:
    """
    Coerce a str/numeric pair so both operands share a numeric type.

    Parameters
    ----------
    left : Any
        Left operand.
    right : Any
        Right operand.

    Returns
    -------
    tuple
        Coerced (left, right). If coercion fails, originals are returned.
    """
    if isinstance(right, (int, float)) and isinstance(left, str):
        try:
            return float(left), right
        except (ValueError, TypeError):
            return left, right
    if isinstance(left, (int, float)) and isinstance(right, str):
        try:
            return left, float(right)
        except (ValueError, TypeError):
            return left, right
    return left, right


def _compare(left: Any, op: Op, right: Any) -> bool:
    """
    Compare two values with an operator.

    Parameters
    ----------
    left : Any
        Left operand (from record).
    op : Op
        Comparison operator.
    right : Any
        Right operand (from query).

    Returns
    -------
    bool
        Comparison result.
    """
    if op == Op.EXISTS:
        return True  # If we got here, field exists

    if op == Op.EQ:
        left, right = _coerce_numeric_pair(left, right)
        return left == right

    if op == Op.NE:
        left, right = _coerce_numeric_pair(left, right)
        return left != right

    if op == Op.CONTAINS:
        return str(right).lower() in str(left).lower()

    # Numeric comparisons
    try:
        left_f = float(left)
        right_f = float(right)
    except (ValueError, TypeError):
        return False

    if op == Op.GT:
        return left_f > right_f
    if op == Op.GE:
        return left_f >= right_f
    if op == Op.LT:
        return left_f < right_f
    if op == Op.LE:
        return left_f <= right_f

    return False


def matches_query(record: dict[str, Any], query: Query) -> bool:
    """
    Check if a run record matches a query.

    Parameters
    ----------
    record : dict
        Run record dictionary.
    query : Query
        Parsed query.

    Returns
    -------
    bool
        True if all conditions match (AND logic).

    Examples
    --------
    >>> query = parse_query("params.shots = 1000")
    >>> if matches_query(record, query):
    ...     print("Record matches!")
    """
    if not query.conditions:
        return True

    for cond in query.conditions:
        found, value = _resolve_field(record, cond.field)

        if cond.op == Op.EXISTS:
            if not found:
                return False
            continue

        if not found:
            return False

        if not _compare(value, cond.op, cond.value):
            return False

    return True


def search_records(
    records: Sequence[dict[str, Any]],
    query: Query | str,
    *,
    sort_by: str | None = None,
    descending: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search and optionally sort records matching a query.

    Parameters
    ----------
    records : sequence of dict
        Run records to search.
    query : Query or str
        Parsed query or query string.
    sort_by : str, optional
        Field to sort by (e.g., "metric.fidelity", "created_at").
    descending : bool, default=True
        Sort in descending order.
    limit : int, optional
        Maximum results to return.

    Returns
    -------
    list of dict
        Matching records, optionally sorted and limited.
        Records with missing sort fields are placed at the end.

    Examples
    --------
    >>> # Find high-fidelity runs
    >>> results = search_records(
    ...     records,
    ...     "metric.fidelity > 0.95",
    ...     sort_by="metric.fidelity",
    ...     limit=10,
    ... )

    >>> # Find runs with specific params
    >>> results = search_records(records, "params.shots >= 1000")

    >>> # Find runs with tag
    >>> results = search_records(records, "tags.experiment ~ bell")
    """
    if isinstance(query, str):
        query = parse_query(query)

    results = [r for r in records if matches_query(r, query)]
    logger.debug("Query matched %d of %d records", len(results), len(records))

    if sort_by:
        # Separate records with and without the sort field
        keyed: list[tuple[Any, dict[str, Any]]] = []
        missing: list[dict[str, Any]] = []

        for rec in results:
            found, val = _resolve_field(rec, sort_by)
            if not found or val is None:
                missing.append(rec)
                continue
            try:
                key: Any = float(val)
            except (ValueError, TypeError):
                key = str(val)
            keyed.append((key, rec))

        # Sort records that have the field, missing always at end
        keyed.sort(key=lambda x: x[0], reverse=descending)
        results = [rec for _, rec in keyed] + missing

    if limit is not None and limit > 0:
        results = results[:limit]

    return results
