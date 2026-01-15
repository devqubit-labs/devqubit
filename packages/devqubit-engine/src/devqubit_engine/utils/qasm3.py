# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
OpenQASM 3 canonicalization utilities.

This module provides functions for normalizing OpenQASM 3 source code
to enable stable, cross-SDK hashing and comparison. The canonicalization
is regex-based and does not fully parse OpenQASM 3.

Canonicalization Steps
----------------------
1. **Whitespace normalization**: Remove comments, collapse whitespace
2. **Float normalization**: Round floats to fixed precision
3. **Name normalization**: Rename qubit registers to canonical names
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

from devqubit_engine.utils.common import sha256_bytes


logger = logging.getLogger(__name__)


# Regex patterns for QASM3 normalization
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

# Float pattern: matches numeric literals including scientific notation.
# Allows underscores in digits (OpenQASM allows underscores in numeric literals).
# Negative lookbehind/ahead to avoid matching within identifiers.
_DIGITS = r"\d[\d_]*"
_FLOAT_PATTERN = re.compile(
    rf"(?<![\w.])([+-]?(?:(?:{_DIGITS}\.{_DIGITS}?|\.{_DIGITS})(?:[eE][+-]?{_DIGITS})?"
    rf"|{_DIGITS}[eE][+-]?{_DIGITS}))(?![\w.])"
)

# Qubit declaration pattern: "qubit[<expr>] name" or "qubit name"
# Captures the designator contents as text (best-effort; not a full parser).
_QUBIT_DECL_PATTERN = re.compile(r"\bqubit(?:\s*\[\s*([^\]]+?)\s*\])?\s+(\w+)\b")


def _strip_comments_preserving_strings(source: str) -> str:
    """
    Strip OpenQASM-style comments while preserving string literal contents.

    Notes
    -----
    This is a small lexer-like pass, not a full OpenQASM parser. It removes:

    - Line comments starting with ``//`` until end of line
    - Block comments delimited by ``/*`` and ``*/``

    Comment markers inside single- or double-quoted string literals are preserved.
    """
    out: list[str] = []
    i = 0
    n = len(source)

    in_string: str | None = None
    escape = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_string is not None:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        # Enter string literal
        if ch in ('"', "'"):
            in_string = ch
            out.append(ch)
            i += 1
            continue

        # Line comment: // ... \n
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            # keep the newline (if any) to preserve statement boundaries
            continue

        # Block comment: /* ... */
        if ch == "/" and nxt == "*":
            i += 2
            newlines = 0
            while i < n - 1:
                if source[i] == "\n":
                    newlines += 1
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                # Unterminated block comment: treat rest as comment
                i = n

            if newlines:
                out.append("\n" * newlines)
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _normalize_line_whitespace_preserving_strings(line: str) -> str:
    """
    Collapse runs of spaces/tabs to single spaces outside string literals.
    """
    out: list[str] = []
    buf: list[str] = []

    in_string: str | None = None
    escape = False

    def flush_buf() -> None:
        if not buf:
            return
        segment = "".join(buf)
        out.append(_WHITESPACE_PATTERN.sub(" ", segment))
        buf.clear()

    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        if in_string is not None:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'"):
            flush_buf()
            in_string = ch
            out.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush_buf()
    return "".join(out).strip()


def _normalize_whitespace(source: str) -> str:
    """
    Strip comments and normalize whitespace in OpenQASM 3 source.

    Parameters
    ----------
    source : str
        OpenQASM 3 source code.

    Returns
    -------
    str
        Source with comments removed and whitespace normalized.
    """
    # Normalize newlines across platforms
    source = source.replace("\r\n", "\n").replace("\r", "\n")

    # Remove comments (OpenQASM-style), preserving string contents
    source = _strip_comments_preserving_strings(source)

    # Normalize whitespace line-by-line (outside strings) and drop empty lines
    lines = []
    for raw in source.split("\n"):
        norm = _normalize_line_whitespace_preserving_strings(raw)
        if norm:
            lines.append(norm)

    return "\n".join(lines)


def _normalize_floats(source: str, precision: int = 10) -> str:
    """
    Normalize floating-point literals to a stable textual form.

    Parameters
    ----------
    source : str
        OpenQASM 3 source code.
    precision : int, optional
        Number of significant digits. Default is 10.

    Returns
    -------
    str
        Source with floats normalized.
    """
    if precision < 1:
        return source

    def replace_float(match: re.Match[str]) -> str:
        token = match.group(1)
        token_clean = token.replace("_", "")
        try:
            val = float(token_clean)
        except (ValueError, OverflowError):
            return token  # Leave as-is if it doesn't parse

        # Format with specified precision
        s = f"{val:.{precision}g}"

        # Ensure a decimal point for non-exponent integer-looking values
        if "." not in s and "e" not in s.lower():
            s += ".0"

        # Canonicalize signed zero
        if s in ("-0.0", "+0.0"):
            s = "0.0"

        return s

    return _FLOAT_PATTERN.sub(replace_float, source)


def _sub_outside_strings(source: str, pattern: re.Pattern[str], repl: str) -> str:
    """
    Apply a regex substitution only outside single- and double-quoted strings.
    """
    out: list[str] = []
    i = 0
    n = len(source)
    start = 0

    in_string: str | None = None
    escape = False

    while i < n:
        ch = source[i]

        if in_string is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'"):
            # flush non-string region
            out.append(pattern.sub(repl, source[start:i]))
            # capture string region
            in_string = ch
            j = i + 1
            escape = False
            while j < n:
                cj = source[j]
                if escape:
                    escape = False
                elif cj == "\\":
                    escape = True
                elif cj == in_string:
                    j += 1
                    break
                j += 1
            out.append(source[i:j])
            i = j
            start = i
            in_string = None
            continue

        i += 1

    out.append(pattern.sub(repl, source[start:]))
    return "".join(out)


def _normalize_qubit_names(source: str) -> tuple[str, list[str]]:
    """
    Rename qubit registers to canonical names (q0, q1, ...) by declaration order.

    Parameters
    ----------
    source : str
        OpenQASM 3 source code.

    Returns
    -------
    tuple
        - source_out: Updated source with renamed registers
        - warnings: List of warnings produced during normalization
    """
    warnings: list[str] = []

    # Find all qubit declarations (best-effort: includes qubit-typed params too)
    decls = list(_QUBIT_DECL_PATTERN.finditer(source))
    if not decls:
        return source, warnings

    # Build rename map preserving declaration order
    rename_map: dict[str, str] = {}
    order: list[tuple[str, str | None]] = []  # (original_name, size_or_none)

    for match in decls:
        size = match.group(1)  # May be None for scalar qubit
        name = match.group(2)
        if name not in rename_map:
            rename_map[name] = f"q{len(rename_map)}"
            order.append((name, size.strip() if isinstance(size, str) else None))

    out = source

    # First: rewrite declarations
    for old_name, size in order:
        new_name = rename_map[old_name]
        if old_name == new_name:
            continue

        if size is None:
            # Scalar: "qubit old" -> "qubit new"
            pat = re.compile(rf"\bqubit\s+{re.escape(old_name)}\b")
            out = _sub_outside_strings(out, pat, f"qubit {new_name}")
        else:
            # Array: "qubit[N] old" -> "qubit[N] new"
            pat = re.compile(
                rf"\bqubit\s*\[\s*{re.escape(size)}\s*\]\s+{re.escape(old_name)}\b"
            )
            out = _sub_outside_strings(out, pat, f"qubit[{size}] {new_name}")

    # Second: rewrite usages
    for old_name, size in order:
        new_name = rename_map[old_name]
        if old_name == new_name:
            continue

        # Array indexing: "old[" -> "new["
        pat_index = re.compile(rf"\b{re.escape(old_name)}\s*\[")
        out = _sub_outside_strings(out, pat_index, f"{new_name}[")

        if size is None:
            # Scalar usages: replace whole-word occurrences
            pat_scalar = re.compile(rf"\b{re.escape(old_name)}\b")
            out = _sub_outside_strings(out, pat_scalar, new_name)
        else:
            # Array register can be used without indexing (e.g., measure q -> b;)
            pat_array = re.compile(rf"\b{re.escape(old_name)}\b")
            out = _sub_outside_strings(out, pat_array, new_name)

    # Flag heuristic rename for scalar qubits
    if any(size is None for _, size in order):
        warnings.append("scalar_qubit_rename_heuristic")

    return out, warnings


def canonicalize_qasm3(
    source: str,
    *,
    normalize_floats: bool = True,
    float_precision: int = 10,
    normalize_names: bool = True,
) -> dict[str, Any]:
    """
    Canonicalize OpenQASM 3 source for stable, cross-SDK hashing.

    The canonical form enables best-effort comparisons across SDKs that
    emit slightly different textual OpenQASM.

    Parameters
    ----------
    source : str
        OpenQASM 3 source code.
    normalize_floats : bool, optional
        Round floating-point literals to fixed precision. Default is True.
    float_precision : int, optional
        Significant digits for float normalization. Default is 10.
    normalize_names : bool, optional
        Rename qubit registers to canonical names (q0, q1, ...).
        Default is True.

    Returns
    -------
    dict
        Dictionary containing:

        - ``digest`` (str): SHA-256 digest of the canonical form
        - ``canonical_source`` (str): Canonicalized OpenQASM 3 source
        - ``normalization_applied`` (list of str): Steps applied
        - ``warnings`` (list of str): Warnings produced during canonicalization

    Raises
    ------
    TypeError
        If source is not a string.

    Notes
    -----
    This function is regex-based and does not fully parse OpenQASM 3.
    It assumes OpenQASM-style line comments (``//``) and block comments
    (``/* ... */``). Complex programs may not canonicalize correctly.
    """
    if not isinstance(source, str):
        raise TypeError(
            f"OpenQASM 3 source must be a string, got {type(source).__name__}"
        )

    steps: list[str] = []
    warnings: list[str] = []

    # Step 1: Normalize whitespace and remove comments
    canonical = _normalize_whitespace(source)
    steps.append("whitespace_normalized")

    # Step 2: Normalize floating-point literals
    if normalize_floats:
        canonical = _normalize_floats(canonical, float_precision)
        steps.append(f"floats_normalized_p{int(float_precision)}")

    # Step 3: Normalize qubit register names
    if normalize_names:
        try:
            canonical, name_warnings = _normalize_qubit_names(canonical)
            steps.append("qubit_names_normalized")
            warnings.extend(name_warnings)
        except Exception as e:
            warnings.append(f"qubit_name_normalization_failed: {e}")
            logger.warning("Qubit name normalization failed: %s", e)

    # Compute digest of canonical form
    digest = sha256_bytes(canonical.encode("utf-8"))

    logger.debug(
        "Canonicalized QASM3: %d chars -> %d chars, steps=%s",
        len(source),
        len(canonical),
        steps,
    )

    return {
        "digest": digest,
        "canonical_source": canonical,
        "normalization_applied": steps,
        "warnings": warnings,
    }


def coerce_openqasm3_sources(
    source: str | Sequence[str] | Sequence[Mapping[str, Any]] | Mapping[str, str],
    *,
    default_name: str,
) -> list[dict[str, Any]]:
    """
    Normalize various OpenQASM 3 input formats into a list of program items.

    This function accepts multiple input formats for convenience and normalizes
    them into a consistent list-of-dicts format.

    Parameters
    ----------
    source : str, sequence, or mapping
        OpenQASM 3 input in one of these formats:

        - ``str``: Single OpenQASM 3 program
        - ``Sequence[str]``: Multiple programs (names auto-assigned)
        - ``Sequence[Mapping]``: Items with ``{"source": "...", "name": "..."}``
        - ``Mapping[str, str]``: Name-to-source mapping

    default_name : str
        Base name for auto-assigned names (e.g., "program").

    Returns
    -------
    list of dict
        List of program items, each with keys:

        - ``name`` (str): Program name
        - ``source`` (str): OpenQASM 3 source code
        - ``index`` (int): Zero-based index

    Raises
    ------
    TypeError
        If input format is not recognized or an item is malformed.

    Examples
    --------
    Single program:

    >>> coerce_openqasm3_sources("OPENQASM 3.0; ...", default_name="circuit")
    [{'name': 'circuit', 'source': 'OPENQASM 3.0; ...', 'index': 0}]

    Multiple programs:

    >>> coerce_openqasm3_sources(["qasm1", "qasm2"], default_name="prog")
    [{'name': 'prog[0]', 'source': 'qasm1', 'index': 0},
     {'name': 'prog[1]', 'source': 'qasm2', 'index': 1}]

    Named programs:

    >>> coerce_openqasm3_sources({"bell": "...", "ghz": "..."}, default_name="x")
    [{'name': 'bell', 'source': '...', 'index': 0},
     {'name': 'ghz', 'source': '...', 'index': 1}]
    """
    if not isinstance(default_name, str) or not default_name.strip():
        raise TypeError("default_name must be a non-empty string")

    # Case 1: Single string
    if isinstance(source, str):
        return [{"name": default_name, "source": source, "index": 0}]

    # Case 2: Mapping (name -> source)
    if isinstance(source, Mapping):
        items: list[dict[str, Any]] = []
        for i, (name, src) in enumerate(source.items()):
            if not isinstance(src, str):
                raise TypeError(
                    f"OpenQASM3 mapping values must be strings, "
                    f"got {type(src).__name__} for key {name!r}"
                )
            items.append({"name": str(name), "source": src, "index": i})
        return items

    # Case 3: Sequence (list of strings or dicts)
    if isinstance(source, Sequence):
        items = []
        for i, item in enumerate(source):
            if isinstance(item, str):
                items.append(
                    {
                        "name": f"{default_name}[{i}]",
                        "source": item,
                        "index": i,
                    }
                )
            elif isinstance(item, Mapping):
                src = item.get("source")
                if not isinstance(src, str):
                    raise TypeError(
                        f"Each OpenQASM3 item must include a string 'source' key, "
                        f"missing or invalid at index {i}"
                    )
                name = item.get("name")
                items.append(
                    {
                        "name": (
                            str(name)
                            if isinstance(name, str) and name
                            else f"{default_name}[{i}]"
                        ),
                        "source": src,
                        "index": i,
                    }
                )
            else:
                raise TypeError(
                    f"OpenQASM3 sequence items must be str or mapping, "
                    f"got {type(item).__name__} at index {i}"
                )
        return items

    raise TypeError(
        f"OpenQASM3 sources must be str, Mapping[name->source], or Sequence, "
        f"got {type(source).__name__}"
    )
