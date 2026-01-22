# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Text formatting utilities for reports and CLI output.

This module provides low-level text rendering primitives used by result
formatters. It handles visual styling, alignment, and structured output
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


# =============================================================================
# Status levels
# =============================================================================


class StatusLevel(str, Enum):
    """
    Status levels for visual indicators.

    Used to communicate result status through consistent symbols
    across all formatted output.

    Attributes
    ----------
    PASS : str
        Success/passing status.
    FAIL : str
        Failure status.
    WARN : str
        Warning status.
    NA : str
        Not applicable/unavailable.
    """

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    NA = "na"


# =============================================================================
# Text styling
# =============================================================================


@dataclass(frozen=True)
class TextStyle:
    """
    Text style configuration for report rendering.

    Defines characters used for borders, separators, and status symbols.
    Supports both Unicode (default) and ASCII-safe modes.

    Attributes
    ----------
    line : str
        Character for section separator lines.
    line_bold : str
        Character for header/footer borders.
    sym_pass : str
        Symbol indicating success.
    sym_fail : str
        Symbol indicating failure.
    sym_warn : str
        Symbol indicating warning.
    sym_na : str
        Symbol indicating not applicable.

    Examples
    --------
    >>> style = TextStyle()
    >>> style.symbol(StatusLevel.PASS)
    '✓'

    >>> ascii_style = TextStyle.ascii()
    >>> ascii_style.symbol(StatusLevel.PASS)
    '+'
    """

    line: str = "─"
    line_bold: str = "═"
    sym_pass: str = "✓"
    sym_fail: str = "✗"
    sym_warn: str = "!"
    sym_na: str = "-"

    @classmethod
    def ascii(cls) -> TextStyle:
        """
        Create ASCII-safe style for environments without Unicode support.

        Returns
        -------
        TextStyle
            Style using only ASCII characters.
        """
        return cls(
            line="-",
            line_bold="=",
            sym_pass="+",
            sym_fail="x",
            sym_warn="!",
            sym_na="-",
        )

    def symbol(self, level: StatusLevel) -> str:
        """
        Get symbol for a status level.

        Parameters
        ----------
        level : StatusLevel
            Status level to get symbol for.

        Returns
        -------
        str
            Corresponding symbol character.
        """
        symbols = {
            StatusLevel.PASS: self.sym_pass,
            StatusLevel.FAIL: self.sym_fail,
            StatusLevel.WARN: self.sym_warn,
            StatusLevel.NA: self.sym_na,
        }
        return symbols.get(level, " ")


# =============================================================================
# Text renderer
# =============================================================================


class TextRenderer:
    """
    Structured text renderer for building formatted reports.

    Provides methods for creating headers, sections, key-value pairs,
    and other common report elements with consistent formatting.

    Parameters
    ----------
    width : int, default=70
        Total line width for borders and headers.
    indent : int, default=2
        Number of spaces for standard indentation.
    key_width : int, default=14
        Default width for keys in key-value pairs.
    style : TextStyle, optional
        Style configuration. Uses Unicode style if not provided.

    Examples
    --------
    >>> r = TextRenderer(width=60)
    >>> lines = r.header("REPORT")
    >>> lines += r.kv("Status:", "OK")
    >>> print(r.render(lines))
    """

    __slots__ = ("width", "indent", "key_width", "style")

    def __init__(
        self,
        width: int = 70,
        *,
        indent: int = 2,
        key_width: int = 14,
        style: TextStyle | None = None,
    ) -> None:
        self.width = width
        self.indent = indent
        self.key_width = key_width
        self.style = style or TextStyle()

    # -------------------------------------------------------------------------
    # Structural elements
    # -------------------------------------------------------------------------

    def header(self, title: str) -> list[str]:
        """
        Create centered header with bold borders.

        Parameters
        ----------
        title : str
            Header title text.

        Returns
        -------
        list of str
            Header lines (top border, title, bottom border).
        """
        pad = max(0, self.width - len(title) - 2)
        left, right = pad // 2, pad - pad // 2
        return [
            self.style.line_bold * self.width,
            f"{' ' * left} {title} {' ' * right}",
            self.style.line_bold * self.width,
        ]

    def footer(self) -> list[str]:
        """
        Create footer border line.

        Returns
        -------
        list of str
            Single footer line.
        """
        return [self.style.line_bold * self.width]

    def section(self, title: str) -> list[str]:
        """
        Create section header with underline.

        Parameters
        ----------
        title : str
            Section title text.

        Returns
        -------
        list of str
            Blank line, title, and underline.
        """
        return ["", title, self.style.line * min(len(title), self.width)]

    def divider(self) -> list[str]:
        """
        Create thin divider line.

        Returns
        -------
        list of str
            Single divider line.
        """
        return [self.style.line * self.width]

    def blank(self) -> list[str]:
        """
        Return blank line.

        Returns
        -------
        list of str
            Single empty string.
        """
        return [""]

    # -------------------------------------------------------------------------
    # Status elements
    # -------------------------------------------------------------------------

    def status_line(
        self,
        label: str,
        passed: bool,
        *,
        pass_text: str = "PASSED",
        fail_text: str = "FAILED",
    ) -> list[str]:
        """
        Create prominent status line with borders.

        Parameters
        ----------
        label : str
            Status label (e.g., "RESULT", "STATUS").
        passed : bool
            Whether status is passing.
        pass_text : str, default="PASSED"
            Text to show when passing.
        fail_text : str, default="FAILED"
            Text to show when failing.

        Returns
        -------
        list of str
            Status block with borders.
        """
        sym = self.style.sym_pass if passed else self.style.sym_fail
        text = pass_text if passed else fail_text
        return [
            "",
            self.style.line * self.width,
            f"  {label}: {sym} {text}",
            self.style.line * self.width,
        ]

    def item(
        self,
        status: StatusLevel,
        key: str,
        value: str,
        *,
        key_width: int | None = None,
    ) -> list[str]:
        """
        Format status item with symbol.

        Parameters
        ----------
        status : StatusLevel
            Status level for symbol.
        key : str
            Item key/label.
        value : str
            Item value.
        key_width : int, optional
            Override default key width.

        Returns
        -------
        list of str
            Single formatted item line.
        """
        kw = key_width or self.key_width
        sym = self.style.symbol(status)
        return [f"  {sym} {key:<{kw}} {value}"]

    # -------------------------------------------------------------------------
    # Content elements
    # -------------------------------------------------------------------------

    def kv(self, key: str, value: str, *, key_width: int | None = None) -> list[str]:
        """
        Format key-value pair.

        Parameters
        ----------
        key : str
            Key/label text.
        value : str
            Value text.
        key_width : int, optional
            Override default key width.

        Returns
        -------
        list of str
            Single formatted line.
        """
        kw = key_width or self.key_width
        return [f"{' ' * self.indent}{key:<{kw}} {value}"]

    def change(
        self,
        key: str,
        val_a: str,
        val_b: str,
        *,
        suffix: str = "",
        key_width: int | None = None,
    ) -> list[str]:
        """
        Format change line showing transition from old to new value.

        Parameters
        ----------
        key : str
            Item key/label.
        val_a : str
            Original value.
        val_b : str
            New value.
        suffix : str, default=""
            Optional suffix (e.g., percentage change).
        key_width : int, optional
            Override default key width.

        Returns
        -------
        list of str
            Single formatted change line.
        """
        kw = key_width or self.key_width
        return [f"  {key:<{kw}}  {val_a} => {val_b}{suffix}"]

    def text(self, text: str) -> list[str]:
        """
        Return single indented text line.

        Parameters
        ----------
        text : str
            Text content.

        Returns
        -------
        list of str
            Single indented line.
        """
        return [f"{' ' * self.indent}{text}"]

    def overflow(self, remaining: int, *, indent: int | None = None) -> list[str]:
        """
        Format overflow indicator for truncated lists.

        Parameters
        ----------
        remaining : int
            Number of remaining items not shown.
        indent : int, optional
            Custom indentation.

        Returns
        -------
        list of str
            Single overflow indicator line.
        """
        spaces = " " * (indent if indent is not None else self.indent)
        return [f"{spaces}... and {remaining} more"]

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def render(self, lines: Sequence[str]) -> str:
        """
        Join lines into final output string.

        Parameters
        ----------
        lines : sequence of str
            Lines to join.

        Returns
        -------
        str
            Final rendered text.
        """
        return "\n".join(lines)


# =============================================================================
# Value formatting helpers
# =============================================================================


def format_value(val: Any) -> str:
    """
    Format value for display with appropriate precision.

    Handles floats with scientific notation for very small/large values,
    and standard decimal format otherwise.

    Parameters
    ----------
    val : Any
        Value to format.

    Returns
    -------
    str
        Formatted string representation.

    Examples
    --------
    >>> format_value(0.123456)
    '0.1235'
    >>> format_value(0.00001)
    '1.0000e-05'
    >>> format_value(42)
    '42'
    """
    if isinstance(val, float):
        if abs(val) < 0.0001 or abs(val) >= 10000:
            return f"{val:.4e}"
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


def format_pct_change(val_a: Any, val_b: Any) -> str:
    """
    Format percentage change between two numeric values.

    Parameters
    ----------
    val_a : Any
        Original value.
    val_b : Any
        New value.

    Returns
    -------
    str
        Formatted percentage change with sign, or empty string
        if values are not numeric or val_a is zero.

    Examples
    --------
    >>> format_pct_change(100, 150)
    ' (+50.0%)'
    >>> format_pct_change(100, 80)
    ' (-20.0%)'
    >>> format_pct_change(0, 100)
    ''
    """
    if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
        return ""
    if val_a == 0:
        return ""
    pct = ((val_b - val_a) / abs(val_a)) * 100
    sign = "+" if pct > 0 else ""
    return f" ({sign}{pct:.1f}%)"


def truncate_id(text: str, max_len: int = 24) -> str:
    """
    Truncate text with ellipsis if it exceeds maximum length.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_len : int, default=24
        Maximum length including ellipsis.

    Returns
    -------
    str
        Original text if within limit, otherwise truncated with '...'.

    Examples
    --------
    >>> truncate_id("short")
    'short'
    >>> truncate_id("a" * 30, max_len=10)
    'aaaaaaa...'
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
