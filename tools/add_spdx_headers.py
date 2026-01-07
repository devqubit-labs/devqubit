#!/usr/bin/env python3
"""
Insert minimal SPDX headers into devqubit Python sources.

This script adds the following header to Python files that don't already contain
an SPDX license identifier:

    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: 2026 devqubit

It is designed for two use cases:

1) Manual one-off run (no arguments):
   - Scans these directories (if present): src/, packages/, examples/

2) pre-commit hook run (filenames passed as arguments):
   - Processes only the provided file paths.

Notes
-----
- If a file contains a shebang, it stays on the first line.
- If a file contains a PEP 263 encoding cookie, it stays in the first or second
  line region (after optional shebang).
- Files named ``__init__.py`` are skipped by default.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


# Configuration
TARGET_DIRS: tuple[str, ...] = ("src", "packages", "examples")
SKIP_FILENAMES: set[str] = {"__init__.py"}

LICENSE_ID = "Apache-2.0"
COPYRIGHT_TEXT = f"{datetime.now().year} devqubit"

# PEP 263 encoding declaration regex
_ENCODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
_SPDX_MARKER_BYTES = b"SPDX-License-Identifier:"
_UTF8_BOM = b"\xef\xbb\xbf"


def _iter_targets(repo_root: Path) -> Iterable[Path]:
    """Yield Python files under TARGET_DIRS."""
    for d in TARGET_DIRS:
        base = repo_root / d
        if base.exists():
            yield from base.rglob("*.py")


def _build_header(newline: str) -> str:
    """Return the SPDX header using the given newline."""
    return (
        f"# SPDX-License-Identifier: {LICENSE_ID}{newline}"
        f"# SPDX-FileCopyrightText: {COPYRIGHT_TEXT}{newline}"
        f"{newline}"
    )


def _insertion_index(lines: list[str]) -> int:
    """Insert after shebang and encoding cookie (if present)."""
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if idx < len(lines) and _ENCODING_RE.match(lines[idx]):
        idx += 1
    return idx


def add_spdx_header_to_file(path: Path) -> bool:
    """
    Add the SPDX header to `path` if missing.

    Parameters
    ----------
    path
        Path to a Python file.

    Returns
    -------
    bool
        True if the file was modified, otherwise False.
    """
    if path.name in SKIP_FILENAMES:
        return False

    # ---- Fast path: read a small prefix and skip quickly if SPDX is already present.
    try:
        prefix = path.read_bytes()[:8192]
    except OSError:
        return False

    if _SPDX_MARKER_BYTES in prefix:
        return False

    # ---- Slow path: full read/modify/write
    raw = path.read_bytes()
    had_bom = raw.startswith(_UTF8_BOM)

    try:
        text = raw.decode("utf-8-sig")  # strips BOM if present
    except UnicodeDecodeError:
        # Skip non-UTF8 files (avoid corruption).
        return False

    lines = text.splitlines(keepends=True)

    # In case SPDX appears after the first 8KB for some reason
    if _SPDX_MARKER_BYTES in text.encode("utf-8", errors="ignore"):
        return False

    newline = "\r\n" if "\r\n" in text else "\n"
    header = _build_header(newline)

    idx = _insertion_index(lines)
    new_text = "".join(lines[:idx]) + header + "".join(lines[idx:])

    out = new_text.encode("utf-8")
    if had_bom:
        out = _UTF8_BOM + out

    path.write_bytes(out)
    return True


def main() -> int:
    """Run on argv paths (pre-commit) or scan repo targets (manual)."""
    changed = 0

    if len(sys.argv) > 1:
        candidates = (Path(p) for p in sys.argv[1:])
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = _iter_targets(repo_root)

    for p in candidates:
        if p.is_file() and p.suffix == ".py":
            if add_spdx_header_to_file(p):
                changed += 1
                print(f"SPDX added: {p}")

    if len(sys.argv) == 1:
        print(f"\nDone. Updated {changed} file(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
