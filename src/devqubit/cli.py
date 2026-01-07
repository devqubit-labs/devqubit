# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""CLI entry point."""

from __future__ import annotations


def main() -> None:
    """Entry point for devqubit CLI."""
    from devqubit_engine.cli import main as core_main

    core_main()


if __name__ == "__main__":
    main()
