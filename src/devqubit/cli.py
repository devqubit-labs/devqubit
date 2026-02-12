# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Command-line interface entry point.

This module is registered as a console script (``devqubit``) and
delegates to the engine's CLI implementation.
"""

from __future__ import annotations


def main() -> None:
    """Run the ``devqubit`` CLI."""
    from devqubit_engine.cli import main as core_main

    core_main()


if __name__ == "__main__":
    main()
