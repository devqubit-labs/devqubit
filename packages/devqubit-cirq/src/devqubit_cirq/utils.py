# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for Cirq adapter.

Provides version utilities and common helpers used across
the adapter components.
"""

from __future__ import annotations

from typing import Any


def cirq_version() -> str:
    """
    Get the installed Cirq version.

    Returns
    -------
    str
        Cirq version string (e.g., "1.3.0"), or "unknown" if
        Cirq is not installed or version cannot be determined.
    """
    try:
        import cirq

        return getattr(cirq, "__version__", "unknown")
    except ImportError:
        return "unknown"


def get_backend_name(executor: Any) -> str:
    """
    Extract backend name from a Cirq sampler or simulator.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator instance.

    Returns
    -------
    str
        Backend name, typically the class name (e.g., "Simulator",
        "DensityMatrixSimulator").
    """
    return executor.__class__.__name__
