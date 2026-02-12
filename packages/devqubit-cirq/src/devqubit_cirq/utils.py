# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for Cirq adapter.

Provides version utilities and common helpers used across
the adapter components.
"""

from __future__ import annotations

from typing import Any


# Module-level caches – populated on first call, stable for process lifetime.
_cirq_version: str | None = None
_adapter_version: str | None = None


def cirq_version() -> str:
    """
    Get the installed Cirq version (cached after first call).

    Returns
    -------
    str
        Cirq version string (e.g., "1.3.0"), or "unknown" if
        Cirq is not installed or version cannot be determined.
    """
    global _cirq_version
    if _cirq_version is None:
        try:
            import cirq

            _cirq_version = getattr(cirq, "__version__", "unknown")
        except ImportError:
            _cirq_version = "unknown"
    return _cirq_version


def get_adapter_version() -> str:
    """Get adapter version dynamically from package metadata (cached)."""
    global _adapter_version
    if _adapter_version is None:
        try:
            from importlib.metadata import version

            _adapter_version = version("devqubit-cirq")
        except ImportError:
            _adapter_version = "unknown"
    return _adapter_version


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
