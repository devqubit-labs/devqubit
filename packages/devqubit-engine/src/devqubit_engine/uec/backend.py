# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Backend resolution helpers.

This module provides utilities for resolving and classifying quantum backends
from various SDK executors and primitives.
"""

from __future__ import annotations

from typing import Any


def resolve_physical_backend(executor: Any) -> dict[str, Any] | None:
    """
    Resolve the physical backend from a high-level executor.

    This is the universal backend resolution helper that all adapters
    should use to extract the underlying physical backend from wrapped
    or multi-layer executors.

    Parameters
    ----------
    executor : Any
        Executor, primitive, device, or backend object from any SDK.

    Returns
    -------
    dict or None
        Dictionary with resolved backend information:

        - provider: Physical provider name
        - backend_name: Backend identifier
        - backend_id: Stable unique ID (if available)
        - backend_type: Type (hardware/simulator)
        - backend_obj: The actual backend object

        Returns None if resolution fails.

    Examples
    --------
    >>> from qiskit_aer import AerSimulator
    >>> info = resolve_physical_backend(AerSimulator())
    >>> info["backend_type"]
    'simulator'
    """
    if executor is None:
        return None

    result: dict[str, Any] = {
        "provider": "unknown",
        "backend_name": "unknown",
        "backend_id": None,
        "backend_type": "simulator",
        "backend_obj": executor,
    }

    executor_type = type(executor).__name__

    # Check for name attribute (common across SDKs)
    if hasattr(executor, "name"):
        name = getattr(executor, "name", None)
        if callable(name):
            try:
                name = name()
            except Exception:
                name = None
        if name:
            result["backend_name"] = str(name)

    # Detect simulator vs hardware from name/type
    name_lower = result["backend_name"].lower()
    type_lower = executor_type.lower()

    if any(
        s in name_lower
        for s in (
            "ibm_",
            "ionq",
            "rigetti",
            "oqc",
            "aspen",
        )
    ):
        result["backend_type"] = "hardware"
    elif any(
        s in name_lower or s in type_lower
        for s in (
            "sim",
            "emulator",
            "fake",
        )
    ):
        result["backend_type"] = "simulator"

    return result


def classify_backend_type(name: str) -> str:
    """
    Classify backend type from name.

    Parameters
    ----------
    name : str
        Backend name.

    Returns
    -------
    str
        Backend type: "hardware" or "simulator".
    """
    name_lower = name.lower()

    # Hardware indicators
    hardware_indicators = (
        "ibm_",
        "ionq",
        "rigetti",
        "oqc",
        "aspen",
        "quantinuum",
        "iqm",
    )

    if any(indicator in name_lower for indicator in hardware_indicators):
        return "hardware"

    # Simulator indicators
    simulator_indicators = (
        "sim",
        "emulator",
        "fake",
        "mock",
        "test",
    )

    if any(indicator in name_lower for indicator in simulator_indicators):
        return "simulator"

    # Default to unknown (safer to assume simulator)
    return "simulator"
