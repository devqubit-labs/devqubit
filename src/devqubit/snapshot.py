# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Uniform Execution Contract (UEC) snapshot schemas.

This module provides standardized types for capturing quantum experiment state
across all supported SDKs. The UEC defines four canonical snapshot types plus
a unified envelope container.

Basic Usage
-----------
>>> from devqubit.snapshot import ExecutionEnvelope
>>> envelope = ExecutionEnvelope(
...     device=device_snapshot,
...     program=program_snapshot,
...     execution=execution_snapshot,
...     result=result_snapshot,
... )

Validation
----------
>>> from devqubit.snapshot import ValidationResult
>>> result = envelope.validate_schema()
>>> if result.valid:
...     print("Schema valid")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "ExecutionEnvelope",
    "DeviceSnapshot",
    "ProgramSnapshot",
    "ExecutionSnapshot",
    "ResultSnapshot",
    "ValidationResult",
]


if TYPE_CHECKING:
    from devqubit_engine.core.snapshot import (
        DeviceSnapshot,
        ExecutionEnvelope,
        ExecutionSnapshot,
        ProgramSnapshot,
        ResultSnapshot,
        ValidationResult,
    )


_LAZY_IMPORTS = {
    "ExecutionEnvelope": ("devqubit_engine.snapshot", "ExecutionEnvelope"),
    "DeviceSnapshot": ("devqubit_engine.snapshot", "DeviceSnapshot"),
    "ProgramSnapshot": ("devqubit_engine.snapshot", "ProgramSnapshot"),
    "ExecutionSnapshot": ("devqubit_engine.snapshot", "ExecutionSnapshot"),
    "ResultSnapshot": ("devqubit_engine.snapshot", "ResultSnapshot"),
    "ValidationResult": ("devqubit_engine.snapshot", "ValidationResult"),
}


def __getattr__(name: str) -> Any:
    """Lazy import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
