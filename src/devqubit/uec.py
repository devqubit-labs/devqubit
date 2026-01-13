# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Uniform Execution Contract (UEC) snapshot schemas.

This module provides standardized types for capturing quantum experiment state
across all supported SDKs. The UEC defines four canonical snapshot types plus
a unified envelope container.

Basic Usage
-----------
>>> from devqubit.uec import ExecutionEnvelope
>>> envelope = ExecutionEnvelope(
...     device=device_snapshot,
...     program=program_snapshot,
...     execution=execution_snapshot,
...     result=result_snapshot,
... )

Validation
----------
>>> from devqubit.uec import ValidationResult
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
    from devqubit_engine.uec.device import DeviceSnapshot
    from devqubit_engine.uec.envelope import ExecutionEnvelope
    from devqubit_engine.uec.execution import ExecutionSnapshot
    from devqubit_engine.uec.program import ProgramSnapshot
    from devqubit_engine.uec.result import ResultSnapshot
    from devqubit_engine.uec.types import ValidationResult

_LAZY_IMPORTS = {
    "ExecutionEnvelope": ("devqubit_engine.uec.envelope", "ExecutionEnvelope"),
    "DeviceSnapshot": ("devqubit_engine.uec.device", "DeviceSnapshot"),
    "ProgramSnapshot": ("devqubit_engine.uec.program", "ProgramSnapshot"),
    "ExecutionSnapshot": ("devqubit_engine.uec.execution", "ExecutionSnapshot"),
    "ResultSnapshot": ("devqubit_engine.uec.result", "ResultSnapshot"),
    "ValidationResult": ("devqubit_engine.uec.types", "ValidationResult"),
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
