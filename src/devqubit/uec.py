# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Uniform Execution Contract (UEC) snapshot schemas.

The UEC defines a standardized representation for quantum experiment
state across all supported SDKs. SDK adapters produce these snapshots
automatically; this module exposes the types for advanced inspection
and custom adapter development.

.. note::

   You rarely need to create these objects directly. Adapters populate
   them during ``run.wrap()`` executions.

Types
-----
:class:`ExecutionEnvelope`
    Top-level container holding all four snapshots plus metadata.
:class:`DeviceSnapshot`
    Backend identity, topology, and calibration at execution time.
:class:`ProgramSnapshot`
    Logical and physical circuit artifacts with structural hashes.
:class:`ExecutionSnapshot`
    Submission metadata, shots, job IDs, transpilation info.
:class:`ResultSnapshot`
    Normalized measurement results (counts, quasi-probabilities,
    or expectation values).
:class:`ValidationResult`
    Schema validation outcome from :meth:`ExecutionEnvelope.validate_schema`.

Example
-------
>>> from devqubit.uec import ExecutionEnvelope
>>> envelope = ExecutionEnvelope.create(
...     producer=producer, device=device,
...     program=program, execution=execution, result=result,
... )
>>> validation = envelope.validate_schema()
>>> assert validation.ok
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
    from devqubit_engine.uec.models.device import DeviceSnapshot
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope, ValidationResult
    from devqubit_engine.uec.models.execution import ExecutionSnapshot
    from devqubit_engine.uec.models.program import ProgramSnapshot
    from devqubit_engine.uec.models.result import ResultSnapshot

_LAZY_IMPORTS = {
    "ExecutionEnvelope": ("devqubit_engine.uec.models.envelope", "ExecutionEnvelope"),
    "DeviceSnapshot": ("devqubit_engine.uec.models.device", "DeviceSnapshot"),
    "ProgramSnapshot": ("devqubit_engine.uec.models.program", "ProgramSnapshot"),
    "ExecutionSnapshot": ("devqubit_engine.uec.models.execution", "ExecutionSnapshot"),
    "ResultSnapshot": ("devqubit_engine.uec.models.result", "ResultSnapshot"),
    "ValidationResult": ("devqubit_engine.uec.models.envelope", "ValidationResult"),
}


def __getattr__(name: str) -> Any:
    """Lazy-import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available public attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
