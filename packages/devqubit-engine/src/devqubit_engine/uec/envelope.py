# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Execution envelope - the top-level UEC container.

This module defines ExecutionEnvelope which unifies all four canonical
snapshots (device, program, execution, result) into a single record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.program import ProgramSnapshot
from devqubit_engine.uec.result import ResultSnapshot
from devqubit_engine.uec.types import ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class ExecutionEnvelope:
    """
    Top-level container for a complete quantum execution record.

    The ExecutionEnvelope unifies all four canonical snapshots (device,
    program, execution, result) into a single, self-contained record.

    Parameters
    ----------
    device : DeviceSnapshot, optional
        Device/backend state at execution time.
    program : ProgramSnapshot, optional
        Program artifacts (logical and physical circuits).
    execution : ExecutionSnapshot, optional
        Execution metadata and configuration.
    result : ResultSnapshot, optional
        Execution results.
    adapter : str, optional
        Adapter that created this envelope.
    envelope_id : str, optional
        Unique envelope identifier.
    created_at : str, optional
        Creation timestamp.
    schema_version : str
        Schema version identifier.
    """

    device: DeviceSnapshot | None = None
    program: ProgramSnapshot | None = None
    execution: ExecutionSnapshot | None = None
    result: ResultSnapshot | None = None

    adapter: str | None = None
    envelope_id: str | None = None
    created_at: str | None = None

    schema_version: str = "devqubit.envelope/0.1"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema": self.schema_version}

        if self.device:
            d["device"] = self.device.to_dict()
        if self.program:
            d["program"] = self.program.to_dict()
        if self.execution:
            d["execution"] = self.execution.to_dict()
        if self.result:
            d["result"] = self.result.to_dict()
        if self.adapter:
            d["adapter"] = self.adapter
        if self.envelope_id:
            d["envelope_id"] = self.envelope_id
        if self.created_at:
            d["created_at"] = self.created_at

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionEnvelope:
        device = None
        if isinstance(d.get("device"), dict):
            device = DeviceSnapshot.from_dict(d["device"])

        program = None
        if isinstance(d.get("program"), dict):
            program = ProgramSnapshot.from_dict(d["program"])

        execution = None
        if isinstance(d.get("execution"), dict):
            execution = ExecutionSnapshot.from_dict(d["execution"])

        result = None
        if isinstance(d.get("result"), dict):
            result = ResultSnapshot.from_dict(d["result"])

        return cls(
            device=device,
            program=program,
            execution=execution,
            result=result,
            adapter=d.get("adapter"),
            envelope_id=d.get("envelope_id"),
            created_at=d.get("created_at"),
            schema_version=d.get("schema", "devqubit.envelope/0.1"),
        )

    def validate(self) -> list[str]:
        """
        Validate envelope completeness (semantic validation).

        Returns a list of warnings for missing or incomplete data.
        """
        warnings: list[str] = []

        if not self.device:
            warnings.append("Missing device snapshot")
        elif not self.device.backend_name:
            warnings.append("Device snapshot missing backend_name")

        if not self.program:
            warnings.append("Missing program snapshot")
        elif not self.program.logical and not self.program.physical:
            warnings.append("Program snapshot has no artifacts")

        if not self.execution:
            warnings.append("Missing execution snapshot")

        if not self.result:
            warnings.append("Missing result snapshot")
        elif self.result.success is False and not self.result.error_message:
            warnings.append("Failed result missing error_message")

        return warnings

    def validate_schema(self) -> ValidationResult:
        """
        Validate envelope against JSON Schema.

        Returns ValidationResult with valid flag, errors, and warnings.
        """
        try:
            from devqubit_engine.schema.validation import validate_envelope

            errors = validate_envelope(self.to_dict(), raise_on_error=False)

            if errors:
                return ValidationResult(valid=False, errors=errors, warnings=[])

            return ValidationResult(valid=True, errors=[], warnings=[])

        except ImportError:
            logger.warning(
                "Schema validation not available: "
                "devqubit_engine.schema.validation module not found"
            )
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=["Schema validation module not available"],
            )

        except Exception as e:
            logger.warning("Schema validation failed unexpectedly: %s", e)
            return ValidationResult(valid=False, errors=[e], warnings=[])


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
    """
    if executor is None:
        return None

    result: dict[str, Any] = {
        "provider": "unknown",
        "backend_name": "unknown",
        "backend_id": None,
        "backend_type": "unknown",
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

    if any(s in name_lower or s in type_lower for s in ("sim", "emulator", "fake")):
        result["backend_type"] = "simulator"
    elif any(s in name_lower for s in ("ibm_", "ionq", "rigetti", "oqc", "aspen")):
        result["backend_type"] = "hardware"

    return result
