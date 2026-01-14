# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Execution envelope - the top-level UEC container.

This module defines ExecutionEnvelope which unifies all four canonical
snapshots (device, program, execution, result) into a single record.

Schema Requirements (devqubit.envelope/1.0)
-------------------------------------------
REQUIRED fields:
- schema: "devqubit.envelope/1.0"
- envelope_id: ULID or UUID
- created_at: RFC3339 timestamp
- producer: ProducerInfo
- result: ResultSnapshot (with success/status)

Use ExecutionEnvelope.create() factory to ensure all required fields.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import ProgramSnapshot
from devqubit_engine.uec.result import ResultSnapshot
from devqubit_engine.uec.types import ValidationResult
from devqubit_engine.utils.time_utils import utc_now_iso


logger = logging.getLogger(__name__)


def _generate_envelope_id() -> str:
    """
    Generate a unique envelope ID.

    Returns a UUID4 without hyphens (26 chars, matches schema pattern).
    """
    return uuid.uuid4().hex[:26]


@dataclass
class ExecutionEnvelope:
    """
    Top-level container for a complete quantum execution record.

    The ExecutionEnvelope unifies all four canonical snapshots (device,
    program, execution, result) into a single, self-contained record.

    Parameters
    ----------
    envelope_id : str
        Unique envelope identifier (ULID/UUID format).
    created_at : str
        Creation timestamp (RFC3339 format).
    producer : ProducerInfo
        SDK stack information.
    result : ResultSnapshot
        Execution results (required by schema).
    device : DeviceSnapshot, optional
        Device/backend state at execution time.
    program : ProgramSnapshot, optional
        Program artifacts (logical and physical circuits).
    execution : ExecutionSnapshot, optional
        Execution metadata and configuration.
    schema_version : str
        Schema version identifier.
    metadata : dict, optional
        Additional metadata.

    Notes
    -----
    Use the ``create()`` factory method to ensure all required fields
    are properly initialized with defaults.
    """

    # REQUIRED fields (per schema)
    envelope_id: str
    created_at: str
    producer: ProducerInfo
    result: ResultSnapshot

    # Optional snapshots
    device: DeviceSnapshot | None = None
    program: ProgramSnapshot | None = None
    execution: ExecutionSnapshot | None = None

    # Schema and metadata
    schema_version: str = "devqubit.envelope/1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary matching devqubit.envelope/1.0 schema.
        """
        d: dict[str, Any] = {
            "schema": self.schema_version,
            "envelope_id": self.envelope_id,
            "created_at": self.created_at,
            "producer": self.producer.to_dict(),
            "result": self.result.to_dict(),
        }

        if self.device:
            d["device"] = self.device.to_dict()
        if self.program:
            d["program"] = self.program.to_dict()
        if self.execution:
            d["execution"] = self.execution.to_dict()
        if self.metadata:
            d["metadata"] = self.metadata

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionEnvelope:
        """
        Create ExecutionEnvelope from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with envelope fields.

        Returns
        -------
        ExecutionEnvelope
            Parsed envelope.
        """
        # Required fields
        producer = ProducerInfo.from_dict(d.get("producer", {}))
        result = ResultSnapshot.from_dict(d.get("result", {}))

        # Optional snapshots
        device = None
        if isinstance(d.get("device"), dict):
            device = DeviceSnapshot.from_dict(d["device"])

        program = None
        if isinstance(d.get("program"), dict):
            program = ProgramSnapshot.from_dict(d["program"])

        execution = None
        if isinstance(d.get("execution"), dict):
            execution = ExecutionSnapshot.from_dict(d["execution"])

        return cls(
            envelope_id=str(d.get("envelope_id", _generate_envelope_id())),
            created_at=str(d.get("created_at", utc_now_iso())),
            producer=producer,
            result=result,
            device=device,
            program=program,
            execution=execution,
            schema_version=str(d.get("schema", "devqubit.envelope/1.0")),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        *,
        producer: ProducerInfo,
        result: ResultSnapshot | None = None,
        device: DeviceSnapshot | None = None,
        program: ProgramSnapshot | None = None,
        execution: ExecutionSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionEnvelope:
        """
        Factory method to create envelope with auto-generated ID and timestamp.

        This is the recommended way to create envelopes - ensures all
        required fields are properly initialized.

        Parameters
        ----------
        producer : ProducerInfo
            SDK stack information (required).
        result : ResultSnapshot, optional
            Execution results. If None, creates empty failed result.
        device : DeviceSnapshot, optional
            Device snapshot.
        program : ProgramSnapshot, optional
            Program snapshot.
        execution : ExecutionSnapshot, optional
            Execution snapshot.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        ExecutionEnvelope
            New envelope with generated envelope_id and created_at.
        """
        # Ensure result is never None (schema requires it)
        if result is None:
            result = ResultSnapshot(
                success=False,
                status="failed",
                items=[],
                metadata={"reason": "No result provided"},
            )

        return cls(
            envelope_id=_generate_envelope_id(),
            created_at=utc_now_iso(),
            producer=producer,
            result=result,
            device=device,
            program=program,
            execution=execution,
            metadata=metadata or {},
        )

    def validate(self) -> list[str]:
        """
        Validate envelope completeness (semantic validation).

        Returns a list of warnings for missing or incomplete data.
        This is NOT schema validation - use validate_schema() for that.

        Notes
        -----
        For adapter runs (producer.adapter != "manual"), program hashes
        are expected. Missing hashes will generate warnings.

        Returns
        -------
        list of str
            Warning messages for missing data.
        """
        warnings: list[str] = []

        if not self.envelope_id:
            warnings.append("Missing envelope_id")
        if not self.created_at:
            warnings.append("Missing created_at")

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

        if self.result.success is False and self.result.error is None:
            warnings.append("Failed result missing error details")

        # For adapter runs, validate program hashes (contract enforcement)
        is_adapter_run = (
            self.producer.adapter
            and self.producer.adapter != "manual"
            and not self.metadata.get("manual_run")
        )

        if is_adapter_run and self.program:
            if not self.program.program_hash:
                warnings.append(
                    "Adapter run missing program_hash - adapters must provide "
                    "structural hash"
                )
            if not self.program.parametric_hash:
                warnings.append(
                    "Adapter run missing parametric_hash - adapters must provide "
                    "parametric hash"
                )

        return warnings

    def validate_schema(self) -> ValidationResult:
        """
        Validate envelope against JSON Schema.

        Returns ValidationResult with valid flag, errors, and warnings.

        Returns
        -------
        ValidationResult
            Validation result with errors list.
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
