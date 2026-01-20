# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run fingerprinting utilities.

Fingerprints are computed from Envelope snapshots with volatile fields
(timestamps, job IDs) excluded to ensure stable identifiers for
reproducibility tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.storage.artifacts.lookup import get_artifact_digests
from devqubit_engine.utils.common import sha256_digest


if TYPE_CHECKING:
    from devqubit_engine.tracking.record import RunRecord
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


def compute_fingerprints_from_envelope(envelope: ExecutionEnvelope) -> dict[str, str]:
    """
    Compute stable fingerprints from an ExecutionEnvelope.

    Fingerprints are computed from Envelope snapshots with volatile fields
    excluded. This ensures two runs with the same experimental setup produce
    the same fingerprints regardless of when they were executed.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope containing device, program, and execution snapshots.

    Returns
    -------
    dict
        Fingerprints dictionary with keys:

        - ``program``: Hash of program hashes (structural + parametric)
        - ``device``: Hash of backend identity (name, type, provider)
        - ``intent``: Hash of execution config (shots, options, transpilation)
        - ``run``: Combined hash of (program, device, intent)

    Notes
    -----
    Volatile fields excluded from fingerprinting:

    - ExecutionSnapshot: submitted_at, completed_at, job_ids, task_ids
    - DeviceSnapshot: captured_at

    The ``program`` fingerprint uses adapter-computed hashes when available.
    These hashes capture circuit structure and parameter values in a
    deterministic way that enables "same circuit?" comparison.

    Post-transpilation hashes (``executed_structural_hash``,
    ``executed_parametric_hash``) are available in the envelope for
    advanced use cases but are not included in fingerprints.
    """
    # --- Program fingerprint ---
    # Use hashes computed by adapter (they capture circuit semantics)
    program_data: dict[str, Any] = {}
    if envelope.program:
        program_data = {
            "structural_hash": envelope.program.structural_hash,
            "parametric_hash": envelope.program.parametric_hash,
            "num_circuits": envelope.program.num_circuits,
        }
    fp_program = sha256_digest({"program": program_data})

    # --- Device fingerprint ---
    # Backend identity without volatile captured_at
    device_data: dict[str, Any] = {}
    if envelope.device:
        device_data = {
            "backend_name": envelope.device.backend_name,
            "backend_type": envelope.device.backend_type,
            "provider": envelope.device.provider,
            "backend_id": envelope.device.backend_id,
            "num_qubits": envelope.device.num_qubits,
        }
    fp_device = sha256_digest({"device": device_data})

    # --- Intent fingerprint ---
    # Execution configuration without volatile fields
    # Keep producer_sdk and execution_sdk separate for determinism
    intent_data: dict[str, Any] = {
        "adapter": envelope.producer.adapter,
        "producer_sdk": envelope.producer.sdk,
    }

    if envelope.execution:
        intent_data["shots"] = envelope.execution.shots
        intent_data["options"] = envelope.execution.options
        intent_data["execution_sdk"] = envelope.execution.sdk

        # Include transpilation config (affects executed circuit)
        if envelope.execution.transpilation:
            intent_data["transpilation"] = {
                "mode": (
                    envelope.execution.transpilation.mode.value
                    if hasattr(envelope.execution.transpilation.mode, "value")
                    else str(envelope.execution.transpilation.mode)
                ),
                "optimization_level": envelope.execution.transpilation.optimization_level,
                "layout_method": envelope.execution.transpilation.layout_method,
                "routing_method": envelope.execution.transpilation.routing_method,
                "seed": envelope.execution.transpilation.seed,
            }

    fp_intent = sha256_digest({"intent": intent_data})

    # --- Combined run fingerprint ---
    fp_run = sha256_digest(
        {
            "program": fp_program,
            "device": fp_device,
            "intent": fp_intent,
        }
    )

    fingerprints: dict[str, str] = {
        "program": fp_program,
        "device": fp_device,
        "intent": fp_intent,
        "run": fp_run,
    }

    logger.debug("Computed fingerprints from envelope: run=%s...", fp_run[:16])
    return fingerprints


def compute_fingerprints_from_envelopes(
    envelopes: list[ExecutionEnvelope],
) -> dict[str, str]:
    """
    Compute stable fingerprints from multiple ExecutionEnvelopes.

    For runs with multiple circuit batches (multi-envelope), fingerprints
    are computed by aggregating data from all envelopes in a stable order.

    Parameters
    ----------
    envelopes : list of ExecutionEnvelope
        List of envelopes to compute fingerprints from. Must not be empty.
        Envelopes are sorted by envelope_id for stable ordering.

    Returns
    -------
    dict
        Fingerprints dictionary with keys:

        - ``program``: Hash of aggregated program data from all envelopes
        - ``device``: Hash of aggregated device data from all envelopes
        - ``intent``: Hash of aggregated intent data from all envelopes
        - ``run``: Combined hash of (program, device, intent)

    Raises
    ------
    ValueError
        If envelopes list is empty.

    Notes
    -----
    Aggregation strategy:

    - Sort envelopes by ``envelope_id`` for deterministic ordering
    - Collect program/device/intent data from each envelope
    - Hash the sorted list of data dictionaries

    This ensures that multi-circuit runs produce consistent fingerprints
    regardless of the order envelopes were created.
    """
    if not envelopes:
        raise ValueError("Cannot compute fingerprints from empty envelope list")

    if len(envelopes) == 1:
        return compute_fingerprints_from_envelope(envelopes[0])

    # Sort envelopes by envelope_id for stable ordering
    sorted_envelopes = sorted(envelopes, key=lambda e: e.envelope_id)

    # Aggregate program data
    program_data_list: list[dict[str, Any]] = []
    for envelope in sorted_envelopes:
        program_data: dict[str, Any] = {"envelope_id": envelope.envelope_id}
        if envelope.program:
            program_data.update(
                {
                    "structural_hash": envelope.program.structural_hash,
                    "parametric_hash": envelope.program.parametric_hash,
                    "num_circuits": envelope.program.num_circuits,
                }
            )
        program_data_list.append(program_data)
    fp_program = sha256_digest({"programs": program_data_list})

    # Aggregate device data
    device_data_list: list[dict[str, Any]] = []
    for envelope in sorted_envelopes:
        device_data: dict[str, Any] = {"envelope_id": envelope.envelope_id}
        if envelope.device:
            device_data.update(
                {
                    "backend_name": envelope.device.backend_name,
                    "backend_type": envelope.device.backend_type,
                    "provider": envelope.device.provider,
                    "backend_id": envelope.device.backend_id,
                    "num_qubits": envelope.device.num_qubits,
                }
            )
        device_data_list.append(device_data)
    fp_device = sha256_digest({"devices": device_data_list})

    # Aggregate intent data
    intent_data_list: list[dict[str, Any]] = []
    for envelope in sorted_envelopes:
        intent_data: dict[str, Any] = {
            "envelope_id": envelope.envelope_id,
            "adapter": envelope.producer.adapter,
            "producer_sdk": envelope.producer.sdk,
        }
        if envelope.execution:
            intent_data["shots"] = envelope.execution.shots
            intent_data["options"] = envelope.execution.options
            intent_data["execution_sdk"] = envelope.execution.sdk

            if envelope.execution.transpilation:
                intent_data["transpilation"] = {
                    "mode": (
                        envelope.execution.transpilation.mode.value
                        if hasattr(envelope.execution.transpilation.mode, "value")
                        else str(envelope.execution.transpilation.mode)
                    ),
                    "optimization_level": envelope.execution.transpilation.optimization_level,
                    "layout_method": envelope.execution.transpilation.layout_method,
                    "routing_method": envelope.execution.transpilation.routing_method,
                    "seed": envelope.execution.transpilation.seed,
                }
        intent_data_list.append(intent_data)
    fp_intent = sha256_digest({"intents": intent_data_list})

    # Combined run fingerprint
    fp_run = sha256_digest(
        {
            "program": fp_program,
            "device": fp_device,
            "intent": fp_intent,
        }
    )

    fingerprints: dict[str, str] = {
        "program": fp_program,
        "device": fp_device,
        "intent": fp_intent,
        "run": fp_run,
    }

    logger.debug(
        "Computed fingerprints from %d envelopes: run=%s...",
        len(envelopes),
        fp_run[:16],
    )
    return fingerprints


def compute_fingerprints(
    run: RunRecord,
    envelope: ExecutionEnvelope | None = None,
) -> dict[str, str]:
    """
    Compute stable fingerprints for a run record.

    If an envelope is provided, fingerprints are computed from it.
    Otherwise, falls back to artifact-based fingerprinting (limited).

    Parameters
    ----------
    run : RunRecord
        Run record (used for fallback if no envelope).
    envelope : ExecutionEnvelope, optional
        Envelope to compute fingerprints from. If provided, this is
        the primary source for fingerprint computation.

    Returns
    -------
    dict
        Fingerprints dictionary with keys: program, device, intent, run.

    Notes
    -----
    When envelope is available, fingerprints capture:

    - Program identity via adapter-computed hashes
    - Device identity via backend name/type/provider
    - Execution intent via shots, options, transpilation config

    Fallback (no envelope) provides limited fingerprints based on
    artifact digests only.
    """
    # Use envelope if provided
    if envelope is not None:
        return compute_fingerprints_from_envelope(envelope)

    # Fallback: artifact-based fingerprints (limited)
    logger.debug("Computing fingerprints from artifacts (no envelope)")

    record = run.record

    # Program fingerprint from artifact digests
    program_digests = get_artifact_digests(run, role="program")
    fp_program = sha256_digest({"program_artifacts": program_digests})

    # Device fingerprint from backend info in record
    backend = record.get("backend") or {}
    if not isinstance(backend, dict):
        backend = {}

    device_snapshot_digests = get_artifact_digests(run, role="device_snapshot")
    device_raw_digests = get_artifact_digests(run, role="device_raw")
    device_digests = sorted(set(device_snapshot_digests + device_raw_digests))

    fp_device = sha256_digest(
        {
            "backend": {
                "name": backend.get("name"),
                "type": backend.get("type"),
                "provider": backend.get("provider"),
            },
            "device_snapshots": device_digests,
        }
    )

    # Intent fingerprint from adapter and envelope artifacts
    intent_data: dict[str, Any] = {
        "adapter": record.get("adapter"),
    }

    # Include shots from params if logged
    params = run.params
    if "shots" in params:
        intent_data["shots"] = params["shots"]

    # Include envelope artifact digests
    envelope_digests = get_artifact_digests(run, role="envelope")
    if envelope_digests:
        intent_data["envelope_digests"] = sorted(envelope_digests)

    fp_intent = sha256_digest(intent_data)

    # Combined run fingerprint
    fp_run = sha256_digest(
        {
            "program": fp_program,
            "device": fp_device,
            "intent": fp_intent,
        }
    )

    fingerprints: dict[str, str] = {
        "program": fp_program,
        "device": fp_device,
        "intent": fp_intent,
        "run": fp_run,
    }

    logger.debug("Computed fingerprints from artifacts: run=%s...", fp_run[:16])
    return fingerprints
