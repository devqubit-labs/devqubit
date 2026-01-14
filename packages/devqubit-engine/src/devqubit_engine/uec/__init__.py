# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Uniform Execution Contract (UEC) snapshot schemas.

This module provides standardized types for capturing quantum experiment
state across all supported SDKs. The UEC defines canonical snapshot types
that every adapter must produce, plus a unified envelope container.

Key Components
--------------
resolver
    The primary entry point for obtaining envelope data.
    Use ``resolve_envelope()`` to get ExecutionEnvelope from any run,
    whether it was created with an adapter or manually.

    Import directly: ``from devqubit_engine.uec.resolver import resolve_envelope``

envelope
    ExecutionEnvelope - top-level container unifying all snapshots.

Snapshot Hierarchy
------------------
ExecutionEnvelope
    Top-level container unifying all snapshots for a single execution.

    ProducerInfo
        SDK stack information for debug/compatibility.

    DeviceSnapshot
        Point-in-time capture of quantum backend state.

        DeviceCalibration
            Calibration data with per-qubit and per-gate metrics.

        FrontendConfig
            Frontend/primitive configuration for multi-layer stacks.

    ProgramSnapshot
        Program artifacts with logical/physical distinction.

    ExecutionSnapshot
        Submission, compilation, and job tracking metadata.

    ResultSnapshot
        Raw result references and normalized summaries.

        ResultItem
            Per-item results for batch executions.

        CountsFormat
            Bit ordering and source SDK metadata.

        QuasiProbability
            Quasi-distributions from error mitigation.

Requirements
------------
- ``producer`` is REQUIRED (SDK stack tracking)
- ``result.success`` and ``result.status`` are REQUIRED
- ``result.items[]`` is always a list (even single executions)
- ``counts.format`` MUST describe bit ordering
- ``quasi_probabilities`` are first-class (IBM Runtime)

Usage
-----
>>> from devqubit_engine.uec.resolver import resolve_envelope
>>> envelope = resolve_envelope(record, store)
>>> # envelope is always ExecutionEnvelope, never None

>>> from devqubit_engine.uec.envelope import ExecutionEnvelope
>>> envelope = ExecutionEnvelope.create(producer=producer, result=result)
"""
