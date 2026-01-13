# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Uniform Execution Contract (UEC) snapshot schemas.

This module provides standardized types for capturing quantum experiment state
across all supported SDKs. The UEC defines four canonical snapshot types that
every adapter must produce, plus a unified envelope container.

Snapshot Hierarchy
------------------
ExecutionEnvelope
    Top-level container unifying all snapshots for a single execution.

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
"""
