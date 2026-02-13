# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

from __future__ import annotations


"""
UEC (Uniform Execution Contract) subsystem.

This package provides the canonical interface for quantum execution records.
The ExecutionEnvelope is the top-level container that unifies device, program,
execution, and result snapshots.

Primary Entry Points
--------------------
- :func:`resolve_envelope` — Get envelope from run (UEC-first strategy).
- :func:`resolve_counts` — Extract canonical counts from run.
- :func:`resolve_device_snapshot` — Extract device snapshot from run.

Data Models
-----------
- :class:`ExecutionEnvelope` — Top-level envelope container.
- :class:`DeviceSnapshot` — Device state at execution time.
- :class:`ProgramSnapshot` — Program artifacts and hashes.
- :class:`ExecutionSnapshot` — Execution configuration and job metadata.
- :class:`ResultSnapshot` — Execution results (counts, expectations, etc.).
- :class:`ProducerInfo` — SDK stack information.
"""
