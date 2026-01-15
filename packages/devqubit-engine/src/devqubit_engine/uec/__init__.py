# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC (Uniform Execution Contract) subsystem.

This package provides the canonical interface for quantum execution records.
The ExecutionEnvelope is the top-level container that unifies device, program,
execution, and result snapshots.

Subpackages
-----------
- ``models`` - Data models (snapshots and containers)
- ``resolution`` - Envelope loading, synthesis, and resolution

Primary Entry Points
--------------------
- :func:`resolve_envelope` - Get envelope from run (UEC-first strategy)
- :class:`ExecutionEnvelope` - Top-level envelope container

Data Types
----------
- :class:`DeviceSnapshot` - Device state at execution time
- :class:`ProgramSnapshot` - Program artifacts
- :class:`ExecutionSnapshot` - Execution configuration
- :class:`ResultSnapshot` - Execution results
- :class:`ArtifactRef` - Content-addressed artifact reference
"""
