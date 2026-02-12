# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
IBM Qiskit Runtime adapter for devqubit.

Provides automatic circuit capture, transpilation management, device
calibration snapshots, and result logging for Qiskit Runtime V2
primitives (``SamplerV2``, ``EstimatorV2``).  Registered as a
``devqubit.adapters`` entry point and discovered automatically by
:meth:`Run.wrap`.

This package is an internal implementation detail of
``devqubit[qiskit-runtime]``.  Users should import from :mod:`devqubit`,
not from this package directly.
"""
