# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
PennyLane adapter for devqubit.

Provides automatic tape capture, device snapshot, and result logging for
PennyLane devices via in-place device patching.  Supports QNode
workflows, parameter sweeps, and multi-layer stacks (e.g. PennyLane →
Braket, PennyLane → Qiskit).  Registered as a ``devqubit.adapters``
entry point and discovered automatically by :meth:`Run.wrap`.

This package is an internal implementation detail of
``devqubit[pennylane]``.  Users should import from :mod:`devqubit`,
not from this package directly.
"""
