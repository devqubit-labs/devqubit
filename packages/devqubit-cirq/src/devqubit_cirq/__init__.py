# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Google Cirq adapter for devqubit.

Provides automatic circuit capture (Cirq JSON + OpenQASM 3), simulator
snapshots, and result logging for Cirq samplers and simulators.
Supports ``run``, ``run_sweep``, and ``simulate`` execution modes.
Registered as a ``devqubit.adapters`` entry point and discovered
automatically by :meth:`Run.wrap`.

This package is an internal implementation detail of ``devqubit[cirq]``.
Users should import from :mod:`devqubit`, not from this package directly.
"""
