# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Amazon Braket adapter for devqubit.

Provides automatic circuit capture (OpenQASM 3), device property
snapshots, and result logging for Braket local simulators and managed
QPUs.  Registered as a ``devqubit.adapters`` entry point and discovered
automatically by :meth:`Run.wrap`.

This package is an internal implementation detail of
``devqubit[braket]``.  Users should import from :mod:`devqubit`,
not from this package directly.
"""
