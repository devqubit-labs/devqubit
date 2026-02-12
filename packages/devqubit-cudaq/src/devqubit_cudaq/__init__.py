# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
NVIDIA CUDA-Q adapter for devqubit.

Provides automatic kernel capture (JSON, MLIR/Quake, QIR), target
snapshots, and result logging for CUDA-Q ``sample`` and ``observe``
workflows.  Registered as a ``devqubit.adapters`` entry point and
discovered automatically by :meth:`Run.wrap`.

This package is an internal implementation detail of ``devqubit[cudaq]``.
Users should import from :mod:`devqubit`, not from this package directly.
"""
