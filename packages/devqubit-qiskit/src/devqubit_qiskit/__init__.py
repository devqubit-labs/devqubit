"""
Qiskit adapter for devqubit.

Provides automatic circuit capture (QPY + OpenQASM 3), backend snapshot,
and result logging for Qiskit and Aer backends.  Registered as a
``devqubit.adapters`` entry point and discovered automatically by
:meth:`Run.wrap`.

This package is an internal implementation detail of ``devqubit[qiskit]``.
Users should import from :mod:`devqubit`, not from this package directly.
"""
