# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Base exception hierarchy for devqubit-engine.

All public exceptions raised by devqubit-engine inherit from
:class:`DevQubitError`, enabling catch-all error handling at the
package boundary.

Examples
--------
Catch any devqubit error:

>>> from devqubit_engine.errors import DevQubitError
>>> try:
...     result = some_engine_operation()
... except DevQubitError:
...     print("engine operation failed")

Catch a specific error, with fallback:

>>> from devqubit_engine.errors import DevQubitError
>>> from devqubit_engine.storage.errors import RunNotFoundError
>>> try:
...     record = registry.load("nonexistent")
... except RunNotFoundError as exc:
...     print(f"run missing: {exc.run_id}")
... except DevQubitError:
...     print("other engine error")
"""

from __future__ import annotations


class DevQubitError(Exception):
    """
    Base exception for all devqubit-engine operations.

    Every public exception in devqubit-engine is a subclass of this
    type, so ``except DevQubitError`` is guaranteed to intercept any
    error originating from the engine.
    """
