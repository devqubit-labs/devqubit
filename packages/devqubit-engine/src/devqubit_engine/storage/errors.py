# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Storage and registry exceptions."""

from __future__ import annotations

from devqubit_engine.errors import DevQubitError


class RegistryError(DevQubitError):
    """Base exception for registry operations."""


class StorageError(DevQubitError):
    """Base exception for storage operations."""


class ObjectNotFoundError(StorageError):
    """
    Raised when a requested object does not exist in the store.

    Parameters
    ----------
    digest : str
        Content-address digest of the missing object.
    """

    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__(f"Object not found: {digest}")


class RunNotFoundError(StorageError):
    """
    Raised when a requested run does not exist in the registry.

    Parameters
    ----------
    run_id : str
        Identifier of the missing run.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run not found: {run_id}")
