# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class ObjectNotFoundError(StorageError):
    """Raised when a requested object doesn't exist in the store."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__(f"Object not found: {digest}")


class RunNotFoundError(StorageError):
    """Raised when a requested run doesn't exist in the registry."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run not found: {run_id}")
