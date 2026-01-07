# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison and verification utilities.

Basic comparison is available from the main module:

>>> from devqubit import compare, verify_against_baseline

This submodule provides additional types and utilities.

Result Types
------------
>>> from devqubit.compare import ComparisonResult, VerifyResult, VerifyPolicy
>>> result = diff("run_a", "run_b")
>>> assert isinstance(result, ComparisonResult)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    # Result types
    "ComparisonResult",
    "VerifyResult",
    "VerifyPolicy",
]


if TYPE_CHECKING:
    from devqubit_engine.compare.results import (
        ComparisonResult,
        VerifyResult,
    )
    from devqubit_engine.compare.verify import VerifyPolicy


_LAZY_IMPORTS = {
    "ComparisonResult": ("devqubit_engine.compare.results", "ComparisonResult"),
    "VerifyResult": ("devqubit_engine.compare.results", "VerifyResult"),
    "VerifyPolicy": ("devqubit_engine.compare.verify", "VerifyPolicy"),
}


def __getattr__(name: str) -> Any:
    """Lazy import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
