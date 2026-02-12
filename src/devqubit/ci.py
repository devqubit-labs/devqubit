# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
CI/CD integration utilities.

Helpers for embedding devqubit verification results into continuous
integration pipelines. Supports JUnit XML reports and GitHub Actions
workflow annotations.

JUnit Reports
-------------
>>> from devqubit.compare import verify_baseline
>>> from devqubit.ci import write_junit
>>> result = verify_baseline("nightly-run", project="bell-state")
>>> write_junit(result, "results.xml")

GitHub Actions Annotations
--------------------------
>>> from devqubit.ci import github_annotations
>>> print(github_annotations(result))
::notice title=Verification Passed::Candidate matches baseline

Custom Test Names
-----------------
>>> write_junit(
...     result,
...     "results.xml",
...     testsuite_name="quantum-tests",
...     testcase_name="bell-state-verification",
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "write_junit",
    "result_to_junit",
    "github_annotations",
]


if TYPE_CHECKING:
    from devqubit_engine.compare.ci import result_to_junit, write_junit


_LAZY_IMPORTS = {
    "write_junit": ("devqubit_engine.compare.ci", "write_junit"),
    "result_to_junit": ("devqubit_engine.compare.ci", "result_to_junit"),
}


def github_annotations(result: Any) -> str:
    """
    Format a verification result as GitHub Actions workflow commands.

    Parameters
    ----------
    result : VerifyResult
        Result from :func:`~devqubit.compare.verify_baseline`.

    Returns
    -------
    str
        One or more ``::notice`` / ``::error`` workflow commands suitable
        for printing to stdout in a GitHub Actions step.

    Examples
    --------
    >>> from devqubit.compare import verify_baseline
    >>> from devqubit.ci import github_annotations
    >>> result = verify_baseline("nightly-run", project="bell-state")
    >>> print(github_annotations(result))
    """
    from devqubit_engine.compare.ci import result_to_github_annotations

    return result_to_github_annotations(result)


def __getattr__(name: str) -> Any:
    """Lazy-import handler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available public attributes."""
    return sorted(set(__all__) | set(_LAZY_IMPORTS.keys()))
