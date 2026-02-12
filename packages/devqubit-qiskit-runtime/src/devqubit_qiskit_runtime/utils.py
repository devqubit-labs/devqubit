# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for Qiskit Runtime adapter.

Provides version utilities, common helpers used across the adapter components
following the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import logging
from typing import Any

import qiskit


logger = logging.getLogger(__name__)


# =============================================================================
# SDK Version Utilities
# =============================================================================


_sdk_versions: dict[str, str] | None = None


def collect_sdk_versions() -> dict[str, str]:
    """
    Collect version strings for all relevant SDK packages.

    Results are cached for the lifetime of the process since SDK
    versions cannot change at runtime.

    Returns
    -------
    dict
        Mapping of package name to version string.
    """
    global _sdk_versions
    if _sdk_versions is not None:
        return _sdk_versions

    versions: dict[str, str] = {
        "qiskit": getattr(qiskit, "__version__", "unknown"),
    }

    for pkg_name, import_name in [
        ("qiskit_ibm_runtime", "qiskit_ibm_runtime"),
        ("qiskit_aer", "qiskit_aer"),
        ("qiskit_ibm_provider", "qiskit_ibm_provider"),
    ]:
        try:
            mod = __import__(import_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError as e:
            logger.debug("Optional import unavailable: %s", e)

    _sdk_versions = versions
    return _sdk_versions


def get_adapter_version() -> str:
    """
    Get adapter version dynamically from package metadata.

    Returns
    -------
    str
        Version string of the devqubit-qiskit-runtime package,
        or "unknown" if not installed.
    """
    try:
        from importlib.metadata import version

        return version("devqubit-qiskit-runtime")
    except ImportError as e:
        logger.debug("Optional qiskit import unavailable: %s", e)
        return "unknown"


# =============================================================================
# Backend Utilities
# =============================================================================


def get_backend_obj(primitive: Any) -> Any | None:
    """
    Best-effort extraction of the backend object from a Runtime primitive.

    This is the canonical function for backend extraction used across
    all adapter modules.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance (Sampler/Estimator).

    Returns
    -------
    Any or None
        Backend-like object if found, else None.

    Notes
    -----
    Tries multiple approaches in order:
    1. backend() method call
    2. backend attribute access
    3. mode attribute (V2 primitives can use Backend/Session/Batch as mode)
    4. mode.backend attribute/method
    """
    # Try backend as method or attribute
    backend = getattr(primitive, "backend", None)
    if backend is not None:
        if callable(backend):
            try:
                return backend()
            except (TypeError, ValueError) as e:
                logger.debug("Unexpected error: %s", e)
        else:
            return backend

    # V2 primitives: mode can be Backend/Session/Batch
    mode = getattr(primitive, "mode", None)
    if mode is not None:
        # Mode itself might be backend-like
        if (
            hasattr(mode, "target")
            or hasattr(mode, "num_qubits")
            or hasattr(mode, "properties")
        ):
            return mode

        # Mode might have a backend attribute/method
        mode_backend = getattr(mode, "backend", None)
        if callable(mode_backend):
            try:
                return mode_backend()
            except (AttributeError, TypeError) as e:
                logger.debug("Failed to call mode backend accessor: %s", e)
        if mode_backend is not None:
            return mode_backend

    return None


def get_backend_name(primitive: Any) -> str:
    """
    Get backend name from a Runtime primitive.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance (Sampler/Estimator).

    Returns
    -------
    str
        Backend name or class name as fallback.

    Examples
    --------
    >>> sampler = SamplerV2(backend)
    >>> get_backend_name(sampler)
    'ibm_brisbane'
    """
    # Try _backend first (common internal attribute)
    backend = getattr(primitive, "_backend", None)

    # Fall back to canonical extraction
    if backend is None:
        backend = get_backend_obj(primitive)

    # Try to get name from backend
    if backend is not None:
        name = getattr(backend, "name", None)
        if name is not None:
            return str(name() if callable(name) else name)

    # Fallback to primitive class name
    return primitive.__class__.__name__


# =============================================================================
# Primitive Utilities
# =============================================================================


def get_primitive_type(executor: Any) -> str:
    """
    Determine if primitive is sampler or estimator.

    Parameters
    ----------
    executor : Any
        Runtime primitive instance.

    Returns
    -------
    str
        'sampler' or 'estimator'.

    Examples
    --------
    >>> get_primitive_type(SamplerV2(backend))
    'sampler'
    >>> get_primitive_type(EstimatorV2(backend))
    'estimator'
    """
    cls = executor.__class__.__name__.lower()
    return "estimator" if "estimator" in cls else "sampler"


def is_runtime_primitive(executor: Any) -> bool:
    """
    Check if an executor is a Qiskit Runtime primitive.

    Parameters
    ----------
    executor : Any
        Potential executor instance.

    Returns
    -------
    bool
        True if executor is a Runtime Sampler or Estimator.

    Examples
    --------
    >>> is_runtime_primitive(SamplerV2(backend))
    True
    >>> is_runtime_primitive(AerSimulator())
    False
    """
    if executor is None or not hasattr(executor, "run"):
        return False
    mod = getattr(executor, "__module__", "") or ""
    if not mod.startswith("qiskit_ibm_runtime"):
        return False
    cls = executor.__class__.__name__.lower()
    return "sampler" in cls or "estimator" in cls


def extract_job_id(job: Any) -> str | None:
    """
    Extract job ID from a Runtime job instance.

    Parameters
    ----------
    job : Any
        Runtime job instance.

    Returns
    -------
    str or None
        Job ID if available, None otherwise.

    Examples
    --------
    >>> job = sampler.run([circuit])
    >>> extract_job_id(job)
    'cq1234567890'
    """
    try:
        jid = getattr(job, "job_id", None)
        if jid is None:
            return None
        return str(jid() if callable(jid) else jid)
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to get job_id: %s", e)
        return None
