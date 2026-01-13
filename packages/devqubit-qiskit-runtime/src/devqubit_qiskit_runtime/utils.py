# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for Qiskit Runtime adapter.

Provides version utilities and common helpers used across
the adapter components following the devqubit Uniform Execution
Contract (UEC).
"""

from __future__ import annotations

from typing import Any

import qiskit


def collect_sdk_versions() -> dict[str, str]:
    """
    Collect version strings for all relevant SDK packages.

    This follows the UEC requirement for tracking all SDK versions
    in the execution environment.

    Returns
    -------
    dict
        Mapping of package name to version string.

    Examples
    --------
    >>> versions = collect_sdk_versions()
    >>> versions["qiskit"]
    '1.0.0'
    >>> versions["qiskit_ibm_runtime"]
    '0.20.0'
    """
    versions: dict[str, str] = {
        "qiskit": getattr(qiskit, "__version__", "unknown"),
    }

    # Optional packages
    for pkg_name, import_name in [
        ("qiskit_ibm_runtime", "qiskit_ibm_runtime"),
        ("qiskit_aer", "qiskit_aer"),
        ("qiskit_ibm_provider", "qiskit_ibm_provider"),
    ]:
        try:
            mod = __import__(import_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return versions


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
    """
    # Try backend() method
    backend_fn = getattr(primitive, "backend", None)
    if callable(backend_fn):
        try:
            return backend_fn()
        except Exception:
            pass

    # Try backend as attribute
    backend_attr = getattr(primitive, "backend", None)
    if backend_attr is not None and not callable(backend_attr):
        return backend_attr

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
            except Exception:
                pass
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


def get_adapter_version() -> str:
    """Get adapter version dynamically from package metadata."""
    try:
        from importlib.metadata import version

        return version("devqubit-qiskit-runtime")
    except Exception:
        return "unknown"


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
    except Exception:
        return None


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


def get_session_id(primitive: Any) -> str | None:
    """
    Extract session ID from a Runtime primitive if in a session.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    str or None
        Session ID if available, None otherwise.
    """
    session = getattr(primitive, "session", None)
    if session is None:
        return None

    try:
        sid = getattr(session, "session_id", None)
        if sid is not None:
            return str(sid() if callable(sid) else sid)
    except Exception:
        pass

    return None
