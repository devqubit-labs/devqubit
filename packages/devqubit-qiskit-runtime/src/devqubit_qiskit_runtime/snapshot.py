# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot creation for Qiskit Runtime primitives.

Creates structured DeviceSnapshot objects from Runtime primitives,
capturing backend configuration, calibration, and primitive frontend
configuration following the devqubit Uniform Execution Contract (UEC).

The UEC uses a multi-layer stack model where:
- Frontend: The primitive (e.g., SamplerV2) that the user interacts with
- Resolved backend: The physical backend (e.g., ibm_brisbane) where execution occurs

This module composes:
1. A FrontendConfig describing the primitive layer
2. A DeviceSnapshot from the underlying backend (reusing devqubit_qiskit.snapshot)
3. Runtime-specific metadata (options, session info)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.device import DeviceSnapshot, FrontendConfig
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_engine.utils.time_utils import utc_now_iso
from devqubit_qiskit.snapshot import create_device_snapshot as create_backend_snapshot
from devqubit_qiskit_runtime.utils import (
    collect_sdk_versions,
    get_backend_name,
    get_backend_obj,
    get_primitive_type,
)


if TYPE_CHECKING:
    from devqubit_engine.core.run import Run

logger = logging.getLogger(__name__)


def _detect_physical_provider(primitive: Any) -> str:
    """
    Detect physical provider from Runtime primitive (not SDK).

    UEC requires provider to be the physical backend provider,
    not the SDK name. SDK goes in producer.frontends[].

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    str
        Physical provider: "ibm_quantum", "fake", or "local".
    """
    backend = get_backend_obj(primitive)
    if backend is None:
        # No backend resolved - check primitive module
        module_name = getattr(primitive, "__module__", "").lower()
        if "ibm" in module_name:
            return "ibm_quantum"
        return "local"

    module_name = type(backend).__module__.lower()
    backend_name = get_backend_name(primitive).lower()

    # IBM quantum hardware
    if "ibm" in module_name or "ibm_" in backend_name:
        # Check for fake backends
        if "fake" in module_name or "fake" in backend_name:
            return "fake"
        return "ibm_quantum"

    # Aer simulator (local)
    if "aer" in module_name:
        return "aer"

    return "local"


def _extract_backend_id(primitive: Any) -> str | None:
    """
    Extract a stable backend identifier from a Runtime primitive.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    str or None
        Backend ID if available.
    """
    backend = get_backend_obj(primitive)
    if backend is None:
        return None

    # Try IBM-specific backend_id
    try:
        if hasattr(backend, "backend_id"):
            bid = backend.backend_id
            if callable(bid):
                bid = bid()
            if bid:
                return str(bid)
    except Exception:
        pass

    # Try instance ID
    try:
        if hasattr(backend, "instance"):
            inst = backend.instance
            if inst:
                return str(inst)
    except Exception:
        pass

    return None


def _build_frontend_config(primitive: Any) -> FrontendConfig:
    """
    Build FrontendConfig for the Runtime primitive layer.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    FrontendConfig
        Configuration describing the primitive frontend.
    """
    primitive_class = primitive.__class__.__name__
    primitive_type = get_primitive_type(primitive)
    sdk_versions = collect_sdk_versions()

    config: dict[str, Any] = {"primitive_type": primitive_type}

    # Extract resilience and execution options
    if hasattr(primitive, "options"):
        opts = primitive.options
        for attr in ("resilience_level", "default_shots", "optimization_level"):
            if hasattr(opts, attr):
                try:
                    val = getattr(opts, attr)
                    config[attr] = int(val) if val is not None else None
                except Exception:
                    pass

        # Extract nested options
        for nested in ("resilience", "execution", "twirling"):
            if hasattr(opts, nested):
                try:
                    config[f"options_{nested}"] = to_jsonable(getattr(opts, nested))
                except Exception:
                    pass

    return FrontendConfig(
        name=primitive_class,
        sdk="qiskit-ibm-runtime",
        sdk_version=sdk_versions.get("qiskit_ibm_runtime", "unknown"),
        config=config if config else {},
    )


def _extract_options(primitive: Any) -> dict[str, Any]:
    """
    Extract primitive options for raw_properties.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict
        Options properties (JSON-serializable).
    """
    props: dict[str, Any] = {}

    if not hasattr(primitive, "options"):
        return props

    opts = primitive.options

    # Extract nested option attributes
    for nested_attr in (
        "resilience",
        "resilience_level",
        "execution",
        "environment",
        "simulator",
        "twirling",
    ):
        if hasattr(opts, nested_attr):
            try:
                raw_val = getattr(opts, nested_attr)
                props[f"options_{nested_attr}"] = to_jsonable(raw_val)
            except Exception:
                try:
                    props[f"options_{nested_attr}"] = repr(raw_val)[:500]
                except Exception:
                    pass

    # Extract common scalar options
    for scalar_attr in ("optimization_level", "default_shots"):
        if hasattr(opts, scalar_attr):
            try:
                val = getattr(opts, scalar_attr)
                props[scalar_attr] = int(val) if val is not None else None
            except Exception:
                pass

    return props


def _extract_session_info(primitive: Any) -> dict[str, Any] | None:
    """
    Extract session information.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict or None
        Session info or None if no session.
    """
    session = getattr(primitive, "session", None)
    if session is None:
        return None

    session_info: dict[str, Any] = {}
    for sattr in ("session_id", "backend", "max_time"):
        if hasattr(session, sattr):
            try:
                val = getattr(session, sattr)
                session_info[sattr] = to_jsonable(val() if callable(val) else val)
            except Exception:
                try:
                    val = getattr(session, sattr)
                    session_info[sattr] = repr(val() if callable(val) else val)[:200]
                except Exception:
                    pass

    return session_info if session_info else None


def _extract_mode_info(primitive: Any) -> dict[str, Any] | None:
    """
    Extract mode information (Session/Batch/Backend).

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict or None
        Mode info or None if not available.
    """
    mode = getattr(primitive, "mode", None)
    if mode is None:
        return None

    mode_info: dict[str, Any] = {"type": type(mode).__name__}

    # Session/Batch ID
    for id_attr in ("session_id", "batch_id", "id"):
        if hasattr(mode, id_attr):
            try:
                val = getattr(mode, id_attr)
                if callable(val):
                    val = val()
                if val:
                    mode_info["id"] = str(val)
                    break
            except Exception:
                pass

    # Max time
    if hasattr(mode, "max_time"):
        try:
            mt = mode.max_time
            if callable(mt):
                mt = mt()
            mode_info["max_time"] = mt
        except Exception:
            pass

    return mode_info if len(mode_info) > 1 else None


def create_device_snapshot(
    primitive: Any,
    *,
    refresh_properties: bool = False,
    tracker: Run | None = None,
) -> DeviceSnapshot:
    """
    Create a DeviceSnapshot from a Runtime primitive.

    Captures the multi-layer stack following the UEC:
    - Frontend layer: The primitive (SamplerV2, EstimatorV2)
    - Resolved backend: The underlying IBM backend

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance (Sampler/Estimator).
    refresh_properties : bool, optional
        If True, attempt to refresh backend calibration properties.
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.

    Returns
    -------
    DeviceSnapshot
        Structured device snapshot with frontend configuration.

    Raises
    ------
    ValueError
        If primitive is None.

    Notes
    -----
    When ``tracker`` is provided, raw backend and primitive properties are
    logged as a separate artifact for lossless capture. This includes:
    - Primitive class and module info
    - Backend properties from the resolved backend
    - Primitive options (resilience, execution, twirling)
    - Session/mode information
    """
    if primitive is None:
        raise ValueError("Cannot create device snapshot from None primitive")

    captured_at = utc_now_iso()
    primitive_class = primitive.__class__.__name__
    sdk_versions = collect_sdk_versions()

    # Build frontend config for the primitive layer
    frontend = _build_frontend_config(primitive)

    # Runtime metadata for raw_properties artifact
    raw_properties: dict[str, Any] = {
        "primitive_class": primitive_class,
        "primitive_module": getattr(primitive, "__module__", ""),
    }

    backend = get_backend_obj(primitive)
    backend_id = _extract_backend_id(primitive)

    # Compose backend snapshot (reuse qiskit adapter's snapshot)
    base: DeviceSnapshot | None = None
    if backend is not None:
        raw_properties["backend_class"] = backend.__class__.__name__
        # Don't pass tracker here - we'll log our own combined raw_properties
        try:
            base = create_backend_snapshot(
                backend,
                refresh_properties=refresh_properties,
                tracker=None,
            )
        except Exception as e:
            logger.warning("Failed to create backend snapshot: %s", e)

    # Add Runtime-only info to raw_properties
    raw_properties.update(_extract_options(primitive))

    session_info = _extract_session_info(primitive)
    if session_info:
        raw_properties["session_info"] = session_info

    mode_info = _extract_mode_info(primitive)
    if mode_info:
        raw_properties["mode_info"] = mode_info

    # Determine backend name and type
    backend_name = (
        base.backend_name if base and base.backend_name else get_backend_name(primitive)
    )

    backend_type = "hardware"
    if base and base.backend_type:
        backend_type = base.backend_type
    elif "simulator" in backend_name.lower() or "fake" in backend_name.lower():
        backend_type = "simulator"

    # Log raw_properties as artifact if tracker is provided
    raw_properties_ref = None
    if tracker is not None and len(raw_properties) > 2:
        try:
            raw_properties_ref = tracker.log_json(
                name="runtime_raw_properties",
                obj=to_jsonable(raw_properties),
                role="device_raw",
                kind="device.qiskit_runtime.raw_properties.json",
            )
            logger.debug("Logged raw runtime properties artifact")
        except Exception as e:
            logger.warning("Failed to log raw_properties artifact: %s", e)

    # Detect physical provider (not SDK)
    physical_provider = _detect_physical_provider(primitive)

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=physical_provider,
        backend_id=backend_id,
        num_qubits=base.num_qubits if base else None,
        connectivity=base.connectivity if base else None,
        native_gates=base.native_gates if base else None,
        calibration=base.calibration if base else None,
        frontend=frontend,
        sdk_versions=sdk_versions,
        raw_properties_ref=raw_properties_ref,
    )


def resolve_runtime_backend(executor: Any) -> dict[str, Any] | None:
    """
    Resolve the physical backend from a Runtime primitive.

    This is the Runtime implementation of the universal backend resolution
    helper specified in the UEC.

    Parameters
    ----------
    executor : Any
        Runtime primitive or executor object.

    Returns
    -------
    dict or None
        Dictionary with resolved backend information, or None if resolution fails.
    """
    if executor is None:
        return None

    try:
        backend = get_backend_obj(executor)
    except Exception:
        backend = None

    if backend is None:
        return None

    try:
        backend_name = get_backend_name(executor)
    except Exception:
        backend_name = "unknown"

    backend_name_lower = backend_name.lower()

    backend_type = "hardware"
    if "simulator" in backend_name_lower or "fake" in backend_name_lower:
        backend_type = "simulator"

    try:
        backend_id = _extract_backend_id(executor)
    except Exception:
        backend_id = None

    try:
        primitive_type = get_primitive_type(executor)
    except Exception:
        primitive_type = "unknown"

    return {
        "provider": _detect_physical_provider(executor),
        "backend_name": backend_name,
        "backend_id": backend_id,
        "backend_type": backend_type,
        "backend_obj": backend,
        "primitive_type": primitive_type,
    }
