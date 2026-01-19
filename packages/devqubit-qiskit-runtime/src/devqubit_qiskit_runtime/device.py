# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot and envelope utilities for Qiskit Runtime adapter.

Creates structured DeviceSnapshot objects from Runtime primitives,
capturing backend configuration, calibration, and primitive frontend
configuration following the devqubit Uniform Execution Contract (UEC).

The UEC uses a multi-layer stack model where:
- Frontend: The primitive (e.g., SamplerV2) that the user interacts with
- Resolved backend: The physical backend (e.g., ibm_brisbane) where execution occurs

This module composes:
1. A FrontendConfig describing the primitive layer
2. A DeviceSnapshot from the underlying backend (reusing devqubit_qiskit.device)
3. Runtime-specific metadata (options, session info)
4. Envelope lifecycle management (finalization, failure handling)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.models.device import DeviceSnapshot, FrontendConfig
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.result import ResultSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_qiskit.device import create_device_snapshot as create_backend_snapshot
from devqubit_qiskit_runtime.utils import (
    collect_sdk_versions,
    get_backend_name,
    get_backend_obj,
    get_primitive_type,
)


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Detection
# =============================================================================


def detect_provider(primitive: Any) -> str:
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
        Physical provider: "ibm_quantum", "fake", "aer", or "local".

    Notes
    -----
    This function examines the backend object attached to the primitive
    to determine the physical provider. The logic is:

    1. If no backend is found, check primitive module for "ibm" string
    2. Check backend module and name for IBM, fake, or Aer indicators
    3. Default to "local" if no provider can be determined

    Examples
    --------
    >>> from qiskit_ibm_runtime import SamplerV2
    >>> sampler = SamplerV2(backend)
    >>> detect_provider(sampler)
    'ibm_quantum'
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


# =============================================================================
# Frontend Configuration
# =============================================================================


def _build_frontend_config(primitive: Any) -> FrontendConfig:
    """
    Build FrontendConfig for the Runtime primitive layer.

    The frontend configuration captures information about the primitive
    that the user interacts with, including options and SDK version.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    FrontendConfig
        Configuration describing the primitive frontend with:
        - name: Primitive class name (e.g., "SamplerV2")
        - sdk: "qiskit-ibm-runtime"
        - sdk_version: Version of qiskit_ibm_runtime
        - config: Primitive options (resilience_level, default_shots, etc.)

    Notes
    -----
    The config dictionary includes:
    - primitive_type: "sampler" or "estimator"
    - resilience_level: Error mitigation level (if set)
    - default_shots: Default shot count (if set)
    - optimization_level: Transpilation optimization level (if set)
    - Nested option groups (resilience, execution, twirling)
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


# =============================================================================
# Property Extraction Helpers
# =============================================================================


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
        Backend ID if available, None otherwise.

    Notes
    -----
    Tries multiple approaches in order:
    1. backend.backend_id (IBM-specific)
    2. backend.instance (IBM Cloud instance ID)
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


def _extract_options(primitive: Any) -> dict[str, Any]:
    """
    Extract primitive options for raw_properties artifact.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict
        Options properties (JSON-serializable) including:
        - options_resilience: Resilience settings
        - options_execution: Execution settings
        - options_twirling: Twirling settings
        - optimization_level: Transpilation optimization level
        - default_shots: Default shot count
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
    Extract session information from a Runtime primitive.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict or None
        Session info dictionary with keys:
        - session_id: Unique session identifier
        - backend: Backend name for the session
        - max_time: Maximum session duration
        Returns None if no session is active.
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
    Extract mode information (Session/Batch/Backend) from primitive.

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    dict or None
        Mode info dictionary with keys:
        - type: Mode class name (Session, Batch, or Backend)
        - id: Session/Batch ID if available
        - max_time: Maximum time if available
        Returns None if no mode is set.
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


# =============================================================================
# Device Snapshot Creation
# =============================================================================


def create_device_snapshot(
    primitive: Any,
    *,
    refresh_properties: bool = False,
    tracker: "Run | None" = None,
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
        Default is False.
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.
        If provided, raw backend and primitive properties are logged
        as a separate artifact for lossless capture.

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

    The snapshot reuses ``devqubit_qiskit.device.create_device_snapshot``
    for the backend layer, ensuring consistent calibration extraction
    across both Qiskit adapters.

    Examples
    --------
    >>> from qiskit_ibm_runtime import SamplerV2
    >>> sampler = SamplerV2(backend)
    >>> snapshot = create_device_snapshot(sampler)
    >>> snapshot.provider
    'ibm_quantum'
    >>> snapshot.frontend.sdk
    'qiskit-ibm-runtime'
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
    physical_provider = detect_provider(primitive)

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


# =============================================================================
# Backend Resolution
# =============================================================================


def resolve_runtime_backend(executor: Any) -> dict[str, Any] | None:
    """
    Resolve the physical backend from a Runtime primitive.

    This is the Runtime implementation of the universal backend resolution
    helper specified in the UEC. It provides a consistent interface for
    obtaining backend information regardless of how the primitive was
    constructed.

    Parameters
    ----------
    executor : Any
        Runtime primitive or executor object.

    Returns
    -------
    dict or None
        Dictionary with resolved backend information:
        - provider: Physical provider name
        - backend_name: Backend name string
        - backend_id: Backend identifier (if available)
        - backend_type: "hardware" or "simulator"
        - backend_obj: The actual backend object
        - primitive_type: "sampler" or "estimator"
        Returns None if resolution fails.

    Examples
    --------
    >>> from qiskit_ibm_runtime import SamplerV2
    >>> sampler = SamplerV2(backend)
    >>> info = resolve_runtime_backend(sampler)
    >>> info["provider"]
    'ibm_quantum'
    >>> info["primitive_type"]
    'sampler'
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
        "provider": detect_provider(executor),
        "backend_name": backend_name,
        "backend_id": backend_id,
        "backend_type": backend_type,
        "backend_obj": backend,
        "primitive_type": primitive_type,
    }


# =============================================================================
# Envelope Lifecycle Management
# =============================================================================


def create_failure_result_snapshot(
    exception: BaseException,
    backend_name: str,
    primitive_type: str,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot for a failed execution.

    Used when job.result() raises an exception. Ensures envelope
    is always created even on failures (UEC requirement).

    Parameters
    ----------
    exception : BaseException
        The exception that caused the failure.
    backend_name : str
        Backend name for metadata.
    primitive_type : str
        Type of primitive ('sampler' or 'estimator').

    Returns
    -------
    ResultSnapshot
        Failed result snapshot with error details including:
        - success: False
        - status: "failed"
        - error: Exception type and message
        - metadata: Backend and primitive info

    Notes
    -----
    The UEC requires that an envelope be created for every execution
    attempt, even if the execution fails. This function creates the
    appropriate failure snapshot that can be attached to the envelope.

    Examples
    --------
    >>> try:
    ...     result = job.result()
    ... except Exception as e:
    ...     snapshot = create_failure_result_snapshot(
    ...         e, "ibm_brisbane", "sampler"
    ...     )
    """
    return ResultSnapshot.create_failed(
        exception=exception,
        metadata={
            "backend_name": backend_name,
            "primitive_type": primitive_type,
        },
    )


def finalize_envelope_with_result(
    tracker: "Run",
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """
    Finalize envelope with result and log as artifact.

    This function completes the envelope lifecycle by attaching the
    result snapshot and logging the complete envelope to the tracker.

    Parameters
    ----------
    tracker : Run
        Tracker instance for logging the envelope.
    envelope : ExecutionEnvelope
        Envelope to finalize. Must not be None.
    result_snapshot : ResultSnapshot
        Result to add to envelope. May be None (warning logged).

    Raises
    ------
    ValueError
        If envelope is None.

    Notes
    -----
    This function:
    1. Attaches the result snapshot to the envelope
    2. Sets the completion timestamp on the execution snapshot
    3. Validates and logs the envelope using the tracker's canonical method

    The envelope is logged as a "devqubit.envelope.json" artifact.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if result_snapshot is None:
        logger.warning("Finalizing envelope with None result_snapshot")

    # Add result to envelope
    envelope.result = result_snapshot

    # Set completion time
    if envelope.execution is not None:
        envelope.execution.completed_at = utc_now_iso()

    # Validate and log envelope using tracker's canonical method
    tracker.log_envelope(envelope=envelope)
