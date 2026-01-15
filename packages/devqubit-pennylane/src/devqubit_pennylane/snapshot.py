# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot creation for PennyLane devices.

Creates structured DeviceSnapshot objects from PennyLane devices,
capturing device configuration, wires, and execution settings following
the devqubit Uniform Execution Contract (UEC).

PennyLane acts as a frontend to multiple execution providers (Braket, Qiskit,
native simulators). This module implements a multi-layer stack model:

- Frontend layer: PennyLane device interface (what the user sees)
- Resolved backend: The actual execution platform (Braket QPU, IBM backend, etc.)

The adapter detects the execution provider from device name patterns and
extracts backend-specific information when available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_engine.uec.device import DeviceSnapshot, FrontendConfig
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_pennylane.utils import (
    collect_sdk_versions,
    extract_shots_info,
    get_device_name,
)


if TYPE_CHECKING:
    from devqubit_engine.core.run import Run

logger = logging.getLogger(__name__)


# Default timeout for remote backend resolution (seconds)
_REMOTE_RESOLUTION_TIMEOUT = 10.0


def _detect_execution_provider(device: Any) -> str:
    """
    Detect the PHYSICAL execution provider from device name/type.

    Per UEC, provider should be the physical execution platform,
    not the SDK name.

    Parameters
    ----------
    device : Any
        PennyLane device.

    Returns
    -------
    str
        Physical provider identifier: 'local', 'aws_braket', 'ibm_quantum'.
    """
    device_name = get_device_name(device).lower()
    short_name = getattr(device, "short_name", "").lower()
    module = getattr(device, "__module__", "").lower()

    # Check for Braket plugin -> physical provider is AWS or local
    if (
        device_name.startswith("braket.")
        or short_name.startswith("braket.")
        or "braket" in module
    ):
        # Check if it's a local simulator
        if "local" in device_name or "localsimulator" in module:
            return "local"
        return "aws_braket"

    # Check for Qiskit plugin -> physical provider depends on backend
    if (
        device_name.startswith("qiskit.")
        or short_name.startswith("qiskit.")
        or "qiskit" in module
    ):
        # Check backend for IBM quantum vs local
        backend = getattr(device, "backend", None) or getattr(device, "_backend", None)
        if backend:
            backend_name = str(getattr(backend, "name", "")).lower()
            if "ibm_" in backend_name or "ibmq" in backend_name:
                return "ibm_quantum"
            if "aer" in backend_name or "fake" in backend_name:
                return "local"
        return "ibm_quantum"  # Default for Qiskit plugin

    # PennyLane native devices are always local simulators
    if any(
        pattern in device_name
        for pattern in ("default.", "lightning.", "numpy", "mixed", "qutrit")
    ):
        return "local"

    # Check module for other hints
    if "pennylane" in module:
        return "local"

    return "local"  # Default to local


def _detect_sdk_frontend(device: Any) -> str:
    """
    Detect the SDK frontend from device name/type.

    This returns the SDK name for tracking purposes (goes in producer.frontends).

    Parameters
    ----------
    device : Any
        PennyLane device.

    Returns
    -------
    str
        SDK identifier: 'braket', 'qiskit', 'pennylane'.
    """
    device_name = get_device_name(device).lower()
    short_name = getattr(device, "short_name", "").lower()
    module = getattr(device, "__module__", "").lower()

    if (
        device_name.startswith("braket.")
        or short_name.startswith("braket.")
        or "braket" in module
    ):
        return "braket"

    if (
        device_name.startswith("qiskit.")
        or short_name.startswith("qiskit.")
        or "qiskit" in module
    ):
        return "qiskit"

    return "pennylane"


def _detect_backend_type(device: Any, provider: str) -> str:
    """
    Detect the backend type (simulator vs hardware).

    Parameters
    ----------
    device : Any
        PennyLane device.
    provider : str
        Physical provider (local, aws_braket, ibm_quantum).

    Returns
    -------
    str
        Backend type: 'simulator' or 'hardware'.
    """
    device_name = get_device_name(device).lower()

    # Local provider is always simulator
    if provider == "local":
        return "simulator"

    # AWS Braket: check for simulator indicators
    if provider == "aws_braket":
        if "local" in device_name or "simulator" in device_name:
            return "simulator"
        # Check ARN for simulator service
        arn = getattr(device, "device_arn", None) or getattr(device, "_device_arn", "")
        if arn and ("simulator" in str(arn).lower() or "sv1" in str(arn).lower()):
            return "simulator"
        return "hardware"

    # IBM Quantum: check for simulator indicators
    if provider == "ibm_quantum":
        if "aer" in device_name or "simulator" in device_name or "fake" in device_name:
            return "simulator"
        # Check backend object
        backend = getattr(device, "backend", None) or getattr(device, "_backend", None)
        if backend:
            backend_name = str(getattr(backend, "name", "")).lower()
            if (
                "simulator" in backend_name
                or "aer" in backend_name
                or "fake" in backend_name
            ):
                return "simulator"
        return "hardware"

    return "simulator"  # Default to simulator for safety


def _resolve_braket_backend(
    device: Any,
    *,
    resolve_remote: bool = False,
    timeout: float = _REMOTE_RESOLUTION_TIMEOUT,
) -> DeviceSnapshot | None:
    """
    Resolve Braket backend from PennyLane-Braket device.

    Parameters
    ----------
    device : Any
        PennyLane device with Braket backend.
    resolve_remote : bool
        If True, attempt to resolve remote devices via AwsDevice API.
        This can be slow and requires network/credentials.
    timeout : float
        Timeout in seconds for remote resolution.

    Returns
    -------
    DeviceSnapshot or None
        Resolved Braket device snapshot, or None if resolution fails.
    """
    # Try to get device ARN (always available, even without resolution)
    arn = getattr(device, "device_arn", None) or getattr(device, "_device_arn", None)
    if not arn:
        return None

    # First try to get the underlying AwsDevice if already available
    aws_device = getattr(device, "_device", None) or getattr(device, "device", None)

    try:
        from devqubit_braket.snapshot import (
            create_device_snapshot as create_braket_snapshot,
        )

        if aws_device is not None:
            # Don't pass tracker - we'll handle raw_properties ourselves
            return create_braket_snapshot(aws_device, tracker=None)

        # Only attempt remote resolution if explicitly enabled
        if resolve_remote:
            try:
                import signal
                from contextlib import contextmanager

                @contextmanager
                def timeout_context(seconds: float):
                    """Context manager for timeout (Unix only)."""

                    def handler(signum, frame):
                        raise TimeoutError("Remote resolution timed out")

                    # Only set alarm on Unix
                    if hasattr(signal, "SIGALRM"):
                        old_handler = signal.signal(signal.SIGALRM, handler)
                        signal.setitimer(signal.ITIMER_REAL, seconds)
                        try:
                            yield
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        # No timeout on Windows
                        yield

                from braket.aws import AwsDevice

                with timeout_context(timeout):
                    aws_device = AwsDevice(arn)
                    return create_braket_snapshot(aws_device, tracker=None)

            except TimeoutError:
                logger.warning(
                    "Braket device resolution timed out after %.1fs for ARN: %s",
                    timeout,
                    arn,
                )
            except Exception as e:
                logger.debug("Failed to create AwsDevice from ARN: %s", e)

    except ImportError:
        logger.debug("devqubit_braket not available for backend resolution")

    return None


def _extract_braket_info(device: Any) -> dict[str, Any]:
    """
    Extract Braket-specific information from device.

    Always extracts device_arn which is the critical identifier.

    Parameters
    ----------
    device : Any
        PennyLane-Braket device.

    Returns
    -------
    dict
        Braket-specific properties.
    """
    props: dict[str, Any] = {}

    # Device ARN - critical identifier, always extract
    arn = getattr(device, "device_arn", None) or getattr(device, "_device_arn", None)
    if arn:
        props["device_arn"] = str(arn)

    # S3 bucket for results
    for attr in ("s3_destination_folder", "_s3_folder"):
        val = getattr(device, attr, None)
        if val:
            props["s3_destination"] = to_jsonable(val)
            break

    # Parallel execution
    parallel = getattr(device, "parallel", None)
    if parallel is not None:
        props["parallel"] = bool(parallel)

    max_parallel = getattr(device, "max_parallel", None)
    if max_parallel is not None:
        props["max_parallel"] = int(max_parallel)

    return props


def _resolve_qiskit_backend(device: Any) -> DeviceSnapshot | None:
    """
    Resolve Qiskit backend from PennyLane-Qiskit device.

    Parameters
    ----------
    device : Any
        PennyLane device with Qiskit backend.

    Returns
    -------
    DeviceSnapshot or None
        Resolved Qiskit device snapshot, or None if resolution fails.
    """
    # Try to get backend object
    backend = getattr(device, "backend", None) or getattr(device, "_backend", None)
    if backend is None:
        return None

    try:
        # Try to use existing Qiskit snapshot builder
        from devqubit_qiskit.snapshot import (
            create_device_snapshot as create_qiskit_snapshot,
        )

        # Don't pass tracker - we'll handle raw_properties ourselves
        return create_qiskit_snapshot(backend, tracker=None)
    except ImportError:
        logger.debug("devqubit_qiskit not available for backend resolution")

    return None


def _extract_qiskit_info(device: Any) -> dict[str, Any]:
    """
    Extract Qiskit-specific information from device.

    Parameters
    ----------
    device : Any
        PennyLane-Qiskit device.

    Returns
    -------
    dict
        Qiskit-specific properties.
    """
    props: dict[str, Any] = {}

    # Backend object
    backend = getattr(device, "backend", None) or getattr(device, "_backend", None)
    if backend:
        backend_name = getattr(backend, "name", None)
        if backend_name:
            props["qiskit_backend_name"] = str(
                backend_name() if callable(backend_name) else backend_name
            )

        # Provider info
        provider = getattr(backend, "provider", None)
        if provider:
            provider_name = getattr(provider, "__class__", type(provider)).__name__
            props["qiskit_provider"] = provider_name

    # Noise model
    noise_model = getattr(device, "noise_model", None)
    if noise_model is not None:
        props["has_noise_model"] = True

    return props


def _extract_wires(device: Any) -> tuple[int | None, dict[str, Any]]:
    """
    Extract wire information from device.

    Parameters
    ----------
    device : Any
        PennyLane device.

    Returns
    -------
    num_qubits : int or None
        Number of qubits.
    props : dict
        Wire-related properties.
    """
    num_qubits: int | None = None
    props: dict[str, Any] = {}

    try:
        if hasattr(device, "wires"):
            wires = list(device.wires)
            props["wires"] = wires
            num_qubits = len(wires)
    except Exception:
        pass

    return num_qubits, props


def _extract_shots(device: Any) -> dict[str, Any]:
    """
    Extract comprehensive shots configuration from device.

    Uses the ShotsInfo dataclass for proper handling of shot vectors.

    Parameters
    ----------
    device : Any
        PennyLane device.

    Returns
    -------
    dict
        Shots configuration including shot_vector if present.
    """
    shots_info = extract_shots_info(device)
    return shots_info.to_dict()


def _extract_diff_method(device: Any) -> dict[str, Any]:
    """Extract differentiation method from device."""
    props: dict[str, Any] = {}

    for attr in ("diff_method", "_diff_method"):
        try:
            if hasattr(device, attr):
                val = getattr(device, attr)
                if val is not None:
                    props["diff_method"] = str(val)
                    break
        except Exception:
            pass

    return props


def _extract_interface(device: Any) -> dict[str, Any]:
    """Extract interface from device."""
    props: dict[str, Any] = {}

    for attr in ("interface", "_interface"):
        try:
            if hasattr(device, attr):
                val = getattr(device, attr)
                if val is not None:
                    props["interface"] = str(val)
                    break
        except Exception:
            pass

    return props


def _extract_data_types(device: Any) -> dict[str, Any]:
    """Extract data type configuration from device."""
    props: dict[str, Any] = {}

    for dtype_attr in ("c_dtype", "r_dtype", "dtype"):
        try:
            if hasattr(device, dtype_attr):
                val = getattr(device, dtype_attr)
                if val is not None:
                    props[dtype_attr] = str(val)
        except Exception:
            pass

    return props


def _extract_seed(device: Any) -> dict[str, Any]:
    """Extract seed from device."""
    props: dict[str, Any] = {}

    try:
        if hasattr(device, "seed"):
            seed = device.seed
            if seed is not None:
                props["seed"] = to_jsonable(seed)
    except Exception:
        pass

    return props


def _extract_capabilities(device: Any) -> dict[str, Any]:
    """Extract device capabilities."""
    props: dict[str, Any] = {}

    # Execute kwargs
    try:
        if hasattr(device, "execute_kwargs"):
            props["execute_kwargs"] = to_jsonable(device.execute_kwargs)
    except Exception:
        pass

    # Supported operations
    try:
        if hasattr(device, "operations"):
            ops = device.operations
            if ops is not None:
                props["num_supported_operations"] = len(ops)
    except Exception:
        pass

    # Supported observables
    try:
        if hasattr(device, "observables"):
            obs = device.observables
            if obs is not None:
                props["num_supported_observables"] = len(obs)
    except Exception:
        pass

    return props


def _build_frontend_config(device: Any) -> FrontendConfig:
    """
    Build FrontendConfig for the PennyLane layer.

    Parameters
    ----------
    device : Any
        PennyLane device.

    Returns
    -------
    FrontendConfig
        Configuration describing the PennyLane frontend.
    """
    device_class = device.__class__.__name__
    sdk_versions = collect_sdk_versions()

    # Extract config options using ShotsInfo
    config: dict[str, Any] = {}

    shots_info = extract_shots_info(device)
    if shots_info.analytic:
        config["analytic_mode"] = True
    else:
        config["shots"] = shots_info.total_shots
        if shots_info.shot_vector:
            config["shot_vector"] = shots_info.shot_vector

    # Diff method
    diff_method = getattr(device, "diff_method", None) or getattr(
        device, "_diff_method", None
    )
    if diff_method:
        config["diff_method"] = str(diff_method)

    # Interface
    interface = getattr(device, "interface", None) or getattr(
        device, "_interface", None
    )
    if interface:
        config["interface"] = str(interface)

    return FrontendConfig(
        name=device_class,
        sdk="pennylane",
        sdk_version=sdk_versions.get("pennylane", "unknown"),
        config=config if config else {},
    )


def _build_raw_properties(
    device: Any, provider: str, sdk_frontend: str
) -> dict[str, Any]:
    """
    Build the complete raw_properties dictionary for a device.

    Parameters
    ----------
    device : Any
        PennyLane device instance.
    provider : str
        Physical execution provider (local, aws_braket, ibm_quantum).
    sdk_frontend : str
        SDK frontend (braket, qiskit, pennylane).

    Returns
    -------
    dict
        Complete raw properties dictionary.
    """
    raw_properties: dict[str, Any] = {
        "device_class": device.__class__.__name__,
        "device_module": getattr(device, "__module__", ""),
        "short_name": getattr(device, "short_name", None),
        "execution_provider": provider,
        "sdk_frontend": sdk_frontend,
    }

    # Extract wire info
    _, wire_props = _extract_wires(device)
    raw_properties.update(wire_props)

    # Extract PennyLane-specific properties
    raw_properties.update(_extract_shots(device))
    raw_properties.update(_extract_diff_method(device))
    raw_properties.update(_extract_interface(device))
    raw_properties.update(_extract_data_types(device))
    raw_properties.update(_extract_seed(device))
    raw_properties.update(_extract_capabilities(device))

    # Extract SDK-specific properties based on frontend
    if sdk_frontend == "braket":
        raw_properties.update(_extract_braket_info(device))
    elif sdk_frontend == "qiskit":
        raw_properties.update(_extract_qiskit_info(device))

    return raw_properties


def create_device_snapshot(
    device: Any,
    *,
    resolve_remote_backend: bool = False,
    tracker: Run | None = None,
) -> DeviceSnapshot:
    """
    Create a DeviceSnapshot from a PennyLane device.

    Captures the multi-layer stack following the UEC:
    - Frontend layer: PennyLane device interface
    - Resolved backend: Braket/Qiskit/native backend

    When PennyLane is used as a frontend to Braket or Qiskit, this function
    attempts to resolve the underlying backend and merge its calibration
    and topology information.

    Parameters
    ----------
    device : Any
        PennyLane device instance.
    resolve_remote_backend : bool
        If True, attempt to resolve remote backends via their APIs.
        This can be slow and requires network access. Default False.
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.
        If provided, raw properties are logged and referenced via ``raw_properties_ref``.

    Returns
    -------
    DeviceSnapshot
        Structured device snapshot with frontend configuration.

    Raises
    ------
    ValueError
        If device is None.

    Notes
    -----
    When ``tracker`` is provided, raw device properties are logged as a separate
    artifact for lossless capture. This includes wire info, shots configuration,
    differentiation method, interface settings, and provider-specific data.

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=3)
    >>> snapshot = create_device_snapshot(dev)
    >>> snapshot.provider
    'pennylane'

    >>> # With Braket backend (no remote resolution)
    >>> dev = qml.device("braket.aws.qubit", wires=2, device_arn="...")
    >>> snapshot = create_device_snapshot(dev)
    >>> snapshot.provider
    'aws_braket'
    """
    if device is None:
        raise ValueError("Cannot create device snapshot from None device")

    captured_at = utc_now_iso()
    backend_name = get_device_name(device)
    sdk_versions = collect_sdk_versions()

    # Detect physical execution provider and SDK frontend
    try:
        provider = _detect_execution_provider(device)
    except Exception as e:
        logger.debug("Failed to detect execution provider: %s", e)
        provider = "local"

    try:
        sdk_frontend = _detect_sdk_frontend(device)
    except Exception as e:
        logger.debug("Failed to detect SDK frontend: %s", e)
        sdk_frontend = "pennylane"

    try:
        backend_type = _detect_backend_type(device, provider)
    except Exception as e:
        logger.debug("Failed to detect backend type: %s", e)
        backend_type = "simulator"

    # Build frontend config for PennyLane layer
    try:
        frontend = _build_frontend_config(device)
    except Exception as e:
        logger.debug("Failed to build frontend config: %s", e)
        frontend = None

    # Try to resolve execution backend for Braket/Qiskit (based on SDK frontend)
    resolved_backend: DeviceSnapshot | None = None
    backend_id: str | None = None

    if sdk_frontend == "braket":
        try:
            resolved_backend = _resolve_braket_backend(
                device, resolve_remote=resolve_remote_backend
            )
        except Exception as e:
            logger.debug("Failed to resolve Braket backend: %s", e)
        # Always extract ARN as backend_id
        try:
            arn = getattr(device, "device_arn", None) or getattr(
                device, "_device_arn", None
            )
            if arn:
                backend_id = str(arn)
        except Exception:
            pass

    elif sdk_frontend == "qiskit":
        try:
            resolved_backend = _resolve_qiskit_backend(device)
        except Exception as e:
            logger.debug("Failed to resolve Qiskit backend: %s", e)
        # Try to get backend ID from Qiskit backend
        try:
            backend_obj = getattr(device, "backend", None) or getattr(
                device, "_backend", None
            )
            if backend_obj:
                bid = getattr(backend_obj, "backend_id", None)
                if bid:
                    backend_id = str(bid() if callable(bid) else bid)
        except Exception:
            pass

    # Extract wire info
    try:
        num_qubits, _ = _extract_wires(device)
    except Exception as e:
        logger.debug("Failed to extract wires: %s", e)
        num_qubits = None

    # Merge resolved backend info if available
    connectivity = None
    native_gates = None
    calibration = None

    if resolved_backend:
        # Use resolved backend's topology and calibration
        connectivity = resolved_backend.connectivity
        native_gates = resolved_backend.native_gates
        calibration = resolved_backend.calibration

        # Use resolved backend's qubit count if we don't have it
        if num_qubits is None and resolved_backend.num_qubits:
            num_qubits = resolved_backend.num_qubits

        # Merge resolved backend's SDK versions
        if resolved_backend.sdk_versions:
            sdk_versions.update(resolved_backend.sdk_versions)

    # Log raw_properties as artifact if tracker is provided
    raw_properties_ref = None
    if tracker is not None:
        try:
            raw_properties = _build_raw_properties(device, provider, sdk_frontend)
            raw_properties_ref = tracker.log_json(
                name="device_raw_properties",
                obj=raw_properties,
                role="device_raw",
                kind="device.pennylane.raw_properties.json",
            )
            logger.debug("Logged raw PennyLane device properties artifact")
        except Exception as e:
            logger.warning("Failed to log raw_properties artifact: %s", e)

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        backend_id=backend_id,
        num_qubits=num_qubits,
        connectivity=connectivity,
        native_gates=native_gates,
        calibration=calibration,
        frontend=frontend,
        sdk_versions=sdk_versions,
        raw_properties_ref=raw_properties_ref,
    )


def resolve_pennylane_backend(device: Any) -> dict[str, Any] | None:
    """
    Resolve the physical backend from a PennyLane device.

    This is the PennyLane implementation of the universal backend resolution
    helper specified in the UEC.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    dict or None
        Dictionary with resolved backend information:
        - ``provider``: Physical execution provider (local/aws_braket/ibm_quantum)
        - ``sdk_frontend``: SDK frontend (braket/qiskit/pennylane)
        - ``backend_name``: Device name
        - ``backend_id``: Stable unique ID (ARN for Braket, etc.)
        - ``backend_type``: Type (hardware/simulator)
        - ``backend_obj``: The underlying backend object (if available)

        Returns None if resolution fails.

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("braket.aws.qubit", wires=2, device_arn="...")
    >>> info = resolve_pennylane_backend(dev)
    >>> info["provider"]
    'aws_braket'
    """
    if device is None:
        return None

    try:
        provider = _detect_execution_provider(device)
    except Exception:
        provider = "local"

    try:
        sdk_frontend = _detect_sdk_frontend(device)
    except Exception:
        sdk_frontend = "pennylane"

    try:
        backend_name = get_device_name(device)
    except Exception:
        backend_name = "unknown"

    try:
        backend_type = _detect_backend_type(device, provider)
    except Exception:
        backend_type = "simulator"

    result: dict[str, Any] = {
        "provider": provider,
        "sdk_frontend": sdk_frontend,
        "backend_name": backend_name,
        "backend_id": None,
        "backend_type": backend_type,
        "backend_obj": None,
    }

    # Use sdk_frontend for backend-specific resolution
    if sdk_frontend == "braket":
        try:
            arn = getattr(device, "device_arn", None) or getattr(
                device, "_device_arn", None
            )
            result["backend_id"] = str(arn) if arn else None
            result["backend_obj"] = getattr(device, "_device", None)
        except Exception:
            pass

    elif sdk_frontend == "qiskit":
        try:
            backend = getattr(device, "backend", None) or getattr(
                device, "_backend", None
            )
            result["backend_obj"] = backend
            if backend:
                bid = getattr(backend, "backend_id", None)
                if bid:
                    result["backend_id"] = str(bid() if callable(bid) else bid)
        except Exception:
            pass

    return result
