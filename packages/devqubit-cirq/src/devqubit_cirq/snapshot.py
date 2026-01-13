# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Device snapshot creation for Cirq samplers and simulators.

Creates structured DeviceSnapshot objects from Cirq executors,
capturing device configuration and sampler attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_cirq.utils import cirq_version, get_backend_name
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.utils.time_utils import utc_now_iso


if TYPE_CHECKING:
    from devqubit_engine.core.run import Run

logger = logging.getLogger(__name__)


def _detect_execution_provider(executor: Any) -> str:
    """
    Detect the physical execution provider for a Cirq executor.

    This distinguishes between the SDK (always "cirq") and the physical
    execution platform where circuits actually run.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    str
        Physical provider identifier:
        - "local" - Local simulators (Simulator, DensityMatrixSimulator, etc.)
        - "google_quantum" - Google Quantum Engine / Quantum AI
        - "ionq" - IonQ hardware via cirq_ionq
        - "aqt" - Alpine Quantum Technologies
        - "pasqal" - Pasqal neutral atoms
        - "rigetti" - Rigetti via cirq_rigetti
    """
    class_name = executor.__class__.__name__.lower()
    module = getattr(executor, "__module__", "").lower()

    # Google Quantum Engine / Quantum AI
    if "engine" in module or "google" in module:
        return "google_quantum"
    if "processor" in class_name and "google" in module:
        return "google_quantum"

    # IonQ
    if "ionq" in module or "ionq" in class_name:
        return "ionq"

    # AQT (Alpine Quantum Technologies)
    if "aqt" in module or "aqt" in class_name:
        return "aqt"

    # Pasqal
    if "pasqal" in module or "pasqal" in class_name:
        return "pasqal"

    # Rigetti
    if "rigetti" in module or "rigetti" in class_name:
        return "rigetti"

    # Default: local simulator
    return "local"


def _resolve_backend_type(executor: Any) -> str:
    """
    Resolve backend_type to a schema-valid value.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    str
        One of: "simulator", "hardware".
    """
    class_name = executor.__class__.__name__.lower()
    module = getattr(executor, "__module__", "").lower()

    # Check for hardware indicators (Google Quantum Engine, IonQ, etc.)
    hardware_indicators = ("engine", "processor", "ionq", "aqt", "pasqal", "quantinuum")
    if any(ind in class_name or ind in module for ind in hardware_indicators):
        return "hardware"

    # Default to simulator for Cirq (most common case)
    return "simulator"


def _extract_num_qubits(executor: Any) -> int | None:
    """
    Extract qubit count from executor's device if available.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    int or None
        Number of qubits, or None if not available.
    """
    try:
        device = getattr(executor, "device", None)
        if device is None:
            return None

        # Try direct qubits attribute
        if hasattr(device, "qubits"):
            return len(list(device.qubits))

        # Try metadata.qubit_set (Google devices)
        metadata = getattr(device, "metadata", None)
        if metadata is not None and hasattr(metadata, "qubit_set"):
            return len(list(metadata.qubit_set))

    except Exception:
        pass

    return None


def _build_qubit_index_map(qubits: Any) -> dict[str, int]:
    """
    Build stable qubit-to-index mapping using string representation.

    Creates a deterministic mapping by sorting qubits by their string
    representation, avoiding hash-based indexing which is unstable
    across Python runs.

    Parameters
    ----------
    qubits : iterable
        Collection of qubit objects.

    Returns
    -------
    dict
        Mapping from qubit string representation to stable integer index.
    """
    # Use string keys to avoid hash instability with qubit objects
    unique_strs = sorted(set(str(q) for q in qubits))
    return {s: idx for idx, s in enumerate(unique_strs)}


def _extract_connectivity(executor: Any) -> list[tuple[int, int]] | None:
    """
    Extract connectivity from executor's device if available.

    Uses stable qubit indexing based on string representation sort order,
    avoiding hash-based indices which are non-deterministic across runs.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    list of tuple or None
        Edge list of connected qubit pairs as (index1, index2), or None
        if connectivity information is not available.

    Notes
    -----
    For GridQubits, indices are assigned by sorting on string representation
    (e.g., "q(0,0)", "q(0,1)", "q(1,0)"). For LineQubits, indices correspond
    to their x-coordinate order.
    """
    try:
        device = getattr(executor, "device", None)
        if device is None:
            return None

        # Try metadata.qubit_pairs (Google devices)
        metadata = getattr(device, "metadata", None)
        if metadata is not None and hasattr(metadata, "qubit_pairs"):
            pairs = list(metadata.qubit_pairs)
            if pairs:
                # Collect all qubits from pairs (use list to avoid hash issues)
                all_qubits: list[Any] = []
                for q1, q2 in pairs:
                    all_qubits.extend([q1, q2])

                # Build stable index mapping (string-keyed)
                qubit_index = _build_qubit_index_map(all_qubits)

                # Convert pairs to index tuples using string lookup
                edges: list[tuple[int, int]] = []
                for q1, q2 in pairs:
                    idx1 = qubit_index.get(str(q1))
                    idx2 = qubit_index.get(str(q2))
                    if idx1 is not None and idx2 is not None:
                        edges.append((idx1, idx2))

                return edges if edges else None

        # Try device.qubits + qubit_set for connectivity graph
        if hasattr(device, "qubits"):
            all_qubits = list(device.qubits)
            qubit_index = _build_qubit_index_map(all_qubits)

            # Some devices have get_edges or similar
            if hasattr(device, "get_edges"):
                try:
                    raw_edges = device.get_edges()
                    edges = []
                    for q1, q2 in raw_edges:
                        idx1 = qubit_index.get(str(q1))
                        idx2 = qubit_index.get(str(q2))
                        if idx1 is not None and idx2 is not None:
                            edges.append((idx1, idx2))
                    return edges if edges else None
                except Exception:
                    pass

    except Exception:
        pass

    return None


def _extract_native_gates(executor: Any) -> list[str] | None:
    """
    Extract native gate set from executor's device if available.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    list of str or None
        Native gate names, or None if not available.
    """
    try:
        device = getattr(executor, "device", None)
        if device is None:
            return None

        # Try metadata.gateset (Google devices)
        metadata = getattr(device, "metadata", None)
        if metadata is not None and hasattr(metadata, "gateset"):
            gateset = metadata.gateset
            if hasattr(gateset, "gates"):
                return [type(g).__name__ for g in gateset.gates]

        # Try device.supported_gates
        if hasattr(device, "supported_gates"):
            gates = device.supported_gates
            if gates:
                return [type(g).__name__ for g in gates]

    except Exception:
        pass

    return None


def _build_raw_properties(executor: Any) -> dict[str, Any]:
    """
    Build raw_properties dictionary for artifact logging.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator.

    Returns
    -------
    dict
        Raw properties for lossless capture.
    """
    raw_properties: dict[str, Any] = {
        "executor_class": executor.__class__.__name__,
        "executor_module": getattr(executor, "__module__", ""),
    }

    # Device info
    try:
        device = getattr(executor, "device", None)
        if device is not None:
            raw_properties["device_class"] = device.__class__.__name__
            raw_properties["device_module"] = getattr(device, "__module__", "")

            # Try to get device metadata
            metadata = getattr(device, "metadata", None)
            if metadata is not None:
                raw_properties["has_metadata"] = True

                # Processor ID (Google devices)
                processor_id = getattr(metadata, "processor_id", None)
                if processor_id:
                    raw_properties["processor_id"] = str(processor_id)

                # Qubit set info
                if hasattr(metadata, "qubit_set"):
                    try:
                        qubits = list(metadata.qubit_set)
                        raw_properties["qubit_labels"] = [str(q) for q in qubits]
                    except Exception:
                        pass

                # Gate set info
                if hasattr(metadata, "gateset"):
                    try:
                        gateset = metadata.gateset
                        if hasattr(gateset, "gates"):
                            raw_properties["gateset"] = [
                                type(g).__name__ for g in gateset.gates
                            ]
                    except Exception:
                        pass

            # Qubits from device directly
            if hasattr(device, "qubits"):
                try:
                    qubits = list(device.qubits)
                    if "qubit_labels" not in raw_properties:
                        raw_properties["qubit_labels"] = [str(q) for q in qubits]
                except Exception:
                    pass

    except Exception:
        pass

    # Noise model info (DensityMatrixSimulator, etc.)
    try:
        noise_model = getattr(executor, "noise", None)
        if noise_model is not None:
            raw_properties["has_noise_model"] = True
            raw_properties["noise_model_class"] = noise_model.__class__.__name__
    except Exception:
        pass

    # Simulator-specific settings
    try:
        # Split state (for DensityMatrixSimulator)
        split_untangled = getattr(executor, "split_untangled_states", None)
        if split_untangled is not None:
            raw_properties["split_untangled_states"] = bool(split_untangled)

        # Seed
        seed = getattr(executor, "seed", None)
        if seed is not None:
            raw_properties["seed"] = seed if isinstance(seed, int) else str(seed)

    except Exception:
        pass

    return raw_properties


def create_device_snapshot(
    executor: Any,
    *,
    tracker: Run | None = None,
) -> DeviceSnapshot:
    """
    Create a DeviceSnapshot from a Cirq sampler or simulator.

    Captures the current state of the executor including device
    information, SDK version, and topology.

    Parameters
    ----------
    executor : Any
        Cirq sampler or simulator instance.
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.
        If provided, raw properties are logged and referenced via ``raw_properties_ref``.

    Returns
    -------
    DeviceSnapshot
        Structured device snapshot for tracking.

    Raises
    ------
    ValueError
        If executor is None.

    Notes
    -----
    When ``tracker`` is provided, raw executor properties are logged as a separate
    artifact for lossless capture. This includes device metadata, qubit labels,
    gate sets, noise model info, and simulator-specific settings.

    Examples
    --------
    >>> import cirq
    >>> simulator = cirq.Simulator()
    >>> snapshot = create_device_snapshot(simulator)
    >>> snapshot.backend_name
    'Simulator'
    >>> snapshot.provider
    'local'
    """
    if executor is None:
        raise ValueError("Cannot create device snapshot from None executor")

    # Extract backend info with fallbacks
    try:
        backend_name = get_backend_name(executor)
    except Exception as e:
        logger.debug("Failed to get backend name: %s", e)
        backend_name = executor.__class__.__name__

    try:
        backend_type = _resolve_backend_type(executor)
    except Exception as e:
        logger.debug("Failed to resolve backend type: %s", e)
        backend_type = "simulator"

    # P1 FIX: Detect physical provider (not SDK frontend)
    try:
        physical_provider = _detect_execution_provider(executor)
    except Exception as e:
        logger.debug("Failed to detect execution provider: %s", e)
        physical_provider = "local"

    try:
        num_qubits = _extract_num_qubits(executor)
    except Exception as e:
        logger.debug("Failed to extract num_qubits: %s", e)
        num_qubits = None

    try:
        connectivity = _extract_connectivity(executor)
    except Exception as e:
        logger.debug("Failed to extract connectivity: %s", e)
        connectivity = None

    try:
        native_gates = _extract_native_gates(executor)
    except Exception as e:
        logger.debug("Failed to extract native gates: %s", e)
        native_gates = None

    try:
        sdk_version = cirq_version()
    except Exception:
        sdk_version = "unknown"

    # Log raw_properties as artifact if tracker is provided
    raw_properties_ref = None
    if tracker is not None:
        try:
            raw_properties = _build_raw_properties(executor)
            # Include execution provider in raw properties
            raw_properties["execution_provider"] = physical_provider
            raw_properties_ref = tracker.log_json(
                name="device_raw_properties",
                obj=raw_properties,
                role="device_raw",
                kind="device.cirq.raw_properties.json",
            )
            logger.debug("Logged raw Cirq device properties artifact")
        except Exception as e:
            logger.warning("Failed to log raw_properties artifact: %s", e)

    return DeviceSnapshot(
        captured_at=utc_now_iso(),
        backend_name=backend_name,
        backend_type=backend_type,
        provider=physical_provider,  # P1 FIX: Physical provider, not "cirq"
        num_qubits=num_qubits,
        connectivity=connectivity,
        native_gates=native_gates,
        calibration=None,  # Cirq simulators typically have no calibration data
        sdk_versions={"cirq": sdk_version},
        raw_properties_ref=raw_properties_ref,
    )
