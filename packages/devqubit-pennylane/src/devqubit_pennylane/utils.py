# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for PennyLane adapter.

Provides version utilities, shots handling, and common helpers used across
the adapter components following the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


@dataclass
class ShotsInfo:
    """
    Structured shots information from a PennyLane device or tape.

    Properly handles the PennyLane `Shots` class which can contain
    shot vectors (partitioned shots) and has `total_shots=None` for
    analytic mode (distinct from `shots=None`).

    Attributes
    ----------
    total_shots : int or None
        Total number of shots, None for analytic mode.
    shot_vector : list of tuple or None
        Shot vector as list of (shots, copies) tuples if partitioned.
    analytic : bool
        True if running in analytic (exact) mode.
    has_partitioned_shots : bool
        True if shots are partitioned across multiple batches.
    num_copies : int
        Number of shot copies (for partitioned shots).
    """

    total_shots: int | None
    shot_vector: list[tuple[int, int]] | None
    analytic: bool
    has_partitioned_shots: bool
    num_copies: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "total_shots": self.total_shots,
            "analytic": self.analytic,
        }
        if self.shot_vector:
            d["shot_vector"] = self.shot_vector
        if self.has_partitioned_shots:
            d["has_partitioned_shots"] = True
            d["num_copies"] = self.num_copies
        return d


def extract_shots_info(device_or_tape: Any) -> ShotsInfo:
    """
    Extract comprehensive shots information from a device or tape.

    Properly handles the PennyLane `Shots` class (introduced in PennyLane >= 0.33),
    distinguishing between:
    - `shots=None`: device attribute not set
    - `Shots(total_shots=None)`: explicit analytic mode
    - `Shots(total_shots=1000)`: finite shots mode
    - Shot vectors with partitioned shots

    Parameters
    ----------
    device_or_tape : Any
        PennyLane device or tape instance.

    Returns
    -------
    ShotsInfo
        Structured shots information.

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=2, shots=1000)
    >>> info = extract_shots_info(dev)
    >>> info.total_shots
    1000
    >>> info.analytic
    False

    >>> dev_analytic = qml.device("default.qubit", wires=2)
    >>> info = extract_shots_info(dev_analytic)
    >>> info.analytic
    True
    """
    try:
        shots = getattr(device_or_tape, "shots", None)

        # Case 1: shots attribute not set at all
        if shots is None:
            return ShotsInfo(
                total_shots=None,
                shot_vector=None,
                analytic=True,
                has_partitioned_shots=False,
                num_copies=1,
            )

        # Case 2: Shots object (PennyLane >= 0.33)
        if hasattr(shots, "total_shots"):
            total = shots.total_shots  # Can be None for analytic

            # Extract shot vector if available
            shot_vector: list[tuple[int, int]] | None = None
            has_partitioned = getattr(shots, "has_partitioned_shots", False)
            num_copies = 1

            if hasattr(shots, "shot_vector") and shots.shot_vector:
                # shot_vector is typically list of ShotCopies namedtuples
                sv = shots.shot_vector
                shot_vector = []
                for item in sv:
                    if hasattr(item, "shots") and hasattr(item, "copies"):
                        shot_vector.append((item.shots, item.copies))
                        num_copies += item.copies - 1
                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                        shot_vector.append((int(item[0]), int(item[1])))
                        num_copies += int(item[1]) - 1
                    else:
                        # Single shot value
                        shot_vector.append((int(item), 1))

            return ShotsInfo(
                total_shots=total,
                shot_vector=shot_vector if shot_vector else None,
                analytic=(total is None),
                has_partitioned_shots=has_partitioned,
                num_copies=num_copies,
            )

        # Case 3: Legacy integer shots
        if isinstance(shots, int):
            return ShotsInfo(
                total_shots=shots if shots > 0 else None,
                shot_vector=None,
                analytic=(shots <= 0),
                has_partitioned_shots=False,
                num_copies=1,
            )

        # Case 4: Iterable of shots (shot vector as list)
        if hasattr(shots, "__iter__"):
            shot_list = list(shots)
            if shot_list:
                total = sum(int(s) for s in shot_list)
                shot_vector = [(int(s), 1) for s in shot_list]
                return ShotsInfo(
                    total_shots=total,
                    shot_vector=shot_vector,
                    analytic=False,
                    has_partitioned_shots=len(shot_list) > 1,
                    num_copies=len(shot_list),
                )

    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Attribute access failed: %s", e)

    # Fallback: assume analytic
    return ShotsInfo(
        total_shots=None,
        shot_vector=None,
        analytic=True,
        has_partitioned_shots=False,
        num_copies=1,
    )


def get_shots(device: Any) -> int | None:
    """
    Get the total shots from a device.

    This is a convenience wrapper around `extract_shots_info` for simple cases.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    int or None
        Number of shots, or None for analytic mode.
    """
    return extract_shots_info(device).total_shots


logger = logging.getLogger(__name__)

_pennylane_version: str | None = None


def pennylane_version() -> str:
    """
    Get the installed PennyLane version (cached after first call).

    Returns
    -------
    str
        PennyLane version string (e.g., "0.35.0"), or "unknown" if
        PennyLane is not installed or version cannot be determined.
    """
    global _pennylane_version
    if _pennylane_version is not None:
        return _pennylane_version
    try:
        import pennylane as qml

        _pennylane_version = getattr(qml, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)
        _pennylane_version = "unknown"
    return _pennylane_version


_adapter_version: str | None = None


def get_adapter_version() -> str:
    """Get adapter version dynamically from package metadata (cached)."""
    global _adapter_version
    if _adapter_version is not None:
        return _adapter_version
    try:
        from importlib.metadata import version

        _adapter_version = version("devqubit-pennylane")
    except ImportError as e:
        logger.debug("Package version lookup failed: %s", e)
        _adapter_version = "unknown"
    return _adapter_version


_sdk_versions: dict[str, str] | None = None


def collect_sdk_versions() -> dict[str, str]:
    """
    Collect version strings for all relevant SDK packages (cached).

    This follows the UEC requirement for tracking all SDK versions
    in the execution environment.

    Returns
    -------
    dict
        Mapping of package name to version string.
    """
    global _sdk_versions
    if _sdk_versions is not None:
        return dict(_sdk_versions)

    versions: dict[str, str] = {}

    # Core PennyLane
    versions["pennylane"] = pennylane_version()

    # PennyLane-Lightning (GPU/CPU optimized)
    try:
        import pennylane_lightning

        versions["pennylane_lightning"] = getattr(
            pennylane_lightning, "__version__", "unknown"
        )
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # Braket plugin
    try:
        import pennylane_braket

        versions["pennylane_braket"] = getattr(
            pennylane_braket, "__version__", "unknown"
        )
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # Amazon Braket SDK (if Braket plugin is used)
    try:
        import braket

        versions["amazon_braket_sdk"] = getattr(braket, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # Qiskit plugin
    try:
        import pennylane_qiskit

        versions["pennylane_qiskit"] = getattr(
            pennylane_qiskit, "__version__", "unknown"
        )
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # Qiskit (if Qiskit plugin is used)
    try:
        import qiskit

        versions["qiskit"] = getattr(qiskit, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # Cirq plugin
    try:
        import pennylane_cirq

        versions["pennylane_cirq"] = getattr(pennylane_cirq, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # JAX (for autodiff)
    try:
        import jax

        versions["jax"] = getattr(jax, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # PyTorch (for autodiff)
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    # TensorFlow (for autodiff)
    try:
        import tensorflow as tf

        versions["tensorflow"] = getattr(tf, "__version__", "unknown")
    except ImportError as e:
        logger.debug("Optional import unavailable: %s", e)

    _sdk_versions = versions
    return dict(_sdk_versions)


def get_device_name(device: Any) -> str:
    """
    Get the name of a PennyLane device.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    str
        Device name or class name as fallback.

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=2)
    >>> get_device_name(dev)
    'default.qubit'
    """
    # Try name attribute first
    name = getattr(device, "name", None)
    if name:
        return str(name)

    # Try short_name
    short_name = getattr(device, "short_name", None)
    if short_name:
        return str(short_name)

    # Fallback to class name
    return device.__class__.__name__


def get_device_short_name(device: Any) -> str:
    """
    Get the short name of a PennyLane device.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    str
        Short device name.
    """
    short_name = getattr(device, "short_name", None)
    if short_name:
        return str(short_name)
    return get_device_name(device)


def is_pennylane_device(executor: Any) -> bool:
    """
    Check if an executor is a PennyLane device.

    This is the canonical check used throughout the adapter. Other modules
    should use this function rather than implementing their own checks.

    Parameters
    ----------
    executor : Any
        Potential executor instance.

    Returns
    -------
    bool
        True if executor is a PennyLane device.

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=2)
    >>> is_pennylane_device(dev)
    True
    """
    if executor is None:
        return False

    module = getattr(executor, "__module__", "") or ""
    if "pennylane" not in module:
        return False

    return hasattr(executor, "execute") or hasattr(executor, "batch_execute")


def get_wires(device: Any) -> list[Any] | None:
    """
    Get the wires from a device.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    list or None
        List of wire labels, or None if not available.
    """
    try:
        if hasattr(device, "wires"):
            return list(device.wires)
    except (AttributeError, TypeError) as e:
        logger.debug("Attribute access failed: %s", e)
    return None


def get_num_wires(device: Any) -> int | None:
    """
    Get the number of wires from a device.

    Parameters
    ----------
    device : Any
        PennyLane device instance.

    Returns
    -------
    int or None
        Number of wires, or None if not available.
    """
    wires = get_wires(device)
    return len(wires) if wires is not None else None
