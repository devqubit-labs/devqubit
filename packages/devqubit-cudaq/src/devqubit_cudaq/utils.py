# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for the CUDA-Q adapter.

Provides version utilities, target introspection, and common helpers used
across the adapter components following the devqubit Uniform Execution
Contract (UEC).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# Version caching
# ============================================================================

_cudaq_version: str | None = None


def cudaq_version() -> str:
    """
    Get the installed CUDA-Q version (cached after first call).

    Returns
    -------
    str
        CUDA-Q version string (e.g., ``"0.13.0"``), or ``"unknown"`` if
        cudaq is not installed or version cannot be determined.
    """
    global _cudaq_version
    if _cudaq_version is not None:
        return _cudaq_version
    try:
        import cudaq

        _cudaq_version = getattr(cudaq, "__version__", "unknown")
    except ImportError:
        _cudaq_version = "unknown"
    return _cudaq_version


_adapter_version: str | None = None


def get_adapter_version() -> str:
    """
    Get adapter version from package metadata (cached).

    Returns
    -------
    str
        Adapter version string, or ``"unknown"`` if metadata lookup fails.
    """
    global _adapter_version
    if _adapter_version is not None:
        return _adapter_version
    try:
        from importlib.metadata import version

        _adapter_version = version("devqubit-cudaq")
    except Exception:
        _adapter_version = "unknown"
    return _adapter_version


_sdk_versions: dict[str, str] | None = None


def collect_sdk_versions() -> dict[str, str]:
    """
    Collect version strings for all relevant SDK packages (cached).

    Follows the UEC requirement for tracking all SDK versions in the
    execution environment.

    Returns
    -------
    dict
        Mapping of package name to version string.  Always includes
        ``cudaq``; optional entries for ``cuquantum``, ``numpy``, etc.
    """
    global _sdk_versions
    if _sdk_versions is not None:
        return dict(_sdk_versions)

    versions: dict[str, str] = {}
    versions["cudaq"] = cudaq_version()

    for pkg, import_name in (
        ("cuquantum", "cuquantum"),
        ("numpy", "numpy"),
    ):
        try:
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    _sdk_versions = versions
    return dict(_sdk_versions)


# ============================================================================
# Target introspection
# ============================================================================


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """
    Structured information about a CUDA-Q target.

    Attributes
    ----------
    name : str
        Target name (e.g., ``"qpp-cpu"``, ``"nvidia"``, ``"ionq"``).
    simulator : str
        Simulator backend name; empty for hardware QPUs.
    platform : str
        Platform name (e.g., ``"default"``, ``"mqpu"``).
    description : str
        Human-readable target description.
    num_qpus : int
        Number of QPUs available in this target.
    is_simulator : bool
        ``True`` if the target is a simulator.
    is_remote : bool
        ``True`` if the target routes to a remote/cloud backend.
    is_emulated : bool
        ``True`` if the target is an emulated backend.
    """

    name: str
    simulator: str = ""
    platform: str = ""
    description: str = ""
    num_qpus: int = 1
    is_simulator: bool = True
    is_remote: bool = False
    is_emulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to serialisable dictionary."""
        return {
            "name": self.name,
            "simulator": self.simulator,
            "platform": self.platform,
            "description": self.description,
            "num_qpus": self.num_qpus,
            "is_simulator": self.is_simulator,
            "is_remote": self.is_remote,
            "is_emulated": self.is_emulated,
        }


# Known hardware-backed targets (fallback for targets without API methods)
_HARDWARE_TARGETS: frozenset[str] = frozenset(
    {
        "ionq",
        "quantinuum",
        "iqm",
        "oqc",
        "braket",
        "infleqtion",
        "pasqal",
        "orca",
        "quera",
        "anyon",
        "qci",
    }
)


def get_target_info() -> TargetInfo:
    """
    Get structured information about the active CUDA-Q target.

    Uses ``Target.is_remote()``, ``Target.is_remote_simulator()``, and
    ``Target.is_emulated()`` when available, falling back to heuristics
    only for older CUDA-Q versions that lack these methods.

    Returns
    -------
    TargetInfo
        Current target information with simulator/hardware classification.
    """
    try:
        import cudaq

        target = cudaq.get_target()
        name = getattr(target, "name", "unknown")
        simulator = getattr(target, "simulator", "")
        platform = getattr(target, "platform", "")
        description = getattr(target, "description", "")

        num_qpus = 1
        try:
            nq = target.num_qpus
            num_qpus = nq() if callable(nq) else int(nq)
        except Exception:
            pass

        # Prefer Target API methods for classification
        is_remote = _safe_call(target, "is_remote", default=None)
        is_remote_sim = _safe_call(target, "is_remote_simulator", default=None)
        is_emulated = _safe_call(target, "is_emulated", default=None)

        if is_remote is not None:
            # API methods available — use them directly
            if is_remote_sim:
                is_simulator = True
            elif is_remote:
                is_simulator = False
            else:
                # Local target — simulator if it has a simulator backend name
                is_simulator = bool(simulator)
        else:
            # Fallback heuristics for older CUDA-Q
            name_lower = name.lower()
            is_remote = any(hw in name_lower for hw in _HARDWARE_TARGETS)
            is_simulator = bool(simulator) or not is_remote
            if is_remote and not simulator:
                is_simulator = False

        return TargetInfo(
            name=name,
            simulator=simulator,
            platform=platform,
            description=description,
            num_qpus=num_qpus,
            is_simulator=is_simulator,
            is_remote=bool(is_remote),
            is_emulated=bool(is_emulated) if is_emulated is not None else False,
        )
    except Exception as exc:
        logger.debug("Failed to get CUDA-Q target info: %s", exc)
        return TargetInfo(name="unknown")


def _safe_call(obj: Any, method_name: str, *, default: Any = None) -> Any:
    """Call a method on *obj* if it exists, returning *default* on failure."""
    fn = getattr(obj, method_name, None)
    if fn is None:
        return default
    try:
        return fn() if callable(fn) else fn
    except Exception:
        return default


def get_target_name() -> str:
    """
    Get the name of the current CUDA-Q target.

    Returns
    -------
    str
        Target name string.
    """
    return get_target_info().name


# ============================================================================
# Kernel introspection
# ============================================================================


def is_cudaq_kernel(obj: Any) -> bool:
    """
    Check if an object is a CUDA-Q kernel.

    Recognises both decorator-style (``@cudaq.kernel``) and builder-style
    (``cudaq.make_kernel()``) kernels.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        ``True`` if the object is a CUDA-Q kernel.
    """
    if obj is None:
        return False

    type_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "") or ""

    if "PyKernelDecorator" in type_name:
        return True
    if type_name == "PyKernel" and "cudaq" in module:
        return True
    if type_name == "Kernel" and "cudaq" in module:
        return True
    if getattr(obj, "__wrapped_cudaq_kernel__", False):
        return True
    return False


def is_cudaq_module(obj: Any) -> bool:
    """
    Check if an object is the ``cudaq`` module.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        ``True`` if *obj* is the cudaq module.
    """
    if not inspect.ismodule(obj):
        return False
    return getattr(obj, "__name__", "") == "cudaq"


def get_kernel_name(kernel: Any) -> str:
    """
    Extract a human-readable name from a CUDA-Q kernel.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel object.

    Returns
    -------
    str
        Kernel name, or ``"unknown"`` as fallback.
    """
    for attr in ("name", "__name__"):
        name = getattr(kernel, attr, None)
        if name:
            return str(name)
    return type(kernel).__name__


def get_kernel_num_qubits(kernel: Any) -> int | None:
    """
    Try to determine the qubit count from a kernel.

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel object.

    Returns
    -------
    int or None
        Number of qubits, or ``None`` if it cannot be determined
        statically.
    """
    nq = getattr(kernel, "num_qubits", None)
    if nq is not None:
        try:
            return int(nq() if callable(nq) else nq)
        except Exception:
            pass
    return None
