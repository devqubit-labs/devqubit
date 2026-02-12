# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Utility functions for the CUDA-Q adapter."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from devqubit_engine.config import get_config
from devqubit_engine.utils.env import collect_gpu_info, collect_sdk_env_vars


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
        Version string, or ``"unknown"`` if cudaq is not installed.
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
        Adapter version string, or ``"unknown"`` if lookup fails.
    """
    global _adapter_version
    if _adapter_version is not None:
        return _adapter_version
    try:
        from importlib.metadata import version

        _adapter_version = version("devqubit-cudaq")
    except ImportError:
        _adapter_version = "unknown"
    return _adapter_version


_sdk_versions: dict[str, str] | None = None


def collect_sdk_versions() -> dict[str, str]:
    """
    Collect version strings for all relevant SDK packages (cached).

    Returns
    -------
    dict
        Package name => version string.  Always includes ``cudaq``.
    """
    global _sdk_versions
    if _sdk_versions is not None:
        return dict(_sdk_versions)

    versions: dict[str, str] = {"cudaq": cudaq_version()}

    for pkg, import_name in (
        ("cuquantum", "cuquantum"),
        ("numpy", "numpy"),
    ):
        try:
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError as e:
            logger.debug("Optional import unavailable: %s", e)

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


def _safe_call(obj: Any, method_name: str, *, default: Any = None) -> Any:
    """Call a method on *obj* if it exists, returning *default* on failure."""
    fn = getattr(obj, method_name, None)
    if fn is None:
        return default
    try:
        return fn() if callable(fn) else fn
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("safe_call_attr failed: %s", e)
        return default


def get_target_info() -> TargetInfo:
    """
    Get structured information about the active CUDA-Q target.

    Uses ``Target.is_remote()``, ``Target.is_remote_simulator()``, and
    ``Target.is_emulated()`` when available, falling back to heuristics
    for older CUDA-Q versions.

    Returns
    -------
    TargetInfo
        Current target information.
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
        except (TypeError, ValueError) as e:
            logger.debug("Failed to get num_qpus: %s", e)

        is_remote = _safe_call(target, "is_remote", default=None)
        is_remote_sim = _safe_call(target, "is_remote_simulator", default=None)
        is_emulated = _safe_call(target, "is_emulated", default=None)

        if is_remote is not None:
            if is_remote_sim:
                is_simulator = True
            elif is_remote:
                is_simulator = False
            else:
                is_simulator = bool(simulator)
        else:
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


def get_target_name() -> str:
    """Get the name of the current CUDA-Q target."""
    return get_target_info().name


# ============================================================================
# Kernel introspection
# ============================================================================


def is_cudaq_kernel(obj: Any) -> bool:
    """
    Check if an object is a CUDA-Q kernel.

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
        Number of qubits, or ``None`` if it cannot be determined statically.
    """
    nq = getattr(kernel, "num_qubits", None)
    if nq is not None:
        try:
            return int(nq() if callable(nq) else nq)
        except (TypeError, ValueError) as e:
            logger.debug("Int conversion failed: %s", e)
    return None


# ============================================================================
# Environment & GPU snapshots (delegates to engine)
# ============================================================================


def collect_env_snapshot() -> dict[str, str]:
    """
    Collect CUDA-Q–relevant environment variables for the device snapshot.

    Delegates to the engine's ``collect_sdk_env_vars`` which owns the
    variable registry and handles redaction via ``RedactionConfig``.

    Returns
    -------
    dict
        Variable name => value (or redacted placeholder).
    """
    return collect_sdk_env_vars("cudaq")


def collect_gpu_snapshot() -> dict[str, Any]:
    """
    Collect GPU information for the device snapshot.

    Extends the engine's ``collect_gpu_info`` with the CUDA-Q–native
    ``cudaq.num_available_gpus()`` count when available.

    Returns
    -------
    dict
        GPU snapshot with count and device list (if available).
    """
    gpu_info = collect_gpu_info()

    try:
        import cudaq

        num_gpus_fn = getattr(cudaq, "num_available_gpus", None)
        if num_gpus_fn is not None:
            gpu_info["num_available_gpus"] = int(num_gpus_fn())
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Failed to get GPU info: %s", e)

    return gpu_info


def safe_list_targets() -> list[dict[str, Any]]:
    """
    List all available CUDA-Q targets with basic metadata.

    Returns
    -------
    list of dict
        Each entry contains ``name``, ``description``, and ``platform``.
    """
    targets: list[dict[str, Any]] = []
    try:
        import cudaq

        for target in cudaq.get_targets():
            entry: dict[str, Any] = {"name": getattr(target, "name", "unknown")}
            desc = getattr(target, "description", "")
            if desc:
                entry["description"] = desc
            platform = getattr(target, "platform", "")
            if platform:
                entry["platform"] = platform
            simulator = getattr(target, "simulator", "")
            if simulator:
                entry["simulator"] = simulator
            targets.append(entry)
    except Exception as exc:
        logger.debug("Failed to list CUDA-Q targets: %s", exc)
    return targets


def sanitize_runtime_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Redact secrets from runtime configuration event kwargs.

    Parameters
    ----------
    events : list of dict
        Raw runtime configuration events.

    Returns
    -------
    list of dict
        Events with sensitive kwarg values redacted.
    """
    redaction = get_config().redaction
    sanitized: list[dict[str, Any]] = []
    for event in events:
        clean = dict(event)
        kwargs = clean.get("kwargs")
        if isinstance(kwargs, dict):
            clean["kwargs"] = redaction.redact_env(kwargs)
        sanitized.append(clean)
    return sanitized
