# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Utility functions for the CUDA-Q adapter.

Provides version utilities, target introspection, environment snapshot
collection, and common helpers used across the adapter components following
the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import inspect
import logging
import os
import subprocess
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


# ============================================================================
# Environment & GPU snapshot helpers
# ============================================================================

# CUDA-Q–relevant environment variables to capture in device snapshots.
# Sensitive values (credentials, tokens, etc.) are automatically redacted
# by the engine's ``RedactionConfig`` — the adapter does not need its own
# allowlist / blocklist logic.
_CUDAQ_ENV_VARS: tuple[str, ...] = (
    # Runtime configuration
    "CUDA_VISIBLE_DEVICES",
    "CUDAQ_DEFAULT_SIMULATOR",
    "CUDAQ_LOG_LEVEL",
    "CUDAQ_MQPU_NGPUS",
    "CUDAQ_SER_CODE_EXEC",
    "CUQUANTUM_ROOT",
    "OMP_NUM_THREADS",
    # Provider credentials (redacted by engine patterns)
    "IONQ_API_KEY",
    "CUDAQ_QUANTINUUM_CREDENTIALS",
    "IQM_SERVER_URL",
    "IQM_TOKENS_FILE",
    "OQC_URL",
    "OQC_EMAIL",
    "OQC_PASSWORD",
    "NVQC_API_KEY",
)


def _get_redaction_config() -> Any:
    """Get the engine's ``RedactionConfig``, with safe fallback."""
    try:
        from devqubit_engine.config import get_config

        return get_config().redaction
    except Exception:
        # Fallback: import class directly and use defaults
        from devqubit_engine.config import RedactionConfig

        return RedactionConfig()


def collect_env_snapshot() -> dict[str, str]:
    """
    Collect CUDA-Q–relevant environment variables for the device snapshot.

    Variables that match the engine's redaction patterns (tokens, passwords,
    API keys, cloud-provider prefixes, etc.) are replaced with the engine's
    ``RedactionConfig.replacement`` value (default ``"[REDACTED]"``).

    Returns
    -------
    dict
        Flat mapping of variable name → value (or redacted placeholder).
        Empty dict when no relevant variables are set.
    """
    raw: dict[str, str] = {}
    for var in _CUDAQ_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            raw[var] = val

    if not raw:
        return {}

    redaction = _get_redaction_config()
    return redaction.redact_env(raw)


def collect_gpu_snapshot() -> dict[str, Any]:
    """
    Collect GPU information for the device snapshot.

    Uses ``cudaq.num_available_gpus()`` when available, and falls back
    to reading ``CUDA_VISIBLE_DEVICES`` and best-effort ``nvidia-smi``.

    Returns
    -------
    dict
        GPU snapshot with count and device list (if available).
    """
    gpu_info: dict[str, Any] = {}

    # CUDA-Q native GPU count
    try:
        import cudaq

        num_gpus_fn = getattr(cudaq, "num_available_gpus", None)
        if num_gpus_fn is not None:
            gpu_info["num_available_gpus"] = int(num_gpus_fn())
    except Exception:
        pass

    # CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        gpu_info["CUDA_VISIBLE_DEVICES"] = cuda_visible

    # Best-effort nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info["nvidia_smi_devices"] = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return gpu_info


def safe_list_targets() -> list[dict[str, Any]]:
    """
    List all available CUDA-Q targets with basic metadata.

    Returns
    -------
    list of dict
        Each entry contains ``name``, ``description``, and ``platform``
        for an available target.
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

    Uses the engine's ``RedactionConfig`` to decide which kwarg keys
    contain sensitive values (tokens, passwords, API keys, etc.).

    Parameters
    ----------
    events : list of dict
        Raw runtime configuration events.

    Returns
    -------
    list of dict
        Events with sensitive kwarg values replaced by the engine's
        redaction placeholder.
    """
    redaction = _get_redaction_config()
    sanitized: list[dict[str, Any]] = []
    for event in events:
        clean = dict(event)
        kwargs = clean.get("kwargs")
        if isinstance(kwargs, dict):
            clean["kwargs"] = {
                k: (redaction.replacement if redaction.should_redact(k) else v)
                for k, v in kwargs.items()
            }
        sanitized.append(clean)
    return sanitized
