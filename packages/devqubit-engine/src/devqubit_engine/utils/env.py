# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Environment capture utilities.

This module provides functions for capturing the current execution environment,
including Python version, platform information, installed packages, git
repository state, SDK-specific environment variables, and GPU information.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from typing import Any


logger = logging.getLogger(__name__)

# Timeout for subprocess calls (seconds)
_SUBPROCESS_TIMEOUT = 30.0
_GIT_TIMEOUT = 5.0

# Per-process cache for pip freeze output keyed by (pid, executable)
_pip_cache: dict[tuple[int, str], list[str] | None] = {}


def _pip_freeze() -> list[str] | None:
    """
    Get installed packages via pip freeze.

    Runs ``pip freeze --local`` to capture the list of installed packages
    in the current environment. Results are cached per process and
    Python executable to avoid redundant subprocess calls.

    Returns
    -------
    list of str or None
        List of installed packages in pip freeze format
        (e.g., ``["numpy==1.24.0", "scipy==1.10.0"]``),
        or None if pip freeze fails or times out.

    Notes
    -----
    This function has a 30-second timeout to prevent hanging on
    slow or unresponsive pip installations.
    """
    cache_key = (os.getpid(), sys.executable)
    if cache_key in _pip_cache:
        return _pip_cache[cache_key]

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode == 0:
            packages = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and not line.startswith("#")
            ]
            logger.debug("Captured %d packages from pip freeze", len(packages))
            _pip_cache[cache_key] = packages
            return packages
        else:
            logger.debug("pip freeze failed with return code %d", result.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("pip freeze timed out after %.1f seconds", _SUBPROCESS_TIMEOUT)
    except FileNotFoundError:
        logger.debug("pip not found in PATH")
    except OSError as e:
        logger.debug("pip freeze failed: %s", e)

    _pip_cache[cache_key] = None
    return None


def capture_environment(include_pip: bool | None = None) -> dict[str, Any]:
    """
    Capture current execution environment information.

    Collects Python version, platform details, and optionally the list
    of installed packages for reproducibility tracking.

    Parameters
    ----------
    include_pip : bool, optional
        Whether to include pip freeze output:

        - ``True``: Always run pip freeze
        - ``False``: Skip pip freeze
        - ``None`` (default): Check ``DEVQUBIT_CAPTURE_PIP`` environment
          variable (enabled if set to "1" or "true")

    Returns
    -------
    dict
        Environment snapshot containing:

        - ``python_version``: Full Python version string
        - ``platform``: Platform identifier (e.g., "Linux-5.15.0-x86_64")
        - ``machine``: Machine type (e.g., "x86_64")
        - ``processor``: Processor name
        - ``env_vars``: Relevant environment variables (if any)
        - ``packages``: List of installed packages (if include_pip is True)
    """
    if include_pip is None:
        env_value = os.environ.get("DEVQUBIT_CAPTURE_PIP", "").strip().lower()
        include_pip = env_value in {"1", "true", "yes", "on"}

    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }

    # Capture relevant environment variables
    relevant_vars = [
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
    ]
    env_vars = {k: os.environ[k] for k in relevant_vars if k in os.environ}
    if env_vars:
        env["env_vars"] = env_vars

    if include_pip:
        packages = _pip_freeze()
        if packages:
            env["packages"] = packages

    logger.debug(
        "Captured environment: platform=%s, pip=%s", env["platform"], include_pip
    )
    return env


def _sanitize_git_url(url: str | None) -> str | None:
    """
    Sanitize a git remote URL by removing embedded credentials.

    Removes userinfo (user:password@) from URLs to prevent accidental
    token/credential leakage in run records.

    Parameters
    ----------
    url : str or None
        Git remote URL, possibly containing credentials.

    Returns
    -------
    str or None
        URL with credentials removed, or None if input was None.

    Examples
    --------
    >>> _sanitize_git_url("https://token@github.com/org/repo.git")
    'https://github.com/org/repo.git'
    >>> _sanitize_git_url("https://user:pass@github.com/org/repo.git")
    'https://github.com/org/repo.git'
    >>> _sanitize_git_url("git@github.com:org/repo.git")
    'git@github.com:org/repo.git'
    """
    if not url:
        return url

    # Handle HTTPS URLs with credentials: https://user:pass@host/...
    # Pattern matches: scheme://[userinfo@]host...
    import re

    # Match URLs with credentials in userinfo section
    pattern = r"^(https?://)([^@]+@)(.+)$"
    match = re.match(pattern, url)
    if match:
        scheme, _, rest = match.groups()
        sanitized = f"{scheme}{rest}"
        logger.debug("Sanitized credentials from git remote URL")
        return sanitized

    return url


def capture_git_provenance(cwd: str | None = None) -> dict[str, Any] | None:
    """
    Capture git repository provenance information.

    Best-effort capture that returns None if not in a git repository
    or if git is not available.

    Parameters
    ----------
    cwd : str, optional
        Working directory to check. Defaults to the current working directory.

    Returns
    -------
    dict or None
        Git provenance containing:

        - ``commit``: Full commit SHA (40 hex characters)
        - ``branch``: Current branch name (e.g., "main", "feature/xyz")
        - ``dirty``: Whether working directory has uncommitted changes
        - ``describe``: Output of ``git describe --tags --always --dirty``
        - ``remote``: Remote origin URL (if configured)

        Returns None if not in a git repository or git is unavailable.
    """

    def _run_git(args: list[str]) -> str | None:
        """Run a git command and return stdout, or None on failure."""
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.debug("git %s timed out", " ".join(args))
        except FileNotFoundError:
            pass  # git not installed
        except OSError as e:
            logger.debug("git %s failed: %s", " ".join(args), e)
        return None

    # Check if we're inside a git repository
    inside = _run_git(["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        logger.debug("Not inside a git repository")
        return None

    # Gather git information
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _run_git(["status", "--porcelain"])
    dirty = bool(status_output)
    remote_raw = _run_git(["config", "--get", "remote.origin.url"])
    remote = _sanitize_git_url(remote_raw)
    describe = _run_git(["describe", "--tags", "--always", "--dirty"])

    result = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "describe": describe,
        "remote": remote,
    }

    logger.debug(
        "Captured git provenance: commit=%s, branch=%s, dirty=%s",
        (commit or "")[:8],
        branch,
        dirty,
    )

    return result


# ============================================================================
# SDK environment & GPU utilities (used by adapters)
# ============================================================================


# ---------------------------------------------------------------------------
# SDK environment variable registry
# ---------------------------------------------------------------------------
#
# Centralised mapping of SDK name => environment variables relevant to that
# SDK.  Adapters never own variable lists — they call
# ``collect_sdk_env_vars("cudaq")`` and the engine resolves + redacts.
#
# Variables shared across multiple SDKs (e.g. CUDA_VISIBLE_DEVICES,
# OMP_NUM_THREADS, provider credentials) appear in every SDK that needs
# them.  New SDKs are added here, not in adapter code.

_COMMON_GPU_VARS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
)

_PROVIDER_CREDENTIAL_VARS: tuple[str, ...] = (
    "IONQ_API_KEY",
    "IQM_SERVER_URL",
    "IQM_TOKENS_FILE",
    "OQC_URL",
    "OQC_EMAIL",
    "OQC_PASSWORD",
)

SDK_ENV_VARS: dict[str, tuple[str, ...]] = {
    "cudaq": (
        *_COMMON_GPU_VARS,
        *_PROVIDER_CREDENTIAL_VARS,
        "CUDAQ_DEFAULT_SIMULATOR",
        "CUDAQ_LOG_LEVEL",
        "CUDAQ_MQPU_NGPUS",
        "CUDAQ_SER_CODE_EXEC",
        "CUQUANTUM_ROOT",
        "CUDAQ_QUANTINUUM_CREDENTIALS",
        "NVQC_API_KEY",
    ),
    "qiskit": (
        *_COMMON_GPU_VARS,
        *_PROVIDER_CREDENTIAL_VARS,
        "QISKIT_IBM_TOKEN",
        "QISKIT_IBM_URL",
        "QISKIT_IBM_INSTANCE",
        "QISKIT_SETTINGS_FILE",
    ),
    "cirq": (
        *_COMMON_GPU_VARS,
        *_PROVIDER_CREDENTIAL_VARS,
        "GOOGLE_CLOUD_PROJECT",
        "CIRQ_TESTING",
    ),
    "pennylane": (
        *_COMMON_GPU_VARS,
        *_PROVIDER_CREDENTIAL_VARS,
        "PENNYLANE_CONFIG_DIR",
    ),
    "braket": (
        *_COMMON_GPU_VARS,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ),
}


def collect_sdk_env_vars(sdk: str, extra_vars: Sequence[str] = ()) -> dict[str, str]:
    """
    Collect environment variables for an SDK with automatic redaction.

    Looks up the variable list from the engine's ``SDK_ENV_VARS`` registry,
    merges any *extra_vars* the caller supplies, reads present values from
    ``os.environ``, and applies ``RedactionConfig`` to mask secrets.

    Parameters
    ----------
    sdk : str
        SDK identifier (e.g. ``"cudaq"``, ``"qiskit"``).
    extra_vars : sequence of str, optional
        Additional variable names beyond the registry entry.

    Returns
    -------
    dict[str, str]
        Variable name => value (or redacted placeholder).
        Only variables that are actually set are included.
    """
    var_names = SDK_ENV_VARS.get(sdk, ())
    if extra_vars:
        seen = set(var_names)
        var_names = (*var_names, *(v for v in extra_vars if v not in seen))

    raw: dict[str, str] = {}
    for var in var_names:
        val = os.environ.get(var)
        if val is not None:
            raw[var] = val

    if not raw:
        return {}

    try:
        from devqubit_engine.config import get_config

        return get_config().redaction.redact_env(raw)
    except Exception:
        from devqubit_engine.config import RedactionConfig

        return RedactionConfig().redact_env(raw)


def collect_gpu_info() -> dict[str, Any]:
    """
    Collect GPU information for device snapshots.

    Reads ``CUDA_VISIBLE_DEVICES`` and runs ``nvidia-smi -L`` to
    enumerate GPU devices.  Adapters can extend the returned dict
    with SDK-specific GPU data (e.g. ``cudaq.num_available_gpus()``).

    Returns
    -------
    dict[str, Any]
        GPU information.  Empty dict when no GPU data is available.
    """
    gpu_info: dict[str, Any] = {}

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        gpu_info["CUDA_VISIBLE_DEVICES"] = cuda_visible

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
