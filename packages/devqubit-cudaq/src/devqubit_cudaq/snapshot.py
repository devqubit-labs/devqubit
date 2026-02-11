# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Target snapshot creation for CUDA-Q adapter.

Creates structured ``DeviceSnapshot`` objects from CUDA-Q targets,
capturing target configuration, execution environment, GPU state, and
runtime settings following the devqubit Uniform Execution Contract (UEC).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from devqubit_cudaq.utils import (
    TargetInfo,
    collect_env_snapshot,
    collect_gpu_snapshot,
    collect_sdk_versions,
    get_target_info,
    safe_list_targets,
    sanitize_runtime_events,
)
from devqubit_engine.uec.models.device import DeviceSnapshot, FrontendConfig
from devqubit_engine.utils.common import utc_now_iso


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run

logger = logging.getLogger(__name__)


# Known hardware target name fragments => provider
_PROVIDER_MAP: dict[str, str] = {
    "ionq": "ionq",
    "quantinuum": "quantinuum",
    "iqm": "iqm",
    "oqc": "oqc",
    "braket": "aws_braket",
    "infleqtion": "infleqtion",
    "pasqal": "pasqal",
    "orca": "orca",
    "quera": "quera",
    "anyon": "anyon",
    "qci": "qci",
}


def _detect_provider(target_info: TargetInfo) -> str:
    """
    Detect the physical execution provider from target info.

    Per UEC, provider should be the physical execution platform,
    not the SDK name.
    """
    name = target_info.name.lower()

    for key, provider in _PROVIDER_MAP.items():
        if key in name:
            return provider

    if target_info.is_simulator:
        return "local"

    return "local"


def _detect_backend_type(target_info: TargetInfo) -> str:
    """Detect the backend type (simulator vs hardware)."""
    return "simulator" if target_info.is_simulator else "hardware"


def _build_frontend_config(target_info: TargetInfo) -> FrontendConfig | None:
    """Build frontend configuration for CUDA-Q."""
    try:
        config: dict[str, Any] = {
            "target_name": target_info.name,
            "platform": target_info.platform,
        }
        if target_info.simulator:
            config["simulator"] = target_info.simulator
        if target_info.num_qpus is not None:
            config["num_qpus"] = target_info.num_qpus
        if target_info.description:
            config["description"] = target_info.description
        if target_info.is_emulated:
            config["is_emulated"] = True

        return FrontendConfig(
            name="cudaq",
            sdk="cudaq",
            sdk_version=collect_sdk_versions().get("cudaq"),
            config=config,
        )
    except Exception as exc:
        logger.debug("Failed to build frontend config: %s", exc)
        return None


def _build_raw_properties(
    target_info: TargetInfo,
    runtime_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Collect comprehensive raw target properties for artifact storage.

    Includes current target state, runtime configuration events,
    available targets, environment variables, and GPU information.
    """
    props: dict[str, Any] = {}

    # Current target
    target_current: dict[str, Any] = {
        "target_name": target_info.name,
        "simulator": target_info.simulator,
        "platform": target_info.platform,
        "description": target_info.description,
        "num_qpus": target_info.num_qpus,
        "is_simulator": target_info.is_simulator,
        "is_remote": target_info.is_remote,
        "is_emulated": target_info.is_emulated,
    }
    props["target_current"] = {k: v for k, v in target_current.items() if v is not None}

    # Runtime configuration events (set_target, set_noise, set_random_seed, etc.)
    if runtime_events:
        props["runtime_events"] = sanitize_runtime_events(runtime_events)

    # Available targets
    try:
        targets = safe_list_targets()
        if targets:
            props["available_targets"] = targets
    except Exception as exc:
        logger.debug("Failed to collect available targets: %s", exc)

    # Environment variables
    try:
        env = collect_env_snapshot()
        if env:
            props["env"] = env
    except Exception as exc:
        logger.debug("Failed to collect env snapshot: %s", exc)

    # GPU information
    try:
        gpu = collect_gpu_snapshot()
        if gpu:
            props["gpu"] = gpu
    except Exception as exc:
        logger.debug("Failed to collect GPU snapshot: %s", exc)

    return props


# ============================================================================
# Public API
# ============================================================================


def create_device_snapshot(
    target_info: TargetInfo | None = None,
    *,
    tracker: Run | None = None,
    runtime_events: list[dict[str, Any]] | None = None,
) -> DeviceSnapshot:
    """
    Create a ``DeviceSnapshot`` from the CUDA-Q target.

    Parameters
    ----------
    target_info : TargetInfo, optional
        Pre-introspected target info. If None, introspects ``cudaq.get_target()``.
    tracker : Run, optional
        Tracker instance for logging raw_properties as artifact.
    runtime_events : list of dict, optional
        Runtime configuration events (set_target calls, set_noise, etc.)
        captured by the executor.

    Returns
    -------
    DeviceSnapshot
        Structured device snapshot.
    """
    if target_info is None:
        target_info = get_target_info()

    captured_at = utc_now_iso()
    sdk_versions = collect_sdk_versions()

    try:
        provider = _detect_provider(target_info)
    except Exception as exc:
        logger.debug("Failed to detect provider: %s", exc)
        provider = "local"

    try:
        backend_type = _detect_backend_type(target_info)
    except Exception as exc:
        logger.debug("Failed to detect backend type: %s", exc)
        backend_type = "simulator"

    try:
        frontend = _build_frontend_config(target_info)
    except Exception as exc:
        logger.debug("Failed to build frontend config: %s", exc)
        frontend = None

    raw_properties_ref = None
    if tracker is not None:
        try:
            raw_properties = _build_raw_properties(target_info, runtime_events)
            raw_properties_ref = tracker.log_json(
                name="device_raw_properties",
                obj=raw_properties,
                role="device_raw",
                kind="device.cudaq.raw_properties.json",
            )
        except Exception as exc:
            logger.warning("Failed to log raw_properties artifact: %s", exc)

    return DeviceSnapshot(
        captured_at=captured_at,
        backend_name=target_info.name,
        backend_type=backend_type,
        provider=provider,
        backend_id=None,
        num_qubits=None,
        connectivity=None,
        native_gates=None,
        calibration=None,
        frontend=frontend,
        sdk_versions=sdk_versions,
        raw_properties_ref=raw_properties_ref,
    )
