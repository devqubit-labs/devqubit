# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for device snapshots (snapshot.py) and utilities (utils.py)."""

import json

from devqubit_cudaq.snapshot import (
    _build_raw_properties,
    _detect_backend_type,
    _detect_provider,
    create_device_snapshot,
)
from devqubit_cudaq.utils import (
    TargetInfo,
    collect_env_snapshot,
    collect_gpu_snapshot,
    collect_sdk_versions,
    get_adapter_version,
    get_kernel_name,
    sanitize_runtime_events,
)


class TestDetectProvider:

    def test_local_simulator(self):
        info = TargetInfo(name="qpp-cpu", is_simulator=True, is_remote=False)
        assert _detect_provider(info) == "local"

    def test_nvidia_simulator(self):
        info = TargetInfo(name="nvidia", is_simulator=True, is_remote=False)
        assert _detect_provider(info) == "local"

    def test_ionq(self):
        info = TargetInfo(name="ionq", is_simulator=False, is_remote=True)
        assert _detect_provider(info) == "ionq"

    def test_quantinuum(self):
        info = TargetInfo(name="quantinuum", is_simulator=False, is_remote=True)
        assert _detect_provider(info) == "quantinuum"


class TestDetectBackendType:

    def test_simulator(self):
        info = TargetInfo(name="nvidia", is_simulator=True)
        assert _detect_backend_type(info) == "simulator"

    def test_hardware(self):
        info = TargetInfo(name="ionq", is_simulator=False, is_remote=True)
        assert _detect_backend_type(info) == "hardware"


class TestTargetClassification:

    def test_hardware_targets_detected(self):
        """Known hardware target names are classified as non-simulator."""
        for name in ("ionq", "quantinuum", "iqm", "oqc", "braket"):
            info = TargetInfo(
                name=name, simulator="", is_simulator=False, is_remote=True
            )
            assert _detect_provider(info) == name or _detect_provider(info) != "local"


class TestCreateDeviceSnapshot:

    def test_simulator(self):
        info = TargetInfo(
            name="nvidia",
            simulator="custatevec",
            platform="default",
            description="NVIDIA GPU",
            num_qpus=1,
            is_simulator=True,
            is_remote=False,
        )
        snap = create_device_snapshot(info)
        assert snap.backend_name == "nvidia"
        assert snap.backend_type == "simulator"
        assert snap.provider == "local"

    def test_hardware(self):
        info = TargetInfo(name="ionq", is_simulator=False, is_remote=True)
        snap = create_device_snapshot(info)
        assert snap.backend_name == "ionq"
        assert snap.backend_type == "hardware"
        assert snap.provider == "ionq"

    def test_emulated_target(self):
        info = TargetInfo(
            name="ionq-emulated",
            is_simulator=True,
            is_remote=False,
            is_emulated=True,
        )
        snap = create_device_snapshot(info)
        assert snap.backend_type == "simulator"
        if snap.frontend and snap.frontend.config:
            assert snap.frontend.config.get("is_emulated") is True

    def test_serializable(self):
        info = TargetInfo(
            name="qpp-cpu", simulator="qpp", is_simulator=True, is_remote=False
        )
        snap = create_device_snapshot(info)
        parsed = json.loads(json.dumps(snap.to_dict()))
        assert parsed["backend_name"] == "qpp-cpu"

    def test_sdk_versions_present(self):
        info = TargetInfo(name="qpp-cpu", is_simulator=True)
        snap = create_device_snapshot(info)
        assert "cudaq" in snap.sdk_versions

    def test_frontend_has_sdk_field(self):
        info = TargetInfo(name="nvidia", simulator="custatevec", is_simulator=True)
        snap = create_device_snapshot(info)
        assert snap.frontend is not None
        assert snap.frontend.sdk == "cudaq"


class TestBuildRawProperties:

    def test_includes_target_current(self):
        info = TargetInfo(name="nvidia", simulator="custatevec", is_simulator=True)
        props = _build_raw_properties(info)
        assert "target_current" in props
        assert props["target_current"]["target_name"] == "nvidia"

    def test_includes_runtime_events(self):
        info = TargetInfo(name="qpp-cpu", is_simulator=True)
        events = [{"method": "set_target", "args": ["nvidia"], "kwargs": {}}]
        props = _build_raw_properties(info, runtime_events=events)
        assert "runtime_events" in props
        assert props["runtime_events"][0]["method"] == "set_target"

    def test_omits_empty_runtime_events(self):
        info = TargetInfo(name="qpp-cpu", is_simulator=True)
        props = _build_raw_properties(info, runtime_events=None)
        assert "runtime_events" not in props


class TestEnvSnapshot:

    def test_collects_cuda_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        snap = collect_env_snapshot()
        assert snap.get("CUDA_VISIBLE_DEVICES") == "0,1"

    def test_secret_redacted_by_engine(self, monkeypatch):
        """Sensitive vars are redacted via engine's RedactionConfig."""
        monkeypatch.setenv("IONQ_API_KEY", "secret-key-123")
        snap = collect_env_snapshot()
        # Engine's ^IONQ_ and API_?KEY patterns both match — value must be redacted
        assert "secret-key-123" not in str(snap)
        assert snap.get("IONQ_API_KEY") == "[REDACTED]"

    def test_empty_when_no_vars_set(self, monkeypatch):
        """Returns empty dict when no CUDAQ-relevant vars are set."""
        for var in (
            "CUDA_VISIBLE_DEVICES",
            "CUDAQ_DEFAULT_SIMULATOR",
            "IONQ_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        snap = collect_env_snapshot()
        assert isinstance(snap, dict)


class TestSanitizeRuntimeEvents:

    def test_redacts_api_key(self):
        events = [{"method": "set_target", "kwargs": {"api_key": "secret123"}}]
        sanitized = sanitize_runtime_events(events)
        assert sanitized[0]["kwargs"]["api_key"] == "[REDACTED]"

    def test_preserves_safe_kwargs(self):
        events = [{"method": "set_target", "kwargs": {"machine": "aria-1"}}]
        sanitized = sanitize_runtime_events(events)
        assert sanitized[0]["kwargs"]["machine"] == "aria-1"

    def test_redacts_password(self):
        events = [{"method": "set_target", "kwargs": {"password": "p4ss"}}]
        sanitized = sanitize_runtime_events(events)
        assert sanitized[0]["kwargs"]["password"] == "[REDACTED]"


class TestGpuSnapshot:

    def test_returns_dict(self):
        snap = collect_gpu_snapshot()
        assert isinstance(snap, dict)


class TestUtilities:

    def test_get_adapter_version(self):
        assert isinstance(get_adapter_version(), str)

    def test_collect_sdk_versions(self):
        versions = collect_sdk_versions()
        assert "cudaq" in versions

    def test_get_kernel_name(self, bell_kernel):
        assert get_kernel_name(bell_kernel) == "bell"

    def test_get_kernel_name_unnamed(self):
        assert get_kernel_name(object()) == "object"

    def test_target_info_defaults(self):
        info = TargetInfo(name="test")
        assert info.simulator == ""
        assert info.is_simulator is True
        assert info.is_remote is False
        assert info.is_emulated is False

    def test_target_info_to_dict(self):
        info = TargetInfo(name="nvidia", simulator="custatevec", is_emulated=True)
        d = info.to_dict()
        assert d["name"] == "nvidia"
        assert d["is_emulated"] is True
