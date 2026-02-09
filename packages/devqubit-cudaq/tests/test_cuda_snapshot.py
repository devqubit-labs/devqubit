# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for device snapshots (snapshot.py) and utilities (utils.py)."""

import json

from devqubit_cudaq.snapshot import (
    _detect_backend_type,
    _detect_provider,
    create_device_snapshot,
)
from devqubit_cudaq.utils import (
    _HARDWARE_TARGETS,
    _SIMULATOR_TARGETS,
    TargetInfo,
    collect_sdk_versions,
    get_adapter_version,
    get_kernel_name,
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

    def test_hardware_targets(self):
        assert "ionq" in _HARDWARE_TARGETS
        assert "quantinuum" in _HARDWARE_TARGETS
        assert "qpp-cpu" not in _HARDWARE_TARGETS

    def test_simulator_targets(self):
        assert "qpp-cpu" in _SIMULATOR_TARGETS
        assert "nvidia" in _SIMULATOR_TARGETS
        assert "ionq" not in _SIMULATOR_TARGETS


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
        assert snap.frontend.name == "cudaq"

    def test_hardware(self):
        info = TargetInfo(name="ionq", is_simulator=False, is_remote=True)
        snap = create_device_snapshot(info)
        assert snap.backend_name == "ionq"
        assert snap.backend_type == "hardware"
        assert snap.provider == "ionq"

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
        assert "devqubit_cudaq" in snap.sdk_versions


class TestUtilities:

    def test_get_adapter_version(self):
        assert isinstance(get_adapter_version(), str)

    def test_collect_sdk_versions(self):
        versions = collect_sdk_versions()
        assert "devqubit_cudaq" in versions

    def test_get_kernel_name(self, bell_kernel):
        assert get_kernel_name(bell_kernel) == "bell"

    def test_get_kernel_name_unnamed(self):
        assert get_kernel_name(object()) == "unknown_kernel"

    def test_target_info_defaults(self):
        info = TargetInfo(name="test")
        assert info.simulator == ""
        assert info.is_simulator is True
        assert info.is_remote is False
