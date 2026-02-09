# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared fixtures for CUDA-Q adapter tests."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.tracking.run import Run


# ---------------------------------------------------------------------------
# Mock classes (internal — tests access these through fixtures only)
# ---------------------------------------------------------------------------


class _MockSampleResult:
    """Mock ``cudaq.SampleResult``."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def items(self):
        return self._counts.items()

    def get_total_shots(self) -> int:
        return sum(self._counts.values())

    def __iter__(self):
        return iter(self._counts)

    def __getitem__(self, key: str) -> int:
        return self._counts[key]

    def __len__(self) -> int:
        return len(self._counts)

    def most_probable(self) -> str:
        return max(self._counts, key=self._counts.get)

    def probability(self, bitstring: str) -> float:
        total = sum(self._counts.values())
        return self._counts.get(bitstring, 0) / total if total > 0 else 0.0

    @property
    def register_names(self):
        return []


class _MockObserveResult:
    """Mock ``cudaq.ObserveResult``."""

    def __init__(self, expectation: float, counts: dict[str, int] | None = None):
        self._expectation = expectation
        self._counts = counts

    def expectation(self) -> float:
        return self._expectation

    def counts(self):
        if self._counts is not None:
            return _MockSampleResult(self._counts)
        return None


class _MockTarget:
    """Mock ``cudaq.Target`` with Target API methods."""

    def __init__(
        self,
        name: str = "qpp-cpu",
        simulator: str = "qpp",
        platform: str = "default",
        description: str = "Default CPU simulator",
        num_qpus: int = 1,
        *,
        _is_remote: bool = False,
        _is_remote_simulator: bool = False,
        _is_emulated: bool = False,
    ):
        self.name = name
        self._simulator = simulator
        self._platform = platform
        self._description = description
        self._num_qpus = num_qpus
        self.__is_remote = _is_remote
        self.__is_remote_simulator = _is_remote_simulator
        self.__is_emulated = _is_emulated

    @property
    def simulator(self):
        return self._simulator

    @property
    def platform(self):
        return self._platform

    @property
    def description(self):
        return self._description

    @property
    def num_qpus(self):
        return self._num_qpus

    def is_remote(self) -> bool:
        return self.__is_remote

    def is_remote_simulator(self) -> bool:
        return self.__is_remote_simulator

    def is_emulated(self) -> bool:
        return self.__is_emulated


class _MockKernel:
    """Mock ``cudaq.PyKernelDecorator`` with realistic ``to_json()``."""

    def __init__(
        self,
        name: str = "bell",
        num_qubits: int = 2,
        instructions: list[dict[str, Any]] | None = None,
        has_arguments: bool = False,
    ):
        self.__name__ = name
        self.name = name
        self._num_qubits = num_qubits
        self._has_arguments = has_arguments
        self._instructions = instructions or self._default_instructions()

    def _default_instructions(self) -> list[dict[str, Any]]:
        if self.__name__ == "bell":
            return [
                {"gate": "h", "qubits": [0]},
                {"gate": "cx", "qubits": [0, 1], "controls": []},
                {"gate": "mz", "qubits": [0]},
                {"gate": "mz", "qubits": [1]},
            ]
        return [{"gate": "h", "qubits": [0]}]

    @property
    def arguments(self):
        if self._has_arguments:
            return [{"name": "theta", "type": "float"}]
        return []

    @property
    def argument_count(self):
        return len(self.arguments)

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.__name__,
                "num_qubits": self._num_qubits,
                "instructions": self._instructions,
            }
        )


def _make_cudaq_module(
    target: _MockTarget | None = None,
    sample_result: Any = None,
    observe_result: Any = None,
) -> types.ModuleType:
    """Build a minimal mock cudaq module."""
    mod = types.ModuleType("cudaq")
    mod.__version__ = "0.9.0"
    mod.get_target = MagicMock(return_value=target or _MockTarget())
    mod.set_target = MagicMock()
    mod.sample = MagicMock(
        return_value=sample_result or _MockSampleResult({"00": 500, "11": 500})
    )
    mod.observe = MagicMock(
        return_value=observe_result or _MockObserveResult(expectation=-0.42)
    )
    mod.draw = MagicMock(return_value=None)
    mod.translate = MagicMock(return_value=None)
    mod.PyKernelDecorator = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# Kernel fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_kernel():
    """2-qubit Bell state kernel: H(0), CX(0,1), MZ."""
    return _MockKernel("bell", num_qubits=2)


@pytest.fixture
def ghz_kernel():
    """3-qubit GHZ kernel: H(0), CX(0,1), CX(1,2), MZ."""
    return _MockKernel(
        "ghz",
        num_qubits=3,
        instructions=[
            {"gate": "h", "qubits": [0]},
            {"gate": "cx", "qubits": [0, 1]},
            {"gate": "cx", "qubits": [1, 2]},
            {"gate": "mz", "qubits": [0]},
            {"gate": "mz", "qubits": [1]},
            {"gate": "mz", "qubits": [2]},
        ],
    )


@pytest.fixture
def parameterized_kernel():
    """Parameterized VQE-style kernel with rx(theta), ry(phi)."""
    return _MockKernel(
        "ansatz",
        num_qubits=2,
        instructions=[
            {"gate": "rx", "qubits": [0], "params": [0.5]},
            {"gate": "ry", "qubits": [1], "params": [0.3]},
            {"gate": "cx", "qubits": [0, 1]},
        ],
        has_arguments=True,
    )


@pytest.fixture
def bare_kernel():
    """Kernel without to_json() support (C++ builder-style)."""

    class _BareKernel:
        __name__ = "legacy"
        name = "legacy"

    return _BareKernel()


# ---------------------------------------------------------------------------
# Result fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_sample_result():
    """Bell state sample: ~50/50 between |00⟩ and |11⟩."""
    return _MockSampleResult({"00": 480, "11": 520})


@pytest.fixture
def uniform_sample_result():
    """Uniform distribution over 2 qubits."""
    return _MockSampleResult({"00": 250, "01": 250, "10": 250, "11": 250})


@pytest.fixture
def observe_result():
    """Observe result with expectation = -0.42."""
    return _MockObserveResult(expectation=-0.42)


@pytest.fixture
def observe_result_with_counts():
    """Observe result with expectation AND shot-based counts."""
    return _MockObserveResult(expectation=0.6, counts={"00": 400, "11": 600})


# ---------------------------------------------------------------------------
# Target fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simulator_target():
    """Local CPU simulator target (qpp-cpu)."""
    return _MockTarget("qpp-cpu", simulator="qpp", platform="default")


@pytest.fixture
def gpu_target():
    """NVIDIA GPU simulator target."""
    return _MockTarget(
        "nvidia",
        simulator="custatevec",
        platform="default",
        description="NVIDIA GPU simulator",
    )


@pytest.fixture
def hardware_target():
    """Remote hardware target (IonQ)."""
    return _MockTarget(
        "ionq",
        simulator="",
        platform="default",
        description="IonQ QPU",
        _is_remote=True,
    )


# ---------------------------------------------------------------------------
# devqubit-engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracking_root(tmp_path):
    """Temporary tracking directory."""
    return tmp_path / ".devqubit"


@pytest.fixture
def store(tracking_root):
    """Temporary object store."""
    return create_store(f"file://{tracking_root}/objects")


@pytest.fixture
def registry(tracking_root):
    """Temporary run registry."""
    return create_registry(f"file://{tracking_root}")


# ---------------------------------------------------------------------------
# Factory fixtures — tests create custom mocks through these, never via import
# ---------------------------------------------------------------------------


@pytest.fixture
def make_kernel():
    """Factory: ``make_kernel(name, num_qubits, instructions, has_arguments)``."""

    def _factory(name="bell", num_qubits=2, instructions=None, has_arguments=False):
        return _MockKernel(name, num_qubits, instructions, has_arguments)

    return _factory


@pytest.fixture
def make_sample_result():
    """Factory: ``make_sample_result(counts_dict)``."""

    def _factory(counts: dict[str, int]):
        return _MockSampleResult(counts)

    return _factory


@pytest.fixture
def make_observe_result():
    """Factory: ``make_observe_result(expectation, counts=None)``."""

    def _factory(expectation: float, counts: dict[str, int] | None = None):
        return _MockObserveResult(expectation, counts)

    return _factory


@pytest.fixture
def make_target():
    """Factory: ``make_target(name, simulator, ...)``."""

    def _factory(
        name="qpp-cpu",
        simulator="qpp",
        platform="default",
        description="",
        num_qpus=1,
        *,
        _is_remote=False,
        _is_remote_simulator=False,
        _is_emulated=False,
    ):
        return _MockTarget(
            name,
            simulator,
            platform,
            description,
            num_qpus,
            _is_remote=_is_remote,
            _is_remote_simulator=_is_remote_simulator,
            _is_emulated=_is_emulated,
        )

    return _factory


@pytest.fixture
def make_executor():
    """
    Factory: ``make_executor(run, target=..., sample_result=..., **kw) -> TrackedCudaqExecutor``.
    """

    def _factory(
        run: Run,
        *,
        target: Any = None,
        sample_result: Any = None,
        observe_result: Any = None,
        **executor_kwargs: Any,
    ):
        from devqubit_cudaq.adapter import TrackedCudaqExecutor

        mod = _make_cudaq_module(
            target=target,
            sample_result=sample_result,
            observe_result=observe_result,
        )
        executor = TrackedCudaqExecutor(tracker=run, executor=mod, **executor_kwargs)
        return executor

    return _factory
