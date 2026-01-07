# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared fixtures for Braket adapter tests."""

from __future__ import annotations

from typing import Any

import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from devqubit_engine.storage.factory import create_registry, create_store


@pytest.fixture
def tracking_root(tmp_path):
    """Create temporary tracking directory."""
    return tmp_path / ".devqubit"


@pytest.fixture
def store(tracking_root):
    """Create temporary object store."""
    return create_store(f"file://{tracking_root}/objects")


@pytest.fixture
def registry(tracking_root):
    """Create temporary run registry."""
    return create_registry(f"file://{tracking_root}")


@pytest.fixture
def local_simulator():
    """Real Braket LocalSimulator."""
    return LocalSimulator()


@pytest.fixture
def bell_circuit():
    """2-qubit Bell state circuit."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit


@pytest.fixture
def ghz_circuit():
    """3-qubit GHZ state circuit."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    return circuit


@pytest.fixture
def non_clifford_circuit():
    """Circuit with non-Clifford gates (T gate)."""
    circuit = Circuit()
    circuit.h(0)
    circuit.t(0)
    circuit.cnot(0, 1)
    return circuit


def build_device_properties(
    *,
    qubit_count: int = 2,
    native_gates: list[str] | None = None,
    connectivity_graph: dict[str, list[int]] | None = None,
    fully_connected: bool = False,
    one_qubit_fidelities: dict[int, list[dict]] | None = None,
    two_qubit_fidelities: dict[str, list[dict]] | None = None,
    calibration_time: str = "2025-01-02T03:04:05Z",
) -> dict:
    """
    Build Braket device properties dict.

    Parameters
    ----------
    qubit_count : int
        Number of qubits.
    native_gates : list of str, optional
        Native gate set.
    connectivity_graph : dict, optional
        Connectivity graph {qubit: [neighbors]}.
    fully_connected : bool
        Whether device is fully connected.
    one_qubit_fidelities : dict, optional
        {qubit: [{"gateName": str, "fidelity": float}]}.
    two_qubit_fidelities : dict, optional
        {edge_key: [{"gateName": str, "fidelity": float}]}.
    calibration_time : str
        Calibration timestamp.

    Returns
    -------
    dict
        Properties dict in Braket format.
    """
    props: dict = {
        "service": {"updatedAt": calibration_time},
        "paradigm": {
            "qubitCount": qubit_count,
            "connectivity": {"fullyConnected": fully_connected},
        },
    }

    if native_gates is not None:
        props["paradigm"]["nativeGateSet"] = native_gates

    if connectivity_graph is not None:
        props["paradigm"]["connectivity"]["connectivityGraph"] = connectivity_graph

    if one_qubit_fidelities or two_qubit_fidelities:
        props["standardized"] = {}

        if one_qubit_fidelities:
            props["standardized"]["oneQubitProperties"] = {
                str(q): {"oneQubitGateFidelity": fids}
                for q, fids in one_qubit_fidelities.items()
            }

        if two_qubit_fidelities:
            props["standardized"]["twoQubitProperties"] = {
                edge_key: {"twoQubitGateFidelity": fids}
                for edge_key, fids in two_qubit_fidelities.items()
            }

    return props


class MockProperties:
    """Mock properties object that mimics pydantic behavior."""

    def __init__(self, d: dict):
        self._d = d

    def dict(self) -> dict:
        return self._d


class MockTask:
    """Mock Braket task for testing."""

    def __init__(
        self,
        result_value: Any = None,
        task_id: str = "mock-task-123",
        raise_on_result: Exception | None = None,
    ):
        self._result_value = result_value
        self._task_id = task_id
        self._raise_on_result = raise_on_result

    @property
    def id(self) -> str:
        return self._task_id

    def result(self, *args, **kwargs) -> Any:
        if self._raise_on_result:
            raise self._raise_on_result
        return self._result_value


class MockTaskBatch:
    """Mock Braket task batch for testing."""

    def __init__(
        self,
        results_list: list[Any] | None = None,
        raise_on_results: Exception | None = None,
    ):
        self._results_list = results_list or []
        self._raise_on_results = raise_on_results

    def results(self, *args, **kwargs) -> list[Any]:
        if self._raise_on_results:
            raise self._raise_on_results
        return self._results_list


class MockResult:
    """Mock Braket result for testing."""

    def __init__(self, counts: dict[str, int] | None = None):
        self._counts = counts or {"00": 50, "11": 50}

    @property
    def measurement_counts(self) -> dict[str, int]:
        return self._counts


class MockDevice:
    """Mock device with configurable behavior for testing."""

    __module__ = "braket.aws.aws_device"

    def __init__(
        self,
        props: dict,
        name: str = "mock_device",
        run_returns: Any = None,
        run_raises: Exception | None = None,
        run_batch_returns: Any = None,
        run_batch_raises: Exception | None = None,
    ):
        self._name = name
        self._arn = f"arn:aws:braket:::device/qpu/mock/{name}"
        self.properties = MockProperties(props)
        self._run_returns = run_returns
        self._run_raises = run_raises
        self._run_batch_returns = run_batch_returns
        self._run_batch_raises = run_batch_raises
        self._run_calls: list[dict] = []
        self._run_batch_calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def arn(self) -> str:
        return self._arn

    @property
    def status(self) -> str:
        return "AVAILABLE"

    @property
    def provider_name(self) -> str:
        return "Amazon Braket"

    @property
    def type(self) -> str:
        return "QPU"

    def run(self, task_spec: Any, *args, **kwargs) -> Any:
        """Record call and return configured value or raise."""
        self._run_calls.append(
            {
                "task_spec": task_spec,
                "args": args,
                "kwargs": kwargs,
            }
        )
        if self._run_raises:
            raise self._run_raises
        if self._run_returns is not None:
            return self._run_returns
        # Default: return a mock task with mock result
        return MockTask(result_value=MockResult())

    def run_batch(self, task_specs: list[Any], *args, **kwargs) -> Any:
        """Record call and return configured value or raise."""
        self._run_batch_calls.append(
            {
                "task_specs": task_specs,
                "args": args,
                "kwargs": kwargs,
            }
        )
        if self._run_batch_raises:
            raise self._run_batch_raises
        if self._run_batch_returns is not None:
            return self._run_batch_returns
        # Default: return a mock batch
        return MockTaskBatch(results_list=[MockResult() for _ in task_specs])


class MockProgramSet:
    """
    Mock ProgramSet for testing submission handling.

    ProgramSet should be sent to device.run() as-is, not converted to list.
    """

    def __init__(self, circuits: list[Any], marker: str = "program_set_marker"):
        self._circuits = circuits
        self._marker = marker
        self._entries = [_MockProgramSetEntry(c) for c in circuits]

    @property
    def entries(self) -> list:
        return self._entries

    @property
    def total_executables(self) -> int:
        return len(self._circuits)

    @property
    def marker(self) -> str:
        """Marker to verify object identity in tests."""
        return self._marker

    def to_ir(self) -> str:
        return "mock_ir"


class _MockProgramSetEntry:
    """Mock entry in a ProgramSet."""

    def __init__(self, circuit: Any):
        self.circuit = circuit


@pytest.fixture
def build_properties():
    """Factory to build Braket device properties."""
    return build_device_properties


@pytest.fixture
def qpu_properties(build_properties):
    """Realistic QPU properties with calibration."""
    return build_properties(
        qubit_count=8,
        native_gates=["x", "rz", "cz"],
        connectivity_graph={
            "0": [1],
            "1": [0, 2],
            "2": [1, 3],
            "3": [2, 4],
            "4": [3, 5],
            "5": [4, 6],
            "6": [5, 7],
            "7": [6],
        },
        one_qubit_fidelities={
            0: [{"gateName": "x", "fidelity": 0.999}],
            1: [{"gateName": "x", "fidelity": 0.998}],
        },
        two_qubit_fidelities={
            "0-1": [{"gateName": "cz", "fidelity": 0.95}],
            "1-2": [{"gateName": "cz", "fidelity": 0.94}],
        },
    )


@pytest.fixture
def device_factory(build_properties):
    """Factory that builds a MockDevice with configurable properties."""

    def _factory(
        *,
        name: str = "mock_device",
        run_returns: Any = None,
        run_raises: Exception | None = None,
        run_batch_returns: Any = None,
        run_batch_raises: Exception | None = None,
        **kwargs,
    ) -> MockDevice:
        props = build_properties(**kwargs)
        return MockDevice(
            props,
            name=name,
            run_returns=run_returns,
            run_raises=run_raises,
            run_batch_returns=run_batch_returns,
            run_batch_raises=run_batch_raises,
        )

    return _factory


@pytest.fixture
def mock_device(qpu_properties):
    """Default MockDevice based on qpu_properties."""
    return MockDevice(qpu_properties)


@pytest.fixture
def mock_program_set(bell_circuit):
    """Mock ProgramSet with identifiable marker."""
    return MockProgramSet([bell_circuit, bell_circuit], marker="test_program_set")


@pytest.fixture
def mock_task():
    """Factory for creating mock tasks."""

    def _factory(
        counts: dict[str, int] | None = None,
        task_id: str = "mock-task-123",
        raise_on_result: Exception | None = None,
    ) -> MockTask:
        result = MockResult(counts) if counts else MockResult()
        return MockTask(
            result_value=result,
            task_id=task_id,
            raise_on_result=raise_on_result,
        )

    return _factory


@pytest.fixture
def mock_result():
    """Factory for creating mock results."""

    def _factory(counts: dict[str, int] | None = None) -> MockResult:
        return MockResult(counts)

    return _factory


@pytest.fixture
def mock_task_batch():
    """Factory for creating mock task batches."""

    def _factory(
        results_list: list[Any] | None = None,
        raise_on_results: Exception | None = None,
    ) -> MockTaskBatch:
        return MockTaskBatch(
            results_list=results_list,
            raise_on_results=raise_on_results,
        )

    return _factory
