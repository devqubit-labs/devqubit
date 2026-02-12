# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for Qiskit Runtime adapter tests.

Uses real Qiskit fake providers wherever possible to minimize mocking.
Only mocks job submission/results since those require network access.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_qiskit_runtime.pubs import extract_circuit_from_pub, iter_pubs
from qiskit import QuantumCircuit


def pytest_configure(config):
    """Register custom markers and warning filters."""
    warnings.filterwarnings(
        "ignore",
        message=r"devqubit:.*ISA-compatible.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"devqubit:.*transpilation_mode.*manual.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"devqubit:.*transpilation option.*backend.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"devqubit:.*unknown.*transpilation.*",
        category=UserWarning,
    )


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
def tracking_root(tmp_path):
    """Create temporary tracking directory."""
    return tmp_path / ".devqubit"


@pytest.fixture
def store(tracking_root):
    """Create a temporary store."""
    return create_store(f"file://{tracking_root}/objects")


@pytest.fixture
def registry(tracking_root):
    """Create a temporary registry."""
    return create_registry(f"file://{tracking_root}")


# =============================================================================
# Circuit Fixtures
# =============================================================================


@pytest.fixture
def bell_circuit():
    """Create a Bell state circuit."""
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def ghz_circuit():
    """Create a 3-qubit GHZ circuit."""
    qc = QuantumCircuit(3, name="ghz")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


@pytest.fixture
def simple_circuit():
    """Create a simple single-qubit circuit."""
    qc = QuantumCircuit(1, name="simple")
    qc.h(0)
    qc.measure_all()
    return qc


@pytest.fixture
def isa_circuit():
    """Create an ISA-compatible circuit (rz, sx, cx only)."""
    qc = QuantumCircuit(2, name="isa")
    qc.rz(0.5, 0)
    qc.sx(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def non_isa_circuit():
    """Create a non-ISA circuit (contains H gate)."""
    qc = QuantumCircuit(2, name="non_isa")
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized circuit."""
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    phi = Parameter("phi")

    qc = QuantumCircuit(2, name="parameterized")
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# =============================================================================
# Backend Fixtures
# =============================================================================


def _get_fake_backend():
    """Try to get a fake backend from qiskit_ibm_runtime."""
    for backend_cls in ["FakeManilaV2", "FakeSherbrooke", "FakeKyoto"]:
        try:
            from qiskit_ibm_runtime import fake_provider

            cls = getattr(fake_provider, backend_cls, None)
            if cls:
                return cls()
        except Exception:
            continue
    return None


@pytest.fixture
def fake_backend():
    """Get a real fake backend from qiskit_ibm_runtime."""
    backend = _get_fake_backend()
    if backend is None:
        pytest.skip("No fake backend available from qiskit_ibm_runtime")
    return backend


@pytest.fixture
def real_target(fake_backend):
    """Get real Target from fake backend."""
    return fake_backend.target


@pytest.fixture
def real_coupling_map(fake_backend):
    """Get real CouplingMap from fake backend."""
    return fake_backend.coupling_map


# =============================================================================
# Mock Result Classes
# =============================================================================


class MockPrimitiveResult:
    """Mock PrimitiveResult."""

    def __init__(self, pub_results: list):
        self._pub_results = pub_results

    def __iter__(self):
        return iter(self._pub_results)

    def __len__(self):
        return len(self._pub_results)

    def __getitem__(self, idx):
        return self._pub_results[idx]


class MockSamplerPubResult:
    """Mock Sampler PubResult with realistic BitArray-like data."""

    def __init__(self, counts: dict, shots: int | None = None):
        self._counts = counts
        self._shots = shots if shots is not None else sum(counts.values())
        self.data = self._create_databin()
        self.metadata = {"shots": self._shots}

    def _create_databin(self):
        class BitArray:
            def __init__(self, counts):
                self._counts = counts

            def get_counts(self):
                return self._counts

            def get_bitstrings(self):
                return [bs for bs, c in self._counts.items() for _ in range(c)]

        class DataBin:
            def __init__(self, counts):
                self.meas = BitArray(counts)

        return DataBin(self._counts)

    def join_data(self):
        """Return joined data for multi-register support."""
        return self.data.meas


class MockEstimatorPubResult:
    """Mock Estimator PubResult with expectation values."""

    def __init__(self, evs, stds=None):
        import numpy as np

        self._evs = np.array(evs) if not isinstance(evs, np.ndarray) else evs
        self._stds = np.array(stds) if stds is not None else np.zeros_like(self._evs)

        class DataBin:
            def __init__(self, evs, stds):
                self.evs = evs
                self.stds = stds

        self.data = DataBin(self._evs, self._stds)
        self.metadata = {}


class MockRuntimeJob:
    """Mock Runtime job."""

    def __init__(self, result: MockPrimitiveResult, job_id: str = None):
        self._result = result
        self._job_id = job_id or f"mock-job-{id(self)}"

    def result(self):
        return self._result

    def job_id(self):
        return self._job_id

    def status(self):
        return "DONE"


# =============================================================================
# Fake Primitive Classes
# =============================================================================


class FakeSamplerV2:
    """Fake SamplerV2 using real backend but mocking execution."""

    __module__ = "qiskit_ibm_runtime.sampler"

    def __init__(self, backend, shots: int = 1024):
        self._backend = backend
        self._mode = backend
        self._shots = shots
        self.options = type("Options", (), {"default_shots": shots})()
        self.session = None

    @property
    def mode(self):
        return self._mode

    @property
    def backend(self):
        return self._backend

    def run(self, pubs, **kwargs):
        pubs_list = iter_pubs(pubs)
        global_shots = kwargs.get("shots", self._shots)

        pub_results = []
        for pub in pubs_list:
            circuit = extract_circuit_from_pub(pub)
            num_clbits = 2
            if circuit is not None:
                num_clbits = circuit.num_clbits or circuit.num_qubits

            pub_shots = global_shots
            if isinstance(pub, (tuple, list)) and len(pub) >= 3:
                per_pub_shots = pub[2]
                if isinstance(per_pub_shots, int) and per_pub_shots > 0:
                    pub_shots = per_pub_shots

            counts = {
                "0" * num_clbits: pub_shots // 2,
                "1" * num_clbits: pub_shots - pub_shots // 2,
            }
            pub_results.append(MockSamplerPubResult(counts, shots=pub_shots))

        return MockRuntimeJob(MockPrimitiveResult(pub_results))


class FakeEstimatorV2:
    """Fake EstimatorV2 using real backend but mocking execution."""

    __module__ = "qiskit_ibm_runtime.estimator"

    def __init__(self, backend, precision: float = 0.01, seed: int = 42):
        self._backend = backend
        self._mode = backend
        self._precision = precision
        self._seed = seed
        self.options = type("Options", (), {"default_precision": precision})()
        self.session = None

    @property
    def mode(self):
        return self._mode

    @property
    def backend(self):
        return self._backend

    def run(self, pubs, **kwargs):
        import numpy as np

        rng = np.random.default_rng(self._seed)
        pubs_list = iter_pubs(pubs)
        pub_results = []

        for i, pub in enumerate(pubs_list):
            observables = getattr(pub, "observables", None)
            if observables is None and isinstance(pub, (tuple, list)) and len(pub) >= 2:
                observables = pub[1]

            n_obs = 1
            if observables is not None and hasattr(observables, "__len__"):
                n_obs = len(observables)

            evs = np.linspace(-0.5, 0.5, n_obs) + rng.uniform(-0.1, 0.1, n_obs)
            stds = np.full(n_obs, self._precision) + rng.uniform(0, 0.005, n_obs)
            pub_results.append(MockEstimatorPubResult(evs, stds))

        return MockRuntimeJob(MockPrimitiveResult(pub_results))


# =============================================================================
# Primitive Fixtures
# =============================================================================


@pytest.fixture
def fake_sampler(fake_backend):
    """Create FakeSamplerV2 with real fake backend."""
    return FakeSamplerV2(fake_backend)


@pytest.fixture
def fake_estimator(fake_backend):
    """Create FakeEstimatorV2 with real fake backend."""
    return FakeEstimatorV2(fake_backend)


@pytest.fixture
def sampler_pub(bell_circuit):
    """Create a SamplerPub tuple."""
    return (bell_circuit,)


@pytest.fixture
def estimator_pub(bell_circuit):
    """Create EstimatorPub tuple with real SparsePauliOp."""
    try:
        from qiskit.quantum_info import SparsePauliOp

        obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
        return (bell_circuit, obs)
    except ImportError:
        return (bell_circuit, "ZZ")


@pytest.fixture
def mock_session():
    """Create a session-like object for testing."""

    class Session:
        session_id = "test-session-12345"
        max_time = 7200

    return Session()


# =============================================================================
# Results & Pubs Fixtures
# =============================================================================


@pytest.fixture
def make_bitarray():
    """Factory for BitArray-like objects."""

    def _factory(counts: dict):
        class BitArray:
            def __init__(self, c):
                self._counts = c

            def get_counts(self):
                return self._counts

            def get_bitstrings(self):
                return [bs for bs, n in self._counts.items() for _ in range(n)]

        return BitArray(counts)

    return _factory


@pytest.fixture
def make_sampler_pub_result(make_bitarray):
    """Factory for mock Sampler PubResult."""

    def _factory(counts: dict, shots: int | None = None):
        ba = make_bitarray(counts)
        total = shots if shots is not None else sum(counts.values())
        data = SimpleNamespace(meas=ba)
        return SimpleNamespace(
            data=data,
            metadata={"shots": total},
            join_data=lambda: ba,
        )

    return _factory


@pytest.fixture
def make_estimator_pub_result():
    """Factory for mock Estimator PubResult."""

    def _factory(evs, stds=None):
        evs_arr = np.asarray(evs)
        stds_arr = np.asarray(stds) if stds is not None else np.zeros_like(evs_arr)
        data = SimpleNamespace(evs=evs_arr, stds=stds_arr)
        return SimpleNamespace(data=data, metadata={})

    return _factory


@pytest.fixture
def make_primitive_result():
    """Factory wrapping pub results into an iterable PrimitiveResult."""

    def _factory(pub_results: list):
        class Result:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        return Result(pub_results)

    return _factory
