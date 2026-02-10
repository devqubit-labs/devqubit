# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for CUDA-Q adapter tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cudaq
import pytest
from devqubit_engine.storage.factory import create_registry, create_store


# ruff: noqa: F821


# ---------------------------------------------------------------------------
# Real CUDA-Q kernels — sample-compatible (with measurements)
# ---------------------------------------------------------------------------


@cudaq.kernel
def bell():
    """2-qubit Bell state: H(0), CX(0,1), MZ."""
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def ghz3():
    """3-qubit GHZ state: H(0), CX(0,1), CX(1,2), MZ."""
    q = cudaq.qvector(3)
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    mz(q)


@cudaq.kernel
def single_qubit():
    """Single-qubit H + measure (Clifford-only)."""
    q = cudaq.qvector(1)
    h(q[0])
    mz(q)


@cudaq.kernel
def rx_param(theta: float):
    """Parameterized single-qubit Rx(θ) + measure."""
    q = cudaq.qvector(1)
    rx(theta, q[0])
    mz(q)


@cudaq.kernel
def non_clifford():
    """Non-Clifford circuit: H, T, measure."""
    q = cudaq.qvector(1)
    h(q[0])
    t(q[0])
    mz(q)


@cudaq.kernel
def parameterized_2p(theta: float, phi: float):
    """2-parameter kernel: Rx(θ), Ry(φ), measure."""
    q = cudaq.qvector(1)
    rx(theta, q[0])
    ry(phi, q[0])
    mz(q)


# ---------------------------------------------------------------------------
# Real CUDA-Q kernels — observe-compatible (NO measurements)
# ---------------------------------------------------------------------------


@cudaq.kernel
def bell_observe():
    """Bell state without measurement — suitable for cudaq.observe()."""
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def single_qubit_observe():
    """Single-qubit H without measurement — suitable for cudaq.observe()."""
    q = cudaq.qvector(1)
    h(q[0])


# ---------------------------------------------------------------------------
# Kernel fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_kernel():
    """2-qubit Bell state kernel (with measurements)."""
    return bell


@pytest.fixture
def ghz_kernel():
    """3-qubit GHZ kernel (with measurements)."""
    return ghz3


@pytest.fixture
def single_qubit_kernel():
    """Single-qubit Hadamard + measure."""
    return single_qubit


@pytest.fixture
def rx_kernel():
    """Parameterized Rx(θ) kernel."""
    return rx_param


@pytest.fixture
def non_clifford_kernel():
    """Non-Clifford circuit (T gate)."""
    return non_clifford


@pytest.fixture
def parameterized_kernel():
    """2-parameter kernel (Rx + Ry) for hash / summarizer tests."""
    return parameterized_2p


@pytest.fixture
def bell_observe_kernel():
    """Bell state without measurements — for observe() tests."""
    return bell_observe


@pytest.fixture
def single_qubit_observe_kernel():
    """Single H — no measurements — for observe() tests."""
    return single_qubit_observe


# ---------------------------------------------------------------------------
# Bare kernel (no to_json / no MLIR — plain object)
# ---------------------------------------------------------------------------


class _BareKernel:
    """Minimal object with a ``name`` attribute but no ``to_json()``."""

    name = "bare"


@pytest.fixture
def bare_kernel():
    """Object that looks like a kernel but has no serialisation support."""
    return _BareKernel()


# ---------------------------------------------------------------------------
# Hamiltonian fixtures (for observe)
# ---------------------------------------------------------------------------


@pytest.fixture
def z0_hamiltonian():
    """Pauli Z on qubit 0."""
    return cudaq.spin.z(0)


@pytest.fixture
def zz_hamiltonian():
    """Z⊗Z two-qubit Hamiltonian."""
    return cudaq.spin.z(0) * cudaq.spin.z(1)


# ---------------------------------------------------------------------------
# Result test doubles — typed, explicit (no MagicMock)
# ---------------------------------------------------------------------------


@dataclass
class SampleResultStub:
    """Typed stand-in for ``cudaq.SampleResult``.

    ``results.py`` detects sample results by checking
    ``"sampleresult" in type(result).__name__.lower()`` and the
    ``.items()`` / ``.most_probable`` / ``.get_total_shots()`` API.

    The class name intentionally contains "SampleResult" so that the
    type-name check works without monkey-patching.
    """

    _counts: dict[str, int] = field(default_factory=dict)

    # -- cudaq.SampleResult API surface used by results.py ----------------

    def items(self) -> Iterator[tuple[str, int]]:
        yield from self._counts.items()

    def most_probable(self) -> str:
        return max(self._counts, key=self._counts.__getitem__, default="")

    def get_total_shots(self) -> int:
        return sum(self._counts.values())

    @property
    def register_names(self) -> list[str]:
        return ["__global__"]

    def __iter__(self):
        return iter(self._counts)

    def __getitem__(self, key: str) -> int:
        return self._counts[key]

    def __len__(self) -> int:
        return len(self._counts)


@dataclass
class ObserveResultStub:
    """Typed stand-in for ``cudaq.ObserveResult``.

    Detected via ``"observeresult" in type().__name__.lower()`` and the
    ``.expectation()`` / ``.counts()`` interface.
    """

    _expectation: float = 0.0
    _counts: dict[str, int] | None = None

    def expectation(self) -> float:
        return self._expectation

    def counts(self) -> SampleResultStub | None:
        if self._counts is not None:
            return SampleResultStub(self._counts)
        return None


# ---------------------------------------------------------------------------
# Result fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_sample_result() -> SampleResultStub:
    """Bell-state sample result: 480×|00⟩ + 520×|11⟩ (1000 shots)."""
    return SampleResultStub({"00": 480, "11": 520})


@pytest.fixture
def uniform_sample_result() -> SampleResultStub:
    """Uniform 2-qubit distribution (1000 shots)."""
    return SampleResultStub({"00": 250, "01": 250, "10": 250, "11": 250})


@pytest.fixture
def make_sample_result():
    """Factory: ``make_sample_result({"01": 300, "10": 700})``."""

    def _make(counts: dict[str, int]) -> SampleResultStub:
        return SampleResultStub(counts)

    return _make


@pytest.fixture
def observe_result() -> ObserveResultStub:
    """Observe result with expectation -0.42."""
    return ObserveResultStub(_expectation=-0.42)


@pytest.fixture
def observe_result_with_counts() -> ObserveResultStub:
    """Observe result with expectation and shot counts."""
    return ObserveResultStub(_expectation=-0.42, _counts={"0": 710, "1": 290})


@pytest.fixture
def make_observe_result():
    """Factory: ``make_observe_result(-0.5)``."""

    def _make(
        expectation: float, counts: dict[str, int] | None = None
    ) -> ObserveResultStub:
        return ObserveResultStub(_expectation=expectation, _counts=counts)

    return _make


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
