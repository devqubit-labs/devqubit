# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Shared test fixtures for CUDA-Q adapter tests."""

from __future__ import annotations

import cudaq
import pytest
from devqubit_engine.storage.factory import create_registry, create_store


# ruff: noqa: F821


# ---------------------------------------------------------------------------
# Real CUDA-Q kernels
# ---------------------------------------------------------------------------


@cudaq.kernel
def _bell_kernel():
    """2-qubit Bell state: H(0), CX(0,1), MZ."""
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def _ghz3_kernel():
    """3-qubit GHZ state: H(0), CX(0,1), CX(1,2), MZ."""
    q = cudaq.qvector(3)
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    mz(q)


@cudaq.kernel
def _single_qubit_kernel():
    """Single-qubit H + measure (Clifford-only)."""
    q = cudaq.qvector(1)
    h(q[0])
    mz(q)


@cudaq.kernel
def _rx_kernel(theta: float):
    """Parameterized single-qubit Rx(θ) + measure."""
    q = cudaq.qvector(1)
    rx(theta, q[0])
    mz(q)


@cudaq.kernel
def _non_clifford_kernel():
    """Non-Clifford circuit: H, T, measure."""
    q = cudaq.qvector(1)
    h(q[0])
    t(q[0])
    mz(q)


# ---------------------------------------------------------------------------
# Kernel fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_kernel():
    """2-qubit Bell state kernel."""
    return _bell_kernel


@pytest.fixture
def ghz_kernel():
    """3-qubit GHZ kernel."""
    return _ghz3_kernel


@pytest.fixture
def single_qubit_kernel():
    """Single-qubit Hadamard + measure."""
    return _single_qubit_kernel


@pytest.fixture
def rx_kernel():
    """Parameterized Rx(θ) kernel."""
    return _rx_kernel


@pytest.fixture
def non_clifford_kernel():
    """Non-Clifford circuit (T gate)."""
    return _non_clifford_kernel


# ---------------------------------------------------------------------------
# Hamiltonian fixtures (for observe)
# ---------------------------------------------------------------------------


@pytest.fixture
def z0_hamiltonian():
    """Pauli Z on qubit 0."""
    return cudaq.spin.z(0)


@pytest.fixture
def zz_hamiltonian():
    """ZxZ two-qubit Hamiltonian."""
    return cudaq.spin.z(0) * cudaq.spin.z(1)


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
