# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for result processing and the failure-envelope path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from devqubit_engine.tracking.run import track
from devqubit_engine.uec.models.result import ResultType
from devqubit_qiskit.adapter import TrackedBackend
from devqubit_qiskit.results import (
    detect_result_type,
    extract_expectation_values,
    extract_quasi_distributions,
    extract_result_metadata,
    normalize_result_counts,
)
from qiskit import QuantumCircuit


# =============================================================================
# Helpers
# =============================================================================


def _run_real(aer_simulator, circuits, shots=100):
    """Execute circuits on real Aer and return the native Qiskit Result."""
    return aer_simulator.run(circuits, shots=shots).result()


def _load_envelope(run_id, store, registry):
    loaded = registry.load(run_id)
    env_artifacts = [a for a in loaded.artifacts if a.kind == "devqubit.envelope.json"]
    if env_artifacts:
        raw = store.get_bytes(env_artifacts[0].digest)
        return loaded, json.loads(raw.decode("utf-8"))
    return loaded, None


# =============================================================================
# detect_result_type
# =============================================================================


class TestDetectResultType:

    def test_counts_from_real_execution(self, aer_simulator, bell_circuit):
        """Real AerSimulator result is detected as COUNTS."""
        result = _run_real(aer_simulator, bell_circuit)
        assert detect_result_type(result) == ResultType.COUNTS

    def test_statevector_from_real_execution(self):
        """Real statevector simulation is detected as STATEVECTOR."""
        from qiskit_aer import AerSimulator

        sv_sim = AerSimulator(method="statevector")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_statevector()

        result = sv_sim.run(qc).result()
        assert detect_result_type(result) == ResultType.STATEVECTOR

    def test_quasi_dist_attribute(self):
        """Object with .quasi_dists → QUASI_DIST.

        Real quasi-dists come from Runtime SamplerV2 (separate package).
        """
        result = SimpleNamespace(quasi_dists=[{0: 0.5, 3: 0.5}])
        assert detect_result_type(result) == ResultType.QUASI_DIST

    def test_estimator_values_attribute(self):
        """Object with .values → EXPECTATION.

        Real expectation values come from Runtime EstimatorV2 (separate package).
        """
        result = SimpleNamespace(values=[0.707, -0.3])
        assert detect_result_type(result) == ResultType.EXPECTATION

    def test_none_returns_other(self):
        assert detect_result_type(None) == ResultType.OTHER

    def test_quasi_dist_priority_over_counts(self):
        """When both quasi_dists and get_counts exist, quasi_dist wins."""
        result = SimpleNamespace(
            quasi_dists=[{0: 0.5, 1: 0.5}],
            get_counts=lambda: {"0": 50, "1": 50},
        )
        assert detect_result_type(result) == ResultType.QUASI_DIST


# =============================================================================
# normalize_result_counts
# =============================================================================


class TestNormalizeResultCounts:

    def test_single_circuit(self, aer_simulator, bell_circuit):
        """Single-circuit real result produces 1 experiment entry."""
        result = _run_real(aer_simulator, bell_circuit, shots=500)
        output = normalize_result_counts(result)

        assert len(output["experiments"]) == 1
        exp = output["experiments"][0]
        assert exp["shots"] == 500
        assert isinstance(exp["counts"], dict)
        assert sum(exp["counts"].values()) == 500

    def test_multi_circuit_batch(self, aer_simulator, batch_circuits):
        """Batch of 3 circuits produces 3 experiment entries."""
        result = _run_real(aer_simulator, batch_circuits, shots=200)
        output = normalize_result_counts(result)

        assert len(output["experiments"]) == 3
        for i, exp in enumerate(output["experiments"]):
            assert exp["index"] == i
            assert exp["shots"] == 200

    def test_none_result_returns_empty(self):
        output = normalize_result_counts(None)
        assert output == {"experiments": []}

    def test_zero_length_results_list(self):
        """result.results=[] must not fall through to single-experiment path.

        This guards the commit-1 fix: ``if not num_experiments`` treated
        0 as falsy, falling into the single-experiment fallback. The fix
        changed it to ``if num_experiments is None``.
        """
        result = SimpleNamespace(
            results=[],
            get_counts=lambda: {"should_not_appear": 999},
        )
        output = normalize_result_counts(result)
        assert output["experiments"] == []


class TestExtractResultMetadata:

    def test_real_aer_result(self, aer_simulator, bell_circuit):
        """Metadata from real Aer result contains expected fields."""
        result = _run_real(aer_simulator, bell_circuit)
        meta = extract_result_metadata(result)

        assert meta["success"] is True
        assert isinstance(meta.get("time_taken"), float)

    def test_none_returns_empty(self):
        assert extract_result_metadata(None) == {}


class TestExtractQuasiDistributions:

    def test_int_keys_to_bitstrings(self):
        """Integer-keyed dict is converted to binary-string keys."""
        result = SimpleNamespace(quasi_dists=[{0: 0.5, 3: 0.5}])
        dists = extract_quasi_distributions(result)

        assert dists is not None
        assert len(dists) == 1
        assert "00" in dists[0] and "11" in dists[0]

    def test_zero_only_key(self):
        """{0: 1.0} must produce at least 1-bit key '0'."""
        result = SimpleNamespace(quasi_dists=[{0: 1.0}])
        dists = extract_quasi_distributions(result)

        assert dists is not None
        assert "0" in dists[0]

    def test_negative_quasi_probabilities(self):
        """Error-mitigated results can have negative values."""
        result = SimpleNamespace(quasi_dists=[{0: 0.6, 1: -0.1, 2: 0.3, 3: 0.2}])
        dists = extract_quasi_distributions(result)

        assert any(v < 0 for v in dists[0].values())

    def test_multiple_distributions(self):
        """Multi-circuit results: one distribution per circuit."""
        result = SimpleNamespace(
            quasi_dists=[
                {0: 0.5, 1: 0.5},
                {0: 0.3, 3: 0.7},
            ]
        )
        dists = extract_quasi_distributions(result)
        assert len(dists) == 2

    def test_none_and_missing_attribute(self):
        assert extract_quasi_distributions(None) is None
        assert extract_quasi_distributions(SimpleNamespace()) is None


# =============================================================================
# extract_expectation_values
# =============================================================================


class TestExtractExpectationValues:

    def test_with_std_errors(self):
        """Values and standard errors are paired correctly."""
        result = SimpleNamespace(
            values=[0.5, -0.3],
            metadata=[{"std_error": [0.01, 0.02]}],
        )
        vals = extract_expectation_values(result)

        assert vals == [(0.5, 0.01), (-0.3, 0.02)]

    def test_without_metadata(self):
        result = SimpleNamespace(values=[1.0], metadata=[])
        vals = extract_expectation_values(result)

        assert vals == [(1.0, None)]

    def test_none_returns_none(self):
        assert extract_expectation_values(None) is None


# =============================================================================
# E2E: job.result() failure → failure envelope
# =============================================================================


class TestFailureEnvelope:

    def test_failure_creates_envelope_with_error_details(
        self,
        bell_circuit,
        aer_simulator,
        store,
        registry,
    ):
        """job.result() raising → envelope with status=failed and error info."""
        with track(project="test", store=store, registry=registry) as run:
            backend = TrackedBackend(backend=aer_simulator, tracker=run)
            job = backend.run(bell_circuit, shots=100)

            # Simulate a hardware timeout AFTER real submission succeeded
            job.job.result = MagicMock(
                side_effect=RuntimeError("Job timed out after 300s")
            )

            with pytest.raises(RuntimeError, match="timed out"):
                job.result()

        _, envelope = _load_envelope(run.run_id, store, registry)

        assert envelope is not None
        assert envelope["result"]["status"] == "failed"
        assert envelope["result"]["success"] is False
        assert envelope["result"]["error"]["type"] == "RuntimeError"
        assert "timed out" in envelope["result"]["error"]["message"]

        # The rest of the envelope must still be fully populated
        assert envelope["device"]["backend_type"] == "simulator"
        assert envelope["program"]["num_circuits"] == 1
        assert envelope["execution"]["submitted_at"] is not None

    def test_failure_then_retry_both_tracked(
        self,
        aer_simulator,
        store,
        registry,
    ):
        """Real workflow: first call fails, user retries, both get envelopes."""
        qc = QuantumCircuit(2, name="retry_test")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with track(project="test", store=store, registry=registry) as run:
            backend = TrackedBackend(
                backend=aer_simulator,
                tracker=run,
                log_every_n=-1,
            )

            # First attempt: fails
            job1 = backend.run(qc, shots=100)
            job1.job.result = MagicMock(side_effect=RuntimeError("Calibration error"))
            with pytest.raises(RuntimeError):
                job1.result()

            # Second attempt: succeeds (fully real execution)
            job2 = backend.run(qc, shots=100)
            counts = job2.result().get_counts()
            assert sum(counts.values()) == 100

        loaded = registry.load(run.run_id)
        envelope_count = sum(
            1 for a in loaded.artifacts if a.kind == "devqubit.envelope.json"
        )
        assert envelope_count == 2
