# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit UEC snapshot types."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from devqubit_engine.uec.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.program import ProgramArtifact
from devqubit_engine.uec.resolver import (
    get_counts_from_envelope,
    load_envelope,
    resolve_envelope,
    synthesize_envelope,
)
from devqubit_engine.uec.result import (
    QuasiProbability,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.uec.types import (
    ArtifactRef,
    ProgramRole,
    ValidationResult,
)


class TestGateCalibration:
    """Tests for GateCalibration dataclass."""

    def test_qubits_tuple_to_list_conversion(self):
        """to_dict converts qubits tuple to list for JSON."""
        gc = GateCalibration(gate="cx", qubits=(0, 1), error=0.005)
        d = gc.to_dict()

        assert d["qubits"] == [0, 1]
        assert isinstance(d["qubits"], list)

    def test_round_trip_preserves_tuple(self):
        """from_dict restores qubits as tuple."""
        gc = GateCalibration(gate="cx", qubits=(0, 1), error=0.01, duration_ns=300.0)
        restored = GateCalibration.from_dict(gc.to_dict())

        assert restored.qubits == (0, 1)
        assert isinstance(restored.qubits, tuple)

    def test_is_two_qubit(self):
        """is_two_qubit correctly identifies multi-qubit gates."""
        single = GateCalibration(gate="x", qubits=(0,))
        two = GateCalibration(gate="cx", qubits=(0, 1))
        three = GateCalibration(gate="ccx", qubits=(0, 1, 2))

        assert not single.is_two_qubit
        assert two.is_two_qubit
        assert three.is_two_qubit  # >= 2 qubits


class TestDeviceCalibration:
    """Tests for DeviceCalibration dataclass."""

    def test_compute_medians(self):
        """compute_medians calculates correct median values."""
        cal = DeviceCalibration(
            qubits=[
                QubitCalibration(qubit=0, t1_us=100.0, t2_us=80.0, readout_error=0.01),
                QubitCalibration(qubit=1, t1_us=120.0, t2_us=90.0, readout_error=0.02),
                QubitCalibration(qubit=2, t1_us=110.0, t2_us=85.0, readout_error=0.015),
            ],
            gates=[
                GateCalibration(gate="cx", qubits=(0, 1), error=0.01),
                GateCalibration(gate="cx", qubits=(1, 2), error=0.02),
            ],
        )
        cal.compute_medians()

        assert cal.median_t1_us == 110.0
        assert cal.median_t2_us == 85.0
        assert cal.median_readout_error == 0.015
        assert cal.median_2q_error == 0.015

    def test_compute_medians_ignores_none(self):
        """compute_medians skips qubits with missing values."""
        cal = DeviceCalibration(
            qubits=[
                QubitCalibration(qubit=0, t1_us=100.0),
                QubitCalibration(qubit=1, t1_us=None),
                QubitCalibration(qubit=2, t1_us=120.0),
            ],
        )
        cal.compute_medians()

        assert cal.median_t1_us == 110.0

    def test_to_dict_auto_computes_medians(self):
        """to_dict triggers median computation if needed."""
        cal = DeviceCalibration(
            qubits=[QubitCalibration(qubit=0, t1_us=100.0)],
        )
        d = cal.to_dict()

        assert "median_t1_us" in d
        assert d["median_t1_us"] == 100.0

    def test_calibration_factory_fixture(self, calibration_factory):
        """Fixture creates valid calibration with computed medians."""
        cal = calibration_factory(num_qubits=5)

        assert len(cal.qubits) == 5
        assert len(cal.gates) == 4
        assert cal.median_t1_us is not None


class TestDeviceSnapshot:
    """Tests for DeviceSnapshot dataclass."""

    def test_connectivity_serialization(self):
        """Connectivity tuples convert to lists and back."""
        snap = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="test",
            backend_type="hardware",
            provider="ibm_quantum",
            connectivity=[(0, 1), (1, 2), (2, 3)],
        )
        d = snap.to_dict()
        restored = DeviceSnapshot.from_dict(d)

        assert d["connectivity"] == [[0, 1], [1, 2], [2, 3]]
        assert restored.connectivity == [(0, 1), (1, 2), (2, 3)]

    def test_get_calibration_summary(self, calibration_factory):
        """get_calibration_summary returns compact metrics."""
        cal = calibration_factory(num_qubits=3)
        snap = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="test",
            backend_type="hardware",
            provider="ibm_quantum",
            calibration=cal,
        )
        summary = snap.get_calibration_summary()

        assert "median_t1_us" in summary
        assert "median_2q_error" in summary

    def test_get_calibration_summary_none_without_calibration(self):
        """get_calibration_summary returns None if no calibration."""
        snap = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="test",
            backend_type="simulator",
            provider="local",
        )
        assert snap.get_calibration_summary() is None


class TestProgramArtifact:
    """Tests for ProgramArtifact dataclass."""

    def test_round_trip(self):
        """ProgramArtifact survives serialization."""
        ref = ArtifactRef(
            kind="qiskit.qpy.circuits",
            digest="sha256:" + "a" * 64,
            media_type="application/vnd.qiskit.qpy",
            role="program",
        )
        artifact = ProgramArtifact(
            ref=ref,
            role=ProgramRole.LOGICAL,
            format="qpy",
            name="bell_circuit",
            index=0,
        )
        restored = ProgramArtifact.from_dict(artifact.to_dict())

        assert restored.role == ProgramRole.LOGICAL
        assert restored.format == "qpy"
        assert restored.name == "bell_circuit"


class TestResultItem:
    """Tests for per-circuit result items."""

    def test_bell_state_counts_item(self, bell_state_counts, qiskit_counts_format):
        """Create result item from Bell state measurement."""
        item = ResultItem.from_counts(
            item_index=0,
            counts=bell_state_counts,
            shots=1000,
            format_info=qiskit_counts_format,
        )

        assert item.success is True
        assert item.item_index == 0
        assert item.counts["counts"]["00"] == 500
        assert item.counts["shots"] == 1000
        assert item.counts["format"]["source_sdk"] == "qiskit"

    def test_failed_circuit_in_batch(self):
        """Individual circuit failure in a batch."""
        item = ResultItem(
            item_index=2,
            success=False,
            error_message="Circuit depth 150 exceeds backend limit 100",
        )

        assert item.success is False
        assert "depth" in item.error_message.lower()


class TestQuasiProbability:
    """Tests for IBM Runtime quasi-distributions."""

    def test_from_integer_keys(self):
        """Convert integer keys to bitstrings (common IBM format)."""
        qp = QuasiProbability.from_quasi_dist({0: 0.48, 3: 0.52}, num_clbits=2)

        assert "00" in qp.distribution
        assert "11" in qp.distribution
        assert abs(qp.sum_probs - 1.0) < 0.01

    def test_negative_probabilities_from_mitigation(self):
        """Error mitigation can produce negative quasi-probabilities."""
        qp = QuasiProbability(
            distribution={"00": 0.52, "01": -0.02, "10": -0.01, "11": 0.51},
        )

        assert qp.distribution["01"] < 0
        assert qp.distribution["10"] < 0


class TestResultError:
    """Tests for structured error capture."""

    def test_from_backend_timeout(self):
        """Capture timeout exception with retryable flag."""
        try:
            raise TimeoutError("Backend did not respond within 300s")
        except TimeoutError as e:
            error = ResultError.from_exception(e)

        assert error.type == "TimeoutError"
        assert "300s" in error.message
        assert error.retryable is True
        assert len(error.stack_hash) == 16

    def test_from_validation_error(self):
        """Non-retryable validation error."""
        try:
            raise ValueError("Invalid parameter: shots must be positive")
        except ValueError as e:
            error = ResultError.from_exception(e)

        assert error.type == "ValueError"
        assert error.retryable is False


class TestResultSnapshot:
    """Tests for complete result snapshots."""

    def test_successful_bell_state_execution(
        self, bell_state_counts, qiskit_counts_format
    ):
        """Happy path: successful Bell state measurement."""
        item = ResultItem.from_counts(0, bell_state_counts, 1000, qiskit_counts_format)
        snap = ResultSnapshot.create_success(items=[item])

        assert snap.success is True
        assert snap.status == "completed"
        assert len(snap.items) == 1
        assert snap.error is None

    def test_failed_execution_preserves_exception_info(self):
        """Failure path: exception captured with full context."""
        try:
            raise RuntimeError("Job cancelled: insufficient credits")
        except RuntimeError as e:
            snap = ResultSnapshot.create_failed(exception=e)

        assert snap.success is False
        assert snap.status == "failed"
        assert snap.error.type == "RuntimeError"
        assert "credits" in snap.error.message

    def test_partial_batch_execution(self, bell_state_counts, qiskit_counts_format):
        """Partial success: some circuits in batch failed."""
        items = [
            ResultItem.from_counts(0, bell_state_counts, 1000, qiskit_counts_format),
            ResultItem(item_index=1, success=False, error_message="Circuit 1 too deep"),
            ResultItem.from_counts(2, {"0": 1000}, 1000, qiskit_counts_format),
        ]
        snap = ResultSnapshot.create_partial(items=items)

        assert snap.success is False
        assert snap.status == "partial"
        assert snap.items[0].success is True
        assert snap.items[1].success is False
        assert snap.items[2].success is True

    def test_invalid_status_rejected(self):
        """Invalid status values are rejected at construction."""
        with pytest.raises(ValueError, match="status must be one of"):
            ResultSnapshot(success=True, status="running")

    def test_round_trip_serialization(self, bell_state_counts, qiskit_counts_format):
        """Result snapshot survives JSON round-trip."""
        item = ResultItem.from_counts(0, bell_state_counts, 1000, qiskit_counts_format)
        original = ResultSnapshot.create_success(
            items=[item],
            metadata={"backend": "ibm_brisbane", "job_id": "abc123"},
        )

        restored = ResultSnapshot.from_dict(original.to_dict())

        assert restored.success == original.success
        assert restored.status == original.status
        assert restored.items[0].counts["counts"]["00"] == 500
        assert restored.metadata["backend"] == "ibm_brisbane"


class TestExecutionEnvelope:
    """Tests for complete execution records."""

    def test_create_minimal_envelope(self, minimal_producer):
        """Minimal valid envelope with required fields only."""
        result = ResultSnapshot(success=True, status="completed", items=[])
        env = ExecutionEnvelope.create(producer=minimal_producer, result=result)

        assert len(env.envelope_id) == 26
        assert env.created_at is not None
        assert env.schema_version == "devqubit.envelope/1.0"
        assert env.producer.adapter == "devqubit-test"

    def test_create_generates_failed_result_if_none(self, minimal_producer):
        """Factory creates failed result if None provided."""
        env = ExecutionEnvelope.create(producer=minimal_producer, result=None)

        assert env.result.success is False
        assert env.result.status == "failed"

    def test_validate_warns_on_missing_snapshots(self, minimal_producer):
        """Validation warns about missing optional snapshots."""
        env = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=ResultSnapshot(success=True, status="completed", items=[]),
        )
        warnings = env.validate()

        assert "Missing device snapshot" in warnings
        assert "Missing program snapshot" in warnings
        assert "Missing execution snapshot" in warnings

    def test_validate_warns_on_failed_without_error(self, minimal_producer):
        """Validation warns when failed result lacks error details."""
        env = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=ResultSnapshot(success=False, status="failed", items=[]),
        )
        warnings = env.validate()

        assert "Failed result missing error details" in warnings

    def test_validate_schema_returns_validation_result(self, minimal_producer):
        """validate_schema returns ValidationResult object."""
        env = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=ResultSnapshot(success=True, status="completed", items=[]),
        )

        mock_module = type(
            "MockModule",
            (),
            {"validate_envelope": lambda *a, **k: []},
        )()

        with patch.dict(
            "sys.modules",
            {"devqubit_engine.core.schema.validation": mock_module},
        ):
            result = env.validate_schema()

            assert isinstance(result, ValidationResult)
            assert result.valid
            assert result.errors == []


class TestArtifactRef:
    """Tests for content-addressed artifact references."""

    def test_invalid_digest_rejected(self):
        """Malformed digest is rejected."""
        with pytest.raises(ValueError, match="Invalid digest format"):
            ArtifactRef(
                kind="test",
                digest="md5:abc123",
                media_type="text/plain",
                role="test",
            )

    def test_short_kind_rejected(self):
        """Kind must be at least 3 characters."""
        with pytest.raises(ValueError, match="at least 3 characters"):
            ArtifactRef(
                kind="ab",
                digest="sha256:" + "a" * 64,
                media_type="text/plain",
                role="test",
            )


class TestValidationResult:
    """Tests for validation result handling."""

    def test_valid_result_is_truthy(self):
        """Valid result evaluates to True in boolean context."""
        result = ValidationResult(valid=True)
        assert result
        assert result.ok
        assert result.error_count == 0

    def test_invalid_result_is_falsy(self):
        """Invalid result evaluates to False."""
        result = ValidationResult(valid=False, errors=["Missing field", "Invalid type"])
        assert not result
        assert not result.ok
        assert result.error_count == 2

    def test_warnings_preserved(self):
        """Warnings are preserved in result."""
        result = ValidationResult(
            valid=True,
            warnings=["Schema validation module not available"],
        )
        assert result.valid
        assert "not available" in result.warnings[0]


class TestResolver:
    """Tests for UEC resolver (UEC-first + Run fallback)."""

    def test_resolve_synthesizes_when_no_envelope(self, store, run_factory):
        """resolve_envelope synthesizes envelope when no artifact exists."""
        run = run_factory(run_id="NO_ENVELOPE", adapter="manual")

        envelope = resolve_envelope(run, store)

        assert envelope is not None
        assert envelope.envelope_id is not None
        assert envelope.metadata.get("synthesized_from_run") is True
        assert envelope.producer.adapter == "manual"

    def test_load_envelope_exact_kind_match(self, store, run_factory):
        """load_envelope uses exact kind match, ignores validation_error."""

        # Create envelope artifact
        envelope_data = {
            "schema_version": "devqubit.envelope/1.0",
            "envelope_id": "TEST_ENVELOPE",
            "created_at": "2024-01-01T00:00:00Z",
            "producer": {"name": "devqubit", "adapter": "test"},
            "result": {"success": True, "status": "completed", "items": []},
        }
        data = json.dumps(envelope_data).encode()
        digest = store.put_bytes(data)

        valid_artifact = ArtifactRef(
            kind="devqubit.envelope.json",  # Exact match
            digest=digest,
            media_type="application/json",
            role="envelope",
        )
        # This should NOT match - it's validation_error
        error_artifact = ArtifactRef(
            kind="devqubit.envelope.validation_error.json",
            digest=digest,
            media_type="application/json",
            role="config",  # Different role!
        )

        run = run_factory(artifacts=[error_artifact, valid_artifact])
        envelope = load_envelope(run, store)

        assert envelope is not None
        assert envelope.envelope_id == "TEST_ENVELOPE"

    def test_build_envelope_captures_counts(
        self, store, run_factory, counts_artifact_factory
    ):
        """synthesize_envelope captures counts from Run artifact."""

        counts = {"00": 500, "11": 500}
        artifact = counts_artifact_factory(counts)
        run = run_factory(run_id="WITH_COUNTS", artifacts=[artifact])

        envelope = synthesize_envelope(run, store)
        extracted = get_counts_from_envelope(envelope)

        assert extracted is not None
        assert extracted["00"] == 500
        assert extracted["11"] == 500
        assert envelope.metadata.get("counts_format_assumed") is True
