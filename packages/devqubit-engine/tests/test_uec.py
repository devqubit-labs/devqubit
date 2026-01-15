# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for UEC (Uniform Execution Contract) module."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.uec.api.extract import (
    get_counts_from_envelope,
    resolve_counts,
    resolve_device_snapshot,
)
from devqubit_engine.uec.api.resolve import load_envelope, resolve_envelope
from devqubit_engine.uec.api.synthesize import synthesize_envelope
from devqubit_engine.uec.errors import MissingEnvelopeError
from devqubit_engine.uec.models.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.result import ResultError, ResultItem, ResultSnapshot


class TestEnvelopeResolution:
    """Tests for UEC-first envelope resolution strategy."""

    def test_load_existing_envelope(self, store, run_factory):
        """Load envelope from existing artifact."""
        envelope_data = {
            "schema_version": "devqubit.envelope/1.0",
            "envelope_id": "ENV_EXISTING",
            "created_at": "2024-01-01T00:00:00Z",
            "producer": {"name": "devqubit", "adapter": "qiskit"},
            "result": {"success": True, "status": "completed", "items": []},
        }
        digest = store.put_bytes(json.dumps(envelope_data).encode())

        artifact = ArtifactRef(
            kind="devqubit.envelope.json",
            digest=digest,
            media_type="application/json",
            role="envelope",
        )
        run = run_factory(run_id="WITH_ENVELOPE", artifacts=[artifact])

        envelope = load_envelope(run, store)

        assert envelope is not None
        assert envelope.envelope_id == "ENV_EXISTING"

    def test_synthesize_for_manual_run(self, store, run_factory):
        """Synthesize envelope when manual run has no envelope artifact."""
        run = run_factory(run_id="MANUAL_RUN", adapter="manual")

        envelope = resolve_envelope(run, store)

        assert envelope is not None
        assert envelope.metadata.get("synthesized_from_run") is True
        assert envelope.producer.adapter == "manual"

    def test_adapter_run_without_envelope_raises(self, store, run_factory):
        """Adapter run without envelope raises MissingEnvelopeError."""
        run = run_factory(run_id="ADAPTER_RUN", adapter="qiskit")

        with pytest.raises(MissingEnvelopeError):
            resolve_envelope(run, store)

    def test_synthesize_captures_counts(
        self, store, run_factory, counts_artifact_factory
    ):
        """Synthesized envelope captures counts from run artifacts."""
        counts = {"00": 500, "11": 500}
        artifact = counts_artifact_factory(counts)
        run = run_factory(
            run_id="WITH_COUNTS",
            adapter="manual",
            artifacts=[artifact],
        )

        envelope = synthesize_envelope(run, store)
        extracted = get_counts_from_envelope(envelope)

        assert extracted == {"00": 500, "11": 500}
        assert envelope.metadata.get("counts_format_assumed") is True


class TestDataExtraction:
    """Tests for extracting data from envelopes."""

    def test_resolve_counts_from_envelope(self, store, run_factory, minimal_producer):
        """resolve_counts extracts counts from envelope result."""
        # Create envelope with counts
        result_item = ResultItem(
            item_index=0,
            success=True,
            counts={
                "counts": {"00": 480, "11": 520},
                "shots": 1000,
                "format": {"source_sdk": "qiskit", "bit_order": "cbit0_right"},
            },
        )
        result = ResultSnapshot.create_success(items=[result_item])
        envelope = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=result,
        )

        # Store envelope as artifact
        digest = store.put_bytes(json.dumps(envelope.to_dict()).encode())
        artifact = ArtifactRef(
            kind="devqubit.envelope.json",
            digest=digest,
            media_type="application/json",
            role="envelope",
        )
        run = run_factory(run_id="ENV_COUNTS", artifacts=[artifact])

        counts = resolve_counts(run, store)

        assert counts == {"00": 480, "11": 520}

    def test_resolve_counts_fallback_for_manual(
        self, store, run_factory, counts_artifact_factory
    ):
        """resolve_counts falls back to run artifact for manual runs."""
        counts_artifact = counts_artifact_factory({"0": 600, "1": 400})
        run = run_factory(
            run_id="MANUAL_COUNTS",
            adapter="manual",
            artifacts=[counts_artifact],
        )

        counts = resolve_counts(run, store)

        assert counts["0"] == 600
        assert counts["1"] == 400

    def test_resolve_device_snapshot_from_envelope(
        self,
        store,
        run_factory,
        calibration_factory,
        minimal_producer,
    ):
        """resolve_device_snapshot extracts device from envelope."""
        calibration = calibration_factory(num_qubits=5)
        device = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="ibm_brisbane",
            backend_type="hardware",
            provider="ibm_quantum",
            num_qubits=127,
            calibration=calibration,
        )

        result = ResultSnapshot(success=True, status="completed", items=[])
        envelope = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=result,
            device=device,
        )

        digest = store.put_bytes(json.dumps(envelope.to_dict()).encode())
        artifact = ArtifactRef(
            kind="devqubit.envelope.json",
            digest=digest,
            media_type="application/json",
            role="envelope",
        )
        run = run_factory(run_id="ENV_DEVICE", artifacts=[artifact])

        snapshot = resolve_device_snapshot(run, store)

        assert snapshot is not None
        assert snapshot.backend_name == "ibm_brisbane"
        assert snapshot.num_qubits == 127
        assert snapshot.calibration is not None


class TestResultHandling:
    """Tests for different execution result scenarios."""

    def test_successful_execution(self):
        """Successful execution with counts."""
        item = ResultItem(
            item_index=0,
            success=True,
            counts={"counts": {"00": 500, "11": 500}, "shots": 1000},
        )
        result = ResultSnapshot.create_success(items=[item])

        assert result.success is True
        assert result.status == "completed"
        assert result.items[0].counts["counts"]["00"] == 500

    def test_failed_execution_with_error(self):
        """Failed execution captures error details."""
        error = ResultError(
            type="RuntimeError",
            message="Insufficient credits",
            details={"credits_needed": 100},
        )
        result = ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=error,
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error.type == "RuntimeError"
        assert "credits" in result.error.message.lower()

    def test_partial_batch_execution(self):
        """Partial success: some circuits in batch failed."""
        items = [
            ResultItem(item_index=0, success=True, counts={"counts": {"0": 1000}}),
            ResultItem(item_index=1, success=False, error_message="Circuit too deep"),
            ResultItem(item_index=2, success=True, counts={"counts": {"1": 1000}}),
        ]
        result = ResultSnapshot.create_partial(items=items)

        assert result.success is False
        assert result.status == "partial"
        assert result.items[0].success is True
        assert result.items[1].success is False
        assert result.items[2].success is True


class TestCalibrationFlow:
    """Tests for calibration data through the UEC pipeline."""

    def test_calibration_medians_computed(self):
        """Calibration computes median values for quick comparison."""
        calibration = DeviceCalibration(
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
        calibration.compute_medians()

        assert calibration.median_t1_us == 110.0
        assert calibration.median_t2_us == 85.0
        assert calibration.median_readout_error == 0.015
        assert calibration.median_2q_error == 0.015

    def test_calibration_survives_serialization(
        self, calibration_factory, minimal_producer
    ):
        """Calibration data survives JSON round-trip through envelope."""
        original_cal = calibration_factory(num_qubits=5)
        device = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="test",
            backend_type="hardware",
            provider="ibm_quantum",
            calibration=original_cal,
        )

        result = ResultSnapshot(success=True, status="completed", items=[])
        envelope = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=result,
            device=device,
        )

        # Round-trip through JSON
        restored = ExecutionEnvelope.from_dict(envelope.to_dict())

        assert restored.device is not None
        assert restored.device.calibration is not None
        assert len(restored.device.calibration.qubits) == 5
        assert restored.device.calibration.median_t1_us == original_cal.median_t1_us

    def test_device_snapshot_get_calibration_summary(self, calibration_factory):
        """Device snapshot provides calibration summary for quick access."""
        cal = calibration_factory(num_qubits=3)
        snapshot = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="ibm_test",
            backend_type="hardware",
            provider="ibm_quantum",
            calibration=cal,
        )

        summary = snapshot.get_calibration_summary()

        assert "median_t1_us" in summary
        assert "median_2q_error" in summary
        assert summary["median_t1_us"] == cal.median_t1_us


class TestEnvelopeValidation:
    """Tests for envelope validation and warnings."""

    def test_validate_warns_on_missing_snapshots(self, minimal_producer):
        """Validation warns about missing optional snapshots."""
        result = ResultSnapshot(success=True, status="completed", items=[])
        envelope = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=result,
        )

        warnings = envelope.validate()

        assert "Missing device snapshot" in warnings
        assert "Missing program snapshot" in warnings

    def test_validate_warns_on_failed_without_error(self, minimal_producer):
        """Validation warns when failed result lacks error details."""
        result = ResultSnapshot(success=False, status="failed", items=[])
        envelope = ExecutionEnvelope.create(
            producer=minimal_producer,
            result=result,
        )

        warnings = envelope.validate()

        assert "Failed result missing error details" in warnings
