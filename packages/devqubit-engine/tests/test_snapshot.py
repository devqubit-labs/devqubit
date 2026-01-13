# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit UEC snapshot types."""

from __future__ import annotations

from unittest.mock import patch

from devqubit_engine.uec.calibration import (
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from devqubit_engine.uec.device import DeviceSnapshot, FrontendConfig
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.program import (
    ProgramArtifact,
    ProgramSnapshot,
    TranspilationInfo,
)
from devqubit_engine.uec.result import NormalizedCounts, ResultSnapshot
from devqubit_engine.uec.types import (
    ArtifactRef,
    ProgramRole,
    ResultType,
    TranspilationMode,
    ValidationResult,
)


class TestQubitCalibration:
    """Tests for QubitCalibration dataclass."""

    def test_to_dict_excludes_none(self):
        """to_dict only includes non-None values."""
        qc = QubitCalibration(qubit=0, t1_us=100.0)
        d = qc.to_dict()

        assert d == {"qubit": 0, "t1_us": 100.0}
        assert "t2_us" not in d

    def test_full_calibration_round_trip(self):
        """Full calibration survives serialization round-trip."""
        original = QubitCalibration(
            qubit=5,
            t1_us=120.5,
            t2_us=85.3,
            readout_error=0.012,
            gate_error_1q=0.001,
            frequency_ghz=5.2,
            anharmonicity_ghz=-0.33,
        )
        restored = QubitCalibration.from_dict(original.to_dict())

        assert restored.qubit == original.qubit
        assert restored.t1_us == original.t1_us
        assert restored.frequency_ghz == original.frequency_ghz


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

        assert not single.is_two_qubit
        assert two.is_two_qubit


class TestDeviceCalibration:
    """Tests for DeviceCalibration dataclass."""

    def test_compute_medians(self):
        """compute_medians calculates correct median values."""
        cal = DeviceCalibration(
            qubits=[
                QubitCalibration(
                    qubit=0,
                    t1_us=100.0,
                    t2_us=80.0,
                    readout_error=0.01,
                ),
                QubitCalibration(
                    qubit=1,
                    t1_us=120.0,
                    t2_us=90.0,
                    readout_error=0.02,
                ),
                QubitCalibration(
                    qubit=2,
                    t1_us=110.0,
                    t2_us=85.0,
                    readout_error=0.015,
                ),
            ],
            gates=[
                GateCalibration(
                    gate="cx",
                    qubits=(0, 1),
                    error=0.01,
                ),
                GateCalibration(
                    gate="cx",
                    qubits=(1, 2),
                    error=0.02,
                ),
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

    def test_round_trip(self):
        """Full calibration survives serialization."""
        cal = DeviceCalibration(
            calibration_time="2024-01-01T10:00:00Z",
            qubits=[QubitCalibration(qubit=0, t1_us=100.0)],
            gates=[GateCalibration(gate="cx", qubits=(0, 1), error=0.01)],
            source="provider",
        )
        restored = DeviceCalibration.from_dict(cal.to_dict())

        assert restored.calibration_time == cal.calibration_time
        assert restored.source == "provider"
        assert len(restored.qubits) == 1

    def test_calibration_factory_fixture(self, calibration_factory):
        """Fixture creates valid calibration with computed medians."""
        cal = calibration_factory(num_qubits=5)

        assert len(cal.qubits) == 5
        assert len(cal.gates) == 4
        assert cal.median_t1_us is not None


class TestFrontendConfig:
    """Tests for FrontendConfig dataclass."""

    def test_round_trip(self):
        """FrontendConfig survives serialization."""
        fc = FrontendConfig(
            name="SamplerV2",
            sdk="qiskit_runtime",
            sdk_version="0.25.0",
            config={"resilience_level": 1},
        )
        restored = FrontendConfig.from_dict(fc.to_dict())

        assert restored.name == "SamplerV2"
        assert restored.config["resilience_level"] == 1


class TestDeviceSnapshot:
    """Tests for DeviceSnapshot dataclass."""

    def test_connectivity_serialization(self):
        """Connectivity tuples convert to lists and back."""
        snap = DeviceSnapshot(
            captured_at="2024-01-01T00:00:00Z",
            backend_name="test",
            backend_type="hardware",
            provider="test",
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
            provider="test",
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
            provider="test",
        )
        assert snap.get_calibration_summary() is None


class TestProgramArtifact:
    """Tests for ProgramArtifact dataclass."""

    def test_round_trip(self):
        """ProgramArtifact survives serialization."""
        ref = ArtifactRef(
            kind="qiskit.qpy.circuits",
            digest="sha256:" + "a" * 64,
            media_type="application/x-qpy",
            role="program",
        )
        pa = ProgramArtifact(
            ref=ref,
            role=ProgramRole.LOGICAL,
            format="qpy",
            name="bell_state",
            index=0,
        )
        restored = ProgramArtifact.from_dict(pa.to_dict())

        assert restored.format == "qpy"
        assert restored.name == "bell_state"


class TestProgramSnapshot:
    """Tests for ProgramSnapshot dataclass."""

    def test_empty_snapshot(self):
        """Empty snapshot serializes with schema and empty lists."""
        snap = ProgramSnapshot()
        d = snap.to_dict()

        assert d["schema"] == "devqubit.program_snapshot/0.1"
        # Empty lists are included in serialization
        assert d["logical"] == []
        assert d["physical"] == []

    def test_non_empty_snapshot(self):
        """Non-empty snapshot includes artifacts."""
        ref = ArtifactRef(
            kind="qiskit.qpy.circuits",
            digest="sha256:" + "a" * 64,
            media_type="application/x-qpy",
            role="program",
        )
        pa = ProgramArtifact(
            ref=ref,
            role=ProgramRole.LOGICAL,
            format="qpy",
            name="bell_state",
            index=0,
        )
        snap = ProgramSnapshot(logical=[pa])
        d = snap.to_dict()

        assert len(d["logical"]) == 1
        assert d["logical"][0]["name"] == "bell_state"


class TestTranspilationInfo:
    """Tests for TranspilationInfo dataclass."""

    def test_mode_enum_serialization(self):
        """TranspilationMode enum serializes to string."""
        ti = TranspilationInfo(mode=TranspilationMode.MANAGED)
        d = ti.to_dict()

        assert d["mode"] == "managed"

    def test_optimization_level_serialization(self):
        """TranspilationInfo serializes optimization_level."""
        ti = TranspilationInfo(
            mode=TranspilationMode.AUTO,
            optimization_level=2,
        )
        d = ti.to_dict()

        assert d["mode"] == "auto"
        assert d["optimization_level"] == 2

    def test_round_trip(self):
        """TranspilationInfo survives serialization round-trip."""
        ti = TranspilationInfo(
            mode=TranspilationMode.AUTO,
            optimization_level=3,
            layout_method="sabre",
            routing_method="stochastic",
        )
        d = ti.to_dict()
        restored = TranspilationInfo.from_dict(d)

        assert restored.mode == TranspilationMode.AUTO
        assert restored.optimization_level == 3
        assert restored.layout_method == "sabre"
        assert restored.routing_method == "stochastic"


class TestExecutionSnapshot:
    """Tests for ExecutionSnapshot dataclass."""

    def test_round_trip_with_transpilation(self):
        """ExecutionSnapshot with transpilation survives serialization."""
        snap = ExecutionSnapshot(
            submitted_at="2024-01-01T10:00:00Z",
            completed_at="2024-01-01T10:05:00Z",
            shots=1000,
            job_ids=["job_123"],
            transpilation=TranspilationInfo(
                mode=TranspilationMode.AUTO,
                optimization_level=1,
            ),
            sdk="qiskit",
        )
        restored = ExecutionSnapshot.from_dict(snap.to_dict())

        assert restored.shots == 1000
        assert restored.transpilation.optimization_level == 1

    def test_minimal_snapshot(self):
        """ExecutionSnapshot with only required fields."""
        snap = ExecutionSnapshot(submitted_at="2024-01-01T10:00:00Z")
        d = snap.to_dict()

        assert d["submitted_at"] == "2024-01-01T10:00:00Z"
        assert "shots" not in d


class TestNormalizedCounts:
    """Tests for NormalizedCounts dataclass."""

    def test_bell_state_counts(self, bell_state_counts):
        """NormalizedCounts handles Bell state results."""
        nc = NormalizedCounts(
            circuit_index=0,
            counts=bell_state_counts,
            shots=1000,
            name="bell",
        )
        d = nc.to_dict()

        assert d["counts"]["00"] == 500
        assert d["shots"] == 1000


class TestResultSnapshot:
    """Tests for ResultSnapshot dataclass."""

    def test_counts_result(self, bell_state_counts):
        """ResultSnapshot with counts."""
        snap = ResultSnapshot(
            result_type=ResultType.COUNTS,
            counts=[NormalizedCounts(circuit_index=0, counts=bell_state_counts)],
            num_experiments=1,
            success=True,
        )
        d = snap.to_dict()

        assert d["result_type"] == "counts"
        assert d["success"] is True

    def test_failed_result(self):
        """ResultSnapshot captures failure info."""
        snap = ResultSnapshot(
            result_type=ResultType.COUNTS,
            success=False,
            error_message="Backend timeout",
        )
        d = snap.to_dict()

        assert d["success"] is False
        assert d["error_message"] == "Backend timeout"

    def test_from_dict_defaults_result_type(self):
        """from_dict defaults result_type to 'counts'."""
        snap = ResultSnapshot.from_dict({})
        assert snap.result_type == "counts"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_is_truthy(self):
        """Valid result evaluates to True."""
        result = ValidationResult(valid=True)
        assert result
        assert result.valid
        assert result.errors == []

    def test_invalid_result_is_falsy(self):
        """Invalid result evaluates to False."""
        result = ValidationResult(valid=False, errors=["error1", "error2"])
        assert not result
        assert not result.valid
        assert len(result.errors) == 2

    def test_warnings_preserved(self):
        """Warnings are preserved in result."""
        result = ValidationResult(
            valid=True,
            warnings=["module not available"],
        )
        assert result.valid
        assert "module not available" in result.warnings


class TestExecutionEnvelope:
    """Tests for ExecutionEnvelope dataclass."""

    def test_to_dict_includes_schema(self):
        """to_dict always includes schema version."""
        env = ExecutionEnvelope()
        d = env.to_dict()

        assert d["schema"] == "devqubit.envelope/0.1"

    def test_round_trip_complete_envelope(self, calibration_factory):
        """Complete envelope survives serialization."""
        env = ExecutionEnvelope(
            device=DeviceSnapshot(
                captured_at="2024-01-01T00:00:00Z",
                backend_name="ibm_brisbane",
                backend_type="hardware",
                provider="ibm_quantum",
                calibration=calibration_factory(num_qubits=3),
            ),
            execution=ExecutionSnapshot(
                submitted_at="2024-01-01T00:00:00Z",
                shots=1000,
            ),
            result=ResultSnapshot(result_type=ResultType.COUNTS, success=True),
            adapter="qiskit",
            envelope_id="env_123",
            created_at="2024-01-01T00:00:00Z",
        )
        restored = ExecutionEnvelope.from_dict(env.to_dict())

        assert restored.device.backend_name == "ibm_brisbane"
        assert restored.adapter == "qiskit"

    def test_validate_missing_snapshots(self):
        """validate() reports missing snapshots."""
        env = ExecutionEnvelope()
        warnings = env.validate()

        assert "Missing device snapshot" in warnings
        assert "Missing program snapshot" in warnings
        assert "Missing execution snapshot" in warnings
        assert "Missing result snapshot" in warnings

    def test_validate_missing_backend_name(self):
        """validate() reports missing backend_name."""
        env = ExecutionEnvelope(
            device=DeviceSnapshot(
                captured_at="2024-01-01T00:00:00Z",
                backend_name="",
                backend_type="simulator",
                provider="test",
            ),
        )
        warnings = env.validate()

        assert "Device snapshot missing backend_name" in warnings

    def test_validate_failed_result_without_error(self):
        """validate() warns on failed result without error_message."""
        env = ExecutionEnvelope(
            result=ResultSnapshot(result_type=ResultType.COUNTS, success=False),
        )
        warnings = env.validate()

        assert "Failed result missing error_message" in warnings

    def test_validate_schema_returns_validation_result(self):
        """validate_schema returns ValidationResult object."""
        env = ExecutionEnvelope()

        # Mock validation module returning no errors
        mock_module = type(
            "MockModule", (), {"validate_envelope": lambda *a, **k: []}
        )()

        with patch.dict(
            "sys.modules",
            {"devqubit_engine.schema.validation": mock_module},
        ):
            result = env.validate_schema()

            assert isinstance(result, ValidationResult)
            assert result.valid
            assert result.errors == []

    def test_validate_schema_returns_errors_in_result(self):
        """validate_schema returns errors in ValidationResult."""
        env = ExecutionEnvelope()

        # Mock validation module returning errors
        mock_module = type(
            "MockModule",
            (),
            {"validate_envelope": lambda *a, **k: ["error1", "error2"]},
        )()

        with patch.dict(
            "sys.modules",
            {"devqubit_engine.schema.validation": mock_module},
        ):
            result = env.validate_schema()

            assert isinstance(result, ValidationResult)
            assert not result.valid
            assert len(result.errors) == 2

    def test_validate_schema_handles_exception(self):
        """validate_schema catches exceptions and returns invalid result."""
        env = ExecutionEnvelope()

        def raise_error(*args, **kwargs):
            raise ValueError("Schema validation error")

        mock_module = type("MockModule", (), {"validate_envelope": raise_error})()

        with patch.dict(
            "sys.modules",
            {"devqubit_engine.schema.validation": mock_module},
        ):
            result = env.validate_schema()

            assert isinstance(result, ValidationResult)
            assert not result.valid
            assert len(result.errors) == 1
            assert isinstance(result.errors[0], ValueError)

    def test_validate_schema_handles_import_error(self):
        """validate_schema returns valid with warning on ImportError."""
        env = ExecutionEnvelope()

        # Set module to None to trigger ImportError
        with patch.dict(
            "sys.modules",
            {"devqubit_engine.schema.validation": None},
        ):
            result = env.validate_schema()

            assert isinstance(result, ValidationResult)
            assert result.valid  # Valid because we can't validate
            assert len(result.warnings) == 1
            assert "not available" in result.warnings[0]
