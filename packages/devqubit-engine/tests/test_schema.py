# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for JSON Schema validation."""

from __future__ import annotations

import pytest
from devqubit_engine.schema.validation import (
    clear_cache,
    validate_envelope,
    validate_run_record,
)


class TestRunRecordValidation:
    """Tests for run record schema validation."""

    @pytest.fixture
    def valid_run_record(self) -> dict:
        """Minimal valid run record."""
        return {
            "schema": "devqubit.run/1.0",
            "run_id": "01HQXYZ1234567890ABC",
            "created_at": "2024-01-15T10:30:00Z",
            "project": {"name": "test_project"},
            "adapter": "qiskit",
            "data": {
                "params": {"shots": 1000},
                "metrics": {"fidelity": 0.95},
                "tags": {"env": "test"},
            },
            "artifacts": [],
        }

    def test_valid_record_passes(self, valid_run_record):
        """Valid run record passes validation."""
        errors = validate_run_record(valid_run_record, raise_on_error=False)

        assert errors == []

    def test_missing_required_field_fails(self, valid_run_record):
        """Missing required field fails validation."""
        del valid_run_record["run_id"]

        with pytest.raises(ValueError, match="run_id"):
            validate_run_record(valid_run_record)

    def test_invalid_run_id_format_fails(self, valid_run_record):
        """run_id must be at least 10 characters."""
        valid_run_record["run_id"] = "SHORT"

        with pytest.raises(ValueError, match="minLength"):
            validate_run_record(valid_run_record)

    def test_finished_run_requires_ended_at(self, valid_run_record):
        """FINISHED status requires ended_at timestamp."""
        valid_run_record["info"] = {"status": "FINISHED"}

        with pytest.raises(ValueError, match="ended_at"):
            validate_run_record(valid_run_record)

    def test_finished_run_with_ended_at_passes(self, valid_run_record):
        """FINISHED status with ended_at passes."""
        valid_run_record["info"] = {
            "status": "FINISHED",
            "ended_at": "2024-01-15T10:35:00Z",
        }

        errors = validate_run_record(valid_run_record, raise_on_error=False)

        assert errors == []

    def test_artifact_ref_requires_valid_digest(self, valid_run_record):
        """Artifact digest must be sha256:hex format."""
        valid_run_record["artifacts"] = [
            {
                "kind": "test.data",
                "digest": "invalid-digest",
                "media_type": "text/plain",
                "role": "test",
            }
        ]

        with pytest.raises(ValueError, match="digest"):
            validate_run_record(valid_run_record)

    def test_valid_artifact_ref_passes(self, valid_run_record):
        """Valid artifact reference passes."""
        valid_run_record["artifacts"] = [
            {
                "kind": "circuit.qasm",
                "digest": "sha256:" + "a" * 64,
                "media_type": "text/plain",
                "role": "program",
            }
        ]

        errors = validate_run_record(valid_run_record, raise_on_error=False)

        assert errors == []

    def test_missing_schema_field_fails(self):
        """Record without schema field fails."""
        record = {"run_id": "TEST123456", "created_at": "2024-01-01T00:00:00Z"}

        with pytest.raises(ValueError, match="schema"):
            validate_run_record(record)


class TestEnvelopeValidation:
    """Tests for execution envelope schema validation."""

    @pytest.fixture
    def valid_envelope(self) -> dict:
        """Minimal valid envelope (manual adapter)."""
        return {
            "schema": "devqubit.envelope/1.0",
            "envelope_id": "01HQXYZ1234567890ABCDEF",
            "created_at": "2024-01-15T10:30:00Z",
            "producer": {
                "name": "devqubit",
                "adapter": "manual",
                "frontends": ["manual"],
            },
            "result": {
                "schema": "devqubit.result_snapshot/1.0",
                "success": True,
                "status": "completed",
                "items": [],
            },
        }

    def test_valid_envelope_passes(self, valid_envelope):
        """Valid envelope passes validation."""
        errors = validate_envelope(valid_envelope, raise_on_error=False)

        assert errors == []

    def test_missing_result_fails(self, valid_envelope):
        """Missing result snapshot fails."""
        del valid_envelope["result"]

        with pytest.raises(ValueError, match="result"):
            validate_envelope(valid_envelope)

    def test_invalid_result_status_fails(self, valid_envelope):
        """Invalid result status fails."""
        valid_envelope["result"]["status"] = "running"  # not in enum

        with pytest.raises(ValueError, match="status"):
            validate_envelope(valid_envelope)

    def test_adapter_run_requires_program_and_execution(self, valid_envelope):
        """Non-manual adapter requires program and execution snapshots."""
        valid_envelope["producer"]["adapter"] = "devqubit-qiskit"
        valid_envelope["producer"]["frontends"] = ["qiskit"]

        # Missing program and execution
        with pytest.raises(ValueError):
            validate_envelope(valid_envelope)

    def test_adapter_run_with_snapshots_passes(self, valid_envelope):
        """Adapter run with required snapshots passes."""
        valid_envelope["producer"]["adapter"] = "devqubit-qiskit"
        valid_envelope["producer"]["frontends"] = ["qiskit"]
        valid_envelope["program"] = {
            "schema": "devqubit.program_snapshot/1.0",
            "structural_hash": "sha256:" + "a" * 64,
            "parametric_hash": "sha256:" + "b" * 64,
        }
        valid_envelope["execution"] = {
            "schema": "devqubit.execution_snapshot/1.0",
            "submitted_at": "2024-01-15T10:30:00Z",
        }

        errors = validate_envelope(valid_envelope, raise_on_error=False)

        assert errors == []

    def test_counts_format_requires_all_fields(self, valid_envelope):
        """Counts format requires source_sdk, bit_order, transformed."""
        valid_envelope["result"]["items"] = [
            {
                "item_index": 0,
                "success": True,
                "counts": {
                    "counts": {"00": 500, "11": 500},
                    "shots": 1000,
                    "format": {
                        "source_sdk": "qiskit",
                        # Missing source_key_format, bit_order, transformed
                    },
                },
            }
        ]

        with pytest.raises(ValueError, match="format"):
            validate_envelope(valid_envelope)

    def test_valid_counts_with_format_passes(self, valid_envelope):
        """Counts with complete format passes."""
        valid_envelope["result"]["items"] = [
            {
                "item_index": 0,
                "success": True,
                "counts": {
                    "counts": {"00": 500, "11": 500},
                    "shots": 1000,
                    "format": {
                        "source_sdk": "qiskit",
                        "source_key_format": "qiskit_little_endian",
                        "bit_order": "cbit0_right",
                        "transformed": False,
                    },
                },
            }
        ]

        errors = validate_envelope(valid_envelope, raise_on_error=False)

        assert errors == []

    def test_failed_result_with_error_passes(self, valid_envelope):
        """Failed result with error details passes."""
        valid_envelope["result"]["success"] = False
        valid_envelope["result"]["status"] = "failed"
        valid_envelope["result"]["error"] = {
            "type": "RuntimeError",
            "message": "Backend unavailable",
        }

        errors = validate_envelope(valid_envelope, raise_on_error=False)

        assert errors == []


class TestValidationErrorHandling:
    """Tests for validation error handling."""

    def test_raise_on_error_true_raises(self):
        """raise_on_error=True raises ValueError."""
        invalid = {"schema": "devqubit.run/1.0"}  # Missing required fields

        with pytest.raises(ValueError):
            validate_run_record(invalid, raise_on_error=True)

    def test_raise_on_error_false_returns_errors(self):
        """raise_on_error=False returns error list."""
        invalid = {"schema": "devqubit.run/1.0"}

        errors = validate_run_record(invalid, raise_on_error=False)

        assert len(errors) > 0

    def test_unsupported_schema_fails(self):
        """Unsupported schema version fails."""
        record = {"schema": "devqubit.run/9.9", "run_id": "TEST123456"}

        with pytest.raises(ValueError, match="Unsupported schema"):
            validate_run_record(record)

    def test_clear_cache_works(self):
        """clear_cache doesn't raise."""
        # Just verify it doesn't error
        clear_cache()
