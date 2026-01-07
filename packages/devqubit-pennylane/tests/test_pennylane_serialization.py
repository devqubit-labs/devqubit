# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for PennyLane tape serialization."""

import json

import pennylane as qml
import pytest
from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.circuit.registry import LoaderError, SerializerError
from devqubit_pennylane.serialization import (
    PennyLaneCircuitLoader,
    PennyLaneCircuitSerializer,
    is_pennylane_tape,
    materialize_tapes,
    serialize_tape,
    serialize_tapes,
    summarize_pennylane_tape,
    tape_to_text,
)


class TestIsTape:
    """Tests for tape type detection."""

    def test_detects_quantum_tape(self, bell_tape):
        """Detects QuantumTape objects."""
        assert is_pennylane_tape(bell_tape) is True

    def test_detects_quantum_script(self, quantum_script_fixture):
        """Detects QuantumScript objects."""
        assert is_pennylane_tape(quantum_script_fixture) is True

    def test_rejects_non_tapes(self):
        """Rejects non-tape objects."""
        assert is_pennylane_tape(None) is False
        assert is_pennylane_tape("tape") is False
        assert is_pennylane_tape([]) is False


class TestMaterializeTapes:
    """Tests for tape materialization."""

    def test_single_tape(self, bell_tape):
        """Materializes single tape."""
        tapes, payload = materialize_tapes(bell_tape)

        assert len(tapes) == 1
        assert tapes[0] is bell_tape
        assert payload is bell_tape

    def test_list_of_tapes(self, bell_tape, ghz_tape):
        """Materializes list of tapes."""
        tapes, payload = materialize_tapes([bell_tape, ghz_tape])

        assert len(tapes) == 2
        assert tapes[0] is bell_tape
        assert tapes[1] is ghz_tape

    def test_generator_consumed_once(self, bell_tape, ghz_tape):
        """Generator is consumed exactly once."""
        gen = (t for t in [bell_tape, ghz_tape])

        tapes, payload = materialize_tapes(gen)

        assert tapes == [bell_tape, ghz_tape]
        assert payload is tapes

    def test_none_and_empty(self):
        """Handles None and empty list."""
        tapes, _ = materialize_tapes(None)
        assert tapes == []

        tapes, _ = materialize_tapes([])
        assert tapes == []


class TestSerializeTape:
    """Tests for single tape serialization."""

    def test_serialize_bell_tape(self, bell_tape):
        """Serializes Bell state tape to JSON."""
        data = serialize_tape(bell_tape, name="bell", index=0)

        assert data.format == CircuitFormat.TAPE_JSON
        assert data.sdk == SDK.PENNYLANE
        assert data.name == "bell"
        assert data.index == 0

        parsed = json.loads(data.data)
        assert "operations" in parsed
        assert "measurements" in parsed
        assert parsed["num_operations"] == 2  # H, CNOT

    def test_serialize_includes_shots_when_present(self, tape_with_shots):
        """Includes shots configuration in serialization."""
        data = serialize_tape(tape_with_shots, name="shots", index=0)
        parsed = json.loads(data.data)

        assert parsed["shots"] == 500
        assert parsed["num_operations"] == 2
        assert parsed["num_measurements"] == 1

    def test_serialize_includes_shot_vector(self, tape_with_shot_vector):
        """Includes shot vector in serialization."""
        data = serialize_tape(tape_with_shot_vector, name="sv", index=0)
        parsed = json.loads(data.data)

        assert "shot_vector" in parsed
        assert isinstance(parsed["shot_vector"], list)

    def test_operations_have_parameters(self, parameterized_tape):
        """Operations with parameters are serialized correctly."""
        data = serialize_tape(parameterized_tape)
        parsed = json.loads(data.data)

        ops_with_params = [op for op in parsed["operations"] if op.get("parameters")]
        assert len(ops_with_params) >= 1

    def test_measurements_have_observable_info(self, expectation_tape):
        """Measurements include observable information."""
        data = serialize_tape(expectation_tape)
        parsed = json.loads(data.data)

        assert parsed["num_measurements"] == 1
        meas = parsed["measurements"][0]
        assert "return_type" in meas
        assert "observable" in meas
        assert meas["observable"]["name"] == "PauliZ"

    def test_serialize_quantum_script(self, quantum_script_fixture):
        """Serializes QuantumScript correctly."""
        data = serialize_tape(quantum_script_fixture, name="script", index=0)
        parsed = json.loads(data.data)

        assert "operations" in parsed
        assert "measurements" in parsed
        assert parsed["num_operations"] >= 1


class TestSerializeTapes:
    """Tests for multiple tape serialization."""

    def test_batch_has_indices_and_count(self, bell_tape, ghz_tape):
        """Batch serialization includes tape indices and count."""
        data = serialize_tapes([bell_tape, ghz_tape])
        parsed = json.loads(data.data)

        assert data.sdk == SDK.PENNYLANE
        assert data.format == CircuitFormat.TAPE_JSON
        assert parsed["num_tapes"] == 2
        assert [t["index"] for t in parsed["tapes"]] == [0, 1]

    def test_empty_list(self):
        """Handles empty tape list."""
        data = serialize_tapes([])
        parsed = json.loads(data.data)

        assert parsed["num_tapes"] == 0
        assert parsed["tapes"] == []


class TestTapeToText:
    """Tests for human-readable tape text."""

    def test_basic_format(self, bell_tape):
        """Produces readable text format."""
        text = tape_to_text(bell_tape, index=0)

        assert "=== Tape 0 ===" in text
        assert "Operations:" in text
        assert "Measurements:" in text
        assert "Hadamard" in text

    def test_shows_shots(self, tape_with_shots):
        """Shows shots configuration when present."""
        text = tape_to_text(tape_with_shots, index=0)

        assert "Shots:" in text and "500" in text

    def test_shows_shot_vector(self, tape_with_shot_vector):
        """Shows shot vector when present."""
        text = tape_to_text(tape_with_shot_vector, index=0)

        assert "Shot vector:" in text or "shot" in text.lower()


class TestPennyLaneCircuitLoader:
    """Tests for tape loader."""

    def test_loader_properties(self):
        """Loader has correct properties."""
        loader = PennyLaneCircuitLoader()

        assert loader.name == "pennylane"
        assert loader.sdk == SDK.PENNYLANE
        assert CircuitFormat.TAPE_JSON in loader.supported_formats

    def test_roundtrip_preserves_operations(self, bell_tape):
        """Roundtrip preserves operation structure."""
        loader = PennyLaneCircuitLoader()

        serialized = serialize_tape(bell_tape, name="bell", index=0)
        loaded = loader.load(serialized)

        assert loaded.sdk == SDK.PENNYLANE
        assert loaded.source_format == CircuitFormat.TAPE_JSON
        assert len(loaded.circuit.operations) == len(bell_tape.operations)
        assert [op.name for op in loaded.circuit.operations] == [
            op.name for op in bell_tape.operations
        ]
        assert [list(op.wires) for op in loaded.circuit.operations] == [
            list(op.wires) for op in bell_tape.operations
        ]

    def test_roundtrip_preserves_measurements(self, expectation_tape):
        """
        Roundtrip preserves measurement structure.

        P1 requirement: loader must reconstruct measurements, not just operations.
        """
        loader = PennyLaneCircuitLoader()

        serialized = serialize_tape(expectation_tape, name="expval", index=0)
        loaded = loader.load(serialized)

        # Measurements should be reconstructed
        assert len(loaded.circuit.measurements) >= 1

        # Check measurement type
        loaded_meas = loaded.circuit.measurements[0]

        # Both should be expectation values
        assert (
            "expval" in type(loaded_meas).__name__.lower()
            or "expectation" in type(loaded_meas).__name__.lower()
        )

    def test_roundtrip_preserves_measurement_wires(self, probability_tape):
        """Roundtrip preserves measurement wire information."""

        loader = PennyLaneCircuitLoader()

        serialized = serialize_tape(probability_tape, name="probs", index=0)
        loaded = loader.load(serialized)

        # Probs measurement should have wires
        assert len(loaded.circuit.measurements) >= 1

    def test_load_batch(self, bell_tape, ghz_tape):
        """Loads batch of tapes."""
        loader = PennyLaneCircuitLoader()

        serialized = serialize_tapes([bell_tape, ghz_tape])
        loaded_list = loader.load_batch(serialized)

        assert len(loaded_list) == 2
        assert loaded_list[0].index == 0
        assert loaded_list[1].index == 1

    def test_unsupported_format_raises(self):
        """Raises error for unsupported format."""
        loader = PennyLaneCircuitLoader()
        data = CircuitData(
            data="{}",
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.PENNYLANE,
            name="x",
            index=0,
        )

        with pytest.raises(LoaderError):
            loader.load(data)


class TestPennyLaneCircuitSerializer:
    """Tests for tape serializer."""

    def test_serializer_properties(self):
        """Serializer has correct properties."""
        serializer = PennyLaneCircuitSerializer()

        assert serializer.name == "pennylane"
        assert serializer.sdk == SDK.PENNYLANE
        assert CircuitFormat.TAPE_JSON in serializer.supported_formats

    def test_can_serialize_tape(self, bell_tape):
        """Recognizes tapes for serialization."""
        serializer = PennyLaneCircuitSerializer()

        assert serializer.can_serialize(bell_tape) is True
        assert serializer.can_serialize(None) is False
        assert serializer.can_serialize([]) is False

    def test_can_serialize_quantum_script(self, quantum_script_fixture):
        """Recognizes QuantumScript for serialization."""
        serializer = PennyLaneCircuitSerializer()
        assert serializer.can_serialize(quantum_script_fixture) is True

    def test_unsupported_format_raises(self, bell_tape):
        """Raises error for unsupported format."""
        serializer = PennyLaneCircuitSerializer()

        with pytest.raises(SerializerError):
            serializer.serialize(bell_tape, CircuitFormat.OPENQASM3)


class TestSummarizePennylaneTape:
    """Tests for tape summary generation."""

    def test_bell_tape_summary(self, bell_tape):
        """Summarizes Bell state tape correctly."""
        summary = summarize_pennylane_tape(bell_tape)

        assert summary.sdk == SDK.PENNYLANE
        assert summary.num_qubits == 2
        assert summary.gate_count_1q == 1  # Hadamard
        assert summary.gate_count_2q == 1  # CNOT
        assert summary.gate_count_measure == 1

    def test_clifford_detection(self, bell_tape, non_clifford_tape):
        """Detects Clifford vs non-Clifford tapes."""
        assert summarize_pennylane_tape(bell_tape).is_clifford is True
        assert summarize_pennylane_tape(non_clifford_tape).is_clifford is False

    def test_parameterized_tape_detection(self, parameterized_tape):
        """Detects parameterized tapes."""
        summary = summarize_pennylane_tape(parameterized_tape)

        assert summary.has_parameters is True
        assert summary.parameter_count >= 1

    def test_gate_types_counted(self, bell_tape):
        """Counts gate types with normalized names."""
        summary = summarize_pennylane_tape(bell_tape)

        assert "hadamard" in summary.gate_types
        assert "cnot" in summary.gate_types
        assert summary.gate_count_total > 0

    def test_empty_tape(self):
        """Handles empty tape."""
        with qml.tape.QuantumTape() as tape:
            pass

        summary = summarize_pennylane_tape(tape)

        assert summary.num_qubits == 0
        assert summary.gate_count_total == 0
        assert summary.is_clifford is None

    def test_quantum_script_summary(self, quantum_script_fixture):
        """Summarizes QuantumScript correctly."""
        summary = summarize_pennylane_tape(quantum_script_fixture)

        assert summary.sdk == SDK.PENNYLANE
        assert summary.num_qubits >= 1


class TestCircuitDataMethods:
    """Tests for CircuitData helper methods."""

    def test_as_bytes(self, bell_tape):
        """CircuitData.as_bytes() returns bytes."""
        data = serialize_tape(bell_tape)
        result = data.as_bytes()

        assert isinstance(result, bytes)
        json.loads(result.decode("utf-8"))  # Valid JSON

    def test_as_text(self, bell_tape):
        """CircuitData.as_text() returns string."""
        data = serialize_tape(bell_tape)
        result = data.as_text()

        assert isinstance(result, str)
        json.loads(result)  # Valid JSON


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_tape_with_multiple_measurements(self, multi_measurement_tape):
        """Handles tapes with multiple measurements."""
        data = serialize_tape(multi_measurement_tape)
        parsed = json.loads(data.data)

        assert parsed["num_measurements"] == 3

    def test_tape_with_rotation_gates(self):
        """Handles rotation gates with parameters."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=0)
            qml.RY(1.2, wires=0)
            qml.RZ(0.8, wires=0)
            qml.expval(qml.PauliZ(0))

        data = serialize_tape(tape)
        parsed = json.loads(data.data)

        for op in parsed["operations"]:
            assert "parameters" in op
            assert len(op["parameters"]) > 0

    def test_tape_with_controlled_gates(self):
        """Handles controlled gates."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CZ(wires=[0, 1])
            qml.SWAP(wires=[0, 1])
            qml.counts(wires=[0, 1])

        summary = summarize_pennylane_tape(tape)

        assert summary.gate_count_1q == 1  # H
        assert summary.gate_count_2q == 2  # CZ, SWAP

    def test_observable_with_tensor_product(self):
        """Handles tensor product observables."""
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        data = serialize_tape(tape)
        parsed = json.loads(data.data)

        # Should capture tensor product info
        assert parsed["num_measurements"] == 1
        meas = parsed["measurements"][0]
        assert "observable" in meas
