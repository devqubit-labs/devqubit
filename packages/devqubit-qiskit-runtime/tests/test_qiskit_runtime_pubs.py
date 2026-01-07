# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime PUB handling."""

from __future__ import annotations

import pytest
from devqubit_qiskit_runtime.pubs import (
    extract_circuit_from_pub,
    extract_circuits_from_pubs,
    extract_pubs_structure,
    is_v2_pub_tuple,
    iter_pubs,
    materialize_pubs,
)


class TestIterPubs:
    """Tests for pub normalization via iter_pubs."""

    def test_none_returns_empty(self):
        """Returns empty list for None."""
        assert iter_pubs(None) == []

    def test_single_circuit_wrapped(self, bell_circuit):
        """Single circuit wrapped in list."""
        result = iter_pubs(bell_circuit)
        assert len(result) == 1
        assert result[0] is bell_circuit

    def test_single_pub_tuple_wrapped(self, bell_circuit):
        """Single PUB tuple wrapped in list."""
        pub = (bell_circuit,)
        result = iter_pubs(pub)
        assert len(result) == 1
        assert result[0] is pub

    def test_list_passthrough(self, bell_circuit, ghz_circuit):
        """List of circuits passes through."""
        circuits = [bell_circuit, ghz_circuit]
        result = iter_pubs(circuits)
        assert len(result) == 2
        assert result == circuits

    def test_generator_consumed_once(self, bell_circuit):
        """Generator PUB inputs materialized once (avoids consumption bugs)."""

        def gen():
            for _ in range(3):
                yield (bell_circuit,)

        pubs = iter_pubs(gen())
        assert len(pubs) == 3
        assert all(isinstance(p, tuple) for p in pubs)


class TestMaterializePubs:
    """Tests for materialize_pubs function."""

    def test_quantum_circuit_not_iterated(self, bell_circuit):
        """QuantumCircuit treated as single item (not iterated over instructions)."""
        pubs = materialize_pubs(bell_circuit)
        assert pubs == [bell_circuit]

    def test_single_pub_tuple(self, bell_circuit):
        """Single PUB tuple wrapped in list."""
        pub = (bell_circuit, None, 2048)
        assert is_v2_pub_tuple(pub) is True
        assert materialize_pubs(pub) == [pub]

    def test_tuple_of_circuits_is_container(self, bell_circuit, ghz_circuit):
        """Tuple of multiple circuits is a container, not a PUB tuple."""
        container = (bell_circuit, ghz_circuit)
        assert is_v2_pub_tuple(container) is False
        assert materialize_pubs(container) == [bell_circuit, ghz_circuit]

    def test_tuple_of_pub_tuples(self, bell_circuit, ghz_circuit):
        """Tuple container of pubs becomes list of pubs."""
        pub_container = ((bell_circuit,), (ghz_circuit,))
        assert materialize_pubs(pub_container) == [(bell_circuit,), (ghz_circuit,)]

    def test_dict_pub(self, bell_circuit):
        """Dict-style pub handled."""
        pub = {"circuit": bell_circuit, "shots": 1000}
        result = materialize_pubs(pub)
        assert len(result) == 1
        assert result[0] is pub


class TestIsV2PubTuple:
    """Tests for V2 PUB tuple detection."""

    def test_circuit_only_tuple(self, bell_circuit):
        """Single circuit tuple is a PUB tuple."""
        assert is_v2_pub_tuple((bell_circuit,)) is True

    def test_circuit_with_params(self, bell_circuit):
        """Tuple with circuit and params is a PUB tuple."""
        assert is_v2_pub_tuple((bell_circuit, {"theta": 0.5})) is True

    def test_circuit_with_observables(self, bell_circuit):
        """Tuple with circuit and SparsePauliOp is a PUB tuple."""
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs = SparsePauliOp.from_list([("ZZ", 1.0)])
            assert is_v2_pub_tuple((bell_circuit, obs)) is True
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_tuple_of_circuits_not_pub(self, bell_circuit, ghz_circuit):
        """Tuple of multiple circuits is NOT a PUB tuple."""
        assert is_v2_pub_tuple((bell_circuit, ghz_circuit)) is False

    def test_empty_tuple_not_pub(self):
        """Empty tuple is not a PUB tuple."""
        assert is_v2_pub_tuple(()) is False

    def test_non_tuple_not_pub(self, bell_circuit):
        """Non-tuple is not a PUB tuple."""
        assert is_v2_pub_tuple(bell_circuit) is False
        assert is_v2_pub_tuple([bell_circuit]) is False


class TestExtractCircuitFromPub:
    """Tests for circuit extraction from various pub formats."""

    def test_from_circuit_directly(self, bell_circuit):
        """Returns circuit when given directly."""
        assert extract_circuit_from_pub(bell_circuit) is bell_circuit

    def test_from_tuple(self, bell_circuit):
        """Extracts from PUB tuple (first element)."""
        assert extract_circuit_from_pub((bell_circuit,)) is bell_circuit
        assert extract_circuit_from_pub((bell_circuit, {"theta": 0.5})) is bell_circuit

    def test_from_list(self, bell_circuit):
        """Extracts from list (first element)."""
        assert extract_circuit_from_pub([bell_circuit, {"theta": 0.5}]) is bell_circuit

    def test_from_dict(self, bell_circuit):
        """Extracts from dict with 'circuit' key."""
        pub = {"circuit": bell_circuit, "shots": 1000}
        assert extract_circuit_from_pub(pub) is bell_circuit

    def test_from_object_with_circuit_attr(self, bell_circuit):
        """Extracts from object with .circuit attribute."""

        class ObjPub:
            def __init__(self, circuit):
                self.circuit = circuit

        assert extract_circuit_from_pub(ObjPub(bell_circuit)) is bell_circuit

    def test_returns_none_for_none(self):
        """Returns None for None input."""
        assert extract_circuit_from_pub(None) is None

    def test_returns_none_for_empty_tuple(self):
        """Returns None for empty tuple."""
        assert extract_circuit_from_pub(()) is None


class TestExtractCircuitsFromPubs:
    """Tests for bulk circuit extraction."""

    def test_extracts_all_circuits(self, bell_circuit, ghz_circuit):
        """Extracts circuits from all pubs."""
        pubs = [(bell_circuit,), (ghz_circuit,)]
        result = extract_circuits_from_pubs(pubs)

        assert len(result) == 2
        assert result[0] is bell_circuit
        assert result[1] is ghz_circuit

    def test_skips_invalid_pubs(self, bell_circuit):
        """Skips pubs that don't have circuits."""

        class NoPub:
            pass

        pubs = [(bell_circuit,), NoPub(), (bell_circuit,)]
        result = extract_circuits_from_pubs(pubs)
        assert len(result) == 2

    def test_mixed_pub_types(self, bell_circuit):
        """Handles mixed pub types."""
        pubs = [
            (bell_circuit,),
            {"circuit": bell_circuit},
            bell_circuit,
        ]
        result = extract_circuits_from_pubs(pubs)
        assert len(result) == 3
        assert all(c is bell_circuit for c in result)


class TestExtractPubsStructure:
    """Tests for pub structure extraction (lightweight summaries)."""

    def test_format_detection(self, bell_circuit):
        """Detects different pub formats correctly."""
        pubs = [
            bell_circuit,
            (bell_circuit, None, 1024),
            {"circuit": bell_circuit, "shots": 1000},
        ]

        struct = extract_pubs_structure(pubs)

        assert [s["format"] for s in struct] == ["circuit", "v2_pub_tuple", "dict_pub"]
        assert all(s["has_circuit"] is True for s in struct)

    def test_v2_tuple_length(self, bell_circuit):
        """Records tuple length for V2 PUB tuples."""
        pub = (bell_circuit, {"theta": 0.5})
        result = extract_pubs_structure([pub])

        assert result[0]["format"] == "v2_pub_tuple"
        assert result[0]["tuple_len"] == 2

    def test_with_observables(self, bell_circuit):
        """Extracts observable count when present."""
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5), ("YY", 0.5)])

            class Pub:
                def __init__(self, circuit, observables):
                    self.circuit = circuit
                    self.observables = observables

            pub = Pub(bell_circuit, obs)
            result = extract_pubs_structure([pub])

            assert result[0]["num_observables"] == 3
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_no_circuit_detected(self):
        """Handles pubs without circuit."""

        class EmptyPub:
            pass

        result = extract_pubs_structure([EmptyPub()])

        assert result[0]["has_circuit"] is False
        assert result[0]["format"] == "unknown"

    def test_estimator_tuple_observable_count(self, bell_circuit):
        """
        Fix #9: Extracts num_observables from estimator tuple PUBs.

        Estimator PUB format: (circuit, observables, [params], [precision])
        """
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
            pub = (bell_circuit, obs)

            # With primitive_type="estimator", should extract observable count
            result = extract_pubs_structure([pub], primitive_type="estimator")

            assert result[0]["format"] == "v2_pub_tuple"
            assert result[0]["num_observables"] == 2
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_estimator_tuple_with_list_observables(self, bell_circuit):
        """Extracts observable count from list of observables in tuple."""
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs1 = SparsePauliOp.from_list([("ZZ", 1.0)])
            obs2 = SparsePauliOp.from_list([("XX", 1.0)])
            obs3 = SparsePauliOp.from_list([("YY", 1.0)])

            pub = (bell_circuit, [obs1, obs2, obs3])
            result = extract_pubs_structure([pub], primitive_type="estimator")

            assert result[0]["num_observables"] == 3
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_dict_pub_observable_count(self, bell_circuit):
        """Extracts observable count from dict-style pub."""
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0), ("ZZ", 0.5)])
            pub = {"circuit": bell_circuit, "observables": obs}

            result = extract_pubs_structure([pub])

            assert result[0]["num_observables"] == 3
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_sampler_tuple_no_observable_extraction(self, bell_circuit):
        """Sampler tuple PUBs don't have observables in second position."""
        import numpy as np

        # Sampler PUB: (circuit, param_values, shots)
        params = np.array([[0.1, 0.2]])
        pub = (bell_circuit, params, 1024)

        # Without primitive_type or with sampler, shouldn't extract "observables"
        result = extract_pubs_structure([pub], primitive_type="sampler")

        # Shouldn't have num_observables for sampler
        assert (
            "num_observables" not in result[0]
            or result[0].get("num_observables") is None
        )

    def test_parameter_detection(self, bell_circuit):
        """Detects parameter values in PUBs."""
        import numpy as np

        class PubWithParams:
            def __init__(self, circuit, params):
                self.circuit = circuit
                self.parameter_values = params

        params = np.array([[0.1, 0.2], [0.3, 0.4]])
        pub = PubWithParams(bell_circuit, params)

        result = extract_pubs_structure([pub])

        assert result[0]["has_parameters"] is True
        assert result[0].get("parameter_shape") == [2, 2]


class TestRealPubFormats:
    """Tests with realistic PUB formats used in practice."""

    def test_sampler_pub_formats(self, bell_circuit):
        """Sampler PUB: (circuit,) or (circuit, param_values, shots)."""
        pub1 = (bell_circuit,)
        pub2 = (bell_circuit, None, 2048)

        assert extract_circuit_from_pub(pub1) is bell_circuit
        assert extract_circuit_from_pub(pub2) is bell_circuit

    def test_estimator_pub_formats(self, bell_circuit):
        """Estimator PUB: (circuit, observables) or with params/precision."""
        try:
            from qiskit.quantum_info import SparsePauliOp

            obs = SparsePauliOp.from_list([("ZZ", 1.0)])

            pub1 = (bell_circuit, obs)
            pub2 = (bell_circuit, obs, None, 0.01)  # With precision

            assert extract_circuit_from_pub(pub1) is bell_circuit
            assert extract_circuit_from_pub(pub2) is bell_circuit
        except ImportError:
            pytest.skip("SparsePauliOp not available")

    def test_parameterized_sampler_pub(self, parameterized_circuit):
        """Sampler PUB with parameter bindings."""
        import numpy as np

        param_values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        pub = (parameterized_circuit, param_values)

        assert extract_circuit_from_pub(pub) is parameterized_circuit

        struct = extract_pubs_structure([pub])
        assert struct[0]["format"] == "v2_pub_tuple"
