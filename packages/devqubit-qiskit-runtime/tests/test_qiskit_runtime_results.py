# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for Qiskit Runtime result processing and pubs utilities.

Covers:
- Sampler result extraction and snapshot construction
- Estimator result extraction with stds alignment
- BitArray extraction from DataBin containers
- Observable counting (SparsePauliOp vs list)
- PUB structure extraction and parameter value extraction
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from devqubit_qiskit_runtime.pubs import (
    _count_observables,
    extract_circuits_from_pubs,
    extract_parameter_values_from_pubs,
    extract_pubs_structure,
    is_v2_pub_tuple,
    materialize_pubs,
)
from devqubit_qiskit_runtime.results import (
    build_estimator_result_snapshot,
    build_sampler_result_snapshot,
    counts_from_bitarrays,
    extract_bitarrays_from_databin,
    extract_estimator_results,
    extract_sampler_results,
    is_bitarray_like,
)
from qiskit import QuantumCircuit


# =====================================================================
# Sampler Result Extraction
# =====================================================================


class TestExtractSamplerResults:
    """Tests for extract_sampler_results."""

    def test_single_pub_counts(
        self,
        make_primitive_result,
        make_sampler_pub_result,
    ):
        """Single PUB produces one experiment with correct counts."""
        result = make_primitive_result(
            [make_sampler_pub_result({"00": 480, "11": 520})]
        )
        payload = extract_sampler_results(result)

        assert payload is not None
        exps = payload["experiments"]
        assert len(exps) == 1
        assert exps[0]["counts"] == {"00": 480, "11": 520}
        assert exps[0]["shots"] == 1000

    def test_multi_pub_counts(
        self,
        make_primitive_result,
        make_sampler_pub_result,
    ):
        """Multiple PUBs produce one experiment entry each."""
        result = make_primitive_result(
            [
                make_sampler_pub_result({"0": 600, "1": 400}),
                make_sampler_pub_result(
                    {"00": 250, "01": 250, "10": 250, "11": 250},
                ),
            ]
        )
        payload = extract_sampler_results(result)

        assert len(payload["experiments"]) == 2
        assert payload["experiments"][0]["index"] == 0
        assert payload["experiments"][1]["index"] == 1
        assert payload["experiments"][1]["shots"] == 1000

    def test_none_result_returns_none(self):
        """None result returns None."""
        assert extract_sampler_results(None) is None

    def test_non_iterable_result_returns_none(self):
        """Non-iterable result returns None."""
        assert extract_sampler_results(42) is None


class TestBuildSamplerResultSnapshot:
    """Tests for build_sampler_result_snapshot."""

    def test_success_snapshot_structure(
        self,
        make_primitive_result,
        make_sampler_pub_result,
    ):
        """Successful result produces completed snapshot with items."""
        result = make_primitive_result(
            [make_sampler_pub_result({"00": 512, "11": 512})]
        )
        snapshot = build_sampler_result_snapshot(
            result,
            backend_name="fake_manila",
        )

        assert snapshot.success is True
        assert snapshot.status == "completed"
        assert len(snapshot.items) == 1
        assert snapshot.items[0].counts is not None
        assert snapshot.items[0].counts["counts"] == {"00": 512, "11": 512}
        assert snapshot.items[0].counts["shots"] == 1024
        assert snapshot.metadata["primitive_type"] == "sampler"
        assert snapshot.metadata["backend_name"] == "fake_manila"

    def test_counts_format_present(
        self,
        make_primitive_result,
        make_sampler_pub_result,
    ):
        """Each item carries CountsFormat metadata."""
        result = make_primitive_result([make_sampler_pub_result({"0": 100})])
        snapshot = build_sampler_result_snapshot(result)

        fmt = snapshot.items[0].counts["format"]
        assert fmt["source_sdk"] == "qiskit-ibm-runtime"
        assert fmt["bit_order"] == "cbit0_right"
        assert fmt["transformed"] is False

    def test_none_result_produces_failed_snapshot(self):
        """None result produces failed snapshot with error."""
        snapshot = build_sampler_result_snapshot(None, backend_name="test")

        assert snapshot.success is False
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert snapshot.error.type == "NullResult"

    def test_raw_result_ref_stored(
        self,
        make_primitive_result,
        make_sampler_pub_result,
    ):
        """raw_result_ref is forwarded into snapshot."""
        result = make_primitive_result([make_sampler_pub_result({"0": 100})])
        snapshot = build_sampler_result_snapshot(
            result,
            raw_result_ref="ref://abc",
        )
        assert snapshot.raw_result_ref == "ref://abc"


# =====================================================================
# Estimator Result Extraction
# =====================================================================


class TestExtractEstimatorResults:
    """Tests for extract_estimator_results."""

    def test_single_pub_expectations(
        self,
        make_primitive_result,
        make_estimator_pub_result,
    ):
        """Single PUB produces expectation values and stds."""
        result = make_primitive_result(
            [make_estimator_pub_result(evs=[0.75], stds=[0.01])]
        )
        payload = extract_estimator_results(result)

        assert payload is not None
        exps = payload["experiments"]
        assert len(exps) == 1
        assert exps[0]["index"] == 0
        np.testing.assert_allclose(
            exps[0]["expectation_values"],
            [0.75],
            atol=1e-6,
        )

    def test_multi_observable_pub(
        self,
        make_primitive_result,
        make_estimator_pub_result,
    ):
        """PUB with multiple observables produces array outputs."""
        evs = [0.25, -0.5, 0.8]
        stds = [0.01, 0.02, 0.005]
        result = make_primitive_result([make_estimator_pub_result(evs=evs, stds=stds)])
        payload = extract_estimator_results(result)

        exp = payload["experiments"][0]
        np.testing.assert_allclose(exp["expectation_values"], evs, atol=1e-6)
        np.testing.assert_allclose(exp["standard_deviations"], stds, atol=1e-6)

    def test_none_result_returns_none(self):
        """None result returns None."""
        assert extract_estimator_results(None) is None


class TestBuildEstimatorResultSnapshot:
    """Tests for build_estimator_result_snapshot."""

    def test_snapshot_items_per_observable(
        self,
        make_primitive_result,
        make_estimator_pub_result,
    ):
        """Each observable produces a separate ResultItem with expectation."""
        evs = [0.5, -0.3]
        stds = [0.01, 0.02]
        result = make_primitive_result([make_estimator_pub_result(evs=evs, stds=stds)])
        snapshot = build_estimator_result_snapshot(
            result,
            backend_name="fake_manila",
        )

        assert snapshot.success is True
        assert len(snapshot.items) == 2
        assert snapshot.items[0].expectation is not None
        assert snapshot.items[0].expectation.value == pytest.approx(0.5, abs=1e-6)
        assert snapshot.items[0].expectation.observable_index == 0
        assert snapshot.items[1].expectation.observable_index == 1

    def test_metadata_experiments_structure(
        self,
        make_primitive_result,
        make_estimator_pub_result,
    ):
        """Snapshot metadata contains experiments structure for adapter."""
        evs = [0.5]
        stds = [0.01]
        result = make_primitive_result([make_estimator_pub_result(evs=evs, stds=stds)])
        snapshot = build_estimator_result_snapshot(result)

        exps = snapshot.metadata["experiments"]
        assert len(exps) == 1
        assert exps[0]["expectations"][0]["value"] == pytest.approx(0.5, abs=1e-6)
        assert exps[0]["expectations"][0]["std_error"] == pytest.approx(
            0.01,
            abs=1e-3,
        )

    def test_none_std_preserved_in_metadata(
        self,
        make_primitive_result,
        make_estimator_pub_result,
    ):
        """None std_error is preserved, not dropped."""
        result = make_primitive_result(
            [make_estimator_pub_result(evs=[0.5], stds=None)]
        )
        snapshot = build_estimator_result_snapshot(result)

        exp = snapshot.metadata["experiments"][0]["expectations"][0]
        assert "std_error" in exp

    def test_none_result_produces_failed_snapshot(self):
        """None result produces failed snapshot."""
        snapshot = build_estimator_result_snapshot(None)
        assert snapshot.success is False
        assert snapshot.error.type == "NullResult"


# =====================================================================
# BitArray Extraction
# =====================================================================


class TestBitArrayExtraction:
    """Tests for BitArray-like detection and DataBin extraction."""

    def test_is_bitarray_like_with_get_counts(self):
        """Object with get_counts is bitarray-like."""
        obj = SimpleNamespace(get_counts=lambda: {})
        assert is_bitarray_like(obj) is True

    def test_is_bitarray_like_with_get_bitstrings(self):
        """Object with get_bitstrings is bitarray-like."""
        obj = SimpleNamespace(get_bitstrings=lambda: [])
        assert is_bitarray_like(obj) is True

    def test_is_bitarray_like_plain_object(self):
        """Plain object is not bitarray-like."""
        assert is_bitarray_like({}) is False
        assert is_bitarray_like(None) is False

    def test_extract_from_databin_with_meas(self, make_bitarray):
        """Canonical 'meas' attribute is found first."""
        databin = SimpleNamespace(meas=make_bitarray({"00": 50, "11": 50}))
        result = extract_bitarrays_from_databin(databin)

        assert len(result) == 1
        assert result[0][0] == "meas"

    def test_extract_from_databin_custom_register(self, make_bitarray):
        """Non-meas register names are found via scan."""
        databin = SimpleNamespace(c0=make_bitarray({"0": 100}))
        result = extract_bitarrays_from_databin(databin)

        assert len(result) == 1
        assert result[0][0] == "c0"

    def test_extract_from_none_returns_empty(self):
        """None DataBin returns empty list."""
        assert extract_bitarrays_from_databin(None) == []


class TestCountsFromBitarrays:
    """Tests for counts_from_bitarrays."""

    def test_single_register(self, make_bitarray):
        """Single BitArray converts to counts dict."""
        ba = make_bitarray({"00": 480, "11": 520})
        counts = counts_from_bitarrays([("meas", ba)])
        assert counts == {"00": 480, "11": 520}

    def test_empty_list(self):
        """Empty list returns None."""
        assert counts_from_bitarrays([]) is None

    def test_multi_register_join(self, make_bitarray):
        """Multiple registers are joined via get_bitstrings."""
        ba1 = make_bitarray({"0": 2, "1": 2})
        ba2 = make_bitarray({"0": 2, "1": 2})

        counts = counts_from_bitarrays([("c0", ba1), ("c1", ba2)])
        assert counts is not None
        assert sum(counts.values()) == 4


# =====================================================================
# Observable Counting
# =====================================================================


class TestCountObservables:
    """Tests for _count_observables, including SparsePauliOp."""

    def test_none_returns_none(self):
        assert _count_observables(None) is None

    def test_list_of_observables(self):
        """List returns its length."""
        assert _count_observables(["ZZ", "XX", "YY"]) == 3

    def test_tuple_of_observables(self):
        """Tuple returns its length."""
        assert _count_observables(("ZZ", "XX")) == 2

    def test_empty_list(self):
        assert _count_observables([]) == 0

    def test_operator_with_paulis_returns_one(self):
        """SparsePauliOp-like (has .paulis) returns 1, not len(terms)."""

        class FakeSparsePauliOp:
            paulis = ["ZZ", "XX", "YY"]

            def __len__(self):
                return 3

        assert _count_observables(FakeSparsePauliOp()) == 1

    def test_operator_with_to_matrix_returns_one(self):
        """Pauli-like (has .to_matrix) returns 1."""

        class FakePauli:
            def to_matrix(self):
                return [[1, 0], [0, -1]]

            def __len__(self):
                return 2

        assert _count_observables(FakePauli()) == 1

    def test_array_like_with_len(self):
        """Non-operator with __len__ returns len()."""

        class ObservablesArray:
            def __len__(self):
                return 5

        assert _count_observables(ObservablesArray()) == 5


# =====================================================================
# PUB Utilities
# =====================================================================


class TestMaterializePubs:
    """Tests for materialize_pubs / iter_pubs."""

    def test_none_returns_empty(self):
        assert materialize_pubs(None) == []

    def test_single_circuit(self):
        """Single QuantumCircuit is wrapped in list."""
        qc = QuantumCircuit(1)
        result = materialize_pubs(qc)
        assert len(result) == 1
        assert result[0] is qc

    def test_list_passthrough(self):
        """List is returned as new list."""
        qc = QuantumCircuit(1)
        pubs = [qc]
        result = materialize_pubs(pubs)
        assert result == pubs
        assert result is not pubs

    def test_v2_pub_tuple(self):
        """V2 PUB tuple is wrapped in list."""
        qc = QuantumCircuit(1)
        pub = (qc, [0.5])
        result = materialize_pubs(pub)
        assert len(result) == 1
        assert result[0] is pub

    def test_generator_materialized(self):
        """Generator is consumed into list."""

        def gen():
            for i in range(3):
                yield QuantumCircuit(1, name=f"c{i}")

        result = materialize_pubs(gen())
        assert len(result) == 3


class TestIsV2PubTuple:
    """Tests for V2 PUB tuple detection."""

    def test_circuit_with_params(self):
        qc = QuantumCircuit(1)
        assert is_v2_pub_tuple((qc, [0.5])) is True

    def test_circuit_only(self):
        qc = QuantumCircuit(1)
        assert is_v2_pub_tuple((qc,)) is True

    def test_multi_circuit_tuple_is_not_pub(self):
        """Tuple of circuits is a container, not a PUB."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        assert is_v2_pub_tuple((qc1, qc2)) is False

    def test_empty_tuple(self):
        assert is_v2_pub_tuple(()) is False

    def test_non_tuple(self):
        assert is_v2_pub_tuple([QuantumCircuit(1)]) is False


class TestExtractCircuitsFromPubs:
    """Tests for extract_circuits_from_pubs."""

    def test_list_of_circuits(self):
        circuits = [QuantumCircuit(1), QuantumCircuit(2)]
        result = extract_circuits_from_pubs(circuits)
        assert len(result) == 2

    def test_list_of_tuples(self):
        qc = QuantumCircuit(2)
        result = extract_circuits_from_pubs([(qc, [0.5])])
        assert len(result) == 1
        assert result[0] is qc

    def test_pre_materialized_list(self):
        """Pre-materialized list works without re-materialization."""
        qc = QuantumCircuit(1)
        result = extract_circuits_from_pubs([qc])
        assert len(result) == 1

    def test_dict_pub(self):
        qc = QuantumCircuit(1)
        result = extract_circuits_from_pubs([{"circuit": qc}])
        assert len(result) == 1
        assert result[0] is qc


class TestExtractPubsStructure:
    """Tests for extract_pubs_structure metadata extraction."""

    def test_circuit_pub(self):
        qc = QuantumCircuit(2)
        struct = extract_pubs_structure([qc])
        assert len(struct) == 1
        assert struct[0]["format"] == "circuit"
        assert struct[0]["has_circuit"] is True

    def test_v2_tuple_pub(self):
        qc = QuantumCircuit(2)
        struct = extract_pubs_structure([(qc, [0.5])])
        assert struct[0]["format"] == "v2_pub_tuple"
        assert struct[0]["tuple_len"] == 2

    def test_estimator_observable_count_operator(self):
        """For Estimator, operator-like observables count as 1."""

        class FakeOp:
            paulis = ["ZZ", "XX"]

            def __len__(self):
                return 2

        qc = QuantumCircuit(2)
        struct = extract_pubs_structure(
            [(qc, FakeOp())],
            primitive_type="estimator",
        )
        assert struct[0].get("num_observables") == 1

    def test_estimator_observable_count_list(self):
        """For Estimator, list of observables returns list length."""
        qc = QuantumCircuit(2)
        struct = extract_pubs_structure(
            [(qc, ["ZZ", "XX", "YY"])],
            primitive_type="estimator",
        )
        assert struct[0].get("num_observables") == 3


class TestExtractParameterValues:
    """Tests for extract_parameter_values_from_pubs."""

    def test_no_params(self):
        qc = QuantumCircuit(1)
        result = extract_parameter_values_from_pubs([qc])
        assert result == [None]

    def test_sampler_tuple_params(self):
        """Sampler PUB: (circuit, params, shots) → index 1."""
        qc = QuantumCircuit(1)
        params = np.array([0.5, 1.0])
        result = extract_parameter_values_from_pubs(
            [(qc, params, 100)],
            primitive_type="sampler",
        )
        assert result[0] is params

    def test_estimator_tuple_params(self):
        """Estimator PUB: (circuit, obs, params, prec) → index 2."""
        qc = QuantumCircuit(1)
        params = np.array([0.5])
        result = extract_parameter_values_from_pubs(
            [(qc, "ZZ", params, 0.01)],
            primitive_type="estimator",
        )
        assert result[0] is params

    def test_object_pub_with_parameter_values(self):
        """Object pub with .parameter_values attribute."""
        params = np.array([0.5])
        pub = SimpleNamespace(
            circuit=QuantumCircuit(1),
            parameter_values=params,
        )
        result = extract_parameter_values_from_pubs([pub])
        assert result[0] is params

    def test_dict_pub_params(self):
        qc = QuantumCircuit(1)
        params = [0.5]
        result = extract_parameter_values_from_pubs(
            [{"circuit": qc, "parameter_values": params}],
        )
        assert result[0] is params
