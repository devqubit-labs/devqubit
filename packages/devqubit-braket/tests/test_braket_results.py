# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for results extraction."""

from __future__ import annotations

from types import SimpleNamespace

from devqubit_braket.results import (
    _extract_program_set_experiments,
    _to_counts_dict,
    extract_counts_payload,
    extract_measurement_counts,
)


# ============================================================================
# Counts Helpers Tests
# ============================================================================


class TestToCountsDict:
    """Tests for _to_counts_dict helper."""

    def test_dict_passthrough(self):
        """Standard dict is normalized to str keys and int values."""
        result = _to_counts_dict({"00": 50, "11": 50})
        assert result == {"00": 50, "11": 50}

    def test_counter_like(self):
        """Counter-like object is converted to dict."""
        from collections import Counter

        c = Counter({"00": 30, "11": 70})
        result = _to_counts_dict(c)
        assert result == {"00": 30, "11": 70}

    def test_none_returns_none(self):
        assert _to_counts_dict(None) is None

    def test_non_dict_returns_none(self):
        """Non-iterable returns None."""
        assert _to_counts_dict(42) is None

    def test_canonicalize_reverses_bitstrings(self):
        """canonicalize=True reverses bitstring keys."""
        result = _to_counts_dict({"01": 50, "10": 50}, canonicalize=True)
        assert result == {"10": 50, "01": 50}


# ============================================================================
# Measurements Tests
# ============================================================================


class TestExtractMeasurementCounts:
    """Tests for extract_measurement_counts."""

    def test_from_measurement_counts_attr(self):
        """Extracts from measurement_counts attribute."""
        obj = SimpleNamespace(measurement_counts={"00": 50, "11": 50})
        assert extract_measurement_counts(obj) == {"00": 50, "11": 50}

    def test_from_counts_attr(self):
        """Extracts from counts attribute (fallback key)."""
        obj = SimpleNamespace(counts={"00": 30, "11": 70})
        assert extract_measurement_counts(obj) == {"00": 30, "11": 70}

    def test_from_callable_measurement_counts(self):
        """Handles callable measurement_counts."""

        class Result:
            def measurement_counts(self):
                return {"00": 25, "11": 75}

        assert extract_measurement_counts(Result()) == {"00": 25, "11": 75}

    def test_none_returns_none(self):
        assert extract_measurement_counts(None) is None

    def test_broken_attr_returns_none(self):
        """Handles exceptions in attribute access."""

        class Broken:
            @property
            def measurement_counts(self):
                raise RuntimeError("boom")

        assert extract_measurement_counts(Broken()) is None

    def test_canonicalize(self):
        """canonicalize=True reverses bitstrings."""
        obj = SimpleNamespace(measurement_counts={"01": 50, "10": 50})
        result = extract_measurement_counts(obj, canonicalize=True)
        assert result == {"10": 50, "01": 50}


# ============================================================================
# Counts Extraction Tests
# ============================================================================


class TestExtractCountsPayload:
    """Tests for extract_counts_payload."""

    def test_single_result(self):
        """Single result produces experiments list with one entry."""
        obj = SimpleNamespace(measurement_counts={"00": 50, "11": 50})
        payload = extract_counts_payload(obj)

        assert payload is not None
        assert len(payload["experiments"]) == 1
        assert payload["experiments"][0]["index"] == 0
        assert payload["experiments"][0]["counts"] == {"00": 50, "11": 50}

    def test_none_returns_none(self):
        assert extract_counts_payload(None) is None

    def test_no_counts_returns_none(self):
        """Object with no counts attributes returns None."""
        assert extract_counts_payload(SimpleNamespace()) is None

    def test_program_set_result(self):
        """Handles nested ProgramSet result structure."""
        measured = SimpleNamespace(counts={"00": 50, "11": 50})
        composite = SimpleNamespace(entries=[measured, measured])
        result = SimpleNamespace(entries=[composite])

        payload = extract_counts_payload(result)

        assert payload is not None
        assert len(payload["experiments"]) == 2
        for i, exp in enumerate(payload["experiments"]):
            assert exp["index"] == i
            assert "program_index" in exp
            assert "executable_index" in exp


# ============================================================================
# ProgramSet Extraction Tests
# ============================================================================


class TestExtractProgramSetExperiments:
    """Tests for _extract_program_set_experiments."""

    def test_non_program_set_returns_none(self):
        """Objects without .entries list return None."""
        assert _extract_program_set_experiments(SimpleNamespace(), False) is None
        assert (
            _extract_program_set_experiments(SimpleNamespace(entries="not_list"), False)
            is None
        )

    def test_empty_entries(self):
        assert (
            _extract_program_set_experiments(SimpleNamespace(entries=[]), False) is None
        )

    def test_inner_entries_not_list_skipped(self):
        """Composite entries without inner .entries list are skipped."""
        composite = SimpleNamespace(entries="not_a_list")
        result = SimpleNamespace(entries=[composite])
        assert _extract_program_set_experiments(result, False) is None

    def test_no_counts_entries_skipped(self):
        """Entries with no extractable counts are skipped."""
        measured = SimpleNamespace()  # no counts attribute
        composite = SimpleNamespace(entries=[measured])
        result = SimpleNamespace(entries=[composite])
        assert _extract_program_set_experiments(result, False) is None
