# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for query language parsing and matching."""

from __future__ import annotations

import pytest
from devqubit_engine.query import (
    Op,
    QueryParseError,
    matches_query,
    parse_query,
    search_records,
)


class TestQueryParsing:
    """Tests for query string parsing."""

    def test_simple_equality(self):
        """Parse simple equality condition."""
        q = parse_query("params.shots = 1000")
        assert len(q.conditions) == 1
        assert q.conditions[0].field == "params.shots"
        assert q.conditions[0].op == Op.EQ
        assert q.conditions[0].value == 1000

    def test_numeric_comparison(self):
        """Parse numeric comparison operators."""
        q = parse_query("metric.fidelity > 0.95")
        assert q.conditions[0].op == Op.GT
        assert q.conditions[0].value == 0.95

        q = parse_query("metric.error <= 0.01")
        assert q.conditions[0].op == Op.LE

    def test_string_equality(self):
        """Parse string value."""
        q = parse_query("status = COMPLETED")
        assert q.conditions[0].value == "COMPLETED"

    def test_quoted_string(self):
        """Parse quoted string with spaces."""
        q = parse_query('tags.name = "my experiment"')
        assert q.conditions[0].value == "my experiment"

    def test_contains_operator(self):
        """Parse contains (~) operator."""
        q = parse_query("tags.device ~ ibm")
        assert q.conditions[0].op == Op.CONTAINS
        assert q.conditions[0].value == "ibm"

    def test_and_conditions(self):
        """Parse AND-joined conditions."""
        q = parse_query("metric.fidelity > 0.9 and params.shots = 1000")
        assert len(q.conditions) == 2
        assert q.conditions[0].field == "metric.fidelity"
        assert q.conditions[1].field == "params.shots"

    def test_multiple_and(self):
        """Parse multiple AND conditions."""
        q = parse_query("a = 1 and b = 2 and c = 3")
        assert len(q.conditions) == 3

    def test_field_preserved_as_is(self):
        """Field names are preserved as written (aliases handled at match time)."""
        # Aliases are handled during matching, not parsing
        q = parse_query("metrics.x = 1")
        assert q.conditions[0].field == "metrics.x"

        q = parse_query("metric.x = 1")
        assert q.conditions[0].field == "metric.x"

    def test_invalid_query_no_operator(self):
        """Invalid query without operator raises error."""
        with pytest.raises(QueryParseError):
            parse_query("params.shots")

    def test_invalid_query_or_not_supported(self):
        """OR operator raises clear error."""
        with pytest.raises(QueryParseError, match="OR not supported"):
            parse_query("a = 1 or b = 2")

    def test_empty_query_returns_empty(self):
        """Empty query returns Query with no conditions."""
        # Empty query is valid - returns all records
        q = parse_query("")
        assert len(q.conditions) == 0

        q = parse_query("   ")
        assert len(q.conditions) == 0


class TestQueryMatching:
    """Tests for matching queries against records."""

    @pytest.fixture
    def sample_record(self):
        """Sample run record for testing."""
        return {
            "run_id": "TEST123",
            "project": {"name": "test_project"},
            "adapter": "qiskit",
            "info": {"status": "COMPLETED"},
            "data": {
                "params": {"shots": 1000, "seed": 42},
                "metrics": {"fidelity": 0.95, "error": 0.05},
                "tags": {"device": "ibm_kyoto", "version": "1.0"},
            },
            "backend": {"name": "ibm_kyoto"},
            "fingerprints": {"run": "abc123"},
        }

    def test_match_param_equals(self, sample_record):
        """Match parameter equality."""
        q = parse_query("params.shots = 1000")
        assert matches_query(sample_record, q)

        q = parse_query("params.shots = 2000")
        assert not matches_query(sample_record, q)

    def test_match_metric_comparison(self, sample_record):
        """Match metric comparisons."""
        assert matches_query(sample_record, parse_query("metric.fidelity > 0.9"))
        assert matches_query(sample_record, parse_query("metric.fidelity >= 0.95"))
        assert not matches_query(sample_record, parse_query("metric.fidelity > 0.95"))
        assert matches_query(sample_record, parse_query("metric.error < 0.1"))

    def test_match_tag_contains(self, sample_record):
        """Match tag contains."""
        assert matches_query(sample_record, parse_query("tags.device ~ ibm"))
        assert matches_query(sample_record, parse_query("tags.device ~ kyoto"))
        assert not matches_query(sample_record, parse_query("tags.device ~ google"))

    def test_match_status(self, sample_record):
        """Match status field."""
        assert matches_query(sample_record, parse_query("status = COMPLETED"))
        assert not matches_query(sample_record, parse_query("status = FAILED"))

    def test_match_project(self, sample_record):
        """Match project field."""
        assert matches_query(sample_record, parse_query("project = test_project"))

    def test_match_backend(self, sample_record):
        """Match backend field."""
        assert matches_query(sample_record, parse_query("backend = ibm_kyoto"))

    def test_match_multiple_conditions(self, sample_record):
        """All conditions must match (AND logic)."""
        q = parse_query("metric.fidelity > 0.9 and params.shots = 1000")
        assert matches_query(sample_record, q)

        q = parse_query("metric.fidelity > 0.9 and params.shots = 2000")
        assert not matches_query(sample_record, q)

    def test_match_not_equals(self, sample_record):
        """Match not-equals operator."""
        assert matches_query(sample_record, parse_query("status != FAILED"))
        assert not matches_query(sample_record, parse_query("status != COMPLETED"))

    def test_missing_field_no_match(self, sample_record):
        """Missing field doesn't match."""
        assert not matches_query(sample_record, parse_query("params.nonexistent = 1"))

    def test_empty_query_matches_all(self, sample_record):
        """Empty query matches all records."""
        q = parse_query("")
        assert matches_query(sample_record, q)


class TestSearchRecords:
    """Tests for searching and sorting records."""

    @pytest.fixture
    def records(self):
        """Multiple records for search testing."""
        return [
            {
                "run_id": "RUN1",
                "created_at": "2024-01-01T00:00:00Z",
                "data": {
                    "params": {"shots": 1000},
                    "metrics": {"fidelity": 0.90},
                    "tags": {},
                },
            },
            {
                "run_id": "RUN2",
                "created_at": "2024-01-02T00:00:00Z",
                "data": {
                    "params": {"shots": 2000},
                    "metrics": {"fidelity": 0.95},
                    "tags": {},
                },
            },
            {
                "run_id": "RUN3",
                "created_at": "2024-01-03T00:00:00Z",
                "data": {
                    "params": {"shots": 1000},
                    "metrics": {"fidelity": 0.85},
                    "tags": {},
                },
            },
        ]

    def test_search_filters(self, records):
        """Search filters records."""
        results = search_records(records, "params.shots = 1000")
        assert len(results) == 2
        assert all(r["data"]["params"]["shots"] == 1000 for r in results)

    def test_search_sort_by_metric(self, records):
        """Search sorts by metric."""
        results = search_records(
            records,
            "params.shots = 1000",
            sort_by="metric.fidelity",
            descending=True,
        )
        assert results[0]["run_id"] == "RUN1"  # 0.90
        assert results[1]["run_id"] == "RUN3"  # 0.85

    def test_search_limit(self, records):
        """Search respects limit."""
        results = search_records(records, "metric.fidelity > 0.8", limit=2)
        assert len(results) == 2

    def test_search_ascending(self, records):
        """Search can sort ascending."""
        results = search_records(
            records,
            "metric.fidelity > 0",
            sort_by="metric.fidelity",
            descending=False,
        )
        assert results[0]["run_id"] == "RUN3"  # 0.85 (lowest)
