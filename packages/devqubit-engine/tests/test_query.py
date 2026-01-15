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

    def test_equality_and_comparison(self):
        """Parse equality and comparison operators."""
        q = parse_query("params.shots = 1000")
        assert q.conditions[0].field == "params.shots"
        assert q.conditions[0].op == Op.EQ
        assert q.conditions[0].value == 1000

        q = parse_query("metric.fidelity > 0.95")
        assert q.conditions[0].op == Op.GT
        assert q.conditions[0].value == 0.95

        q = parse_query("metric.error <= 0.01")
        assert q.conditions[0].op == Op.LE

    def test_contains_operator(self):
        """Parse contains (~) operator for substring matching."""
        q = parse_query("tags.device ~ ibm")
        assert q.conditions[0].op == Op.CONTAINS
        assert q.conditions[0].value == "ibm"

    def test_and_conditions(self):
        """Parse multiple AND-joined conditions."""
        q = parse_query(
            "metric.fidelity > 0.9 and params.shots = 1000 and status = FINISHED"
        )
        assert len(q.conditions) == 3

    def test_quoted_string_with_spaces(self):
        """Parse quoted string values."""
        q = parse_query('tags.name = "my experiment"')
        assert q.conditions[0].value == "my experiment"

    def test_empty_query(self):
        """Empty query returns no conditions (matches all)."""
        assert len(parse_query("").conditions) == 0
        assert len(parse_query("   ").conditions) == 0

    def test_or_not_supported(self):
        """OR operator raises clear error."""
        with pytest.raises(QueryParseError, match="OR not supported"):
            parse_query("a = 1 or b = 2")

    def test_invalid_query_raises(self):
        """Invalid query syntax raises QueryParseError."""
        with pytest.raises(QueryParseError):
            parse_query("params.shots")  # No operator


class TestQueryMatching:
    """Tests for matching queries against records."""

    @pytest.fixture
    def sample_record(self):
        """Sample run record for testing."""
        return {
            "run_id": "TEST123",
            "project": {"name": "test_project"},
            "info": {"status": "FINISHED"},
            "data": {
                "params": {"shots": 1000, "seed": 42},
                "metrics": {"fidelity": 0.95, "error": 0.05},
                "tags": {"device": "ibm_kyoto"},
            },
            "backend": {"name": "ibm_kyoto"},
        }

    def test_param_and_metric_matching(self, sample_record):
        """Match params and metrics with various operators."""
        assert matches_query(sample_record, parse_query("params.shots = 1000"))
        assert not matches_query(sample_record, parse_query("params.shots = 2000"))
        assert matches_query(sample_record, parse_query("metric.fidelity > 0.9"))
        assert matches_query(sample_record, parse_query("metric.fidelity >= 0.95"))
        assert not matches_query(sample_record, parse_query("metric.fidelity > 0.95"))

    def test_contains_matching(self, sample_record):
        """Match substring contains."""
        assert matches_query(sample_record, parse_query("tags.device ~ ibm"))
        assert matches_query(sample_record, parse_query("tags.device ~ kyoto"))
        assert not matches_query(sample_record, parse_query("tags.device ~ google"))

    def test_special_fields(self, sample_record):
        """Match special fields: status, project, backend."""
        assert matches_query(sample_record, parse_query("status = FINISHED"))
        assert matches_query(sample_record, parse_query("project = test_project"))
        assert matches_query(sample_record, parse_query("backend = ibm_kyoto"))

    def test_multiple_conditions_and_logic(self, sample_record):
        """All conditions must match (AND logic)."""
        assert matches_query(
            sample_record,
            parse_query("metric.fidelity > 0.9 and params.shots = 1000"),
        )
        assert not matches_query(
            sample_record,
            parse_query("metric.fidelity > 0.9 and params.shots = 2000"),
        )

    def test_empty_query_matches_all(self, sample_record):
        """Empty query matches all records."""
        assert matches_query(sample_record, parse_query(""))

    def test_missing_field_no_match(self, sample_record):
        """Missing field doesn't match."""
        assert not matches_query(
            sample_record,
            parse_query("params.nonexistent = 1"),
        )


class TestSearchRecords:
    """Tests for searching and sorting record collections."""

    @pytest.fixture
    def records(self):
        """Multiple records for search testing."""
        return [
            {
                "run_id": "RUN1",
                "data": {
                    "params": {"shots": 1000},
                    "metrics": {"fidelity": 0.90},
                    "tags": {},
                },
            },
            {
                "run_id": "RUN2",
                "data": {
                    "params": {"shots": 2000},
                    "metrics": {"fidelity": 0.95},
                    "tags": {},
                },
            },
            {
                "run_id": "RUN3",
                "data": {
                    "params": {"shots": 1000},
                    "metrics": {"fidelity": 0.85},
                    "tags": {},
                },
            },
        ]

    def test_search_filters_records(self, records):
        """Search filters records by query."""
        results = search_records(records, "params.shots = 1000")

        assert len(results) == 2
        assert all(r["data"]["params"]["shots"] == 1000 for r in results)

    def test_search_with_sort_and_limit(self, records):
        """Search sorts and limits results."""
        results = search_records(
            records,
            "params.shots = 1000",
            sort_by="metric.fidelity",
            descending=True,
            limit=1,
        )

        assert len(results) == 1
        assert results[0]["run_id"] == "RUN1"  # fidelity 0.90 > 0.85
