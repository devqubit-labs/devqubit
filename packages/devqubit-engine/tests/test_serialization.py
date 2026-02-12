# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit_engine.utils.serialization — deterministic JSON, edge cases."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from devqubit_engine.utils.serialization import (
    _normalize_float,
    json_dumps,
    to_jsonable,
)


class TestToJsonable:
    """Conversion of arbitrary Python objects to JSON-safe types."""

    def test_primitives_pass_through(self):
        assert to_jsonable(None) is None
        assert to_jsonable("hello") == "hello"
        assert to_jsonable(42) == 42
        assert to_jsonable(True) is True
        assert to_jsonable(False) is False
        assert to_jsonable(3.14) == 3.14

    def test_bool_not_treated_as_int(self):
        """bool is a subclass of int — make sure True != 1 in output type."""
        result = to_jsonable(True)
        assert result is True
        assert not isinstance(result, int) or isinstance(result, bool)

    def test_dict_keys_stringified(self):
        assert to_jsonable({1: "a", 2: "b"}) == {"1": "a", "2": "b"}

    def test_nested_dict(self):
        obj = {"a": {"b": {"c": 1}}}
        assert to_jsonable(obj) == {"a": {"b": {"c": 1}}}

    def test_list_and_tuple(self):
        assert to_jsonable([1, 2, 3]) == [1, 2, 3]
        assert to_jsonable((1, 2, 3)) == [1, 2, 3]

    def test_set_sorted_for_determinism(self):
        result = to_jsonable({3, 1, 2})
        assert result == [1, 2, 3]

    def test_frozenset_sorted(self):
        result = to_jsonable(frozenset(["c", "a", "b"]))
        assert result == ["a", "b", "c"]

    def test_nan_and_infinity(self):
        nan_result = to_jsonable(float("nan"))
        assert math.isnan(nan_result)

        assert to_jsonable(float("inf")) == float("inf")
        assert to_jsonable(float("-inf")) == float("-inf")

    def test_float_normalization(self):
        result = to_jsonable(0.1 + 0.2, normalize_floats=True)
        assert result == 0.3

    def test_dataclass_converted(self):
        @dataclass
        class Point:
            x: int
            y: int

        result = to_jsonable(Point(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_object_with_to_dict(self):
        class Obj:
            def to_dict(self):
                return {"key": "value"}

        assert to_jsonable(Obj()) == {"key": "value"}

    def test_max_depth_truncation(self):
        """Deeply nested structures get truncated, not crash."""
        result = to_jsonable({"a": {"a": 1}}, max_depth=1)
        assert "__truncated__" in str(result)

    def test_bytes_fallback_to_repr(self):
        """Non-serializable types fall back to repr."""
        result = to_jsonable(b"raw bytes")
        assert "__repr__" in result or isinstance(result, str)


class TestNormalizeFloat:
    def test_zero(self):
        assert _normalize_float(0.0) == 0.0

    def test_nan_preserved(self):
        assert math.isnan(_normalize_float(float("nan")))

    def test_inf_preserved(self):
        assert _normalize_float(float("inf")) == float("inf")
        assert _normalize_float(float("-inf")) == float("-inf")

    def test_precision(self):
        # 0.30000000000000004 should normalize to 0.3
        assert _normalize_float(0.1 + 0.2, precision=15) == 0.3

    def test_non_float_passthrough(self):
        assert _normalize_float(42) == 42


class TestJsonDumps:
    def test_deterministic_key_order(self):
        """Keys are always sorted regardless of insertion order."""
        a = json_dumps({"z": 1, "a": 2})
        b = json_dumps({"a": 2, "z": 1})
        assert a == b

    def test_compact_mode(self):
        """compact=True produces minimal output with float normalization."""
        result = json_dumps({"x": 0.1 + 0.2}, compact=True)
        parsed = json.loads(result)
        assert parsed["x"] == 0.3
        # No whitespace
        assert " " not in result

    def test_non_compact_with_indent(self):
        result = json_dumps({"a": 1}, indent=2)
        assert "\n" in result

    def test_roundtrip_preserves_types(self):
        """json_dumps → json.loads preserves int, float, str, list, dict, None, bool."""
        original = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "none": None,
            "bool": False,
        }
        roundtripped = json.loads(json_dumps(original))
        assert roundtripped == original

    def test_compact_determinism_for_fingerprinting(self):
        """Same object always produces identical compact output (critical for hashes)."""
        obj = {"params": {"shots": 1024, "opt_level": 3}, "backend": "sim"}
        a = json_dumps(obj, compact=True)
        b = json_dumps(obj, compact=True)
        assert a == b

    def test_separators_no_trailing_spaces(self):
        """Compact output uses `,` and `:` without spaces."""
        result = json_dumps({"a": 1, "b": 2}, compact=True)
        assert result == '{"a":1,"b":2}'
