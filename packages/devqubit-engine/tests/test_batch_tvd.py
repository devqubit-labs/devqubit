# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for batch TVD correctness: item_index pairing, FWER noise context, mismatch detection."""

from __future__ import annotations

import json

import pytest
from devqubit_engine.compare.diff import diff_runs
from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.utils.distributions import (
    compute_noise_context,
    compute_noise_context_max,
)


def _batch_artifact(store, experiments: list[dict]) -> ArtifactRef:
    """Helper: store a batch counts artifact and return ArtifactRef."""
    payload = {"experiments": [{"counts": c} for c in experiments]}
    digest = store.put_bytes(json.dumps(payload).encode())
    return ArtifactRef(
        kind="result.counts.json",
        digest=digest,
        media_type="application/json",
        role="results",
    )


class TestBatchTVDMetadata:
    """item_index='all' populates tvd_item_index, tvd_batch_size, tvd_aggregation."""

    def test_batch_sets_aggregation_fields(self, store, run_factory):
        """Batch comparison fills new metadata fields correctly."""
        art_a = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 100, "11": 900},
                {"00": 300, "11": 700},
            ],
        )
        art_b = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},  # TVD=0
                {"00": 900, "11": 100},  # TVD=0.8 (worst)
                {"00": 350, "11": 650},  # TVD=0.05
            ],
        )

        run_a = run_factory(run_id="A", artifacts=[art_a])
        run_b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store, item_index="all")

        assert result.tvd == pytest.approx(0.8)
        assert result.tvd_aggregation == "max"
        assert result.tvd_batch_size == 3
        assert result.tvd_item_index == 1  # item_index of worst pair

    def test_batch_noise_context_uses_bootstrap_max(self, store, run_factory):
        """Batch noise context uses bootstrap_max method, not per-item bootstrap."""
        art_a = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 100, "11": 900},
            ],
        )
        art_b = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 900, "11": 100},
            ],
        )

        run_a = run_factory(run_id="A", artifacts=[art_a])
        run_b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(
            run_a,
            run_b,
            store_a=store,
            store_b=store,
            item_index="all",
            noise_n_boot=200,
        )

        assert result.noise_context is not None
        assert result.noise_context.method == "bootstrap_max"
        assert result.noise_context.n_boot == 200

    def test_batch_tvd_fields_in_serialization(self, store, run_factory):
        """New batch TVD fields appear in to_dict() output."""
        art_a = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 100, "11": 900},
            ],
        )
        art_b = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 900, "11": 100},
            ],
        )

        run_a = run_factory(run_id="A", artifacts=[art_a])
        run_b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store, item_index="all")
        d = result.to_dict()

        assert d["tvd_aggregation"] == "max"
        assert d["tvd_batch_size"] == 2
        assert d["tvd_item_index"] == 1

    def test_single_item_does_not_set_batch_fields(
        self, store, run_factory, counts_artifact_factory
    ):
        """Single-item comparison keeps default aggregation fields."""
        art = counts_artifact_factory({"00": 500, "11": 500})
        run_a = run_factory(run_id="A", artifacts=[art])
        run_b = run_factory(run_id="B", artifacts=[art])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store)

        assert result.tvd_aggregation == "single"
        assert result.tvd_batch_size == 1
        assert result.tvd_item_index is None
        # Should not appear in serialization
        assert "tvd_aggregation" not in result.to_dict()


class TestBatchMismatch:
    """Mismatched batch item sets produce a warning and skip TVD."""

    def test_different_batch_sizes_warns_with_indices(self, store, run_factory):
        """Envelopes with different item counts produce a detailed warning."""
        art_a = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 300, "11": 700},
                {"00": 400, "11": 600},
            ],
        )
        art_b = _batch_artifact(
            store,
            [
                {"00": 500, "11": 500},
                {"00": 300, "11": 700},
            ],
        )

        run_a = run_factory(run_id="A", artifacts=[art_a])
        run_b = run_factory(run_id="B", artifacts=[art_b])

        result = diff_runs(run_a, run_b, store_a=store, store_b=store, item_index="all")

        assert result.tvd is None
        assert any("mismatch" in w.lower() for w in result.warnings)
        # Warning should mention which indices are missing
        mismatch_warn = [w for w in result.warnings if "mismatch" in w.lower()][0]
        assert "missing_in_candidate" in mismatch_warn


class TestNoiseContextMax:
    """compute_noise_context_max produces FWER-correct thresholds."""

    def test_max_p95_at_least_as_high_as_any_single(self):
        """max-TVD p95 threshold >= any individual item's p95."""
        pairs = [
            ({"00": 500, "11": 500}, {"00": 480, "11": 520}),
            ({"00": 300, "11": 700}, {"00": 320, "11": 680}),
            ({"00": 600, "11": 400}, {"00": 580, "11": 420}),
        ]

        # Individual p95 for each pair
        individual_p95s = []
        for ca, cb in pairs:
            ctx = compute_noise_context(ca, cb, n_boot=500, seed=42)
            individual_p95s.append(ctx.noise_p95)

        # Max-TVD p95 (FWER-correct)
        max_ctx = compute_noise_context_max(pairs, tvd_max=0.02, n_boot=500, seed=42)

        assert max_ctx.method == "bootstrap_max"
        assert (
            max_ctx.noise_p95 >= max(individual_p95s) - 1e-9
        )  # tolerance for numerics

    def test_reproducible_with_seed(self):
        """Same seed gives deterministic max-TVD results."""
        pairs = [
            ({"00": 500, "11": 500}, {"00": 480, "11": 520}),
            ({"00": 300, "11": 700}, {"00": 320, "11": 680}),
        ]

        ctx1 = compute_noise_context_max(pairs, 0.02, n_boot=200, seed=99)
        ctx2 = compute_noise_context_max(pairs, 0.02, n_boot=200, seed=99)

        assert ctx1.noise_p95 == ctx2.noise_p95
        assert ctx1.p_value == ctx2.p_value

    def test_empty_pairs_returns_heuristic(self):
        """No pairs → heuristic fallback, not a crash."""
        ctx = compute_noise_context_max([], tvd_max=0.5)

        assert ctx.method == "heuristic"
        assert ctx.n_boot == 0
