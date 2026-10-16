"""
Unit tests for the TabPFN subsample ensemble.

The partitioning is the part that is easy to get subtly wrong, and it does not
need torch, so it is tested directly. The one test that does need torch is
skipped where torch is absent, which keeps the rest of the suite runnable on a
machine that has not installed it.

Run with: pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from erf import config as cfg
from erf import tabpfn_model as tp


class TestSubsampleIndices:
    def test_members_are_disjoint_when_the_data_allows(self):
        rng = np.random.default_rng(0)
        members = tp.subsample_indices(100_000, 10_000, 4, rng)
        assert [len(m) for m in members] == [10_000] * 4
        pooled = np.concatenate(members)
        assert len(set(pooled.tolist())) == 40_000

    def test_every_index_is_in_range(self):
        rng = np.random.default_rng(1)
        for n, ctx, k in [(100_000, 10_000, 4), (25_000, 10_000, 4),
                          (12_000, 10_000, 3)]:
            for m in tp.subsample_indices(n, ctx, k, rng):
                assert m.min() >= 0 and m.max() < n

    def test_overlapping_members_still_differ(self):
        # 25,000 rows cannot give four disjoint blocks of 10,000, so the
        # fallback draws independently. They must not come back identical.
        rng = np.random.default_rng(2)
        members = tp.subsample_indices(25_000, 10_000, 4, rng)
        assert [len(m) for m in members] == [10_000] * 4
        assert not np.array_equal(np.sort(members[0]), np.sort(members[1]))

    def test_no_duplicate_rows_inside_a_member(self):
        rng = np.random.default_rng(3)
        for m in tp.subsample_indices(25_000, 10_000, 4, rng):
            assert len(set(m.tolist())) == len(m)

    def test_a_training_set_below_the_context_is_used_whole(self):
        rng = np.random.default_rng(4)
        members = tp.subsample_indices(6_000, 10_000, 4, rng)
        assert len(members) == 1
        assert np.array_equal(members[0], np.arange(6_000))

    def test_partition_is_deterministic_given_the_seed(self):
        a = tp.subsample_indices(50_000, 10_000, 4, np.random.default_rng(7))
        b = tp.subsample_indices(50_000, 10_000, 4, np.random.default_rng(7))
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


class TestConfig:
    def test_features_match_gradient_boosting(self):
        # The comparison only isolates the model if the inputs are identical.
        assert cfg.TABPFN_FEATURES == cfg.XGB_FEATURES

    def test_the_default_run_covers_a_sensible_share_of_training_rows(self):
        covered = cfg.TABPFN_MEMBERS * cfg.TABPFN_CONTEXT
        assert covered <= cfg.TARGETS["train"], "members would have to overlap"
        assert covered / cfg.TARGETS["train"] > 0.3


class TestPredict:
    def test_prediction_is_chunked_safely(self):
        # PREDICT_CHUNK must divide the work without dropping or duplicating
        # rows, including on a final partial block.
        n, out = 20_001, []
        for start in range(0, n, tp.PREDICT_CHUNK):
            out.append((start, min(start + tp.PREDICT_CHUNK, n)))
        assert out[0][0] == 0 and out[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(out, out[1:]))

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
        reason="torch is not installed")
    def test_ensemble_averaging_runs_end_to_end(self):
        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame({
            "erf": rng.normal(0, 3, n),
            "dem_elevation_m": rng.normal(5, 2, n),
            "slope_deg": rng.uniform(0, 5, n),
            "aspect_deg": rng.uniform(0, 360, n),
            "terrain_ruggedness": rng.uniform(0, 2, n),
            "age_at_loss": rng.integers(0, 80, n).astype(float),
            "num_floors": rng.integers(1, 3, n).astype(float),
            "flood_zone_group": rng.choice(["A", "V", "X"], n),
            "occupancy_group": rng.choice(["1", "2"], n),
        })
        df["loss_building"] = np.exp(3 - 0.2 * df["erf"] + rng.normal(0, .3, n))
        train, test = df.iloc[:400], df.iloc[400:]
        p = tp.predict(train, test, n_context=200, n_members=2, seed=0)
        assert p.shape == (len(test),)
        assert np.all(np.isfinite(p))
        # The response is strictly positive and the signal is strong, so a
        # working ensemble should rank it the right way round.
        from erf import spec
        assert spec.gini(test["loss_building"].to_numpy(float), p) > 0.1
