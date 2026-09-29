"""
Unit tests for the parts of the specification that are easy to get subtly wrong.

None of these need the dataset. Run with: pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from erf import config as cfg
from erf import spec


class TestGini:
    def test_perfect_ranking_beats_random(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 1000.0, size=5000)
        assert spec.gini(y, y) > spec.gini(y, rng.permutation(y))

    def test_reversed_prediction_flips_the_sign(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(2.0, 1000.0, size=2000)
        assert spec.gini(y, -y) == pytest.approx(-spec.gini(y, y), abs=1e-6)

    def test_constant_prediction_is_near_zero(self):
        rng = np.random.default_rng(2)
        y = rng.gamma(2.0, 1000.0, size=4000)
        assert abs(spec.gini(y, np.ones_like(y))) < 0.05

    def test_invariant_to_monotone_rescaling(self):
        rng = np.random.default_rng(3)
        y = rng.gamma(2.0, 1000.0, size=3000)
        p = rng.gamma(2.0, 1000.0, size=3000)
        assert spec.gini(y, p) == pytest.approx(spec.gini(y, 3 * p + 10),
                                                abs=1e-9)


class TestBands:
    def test_intervals_are_right_closed(self):
        # The ERF is integer-valued, so this is the distinction that matters.
        # A left-closed cut disagrees with the published banding on 56% of rows.
        assert spec.bands(pd.Series([0.0]))[0] == "-2to0"
        assert spec.bands(pd.Series([2.0]))[0] == "0to2"
        assert spec.bands(pd.Series([-6.0]))[0] == "<-6"

    def test_covers_the_full_range(self):
        got = spec.bands(pd.Series([-30.0, -5.0, -1.0, 1.0, 3.0, 5.0, 30.0]))
        assert got.isna().sum() == 0

    def test_band_order_matches_config(self):
        vals = pd.Series([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        assert list(spec.bands(vals)) == cfg.BAND_ORDER


class TestHingeBasis:
    @staticmethod
    def hinges(erf):
        return {f"erf_hinge_{k:+d}".replace("+", "p").replace("-", "m"):
                max(erf - k, 0) for k in cfg.ERF_KNOTS}

    def test_five_knots_not_seven(self):
        assert cfg.ERF_KNOTS == [-4, -2, 0, 2, 4]
        assert len(cfg.HINGE_COLS) == 5

    def test_below_all_knots_is_all_zero(self):
        assert all(v == 0 for v in self.hinges(-10.0).values())

    def test_values_at_a_worked_point(self):
        h = self.hinges(2.0)
        assert h["erf_hinge_m4"] == 6
        assert h["erf_hinge_m2"] == 4
        assert h["erf_hinge_p0"] == 2
        assert h["erf_hinge_p2"] == 0
        assert h["erf_hinge_p4"] == 0

    def test_column_names_match_config(self):
        assert sorted(self.hinges(1.0)) == sorted(cfg.HINGE_COLS)


class TestFormulae:
    def test_baseline_is_not_zone_only(self):
        # The error that made an earlier round of this work unreproducible.
        assert "occupancy_group" in spec.F_BASELINE
        assert "age_at_loss" in spec.F_BASELINE

    def test_elevation_model_nests_the_baseline(self):
        assert spec.F_ELEVATION.startswith(spec.F_BASELINE)

    def test_elevation_model_adds_the_expected_terms(self):
        added = spec.F_ELEVATION[len(spec.F_BASELINE):]
        assert added.count("erf_hinge") == 5
        assert "erf:C(flood_zone_group)" in added


class TestConfig:
    def test_post_firm_excluded_from_boosting(self):
        # Not the zero-variance field it is sometimes assumed to be, but
        # excluding it is what reproduces the published figure.
        assert "post_firm_bin" not in cfg.XGB_FEATURES
        assert len(cfg.XGB_FEATURES) == 9

    def test_tweedie_power_is_consistent(self):
        assert cfg.TWEEDIE_POWER == 1.5
        assert cfg.XGB_PARAMS["tweedie_variance_power"] == 1.5

    def test_config_alias_does_not_shadow_patsy(self):
        # spec.py must not bind the name C, which patsy needs for C(...) in
        # formulas evaluated against that module's namespace.
        assert not hasattr(spec, "C") or callable(getattr(spec, "C"))
