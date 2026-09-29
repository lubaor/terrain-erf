"""Paths and constants. Nothing here is specific to one machine."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root is the parent of this package. Override with ERF_ROOT if you keep
# the data somewhere else, for example on a fast local disk.
ROOT = Path(os.environ.get("ERF_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
CHECKPOINTS = OUTPUTS / "checkpoints"

for _d in (DATA, OUTPUTS, FIGURES, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data files
MODELLING = DATA / "nfip_with_erf.parquet"          # the analysis dataset
CLAIMS_RAW = DATA / "la_claims.parquet"             # OpenFEMA, Louisiana
DEM = DATA / "louisiana_dem_90m.tif"                # USGS 3DEP tile
TPI = DATA / "claims_tpi.parquet"                   # derived, script 05

# ------------------------------------------------------------ specification
# Every constant below is part of the published specification. Changing one
# changes the headline numbers, so they live here rather than in the scripts.
SEED = 42                    # the reference split
TEST_SIZE = 0.2
N_REPEATS = 20               # repeated splits for interval estimates
TWEEDIE_POWER = 1.5
MAXITER = 100

ERF_KNOTS = [-4, -2, 0, 2, 4]                       # hinge spline, five knots
HINGE_COLS = ["erf_hinge_m4", "erf_hinge_m2", "erf_hinge_p0",
              "erf_hinge_p2", "erf_hinge_p4"]

BAND_ORDER = ["<-6", "-6to-4", "-4to-2", "-2to0",
              "0to2", "2to4", "4to6", ">6"]
BAND_EDGES = [float("-inf"), -6, -4, -2, 0, 2, 4, 6, float("inf")]

# Credibility. The ad-hoc estimator reproduces the originally published table;
# the uniform-exposure estimator is the one a reader would derive independently.
# They differ by at most 0.007 in any relativity.
K_ADHOC = 19.7484
K_TEXTBOOK = 39.5266

DPI = 150

# ------------------------------------------------------------------ targets
# The published values. scripts/02_reproduce.py asserts against these.
TARGETS = {
    "rows": 128_626,
    "train": 102_900,
    "test": 25_726,
    "stage1_gini": 0.0954,
    "stage2_gini": 0.1252,
    "stage3_gini": 0.3356,
    "grand_mean": 61_213,
    "k_adhoc": 19.7,
    "relativities": [0.496, 0.653, 0.862, 1.084, 1.037, 0.805, 0.603, 0.630],
}

# Gradient boosting. post_firm_bin is deliberately absent: it is not the
# zero-variance field it is sometimes assumed to be (72,311 / 56,315), and
# excluding it is what reproduces the published Stage 3 figure. Including it
# raises Stage 3 by about 0.016. See docs/RESULTS.md.
XGB_FEATURES = ["erf", "dem_elevation_m", "slope_deg", "aspect_deg",
                "terrain_ruggedness", "age_at_loss", "num_floors",
                "zone_enc", "occ_enc"]

XGB_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=TWEEDIE_POWER,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    eval_metric="tweedie-nloglik@1.5",
    early_stopping_rounds=30,
)
