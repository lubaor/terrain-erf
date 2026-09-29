"""
The model specification.

This module is the single source of truth for how the dataset is built, how it
is split, how the Gini is computed and how each model is fitted. Every script in
the repo imports from here, so a change made once takes effect everywhere.

The specification is deliberately explicit. An earlier round of this work was
not reproducible because the baseline was documented as "flood zone only" when
it in fact carried occupancy and structure age as well. Once the baseline is
wrong, nothing downstream reconciles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config as cfg

# --------------------------------------------------------------- formulae
F_BASELINE = ("loss_building ~ C(flood_zone_group) + C(occupancy_group) "
              "+ age_at_loss")

F_ELEVATION = (F_BASELINE + " + erf + erf_hinge_m4 + erf_hinge_m2 "
                            "+ erf_hinge_p0 + erf_hinge_p2 + erf_hinge_p4 "
                            "+ erf:C(flood_zone_group)")

F_FULL_TERRAIN = (F_ELEVATION + " + dem_elevation_m + slope_deg + aspect_deg "
                                "+ terrain_ruggedness + num_floors")

# Absolute ground elevation instead of the certified ERF, for the question of
# how much free public data recovers on its own.
F_DEM_ONLY = (F_BASELINE + " + dem_elevation_m "
                           "+ dem_elevation_m:C(flood_zone_group)")

# numpy 2 renamed trapz
_trapz = getattr(np, "trapz", None) or np.trapezoid


# ------------------------------------------------------------------- Gini
def gini(y_true, y_pred) -> float:
    """Rank-discrimination on the held-out set.

    Sort by prediction descending, accumulate actual losses, integrate under the
    resulting Lorenz curve. This exact form matters: a normalised ordered-Lorenz
    variant gives visibly different numbers on the same data.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    cumsum = np.cumsum(y_sorted) / y_sorted.sum()
    n = len(y_sorted)
    lorenz = np.concatenate([[0], cumsum])
    return float(2 * _trapz(lorenz, dx=1 / n) - 1)


# --------------------------------------------------------------- the data
def load(path=None) -> pd.DataFrame:
    """Load the modelling set and build the derived columns.

    The derived columns are not stored in the parquet. They are rebuilt here so
    that the definition lives in one place and in version control.
    """
    df = pd.read_parquet(path or cfg.MODELLING)
    n_in = len(df)

    # structure age at loss
    age = df["year_of_loss"] - df["year_built"].clip(1900, 2023)
    age = age.clip(0, 100)
    df["age_at_loss"] = age.fillna(age.median())

    # occupancy: the five most frequent codes kept, the rest pooled
    top = df["occupancy_type"].value_counts().head(5).index
    occ = df["occupancy_type"].astype(str)
    df["occupancy_group"] = occ.where(occ.isin([str(x) for x in top]),
                                      other="OTHER")

    # hinge-spline basis on the ERF
    for knot in cfg.ERF_KNOTS:
        col = f"erf_hinge_{knot:+d}".replace("+", "p").replace("-", "m")
        df[col] = np.maximum(df["erf"].fillna(0) - knot, 0)

    df["post_firm_bin"] = df["post_firm"].astype(int)

    df = df.dropna(subset=["loss_building", "erf", "flood_zone_group",
                           "occupancy_group"]).reset_index(drop=True)
    if len(df) != n_in:
        print(f"note: dropna removed {n_in - len(df):,} rows")
    return df


def bands(erf: pd.Series) -> pd.Series:
    """ERF band labels.

    Right-closed intervals. The ERF is integer-valued, so a left-closed cut
    disagrees with the published banding on 56% of rows.
    """
    return pd.cut(erf, cfg.BAND_EDGES, labels=cfg.BAND_ORDER, right=True)


def split(df: pd.DataFrame, seed: int = cfg.SEED):
    """The reference split: unstratified, 80/20."""
    return train_test_split(df, test_size=cfg.TEST_SIZE, random_state=seed)


# ------------------------------------------------------------------ models
def fit_glm(formula: str, train: pd.DataFrame):
    return smf.glm(
        formula=formula, data=train,
        family=sm.families.Tweedie(link=sm.families.links.Log(),
                                   var_power=cfg.TWEEDIE_POWER),
    ).fit(maxiter=cfg.MAXITER)


def glm_gini(formula: str, train, test, response: str = "loss_building"):
    """Fit on train, score on test. Returns (gini, fitted model)."""
    f = formula if response == "loss_building" else \
        formula.replace("loss_building", response, 1)
    m = fit_glm(f, train)
    return gini(test[response].to_numpy(float), m.predict(test)), m


def xgb_matrices(train, test, features=None):
    """Label-encode the categoricals on train, fill gaps with the train median.

    Categories unseen in training map to -1 rather than raising, which matters
    for grouped cross-validation where a whole year is held out.
    """
    features = list(features or cfg.XGB_FEATURES)
    tr, te = train.copy(), test.copy()

    le_zone, le_occ = LabelEncoder(), LabelEncoder()
    tr["zone_enc"] = le_zone.fit_transform(tr["flood_zone_group"].astype(str))
    tr["occ_enc"] = le_occ.fit_transform(tr["occupancy_group"].astype(str))
    zmap = {c: i for i, c in enumerate(le_zone.classes_)}
    omap = {c: i for i, c in enumerate(le_occ.classes_)}
    te["zone_enc"] = te["flood_zone_group"].astype(str).map(zmap).fillna(-1)
    te["occ_enc"] = te["occupancy_group"].astype(str).map(omap).fillna(-1)

    X_tr = tr[features].astype(float)
    X_te = te[features].astype(float)
    med = X_tr.median()
    return X_tr.fillna(med), X_te.fillna(med)


def fit_xgb(X_train, y_train, X_eval, y_eval, seed: int = cfg.SEED, **overrides):
    """Gradient-boosted Tweedie model.

    Note on the evaluation set: the published configuration passed the test set
    here, which is improper. Early stopping never triggered, so nothing leaked
    in practice, and re-fitting with an evaluation set carved out of training
    data moves the Gini by about -0.001. Pass your own X_eval to do it properly.
    """
    params = {**cfg.XGB_PARAMS, "random_state": seed, **overrides}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return model


def xgb_gini(train, test, features=None, seed: int = cfg.SEED,
             eval_on_train_split: bool = False):
    """Fit and score a boosted model. Returns (gini, model, X_train, X_test)."""
    y_tr = train["loss_building"].to_numpy(float)
    y_te = test["loss_building"].to_numpy(float)
    X_tr, X_te = xgb_matrices(train, test, features)
    if eval_on_train_split:
        i_tr, i_va = train_test_split(np.arange(len(train)), test_size=0.2,
                                      random_state=seed)
        model = fit_xgb(X_tr.iloc[i_tr], y_tr[i_tr],
                        X_tr.iloc[i_va], y_tr[i_va], seed)
    else:
        model = fit_xgb(X_tr, y_tr, X_te, y_te, seed)
    return gini(y_te, model.predict(X_te)), model, X_tr, X_te


# -------------------------------------------------------------- credibility
def credibility(df: pd.DataFrame, k: float | None = None):
    """Buhlmann-Straub band relativities.

    Computed on the whole dataset rather than the training split, which is how
    the published table was produced. With k unset, the ad-hoc estimator used
    for the original table is re-derived: the simple mean of within-band
    variances over the sample variance of the band means.
    """
    y = df["loss_building"].astype(float)
    band = df["erf_band"].astype(str)
    g = y.groupby(band)
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(),
                      "var": g.var(ddof=1)}).reindex(cfg.BAND_ORDER)

    k_hat = float(t["var"].mean()) / float(np.var(t["mean"].to_numpy(), ddof=1))
    k_use = k_hat if k is None else k

    grand = float(y.mean())
    t["Z"] = t["n"] / (t["n"] + k_use)
    t["cred_mean"] = t["Z"] * t["mean"] + (1 - t["Z"]) * grand
    t["relativity"] = t["cred_mean"] / grand
    return t, k_hat, grand


def credibility_uniform_exposure(df: pd.DataFrame):
    """The standard estimator, for comparison. Returns (k, EPV, VHM)."""
    y = df["loss_building"].astype(float)
    band = df["erf_band"].astype(str)
    g = y.groupby(band)
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(),
                      "var": g.var(ddof=1)}).reindex(cfg.BAND_ORDER)
    n_i = t["n"].to_numpy(float)
    m_i = t["mean"].to_numpy(float)
    v_i = t["var"].to_numpy(float)
    N, r = n_i.sum(), len(n_i)
    epv = float(np.average(v_i, weights=n_i - 1))
    mu = float(np.average(m_i, weights=n_i))
    vhm = float((np.sum(n_i * (m_i - mu) ** 2) - (r - 1) * epv)
                / (N - np.sum(n_i ** 2) / N))
    return epv / vhm, epv, vhm
