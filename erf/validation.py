"""
Validation schemes.

The point of this module is that the choice of scheme changes the answer.
Flood claims arrive in event clusters, so a random split lets claims from one
storm sit on both sides and the model can recognise the event rather than the
risk. `repeated_splits` is the conventional approach and `year_grouped` is the
honest one. Report both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import config as cfg

# Seed 42 first so the reference split is always element zero.
SEEDS = [cfg.SEED] + [s for s in range(cfg.N_REPEATS - 1)]


def repeated_splits(df: pd.DataFrame, seeds=None):
    """Yield (seed, train, test) over repeated random 80/20 splits."""
    from .spec import split
    for seed in (seeds or SEEDS):
        tr, te = split(df, seed)
        yield seed, tr, te


def year_grouped(df: pd.DataFrame, n_splits: int = 5):
    """Yield (fold, train, test) holding out whole loss years together."""
    groups = df["year_of_loss"].astype(int).to_numpy()
    for i, (i_tr, i_te) in enumerate(GroupKFold(n_splits).split(df, groups=groups)):
        yield i, df.iloc[i_tr], df.iloc[i_te]


def random_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = cfg.SEED):
    """Random k-fold, as the like-for-like comparison against year_grouped."""
    for i, (i_tr, i_te) in enumerate(
            KFold(n_splits, shuffle=True, random_state=seed).split(df)):
        yield i, df.iloc[i_tr], df.iloc[i_te]


def leave_one_year_out(df: pd.DataFrame, min_claims: int = 500):
    """Yield (year, train, test) for each year with enough claims to score.

    A Gini computed on a few hundred claims is mostly noise, so years below the
    threshold are skipped rather than averaged in. Restricting further, to say
    2,000 claims, is defensible but the threshold was chosen after seeing the
    all-years result, so treat those intervals as descriptive.
    """
    yr = df["year_of_loss"].astype(int)
    counts = yr.value_counts()
    for year in sorted(counts[counts >= min_claims].index):
        yield int(year), df[yr != year], df[yr == year]


# ------------------------------------------------------------- summarising
def summarise(values) -> dict:
    """Mean, spread and interval over a set of fold or split results."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {
        "reference": float(v[0]) if len(v) else float("nan"),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "lo95": float(np.percentile(v, 2.5)),
        "hi95": float(np.percentile(v, 97.5)),
        "n": int(len(v)),
    }


def fmt(s: dict, nd: int = 4) -> str:
    return (f"ref={s['reference']:.{nd}f}  "
            f"mean {s['mean']:.{nd}f} ± {s['sd']:.{nd}f} "
            f"[{s['lo95']:.{nd}f}, {s['hi95']:.{nd}f}]")


def block_bootstrap(values, weights=None, n_boot: int = 20_000, seed: int = cfg.SEED):
    """Resample whole years, which are the independent unit here.

    Claims within an event are not independent draws, so an interval built by
    resampling claims would be far too narrow. Resampling years respects the
    cluster structure.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    if weights is None:
        means = v[idx].mean(axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        means = (v[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return {
        "mean": float(means.mean()),
        "lo95": float(np.percentile(means, 2.5)),
        "hi95": float(np.percentile(means, 97.5)),
        "crosses_zero": bool(np.percentile(means, 2.5) <= 0 <=
                             np.percentile(means, 97.5)),
    }


def effective_sample_size(df: pd.DataFrame, group: str = "year_of_loss") -> dict:
    """Design-effect style count of independent clusters.

    (sum n)^2 / sum(n^2). If claims within an event were perfectly correlated
    the dataset would carry this much information rather than one observation
    per claim. The truth lies between the two, but the gap is large enough that
    any interval assuming independent claims is materially overstated.
    """
    n = df.groupby(group).size().to_numpy(dtype=float)
    n_eff = float(n.sum() ** 2 / (n ** 2).sum())
    return {
        "nominal_n": int(n.sum()),
        "n_clusters": int(len(n)),
        "effective_n": n_eff,
        "design_effect": float(n.sum() / n_eff),
    }
