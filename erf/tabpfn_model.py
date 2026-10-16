"""
TabPFN as a third model class.

TabPFN is a tabular foundation model. It is pretrained on synthetic tabular
tasks and does its "fitting" in context, so there is no loss function to choose
and nothing to tune. That makes it a useful third point of comparison here: the
GLM carries a Tweedie assumption, gradient boosting carries a Tweedie objective
and a tuned depth, and TabPFN carries neither.

Two things stop it being a drop-in replacement.

No Tweedie. The response is a strictly positive, heavy-tailed severity, and the
other two models are told so. TabPFN cannot be. It is handed the raw response
and infers the shape from context. Fitting on log loss instead was tried and is
worse on held-out Gini, so the raw response is used, which is also the
like-for-like choice against the other two.

A context limit. TabPFN attends over the training rows at prediction time, so
cost grows with training rows multiplied by test rows and the pretraining regime
tops out near ten thousand rows. There are 102,900 training rows in a standard
split. The fix is a subsample ensemble: fit several independent models on
disjoint random subsamples and average their predictions. Disjoint rather than
bootstrapped so that the ensemble sees as much of the data as possible.

The feature matrix is built by `spec.xgb_matrices`, so TabPFN and gradient
boosting receive byte-identical inputs and any difference is the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import spec

# Prediction is chunked so that a held-out year of 64,616 claims does not try to
# build one attention matrix against the whole context.
PREDICT_CHUNK = 8192


def _regressor(seed: int):
    """Build a TabPFN regressor. Imported late so the rest of the repo runs
    without torch installed."""
    import torch
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        random_state=seed,
        device=cfg.TABPFN_DEVICE,
        ignore_pretraining_limits=True,
        fit_mode="fit_with_cache",       # halves per-row prediction cost
        inference_precision=torch.float32,
        **cfg.TABPFN_PARAMS,
    )


def subsample_indices(n: int, n_context: int, n_members: int, rng) -> list:
    """Row indices for each ensemble member.

    Disjoint blocks of a single permutation where the training set is large
    enough, which it is for every split used here. Where it is not, independent
    draws without replacement, which overlap but still differ.
    """
    if n <= n_context:
        return [np.arange(n)]
    perm = rng.permutation(n)
    if n >= n_members * n_context:
        return [perm[i * n_context:(i + 1) * n_context] for i in range(n_members)]
    return [rng.choice(n, n_context, replace=False) for _ in range(n_members)]


def predict(train: pd.DataFrame, test: pd.DataFrame, features=None,
            seed: int = cfg.SEED, n_context: int | None = None,
            n_members: int | None = None, verbose: bool = False) -> np.ndarray:
    """Subsample-ensemble TabPFN predictions for `test`.

    Members are averaged on the response scale. The metric is rank based, so the
    averaging scale only matters through its effect on the ordering, and the
    response scale is the one the other two models are averaged on.
    """
    features = list(features or cfg.TABPFN_FEATURES)
    n_context = n_context or cfg.TABPFN_CONTEXT
    n_members = n_members or cfg.TABPFN_MEMBERS

    X_tr, X_te = spec.xgb_matrices(train, test, features)
    X_tr = X_tr.to_numpy(float)
    X_te = X_te.to_numpy(float)
    y_tr = train["loss_building"].to_numpy(float)

    rng = np.random.default_rng(seed)
    members = subsample_indices(len(X_tr), n_context, n_members, rng)

    total = np.zeros(len(X_te), dtype=float)
    for m, idx in enumerate(members):
        model = _regressor(seed + m)
        model.fit(X_tr[idx], y_tr[idx])
        for start in range(0, len(X_te), PREDICT_CHUNK):
            block = X_te[start:start + PREDICT_CHUNK]
            total[start:start + len(block)] += model.predict(block)
        if verbose:
            print(f"      member {m + 1}/{len(members)} "
                  f"(context {len(idx):,})", flush=True)
    return total / len(members)


def tabpfn_gini(train: pd.DataFrame, test: pd.DataFrame, features=None,
                seed: int = cfg.SEED, verbose: bool = False, **kw):
    """Fit and score. Returns (gini, predictions)."""
    p = predict(train, test, features, seed, verbose=verbose, **kw)
    return spec.gini(test["loss_building"].to_numpy(float), p), p
