"""
Reproduce the published headline numbers.

Run this first. If it passes, the specification and the data agree and every
other script can be trusted. If it fails, stop and find out why rather than
running the rest.

    python scripts/02_reproduce.py

Targets, from the paper:

    rows / train / test        128,626 / 102,900 / 25,726
    baseline Gini              0.0954
    elevation model Gini       0.1252
    gradient boosting Gini     0.3356
    portfolio mean loss        $61,213
    credibility k              19.7
    band relativities          0.496 ... 0.630
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from erf import config as cfg
from erf import spec

TOL_GINI = 5e-4


def check(label, got, want, tol, fmt="{:.4f}"):
    ok = abs(got - want) <= tol
    print(f"  {label:<34}{fmt.format(got):>12}  target {fmt.format(want):>12}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not cfg.MODELLING.exists():
        print(f"missing {cfg.MODELLING}\nSee the data section of README.md.")
        return 2

    df = spec.load()
    train, test = spec.split(df)
    y_test = test["loss_building"].to_numpy(float)
    results = []

    print("\ndataset")
    results.append(check("rows", len(df), cfg.TARGETS["rows"], 0, "{:,.0f}"))
    results.append(check("training claims", len(train), cfg.TARGETS["train"], 0,
                         "{:,.0f}"))
    results.append(check("held-out claims", len(test), cfg.TARGETS["test"], 0,
                         "{:,.0f}"))

    print("\nmodels, held-out Gini")
    g1, _ = spec.glm_gini(spec.F_BASELINE, train, test)
    g2, _ = spec.glm_gini(spec.F_ELEVATION, train, test)
    results.append(check("baseline", g1, cfg.TARGETS["stage1_gini"], TOL_GINI))
    results.append(check("elevation model", g2, cfg.TARGETS["stage2_gini"],
                         TOL_GINI))

    g3, model, _, _ = spec.xgb_gini(train, test)
    # A wider tolerance here: xgboost changes its sampling RNG between releases,
    # so this figure drifts by about 0.001 across versions.
    results.append(check("gradient boosting", g3, cfg.TARGETS["stage3_gini"],
                         2e-3))
    best = getattr(model, "best_iteration", None)
    print(f"  {'best iteration':<34}{best!s:>12}  "
          f"(early stopping {'fired' if best is not None and best < 499 else 'never fired'})")

    print("\ncredibility")
    df = df.assign(erf_band=spec.bands(df["erf"]))
    table, k_hat, grand = spec.credibility(df)
    results.append(check("portfolio mean loss", grand, cfg.TARGETS["grand_mean"],
                         1.0, "${:,.0f}"))
    results.append(check("k (ad-hoc estimator)", k_hat, cfg.TARGETS["k_adhoc"],
                         0.05, "{:.3f}"))
    k_uni, _, _ = spec.credibility_uniform_exposure(df)
    print(f"  {'k (uniform exposure)':<34}{k_uni:>12.3f}  "
          f"(reported alongside; differs by <=0.007 in any relativity)")

    print("\nband relativities")
    worst = 0.0
    for band, got, want in zip(cfg.BAND_ORDER, table["relativity"],
                               cfg.TARGETS["relativities"]):
        worst = max(worst, abs(got - want))
        print(f"  {band:<10}{got:>10.4f}  target {want:>7.3f}  "
              f"diff {got - want:+.4f}")
    results.append(check("max relativity deviation", worst, 0.0, 1e-3))

    passed = all(results)
    print("\n" + "=" * 74)
    print(f"REPRODUCTION {'PASSED' if passed else 'FAILED'} "
          f"({sum(results)}/{len(results)} checks)")
    print("=" * 74)
    if not passed:
        print("\nDo not trust the other scripts until this passes.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
