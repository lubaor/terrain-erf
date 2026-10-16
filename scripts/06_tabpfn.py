"""
TabPFN against the GLM and gradient boosting, under both validation schemes.

The question this script exists to answer is not whether a foundation model
scores well on flood severity. It is whether the result in script 04 survives
one.

Script 04 found that the apparent advantage of gradient boosting over the
interpretable elevation GLM falls by about 84% when whole loss years are held
out instead of a random sample of claims, and becomes indistinguishable from
zero. The obvious objection is that this says something about gradient boosting.
TabPFN is the cleanest available test of that objection. It is pretrained rather
than fitted, it is given no Tweedie objective, it has no depth or learning rate
to tune, and it shares none of the inductive bias of an additive tree ensemble.
If the collapse is a property of event-clustered data rather than of boosting,
TabPFN should lose its advantage in the same way.

    python scripts/06_tabpfn.py             # about two CPU-hours
    python scripts/06_tabpfn.py --quick     # two folds per scheme, to smoke-test

Writes outputs/checkpoints/tabpfn.json
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from erf import config as cfg
from erf import spec
from erf import tabpfn_model as tp
from erf import validation as val

warnings.filterwarnings("ignore")


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def reference_split(df, out):
    """The published 80/20 split, for a number comparable with Stage 3."""
    print("reference split, seed 42")
    tr, te = spec.split(df, cfg.SEED)
    print(f"  train {len(tr):,}  test {len(te):,}")

    g_base, _ = spec.glm_gini(spec.F_BASELINE, tr, te)
    g_elev, _ = spec.glm_gini(spec.F_ELEVATION, tr, te)
    g_xgb, *_ = spec.xgb_gini(tr, te, cfg.XGB_FEATURES)
    print(f"  baseline GLM        {g_base:+.4f}   (published {cfg.TARGETS['stage1_gini']:+.4f})")
    print(f"  elevation GLM       {g_elev:+.4f}   (published {cfg.TARGETS['stage2_gini']:+.4f})")
    print(f"  gradient boosting   {g_xgb:+.4f}   (published {cfg.TARGETS['stage3_gini']:+.4f})")

    t0 = time.time()
    g_tp, _ = tp.tabpfn_gini(tr, te, verbose=True)
    print(f"  TabPFN              {g_tp:+.4f}   ({hms(time.time() - t0)})")

    out["reference_split"] = {
        "baseline": g_base, "elevation": g_elev,
        "gradient boosting": g_xgb, "TabPFN": g_tp,
        "boosting advantage": g_xgb - g_elev,
        "TabPFN advantage": g_tp - g_elev,
    }


def scheme(name, folds, df, out, max_folds=None):
    """Score every model on one validation scheme."""
    print(f"\n{name}")
    per = {k: [] for k in ("baseline", "elevation", "gradient boosting", "TabPFN")}
    t_scheme = time.time()
    for i, tr, te in folds(df):
        if max_folds is not None and i >= max_folds:
            break
        t0 = time.time()
        years = sorted(te["year_of_loss"].astype(int).unique())
        tag = (f"{years[0]}-{years[-1]}" if len(years) > 3
               else ",".join(str(y) for y in years))
        print(f"  fold {i}  test {len(te):>6,}  years {tag}", flush=True)

        per["baseline"].append(spec.glm_gini(spec.F_BASELINE, tr, te)[0])
        per["elevation"].append(spec.glm_gini(spec.F_ELEVATION, tr, te)[0])
        per["gradient boosting"].append(spec.xgb_gini(tr, te, cfg.XGB_FEATURES)[0])
        per["TabPFN"].append(tp.tabpfn_gini(tr, te, verbose=True)[0])

        print(f"    baseline {per['baseline'][-1]:+.4f}  "
              f"elevation {per['elevation'][-1]:+.4f}  "
              f"boosting {per['gradient boosting'][-1]:+.4f}  "
              f"TabPFN {per['TabPFN'][-1]:+.4f}   [{hms(time.time() - t0)}]",
              flush=True)

    elev = np.array(per["elevation"])
    out[name] = {
        **{k: val.summarise(v) for k, v in per.items()},
        "elevation lift": val.summarise(elev - np.array(per["baseline"])),
        "boosting advantage": val.summarise(
            np.array(per["gradient boosting"]) - elev),
        "TabPFN advantage": val.summarise(np.array(per["TabPFN"]) - elev),
        "per_fold": per,
    }
    print(f"  scheme done in {hms(time.time() - t_scheme)}")


def report(out, a_name, b_name):
    print(f"\n{'':<22}{'random':>24}{'year-grouped':>24}")
    print("-" * 70)
    rows = ("baseline", "elevation", "gradient boosting", "TabPFN",
            "elevation lift", "boosting advantage", "TabPFN advantage")
    for row in rows:
        if row == "elevation lift":
            print("-" * 70)
        a, b = out[a_name][row], out[b_name][row]
        print(f"{row:<22}{a['mean']:>+11.4f} ± {a['sd']:.4f}"
              f"{b['mean']:>+11.4f} ± {b['sd']:.4f}")

    print()
    for label in ("boosting advantage", "TabPFN advantage"):
        r, g = out[a_name][label]["mean"], out[b_name][label]["mean"]
        drop = 100 * (1 - g / r) if r else float("nan")
        print(f"  {label:<20} {r:+.4f} random  ->  {g:+.4f} grouped   "
              f"({drop:.0f}% of it goes)")

    # Resample whole years, which is the independent unit, to ask whether what
    # is left under grouped validation is distinguishable from zero.
    print("\n  block bootstrap over folds, year-grouped scheme")
    for label in ("elevation lift", "boosting advantage", "TabPFN advantage"):
        v = np.array(out[b_name][label]["_values"])
        bs = val.block_bootstrap(v)
        print(f"    {label:<20} {bs['mean']:+.4f} "
              f"[{bs['lo95']:+.4f}, {bs['hi95']:+.4f}]  "
              f"{'crosses zero' if bs['crosses_zero'] else 'excludes zero'}")
        out[b_name][label]["bootstrap"] = bs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="two folds per scheme, to smoke-test the wiring")
    args = ap.parse_args()
    max_folds = 2 if args.quick else None

    # Fail now rather than forty minutes in. On Windows torch's DLLs sometimes
    # refuse to load immediately after installation, while the files are still
    # being scanned, and the error surfaces only at the first TabPFN fit.
    import torch  # noqa: F401

    t_start = time.time()
    df = spec.load()
    print(f"{len(df):,} claims, {df['year_of_loss'].nunique()} loss years")
    print(f"TabPFN: {cfg.TABPFN_MEMBERS} members x {cfg.TABPFN_CONTEXT:,} "
          f"context on {cfg.TABPFN_DEVICE}, features identical to boosting\n")

    out = {"config": {
        "context": cfg.TABPFN_CONTEXT, "members": cfg.TABPFN_MEMBERS,
        "device": cfg.TABPFN_DEVICE, "features": cfg.TABPFN_FEATURES,
        "quick": args.quick,
    }}

    reference_split(df, out)

    a_name, b_name = "random 5-fold", "year-grouped 5-fold"
    scheme(a_name, val.random_kfold, df, out, max_folds)
    scheme(b_name, val.year_grouped, df, out, max_folds)

    # keep the raw differences for the bootstrap
    for s in (a_name, b_name):
        p = out[s]["per_fold"]
        e = np.array(p["elevation"])
        out[s]["elevation lift"]["_values"] = (e - np.array(p["baseline"])).tolist()
        out[s]["boosting advantage"]["_values"] = (
            np.array(p["gradient boosting"]) - e).tolist()
        out[s]["TabPFN advantage"]["_values"] = (np.array(p["TabPFN"]) - e).tolist()

    report(out, a_name, b_name)

    path = cfg.CHECKPOINTS / "tabpfn.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    print(f"total {hms(time.time() - t_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
