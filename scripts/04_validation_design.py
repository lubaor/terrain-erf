"""
Validation design: why random splits overstate model complexity.

The most transferable result in this work, and it is not about flood.

Flood claims arrive in event clusters. One storm, one region, one year. A random
split lets claims from the same event sit on both sides, so a flexible model can
recognise the event rather than the risk. Holding out whole loss years removes
that route.

    python scripts/04_validation_design.py

Writes outputs/checkpoints/validation_design.json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from erf import config as cfg
from erf import spec
from erf import validation as val

warnings.filterwarnings("ignore")


def main() -> int:
    df = spec.load()

    print("event concentration")
    ess = val.effective_sample_size(df)
    print(f"  nominal claims            {ess['nominal_n']:>12,}")
    print(f"  loss years                {ess['n_clusters']:>12,}")
    print(f"  effective observations    {ess['effective_n']:>12.2f}")
    print(f"  design effect             {ess['design_effect']:>12,.0f}x")
    print("  Any interval computed as though claims were independent is")
    print("  overstated by orders of magnitude.")

    cells = {
        "baseline": ("glm", spec.F_BASELINE),
        "elevation model": ("glm", spec.F_ELEVATION),
        "gradient boosting": ("xgb", cfg.XGB_FEATURES),
    }

    out = {"effective_sample_size": ess}
    for scheme, folds in (("random 5-fold", val.random_kfold),
                          ("year-grouped 5-fold", val.year_grouped)):
        per = {k: [] for k in cells}
        for i, tr, te in folds(df):
            for name, (kind, arg) in cells.items():
                if kind == "glm":
                    g, _ = spec.glm_gini(arg, tr, te)
                else:
                    g, *_ = spec.xgb_gini(tr, te, arg)
                per[name].append(g)
            print(f"  {scheme} fold {i} done", flush=True)
        base = np.array(per["baseline"])
        elev = np.array(per["elevation model"])
        boost = np.array(per["gradient boosting"])
        out[scheme] = {
            **{k: val.summarise(v) for k, v in per.items()},
            "elevation lift": val.summarise(elev - base),
            "boosting advantage": val.summarise(boost - elev),
        }

    print(f"\n{'':<24}{'random':>26}{'year-grouped':>26}")
    print("-" * 78)
    for row in ("baseline", "elevation model", "gradient boosting",
                "elevation lift", "boosting advantage"):
        a = out["random 5-fold"][row]
        b = out["year-grouped 5-fold"][row]
        print(f"{row:<24}{a['mean']:>+12.4f} ± {a['sd']:.4f}"
              f"{b['mean']:>+12.4f} ± {b['sd']:.4f}")

    lift_r = out["random 5-fold"]["elevation lift"]["mean"]
    lift_g = out["year-grouped 5-fold"]["elevation lift"]["mean"]
    adv_r = out["random 5-fold"]["boosting advantage"]["mean"]
    adv_g = out["year-grouped 5-fold"]["boosting advantage"]["mean"]
    print(f"\nThe elevation lift is unchanged: {lift_r:+.4f} to {lift_g:+.4f}.")
    print(f"The boosting advantage falls {100 * (1 - adv_g / adv_r):.0f}%: "
          f"{adv_r:+.4f} to {adv_g:+.4f}.")

    # ---- leave one year out, the question a pricing actuary actually asks
    print("\nleave-one-year-out, years with at least 500 claims")
    rows = []
    for year, tr, te in val.leave_one_year_out(df):
        g1, _ = spec.glm_gini(spec.F_BASELINE, tr, te)
        g2, _ = spec.glm_gini(spec.F_ELEVATION, tr, te)
        rows.append({"year": year, "n": len(te), "baseline": g1,
                     "elevation": g2, "lift": g2 - g1})
        print(f"  {year}  n={len(te):>6,}  baseline {g1:+.4f}  "
              f"elevation {g2:+.4f}  lift {g2 - g1:+.4f}")

    lifts = np.array([r["lift"] for r in rows])
    ns = np.array([r["n"] for r in rows], dtype=float)
    unweighted = val.block_bootstrap(lifts)
    weighted = val.block_bootstrap(lifts, weights=ns)
    out["leave_one_year_out"] = {
        "per_year": rows,
        "unweighted": unweighted,
        "claim_weighted": weighted,
        "n_positive": int((lifts > 0).sum()),
        "n_years": int(len(lifts)),
    }
    print(f"\n  unweighted     {unweighted['mean']:+.4f} "
          f"[{unweighted['lo95']:+.4f}, {unweighted['hi95']:+.4f}]  "
          f"{'crosses zero' if unweighted['crosses_zero'] else 'excludes zero'}")
    print(f"  claim-weighted {weighted['mean']:+.4f} "
          f"[{weighted['lo95']:+.4f}, {weighted['hi95']:+.4f}]  "
          f"{'crosses zero' if weighted['crosses_zero'] else 'excludes zero'}")
    print(f"  positive in {int((lifts > 0).sum())} of {len(lifts)} years")
    print("\n  A Gini on a five-hundred-claim year is mostly noise, so the two")
    print("  weightings disagree. Both are reported rather than the better one.")

    path = cfg.CHECKPOINTS / "validation_design.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
