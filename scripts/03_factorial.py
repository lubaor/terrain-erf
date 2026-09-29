"""
Separate the effect of the feature set from the effect of the model class.

Comparing an elevation GLM against a boosted model on the full terrain feature
set changes two things at once, so a single lift figure cannot attribute the
improvement to either. This crosses the two factors instead.

    python scripts/03_factorial.py

Writes outputs/checkpoints/factorial.json
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

# The tree equivalent of "zone plus ERF": the same information, encoded for a
# model that does not need a spline basis to bend.
XGB_ZONE_ERF = ["erf", "age_at_loss", "zone_enc", "occ_enc"]


def main() -> int:
    df = spec.load()
    cells = {
        "glm_zone_erf": ("glm", spec.F_ELEVATION),
        "glm_full": ("glm", spec.F_FULL_TERRAIN),
        "xgb_zone_erf": ("xgb", XGB_ZONE_ERF),
        "xgb_full": ("xgb", cfg.XGB_FEATURES),
    }
    per = {k: [] for k in cells}
    for seed, tr, te in val.repeated_splits(df):
        for name, (kind, arg) in cells.items():
            if kind == "glm":
                g, _ = spec.glm_gini(arg, tr, te)
            else:
                g, *_ = spec.xgb_gini(tr, te, arg, seed=seed)
            per[name].append(g)
        print(f"  seed {seed:2d} done", flush=True)

    a = np.array(per["glm_zone_erf"])
    b = np.array(per["glm_full"])
    c = np.array(per["xgb_zone_erf"])
    d = np.array(per["xgb_full"])
    out = {
        "cells": {k: val.summarise(v) for k, v in per.items()},
        "feature_set_effect": val.summarise(((b - a) + (d - c)) / 2),
        "model_class_effect": val.summarise(((c - a) + (d - b)) / 2),
        "interaction": val.summarise((d - c) - (b - a)),
    }

    print(f"\n{'':<20}{'zone + ERF':>16}{'full terrain':>16}")
    print("-" * 52)
    print(f"{'GLM':<20}{a.mean():>16.4f}{b.mean():>16.4f}")
    print(f"{'gradient boosting':<20}{c.mean():>16.4f}{d.mean():>16.4f}")
    print()
    for k in ("feature_set_effect", "model_class_effect", "interaction"):
        print(f"  {k:<22}{val.fmt(out[k])}")
    print("\nNeither change alone accounts for the improvement, and the")
    print("interaction is larger than either main effect. Read this together")
    print("with scripts/04: most of the model-class effect is a split artefact.")

    path = cfg.CHECKPOINTS / "factorial.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
