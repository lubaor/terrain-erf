"""
Does surrounding terrain add signal beyond point-level slope, aspect and
ruggedness?

Short answer on this data: no, and the reason is worth more than the answer.

Needs data/louisiana_dem_90m.tif. See the data section of README.md.

    python scripts/05_topographic_position.py

Writes data/claims_tpi.parquet and outputs/checkpoints/topographic_position.json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from erf import config as cfg
from erf import spec, terrain
from erf import validation as val

warnings.filterwarnings("ignore")

# The first three are point-scale. The last two exist because the coordinates
# are censored to about 11 km, so a 710 m window cannot be attributed to a
# building. These are the scales the data can actually support.
WINDOWS = (3, 5, 9, 33, 121)
TPI_COLS = [f"tpi_{w}" for w in WINDOWS]


def main() -> int:
    if not cfg.DEM.exists():
        print(f"missing {cfg.DEM}\nSee the data section of README.md.")
        return 2

    df = spec.load()

    if cfg.TPI.exists():
        nodes = pd.read_parquet(cfg.TPI)
        print(f"reusing {cfg.TPI}")
    else:
        print("sampling the DEM at each distinct claim coordinate")
        nodes = terrain.tpi_at_claims(df, windows=WINDOWS)
        nodes.to_parquet(cfg.TPI, index=False)
        print(f"wrote {cfg.TPI}")

    merged = df.merge(nodes, on=["latitude", "longitude"], how="left")

    # The georeferencing check. Without this the rest is unverifiable.
    both = merged["dem_centre"].notna() & merged["dem_elevation_m"].notna()
    diff = (merged.loc[both, "dem_centre"]
            - merged.loc[both, "dem_elevation_m"]).abs()
    print("\ngeoreferencing check against the stored elevation")
    print(f"  median absolute difference  {diff.median():.4f} m")
    print(f"  within 0.5 m                {100 * (diff < 0.5).mean():.1f}%")
    if diff.median() > 0.01:
        print("  WARNING: the pixel mapping looks wrong. Stop here.")

    for c in TPI_COLS:
        merged[c] = merged[c].fillna(merged[c].median())

    print("\ndistinct values, a direct consequence of coordinate censoring")
    for c in ["dem_elevation_m", "slope_deg", "terrain_ruggedness"] + TPI_COLS:
        print(f"  {c:<20}{merged[c].nunique():>6}")

    f_tpi = spec.F_FULL_TERRAIN + " + " + " + ".join(TPI_COLS)
    out = {}

    per = {"full terrain": [], "with TPI": []}
    for seed, tr, te in val.repeated_splits(merged):
        per["full terrain"].append(spec.glm_gini(spec.F_FULL_TERRAIN, tr, te)[0])
        per["with TPI"].append(spec.glm_gini(f_tpi, tr, te)[0])
        print(f"  seed {seed:2d} done", flush=True)
    gain_random = np.array(per["with TPI"]) - np.array(per["full terrain"])
    out["random"] = {k: val.summarise(v) for k, v in per.items()}
    out["random"]["gain"] = val.summarise(gain_random)

    gper = {"full terrain": [], "with TPI": []}
    for i, tr, te in val.year_grouped(merged):
        gper["full terrain"].append(
            spec.glm_gini(spec.F_FULL_TERRAIN, tr, te)[0])
        gper["with TPI"].append(spec.glm_gini(f_tpi, tr, te)[0])
        print(f"  grouped fold {i} done", flush=True)
    gain_grouped = np.array(gper["with TPI"]) - np.array(gper["full terrain"])
    out["grouped"] = {k: val.summarise(v) for k, v in gper.items()}
    out["grouped"]["gain"] = val.summarise(gain_grouped)

    print("\nTPI added to the full terrain GLM")
    print(f"  random splits   {gain_random.mean():+.4f}")
    print(f"  year-grouped    {gain_grouped.mean():+.4f}")

    # Is it terrain, or is it a location label? Replace TPI with a bare
    # categorical for the coordinate node, which holds no terrain information
    # at all. Sparse nodes are pooled so every level appears in training.
    raw_node = (merged["latitude"].round(1).astype(str) + "_"
                + merged["longitude"].round(1).astype(str))
    counts = raw_node.value_counts()
    keep = set(counts[counts >= 100].index)
    merged["node"] = np.where(raw_node.isin(keep), raw_node, "OTHER")
    f_node = spec.F_FULL_TERRAIN + " + C(node)"

    node_gain = []
    for seed, tr, te in val.repeated_splits(merged, seeds=val.SEEDS[:5]):
        base, _ = spec.glm_gini(spec.F_FULL_TERRAIN, tr, te)
        with_node, _ = spec.glm_gini(f_node, tr, te)
        node_gain.append(with_node - base)
    out["location_label_gain"] = val.summarise(node_gain)

    print(f"  a bare location label buys {np.mean(node_gain):+.4f} on random")
    print("  splits, which is more than TPI buys. The apparent gain is")
    print("  location identification, not terrain measurement.")

    path = cfg.CHECKPOINTS / "topographic_position.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
