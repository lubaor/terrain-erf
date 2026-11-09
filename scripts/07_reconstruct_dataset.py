"""
Reconstruct the modelling dataset from raw claims, as far as the recipe allows.

This does not reproduce data/nfip_with_erf.parquet, and it is not meant to. The
code that originally built that file from raw claims, a DEM and a
building-footprint layer is not in this repository and was not recovered. What
this script does is implement the recipe documented in the README, run it
against the current raw claims, and report exactly where it lands relative to
the published file.

The gap is the point. A reader should be able to inspect the difference rather
than take a sentence in a README on trust.

Two reasons the numbers will not line up, both of which the script quantifies:

  1  An upstream step that is not documented removes roughly 1,960 rows. It is
     not recoverable from what survives.
  2  The OpenFEMA claims file is live and grows. Louisiana stood at 484,979
     claims when the modelling dataset was built and higher now, so the
     reconstruction drifts further from the published file over time.

    python scripts/01_download_claims.py LA      # first, about 28 MB
    python scripts/07_reconstruct_dataset.py

With data/louisiana_dem_90m.tif present the terrain features are sampled too.
Without it those columns are skipped and the script says so.

Writes outputs/checkpoints/reconstruction.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from erf import config as cfg

# The documented recipe. Each entry is (label, predicate on the raw frame).
ERF_LIMIT_FT = 30

RENAMES = {
    "amountPaidOnBuildingClaim": "loss_building",
    "elevationDifference": "erf",
    "ratedFloodZone": "flood_zone",
    "occupancyType": "occupancy_type",
    "yearOfLoss": "year_of_loss",
    "numberOfFloorsInTheInsuredBuilding": "num_floors",
    "postFIRMConstructionIndicator": "post_firm",
    "latitude": "latitude",
    "longitude": "longitude",
}


def attrition(raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply the documented filters, recording what each one costs."""
    steps, df = [], raw
    steps.append(("raw claims for the state", len(df), 0))

    checks = [
        ("elevationDifference present", df["elevationDifference"].notna()),
        (f"elevationDifference within +/-{ERF_LIMIT_FT} ft",
         df["elevationDifference"].abs() <= ERF_LIMIT_FT),
        ("amountPaidOnBuildingClaim > 0",
         df["amountPaidOnBuildingClaim"].fillna(0) > 0),
        ("coordinates present",
         df["latitude"].notna() & df["longitude"].notna()),
    ]
    for label, mask in checks:
        before = len(df)
        df = df[mask.reindex(df.index).fillna(False)]
        steps.append((label, len(df), before - len(df)))
    return df.reset_index(drop=True), steps


def build_columns(df: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Rename and derive what the raw claims alone can support."""
    out = df.rename(columns=RENAMES)[list(RENAMES.values())].copy()

    # construction year, from the original construction date
    yr = pd.to_datetime(df["originalConstructionDate"], errors="coerce").dt.year
    out["year_built"] = yr.to_numpy()

    out["post_firm"] = out["post_firm"].astype("boolean").fillna(False).astype(bool)

    # The zone grouping is not documented anywhere. Recover it from the
    # published file by observing which raw zone maps to which group, and say
    # so, rather than inventing a rule that happens to fit.
    pairs = (published[["flood_zone", "flood_zone_group"]]
             .dropna().drop_duplicates())
    mapping = dict(zip(pairs["flood_zone"], pairs["flood_zone_group"]))
    out["flood_zone_group"] = out["flood_zone"].map(mapping)
    return out, mapping


def add_terrain(out: pd.DataFrame) -> str:
    """Sample ground elevation from the DEM if the tile is present.

    Only `dem_elevation_m` is recoverable. The published file also carries
    slope_deg, aspect_deg and terrain_ruggedness, and the code that computed
    them is part of the pipeline that was not recovered. Their definitions are
    not in this repository, so they are left out rather than guessed at: a
    reconstruction that invented a slope definition would produce a comparison
    that means nothing either way.
    """
    if not cfg.DEM.exists():
        return f"skipped, {cfg.DEM.name} not present"
    from erf import terrain
    z, lon0, lat0, px = terrain.read_dem(cfg.DEM)
    rows, cols = terrain.to_pixel(out["latitude"].to_numpy(),
                                  out["longitude"].to_numpy(), lon0, lat0, px)
    out["dem_elevation_m"] = terrain.sample(z, rows, cols)
    return "dem_elevation_m sampled; slope, aspect and ruggedness not derivable"


def compare(recon: pd.DataFrame, published: pd.DataFrame) -> dict:
    """Where does the reconstruction land against the published file?"""
    shared = [c for c in recon.columns if c in published.columns]
    rows = []
    for c in shared:
        a, b = recon[c], published[c]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            rows.append({
                "column": c,
                "recon_mean": float(a.mean()),
                "published_mean": float(b.mean()),
                "diff_pct": (float(a.mean() - b.mean()) / float(b.mean()) * 100
                             if float(b.mean()) else float("nan")),
            })
    return {"shared_columns": shared, "numeric": rows}


def main() -> int:
    if not cfg.CLAIMS_RAW.exists():
        print(f"missing {cfg.CLAIMS_RAW}")
        print("run: python scripts/01_download_claims.py LA")
        return 1
    if not cfg.MODELLING.exists():
        print(f"missing {cfg.MODELLING}, which is the comparison target")
        return 1

    raw = pd.read_parquet(cfg.CLAIMS_RAW)
    published = pd.read_parquet(cfg.MODELLING)
    print(f"raw claims    {len(raw):>9,}")
    print(f"published set {len(published):>9,}\n")

    filtered, steps = attrition(raw)
    print("documented filters")
    for label, remaining, lost in steps:
        print(f"  {label:<44} {remaining:>9,}"
              + (f"   -{lost:,}" if lost else ""))

    recon, mapping = build_columns(filtered, published)
    unmapped = int(recon["flood_zone_group"].isna().sum())
    print(f"\nzone grouping recovered from the published file: "
          f"{len(mapping)} zones mapped, {unmapped:,} rows unmapped")

    terrain_note = add_terrain(recon)
    print(f"terrain features: {terrain_note}")

    gap = len(recon) - len(published)
    print(f"\nreconstruction {len(recon):,} rows against published "
          f"{len(published):,}, a gap of {gap:+,}")
    print("  This gap is expected and is not closed by anything in this repo.")
    print("  Part of it is an undocumented upstream step. Part of it is that")
    print("  the claims file has grown since the published set was built.")

    cmp = compare(recon, published)
    print(f"\n{'column':<22}{'reconstruction':>16}{'published':>14}{'diff':>9}")
    print("-" * 61)
    for r in cmp["numeric"]:
        print(f"{r['column']:<22}{r['recon_mean']:>16,.3f}"
              f"{r['published_mean']:>14,.3f}{r['diff_pct']:>8.2f}%")

    out = {
        "raw_rows": len(raw),
        "published_rows": len(published),
        "reconstructed_rows": len(recon),
        "gap": gap,
        "attrition": [{"step": s, "remaining": r, "removed": l}
                      for s, r, l in steps],
        "zones_mapped": len(mapping),
        "rows_with_unmapped_zone": unmapped,
        "terrain": terrain_note,
        "comparison": cmp,
        "note": ("This is an approximation of the modelling dataset, not a "
                 "reproduction of it. The original construction code was not "
                 "recovered."),
    }
    path = cfg.CHECKPOINTS / "reconstruction.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
