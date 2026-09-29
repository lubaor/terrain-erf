"""
Download the NFIP claims for one state from the OpenFEMA API.

    python scripts/01_download_claims.py [STATE]

Defaults to Louisiana. Writes data/la_claims.parquet, about 28 MB for LA.
Each page is cached under data/_openfema_pages so an interrupted run resumes.

This is the raw claims extract. It is not the modelling dataset: see the data
section of README.md for what separates the two.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erf import config as cfg
from erf import data


def main() -> int:
    state = (sys.argv[1] if len(sys.argv) > 1 else "LA").upper()
    out = cfg.DATA / f"{state.lower()}_claims.parquet"
    df = data.download_claims(state=state, out=out)
    print(f"\n{len(df):,} rows, {df.shape[1]} columns")
    print(f"non-null elevationDifference: "
          f"{df['elevationDifference'].notna().sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
