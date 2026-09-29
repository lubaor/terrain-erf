"""
Fetching the source data.

The NFIP claims come from the OpenFEMA API with server-side filtering, which is
both faster and far smaller than the bulk file. The bulk parquet endpoint is
served with chunked encoding, no Content-Length and no Range support, so a
dropped connection cannot be resumed and a 400 MB download has to start again.
Filtering to one state and the columns actually needed turns that into about
28 MB.

The documented bulk URL under /about/reports-and-data/openfema/ returns 403.
The working endpoint is the one used below.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import config as cfg

BASE = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
PAGE = 10_000
UA = "Mozilla/5.0 (compatible; erf-research/1.0)"

# The fields the analysis touches. Requesting everything roughly triples the
# download for no benefit.
FIELDS = [
    "id",
    # join keys and modelling inputs
    "state", "yearOfLoss", "dateOfLoss", "elevationDifference", "occupancyType",
    "numberOfFloorsInTheInsuredBuilding", "originalConstructionDate",
    "latitude", "longitude", "ratedFloodZone", "amountPaidOnBuildingClaim",
    "postFIRMConstructionIndicator",
    # exposure and value controls
    "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
    "buildingReplacementCost", "buildingPropertyValue",
    # occupancy and construction indicators
    "primaryResidenceIndicator", "nonProfitIndicator",
    "smallBusinessIndicatorBuilding", "agricultureStructureIndicator",
    "houseWorship", "rentalPropertyIndicator", "numberOfUnits",
    "elevatedBuildingIndicator", "basementEnclosureCrawlspaceType",
    "floodproofedIndicator", "obstructionType", "causeOfDamage",
    # elevation detail
    "lowestFloorElevation", "baseFloodElevation", "lowestAdjacentGrade",
    "elevationCertificateIndicator",
]


def _get(url: str, retries: int = 5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:                                  # noqa: BLE001
            last = exc
            wait = 3 * (attempt + 1)
            print(f"    retry {attempt + 1}/{retries} in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def _url(**params) -> str:
    return BASE + "?" + urllib.parse.urlencode(params,
                                               quote_via=urllib.parse.quote)


def claim_count(state: str = "LA") -> int:
    payload = json.loads(_get(_url(**{
        "$filter": f"state eq '{state}'", "$top": "1", "$select": "id",
        "$inlinecount": "allpages"})))
    return int(payload["metadata"]["count"])


def download_claims(state: str = "LA", out: Path | None = None,
                    cache_dir: Path | None = None) -> pd.DataFrame:
    """Page through the API and write a parquet.

    Each page is cached to disk, so an interrupted run costs one page rather
    than the whole download.
    """
    out = out or cfg.CLAIMS_RAW
    cache_dir = cache_dir or (cfg.DATA / "_openfema_pages")
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = claim_count(state)
    n_pages = -(-total // PAGE)
    print(f"{state}: {total:,} claims in {n_pages} pages")

    frames = []
    for i, skip in enumerate(range(0, total, PAGE), start=1):
        cached = cache_dir / f"page_{skip:07d}.json"
        if cached.exists():
            raw = cached.read_bytes()
        else:
            raw = _get(_url(**{
                "$filter": f"state eq '{state}'",
                "$select": ",".join(FIELDS),
                "$top": str(PAGE), "$skip": str(skip), "$orderby": "id"}))
            cached.write_bytes(raw)
        frames.append(pd.DataFrame(json.loads(raw)["FimaNfipClaims"]))
        if i % 10 == 0 or i == n_pages:
            print(f"  page {i}/{n_pages}")

    df = pd.concat(frames, ignore_index=True)
    if len(df) != total:
        print(f"warning: expected {total:,} rows, got {len(df):,}")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return df


def field_definition(name: str) -> str:
    """The official description of a claims field, straight from OpenFEMA.

    Useful when a coded field needs interpreting, for instance causeOfDamage or
    elevatedBuildingIndicator.
    """
    url = ("https://www.fema.gov/api/open/v1/OpenFemaDataSetFields?"
           + urllib.parse.urlencode(
               {"$filter": f"openFemaDataSet eq 'FimaNfipClaims' "
                           f"and name eq '{name}'"},
               quote_via=urllib.parse.quote))
    rows = json.loads(_get(url)).get("OpenFemaDataSetFields", [])
    return rows[0]["description"] if rows else ""
