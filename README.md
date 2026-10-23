# Terrain-Informed Ratemaking

An elevation rating factor for flood severity, built from open digital elevation
data and the public NFIP claims record, with the validation machinery needed to
tell whether it actually works.

Research code for a CAS Ratemaking Working Group paper. Everything here uses
open data and open-source tooling, so the analysis can be reproduced without
institutional access.

---

## What this is

Flood insurance is priced mainly on flood zone. Within a single zone, the height
of a building's lowest floor varies by several feet, and that height is one of
the strongest physical drivers of what a flood costs. This repository builds a
continuous elevation covariate, the **Elevation Rating Factor** (ERF, the height
of the lowest floor above or below the Base Flood Elevation, in feet), tests
whether it improves a conventional pricing model, and then tests the test.

The second half is the part worth borrowing. Catastrophe claims arrive in event
clusters, and a random train/test split lets claims from one storm sit on both
sides. On this dataset that inflates the apparent advantage of gradient boosting
over a GLM by a factor of six.

## Headline results

| | Held-out Gini |
|---|---|
| Baseline: flood zone, occupancy, structure age | 0.0954 |
| Baseline plus the ERF | 0.1252 |
| Gradient boosting on the full terrain feature set | 0.3356 |

| Validation scheme | Elevation lift | Boosting advantage | TabPFN advantage |
|---|---|---|---|
| Random 5-fold | +0.0309 | +0.2054 | +0.2046 |
| Grouped on loss year | +0.0320 | +0.0328 | +0.0160 |

The elevation lift is unchanged when whole loss years are held out. The boosting
advantage falls by 84% and its interval includes zero.

The last column is the check on that. TabPFN is a pretrained tabular foundation
model: no Tweedie objective, nothing tuned, and none of the inductive bias of a
tree ensemble. Given the same features it matches gradient boosting under random
splitting and then loses the same advantage under grouped validation. The
collapse is a property of event-clustered data rather than of boosting.

Other findings, with the scripts that produce them:

- The most hazardous flood zone carries the **lowest** average loss, because
  91.3% of its buildings are elevated on piles against 23.9% across the book.
- Damage peaks just below the flood line rather than at the deepest properties,
  for the same reason. Four competing explanations were tested and rejected.
- Free public ground elevation recovers only about a fifth of the certified
  benefit, and under grouped validation that residue is not distinguishable from
  zero.
- Treating each loss year as a cluster, the data carry the information of
  roughly **four** independent observations, not 128,626.

## Layout

```
erf/
  config.py       paths, seeds, the published constants and targets
  spec.py         the model specification: derived columns, split, Gini, fitters
  validation.py   repeated splits, grouped CV, leave-one-year-out, bootstrap
  terrain.py      DEM reading, elevation sampling, topographic position index
  tabpfn_model.py TabPFN subsample ensemble (optional, needs torch)
  data.py         OpenFEMA download
scripts/
  01_download_claims.py        fetch the raw claims for a state
  02_reproduce.py              assert the published numbers      <- run first
  03_factorial.py              separate feature set from model class
  04_validation_design.py      random vs grouped, and effective sample size
  05_topographic_position.py   does surrounding terrain add anything
  06_tabpfn.py                 does the validation result survive a foundation model
tests/                         unit tests, no dataset required
docs/RESULTS.md                the full result set with caveats
```

`erf/spec.py` is the single source of truth. An earlier round of this work was
not reproducible because the baseline was documented as "flood zone only" when
it also carried occupancy and structure age. Once the baseline is wrong, nothing
downstream reconciles. Keeping the specification in one version-controlled
module is the fix.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                              # 27 tests, no data needed
python scripts/02_reproduce.py                      # the gate
```

`02_reproduce.py` checks nine published quantities and exits non-zero if any
fail. If it fails, stop there. Nothing else in the repo is trustworthy until it
passes.

## Data

**Nothing in `data/` is committed.** The sources are public but large, and some
are derived.

| File | Size | How to get it |
|---|---|---|
| `nfip_with_erf.parquet` | 2.3 MB | The modelling dataset. See the note below. |
| `la_claims.parquet` | 28 MB | `python scripts/01_download_claims.py LA` |
| `louisiana_dem_90m.tif` | 130 MB | USGS 3DEP, one Louisiana tile at 1/3 arc-second, retrievable through `py3dep`. Only needed for script 05. |

### A limitation worth stating plainly

`nfip_with_erf.parquet` is the analysis dataset: 128,626 Louisiana claims with
the certified elevation difference, the sampled terrain features and the derived
banding. **The code that originally built it from raw claims, the DEM and a
building-footprint layer is not in this repository.** It predates this codebase
and was not recovered.

Everything from that file onward is fully reproducible here, which is the whole
analysis. The upstream construction is documented rather than executable. The
closest reconstruction, from the raw claims alone, is to filter to the state,
require a non-null elevation difference within ±30 ft, a positive paid building
amount and present coordinates. That yields 130,586 rows against the 128,626 in
the modelling file, so roughly 1,960 rows are removed by an upstream step that
is not documented. The footprint layer is absent entirely, and with coordinates
censored to 0.1 degree it could not be used meaningfully in any case.

If you are reproducing this from scratch, start from the published dataset
rather than trying to rebuild it.

### Coordinate censoring

FEMA rounds claim coordinates to one decimal degree, about 11 km, for privacy.
The 128,626 claims occupy 572 distinct positions. Every DEM-derived feature
therefore describes a neighbourhood rather than a building, and inherits roughly
560 distinct values. This is not a detail. Script 05 shows a terrain variable
that appeared more valuable than the ERF itself and turned out to be a location
label in disguise.

## Reproducibility notes

- The reference split is scikit-learn `train_test_split(test_size=0.2,
  random_state=42)`, unstratified. Interval estimates use 20 repeated splits.
- Gini here is the descending-sort trapezoid form in `spec.gini`. A normalised
  ordered-Lorenz variant gives visibly different numbers on the same data.
- ERF band intervals are **right-closed**. The ERF is integer-valued, so a
  left-closed cut disagrees with the published banding on 56% of rows.
- Versions are pinned in `requirements.txt`. Gradient boosting drifts by about
  0.001 across xgboost releases, so `02_reproduce.py` allows 0.002 there and
  5e-4 on the GLM figures.
- `post_firm_bin` is excluded from the boosted feature set. It is not the
  zero-variance field it is sometimes assumed to be (72,311 against 56,315), but
  excluding it is what reproduces the published figure. Including it adds about
  0.016.

## Citation

> Orlovsky, L. (2026). *Terrain-Informed Ratemaking: Deriving an Elevation
> Rating Factor from Open-Source DEM Data and NFIP Claims.* CAS Ratemaking
> Working Group.

## Data sources and licence

Claims data from the [OpenFEMA](https://www.fema.gov/about/openfema) NFIP
Redacted Claims dataset, public domain. Elevation from the
[USGS 3DEP](https://www.usgs.gov/3d-elevation-program) programme, public domain.

Code released under the Apache License 2.0, Copyright 2026 Earnix.
See `LICENSE` and `NOTICE`.
