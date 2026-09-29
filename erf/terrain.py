"""
Sampling the digital elevation model.

Reads the USGS 3DEP tile directly with tifffile. The raster is uncompressed
float32, so GDAL and rasterio are not needed, which keeps the dependency list
installable anywhere.

A warning that applies to everything in this module: the claim coordinates are
censored to one decimal degree, roughly 11 km. Any quantity computed from the
terrain raster at those coordinates describes a neighbourhood, not a building.
`topographic_position` in particular behaves like a location label rather than a
terrain measurement on this data. See docs/RESULTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from . import config as cfg


def read_dem(path=None):
    """Return (elevation array, lon0, lat0, pixel_size_degrees).

    lon0/lat0 are the upper-left corner of pixel (0, 0).
    """
    with tifffile.TiffFile(path or cfg.DEM) as tf:
        page = tf.pages[0]
        tags = {t.name: t.value for t in page.tags.values()}
        z = page.asarray().astype(np.float64)
    scale = tags["ModelPixelScaleTag"]
    tie = tags["ModelTiepointTag"]
    return z, float(tie[3]), float(tie[4]), float(scale[0])


def to_pixel(lat, lon, lon0, lat0, px):
    """Map geographic coordinates to (row, col) indices."""
    col = np.round((np.asarray(lon) - lon0) / px - 0.5).astype(int)
    row = np.round((lat0 - np.asarray(lat)) / px - 0.5).astype(int)
    return row, col


def sample(z, rows, cols):
    """Elevation at each (row, col), NaN where outside the raster."""
    h, w = z.shape
    out = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[ok] = z[rows[ok], cols[ok]]
    return out


def topographic_position(z, rows, cols, window: int):
    """TPI: centre elevation minus the mean of the surrounding window.

    The centre cell is excluded from the mean, which is the standard definition.
    Positive means the point sits above its surroundings, negative below.

    `window` is the edge length in cells. At this raster's resolution one cell is
    about 92 m north-south and 79 m east-west at Louisiana's latitude.
    """
    h, w = z.shape
    half = window // 2
    out = np.full(len(rows), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if not (0 <= r < h and 0 <= c < w) or not np.isfinite(z[r, c]):
            continue
        r0, r1 = max(0, r - half), min(h, r + half + 1)
        c0, c1 = max(0, c - half), min(w, c + half + 1)
        win = z[r0:r1, c0:c1]
        m = np.isfinite(win)
        n = int(m.sum())
        if n <= 1:
            continue
        out[i] = z[r, c] - (win[m].sum() - z[r, c]) / (n - 1)
    return out


def tpi_at_claims(df: pd.DataFrame, windows=(3, 5, 9, 33, 121), dem_path=None):
    """TPI at every distinct claim coordinate.

    Censoring collapses the claims onto a few hundred grid nodes, so only the
    distinct coordinates are visited. Returns one row per node, ready to merge
    back on latitude and longitude.

    The returned frame also carries `dem_centre`, which should agree with the
    stored `dem_elevation_m`. Check it: that agreement is the only evidence the
    georeferencing is right.
    """
    z, lon0, lat0, px = read_dem(dem_path)
    nodes = (df[["latitude", "longitude"]].drop_duplicates()
             .reset_index(drop=True))
    rows, cols = to_pixel(nodes["latitude"], nodes["longitude"], lon0, lat0, px)
    nodes["dem_centre"] = sample(z, rows, cols)
    for win in windows:
        nodes[f"tpi_{win}"] = topographic_position(z, rows, cols, win)
    return nodes
