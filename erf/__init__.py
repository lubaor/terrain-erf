"""
Terrain-informed ratemaking: an elevation rating factor from open DEM data and
NFIP claims.

    from erf import spec, validation, config

`spec` holds the model specification, `validation` the splitting schemes,
`terrain` the DEM sampling and `data` the OpenFEMA download.
"""
__version__ = "1.0.0"

from . import config, data, spec, terrain, validation  # noqa: F401

__all__ = ["config", "data", "spec", "terrain", "validation"]
