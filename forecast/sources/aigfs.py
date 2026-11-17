"""NOAA AIGFS 0.25°: NCEP's operational AI global model (since December 2025), built on Google
DeepMind's GraphCast and started from NOAA's own analyses.

The files follow NCEP conventions (GRIB2 with ``.idx`` inventories) on the same 0.25° grid as GFS,
so the GFS client downloads them unchanged. Steps are 6-hourly to 16 days. Two files per step:
``sfc`` (2 m temperature and dew point, 10 m and 100 m wind, pressure, precipitation) and ``pres``
(13 pressure levels); the catalog's ``atmos`` product reads ``sfc``. NCEP notes the contents may
change, so a missing field is left empty for that step rather than failing the run.

NOAA publishes the operational runs on NOMADS and copies them to the NOAA Open Data Dissemination
bucket on AWS (labelled "NOAA EAGLE"), which this reads by default. Set AIGFS_BASE_URL to
https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod for NOMADS, which is earlier but
rate-limited and keeps only about two days.
"""

from __future__ import annotations

from datetime import datetime

from .base import Source
from .gfs import GRID

DEFAULT_BASE_URL = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
MAX_FHR = 384
STEP_HOURS = 6
FILES = {"atmos": "sfc", "pres": "pres"}


def forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """AIGFS output steps: every 6 hours from the analysis to f384."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(0, max_fhr + 1, STEP_HOURS))


def file_url(base_url: str, run: datetime, fhr: int, product: str = "atmos") -> str:
    name = FILES.get(product)
    if name is None:
        raise ValueError(f"Unknown AIGFS product {product!r}")
    return f"{base_url}/aigfs.{run:%Y%m%d}/{run:%H}/model/atmos/grib2/aigfs.t{run:%H}z.{name}.f{fhr:03d}.grib2"


def source_info(base_url: str, run: datetime, products=("atmos",)) -> dict:
    return {
        "provider": "NOAA/NCEP Artificial Intelligence Global Forecast System (AIGFS), 0.25°, "
                    "based on Google DeepMind's GraphCast",
        "distribution": "NOAA Open Data Dissemination (AWS bucket noaa-nws-graphcastgfs-pds)",
        "url": f"{base_url}/aigfs.{run:%Y%m%d}/{run:%H}/model/atmos/grib2/",
        "license": "Public domain (NOAA product, CC0)",
    }


SOURCE = Source("aigfs", GRID, MAX_FHR, DEFAULT_BASE_URL, forecast_hours, file_url, source_info,
                stale_after_hours=18)  # 6-hourly, last step on AWS 8-11 h after the run starts
