"""NOAA GEFS-Aerosols 0.25°: air quality from NOAA's global aerosol forecast, the aerosol member of the
Global Ensemble Forecast System (GEFS v12 with GOCART chemistry).

It forecasts dust, sea salt, sulphate, organic matter and black carbon (smoke), with fire emissions
from satellite fire detections. The ``a2d_0p25`` files hold surface PM2.5 and PM10 and aerosol optical
depth (haze) by species, every 3 hours to 120 h, four runs a day, on the same 0.25° grid as GFS. The
files follow NCEP conventions (GRIB2 with ``.idx`` inventories); several messages share a parameter
and level and differ only in the species named in the inventory line, which the catalog matches
exactly (``Variable.grib_extra``). NOAA publishes them on its open-data bucket on AWS, which this
reads by default (AEROSOLS_BASE_URL for another mirror).
"""

from __future__ import annotations

from datetime import datetime

from .base import Source
from .gfs import GRID

DEFAULT_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
MAX_FHR = 120
STEP_HOURS = 3


def forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """GEFS-Aerosols steps: every 3 hours to 120 h."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(0, max_fhr + 1, STEP_HOURS))


def file_url(base_url: str, run: datetime, fhr: int, product: str = "atmos") -> str:
    if product != "atmos":
        raise ValueError(f"Unknown GEFS-Aerosols product {product!r}")
    return f"{base_url}/gefs.{run:%Y%m%d}/{run:%H}/chem/pgrb2ap25/gefs.chem.t{run:%H}z.a2d_0p25.f{fhr:03d}.grib2"


def source_info(base_url: str, run: datetime, products=("atmos",)) -> dict:
    return {
        "provider": "NOAA/NCEP GEFS-Aerosols (GEFS v12 with GOCART aerosols), 0.25°",
        "distribution": "NOAA Open Data Dissemination (AWS bucket noaa-gefs-pds)",
        "url": f"{base_url}/gefs.{run:%Y%m%d}/{run:%H}/chem/pgrb2ap25/",
        "license": "Public domain (U.S. Government work)",
    }


SOURCE = Source("aerosols", GRID, MAX_FHR, DEFAULT_BASE_URL, forecast_hours, file_url, source_info,
                stale_after_hours=15)  # 6-hourly, last step out ~6 h after the run starts
