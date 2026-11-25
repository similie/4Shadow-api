"""Copernicus Marine: surface currents, with tides, from the global ocean analysis and forecast.

The product is GLOBAL_ANALYSISFORECAST_PHY_001_024 (Mercator Ocean International's GLO12 model at 1/12°,
https://doi.org/10.48670/moi-00016). Its ``merged-uv`` dataset gives the total current at the surface,
hourly: the ocean's circulation plus tidal currents (FES2014) plus the drift waves cause (Stokes drift).
The model's own currents have no tides, and in tidal straits and on shelves tides are most of the current.

The data is one rolling time series, rewritten daily as a new forecast comes out, that runs about ten days
past its newest forecast's start; we call that start the run (the series' last hour less 239 hours, to
00 UTC) and read its 240 hours. It is read through the official ``copernicusmarine`` toolbox, which needs
a (free) Copernicus Marine account: COPERNICUSMARINE_SERVICE_USERNAME and ..._PASSWORD. Each hour is the
global 1/12° field (about 42 MB for both components); we keep 0.25°, each point the mean of the 3 x 3
cells around it that are sea, so the coast keeps the currents next to it.

Copernicus Marine's licence allows commercial use; derived products must say "Generated using E.U.
Copernicus Marine Service Information".
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from ..grid import Grid
from .base import ArraySource, StepUnavailable
from .gfs import GRID

log = logging.getLogger(__name__)

PRODUCT_ID = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
DATASET_ID = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
DOI_URL = "https://doi.org/10.48670/moi-00016"
MAX_FHR = 239
CREDENTIALS = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")
ATTRIBUTION = (f"Generated using E.U. Copernicus Marine Service Information ({DOI_URL}). 4Shadow averages it "
               "onto a 0.25° grid and derives speed and direction.")

# The dataset's grid, rows turned to run north -> south: 80° S to 90° N and 180° W to 179.92° E, every 1/12°.
NATIVE = Grid(lat0=90.0, dlat=1 / 12, nlat=2041, lon0=-180.0, dlon=1 / 12, nlon=4320)


def _utc(when: datetime) -> datetime:
    """The UTC wall time, without a zone (as the dataset's times are)."""
    return when.astimezone(timezone.utc).replace(tzinfo=None) if when.tzinfo else when


def forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """Every hour to 239 h."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(max_fhr + 1))


def run_for(last_time: datetime) -> datetime:
    """The newest forecast's start: the series' last hour less 239 hours, to 00 UTC."""
    start = last_time - timedelta(hours=MAX_FHR)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def block_mean(values: np.ndarray, native: Grid, target: Grid) -> np.ndarray:
    """``values`` on ``native`` as ``target``, whose points are native points a whole number of cells
    apart: each target point gets the mean of the native cells within half a target cell of it, ignoring
    missing ones (land), and is missing only where they all are. Rows outside ``native`` are missing."""
    factor = target.dlat / native.dlat
    k = round(factor)
    if k < 1 or abs(factor - k) > 1e-6 or abs(target.dlon / native.dlon - k) > 1e-6:
        raise ValueError("the target grid's spacing must be a whole number of native cells")
    rows_f = (native.lat0 - (target.lat0 - np.arange(target.nlat) * target.dlat)) / native.dlat
    cols_f = np.mod(target.lon0 + np.arange(target.nlon) * target.dlon - native.lon0, 360.0) / native.dlon
    rows, cols = np.rint(rows_f).astype(int), np.rint(cols_f).astype(int)
    if np.abs(rows_f - rows).max() > 1e-6 or np.abs(cols_f - cols).max() > 1e-6:
        raise ValueError("the target grid's points must be native grid points")
    total = np.zeros(target.shape, np.float64)
    count = np.zeros(target.shape, np.int32)
    half = k // 2
    for dr in range(-half, half + 1):
        r = rows + dr
        inside = (r >= 0) & (r < native.nlat)
        r = np.clip(r, 0, native.nlat - 1)
        for dc in range(-half, half + 1):
            block = values[np.ix_(r, np.mod(cols + dc, native.nlon))]
            ok = np.isfinite(block) & inside[:, None]
            total += np.where(ok, block, 0.0)
            count += ok
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


class MarineClient:
    """Reads the dataset through the ``copernicusmarine`` toolbox (imported here: only the ingest needs it)."""

    def __init__(self, dataset_id: str = DATASET_ID) -> None:
        self.dataset_id = dataset_id
        self._run: datetime | None = None
        self._ds = None

    def _open(self, start: datetime | None = None, end: datetime | None = None):
        missing = [k for k in CREDENTIALS if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"No Copernicus Marine login: set {' and '.join(missing)} (a free account at "
                               "https://data.marine.copernicus.eu/register)")
        import copernicusmarine

        window = {}
        if start is not None:
            window = {"start_datetime": f"{_utc(start):%Y-%m-%dT%H:%M:%S}", "end_datetime": f"{_utc(end):%Y-%m-%dT%H:%M:%S}"}
        # chunk_size_limit=0: no dask chunks of its own. By default the toolbox groups 50 hours into each,
        # so reading one hour downloads 50.
        return copernicusmarine.open_dataset(dataset_id=self.dataset_id, service="arco-geo-series",
                                             chunk_size_limit=0, **window)

    def latest_run(self, max_fhr: int = MAX_FHR) -> datetime | None:
        times = self._open().time.values
        if not len(times):
            return None
        last = datetime.fromtimestamp(times[-1].astype("datetime64[s]").astype(int), timezone.utc)
        run = run_for(last)
        return run if run + timedelta(hours=max_fhr) <= last else None

    def step(self, run: datetime, fhr: int, names: list[str], grid: Grid) -> dict[str, np.ndarray]:
        if self._run != run:
            self._ds, self._run = self._open(run, run + timedelta(hours=MAX_FHR)), run
        when = np.datetime64(_utc(run + timedelta(hours=fhr)), "ns")
        if when not in self._ds.time.values:
            raise StepUnavailable(f"{self.dataset_id} has no {when}")
        hour = self._ds[names].sel(time=when).isel(depth=0).load()
        flip = bool(hour.latitude.values[0] < hour.latitude.values[-1])
        out = {}
        for name in names:
            values = np.asarray(hour[name].values, np.float32)
            if values.shape != NATIVE.shape:
                raise RuntimeError(f"{self.dataset_id} {name}: shape {values.shape}, expected {NATIVE.shape}")
            if np.isnan(values).all():
                raise StepUnavailable(f"{self.dataset_id} {name} at {when}: no data yet")
            out[name] = block_mean(values[::-1] if flip else values, NATIVE, grid)
        return out


def source_info(base_url: str, run: datetime, products=()) -> dict:
    return {
        "provider": ("Copernicus Marine Service global ocean analysis and forecast (Mercator Ocean International "
                     "GLO12, 1/12°): surface currents with tidal currents (FES2014) and wave drift"),
        "distribution": "Copernicus Marine Service",
        "url": base_url,
        "dataset": DATASET_ID,
        "license": ("Copernicus Marine Service licence, commercial use allowed "
                    "(https://marine.copernicus.eu/user-corner/service-commitments-and-licence)"),
        "attribution": ATTRIBUTION,
    }


SOURCE = ArraySource("ocean", GRID, MAX_FHR, DOI_URL, forecast_hours, source_info, MarineClient,
                     # Updated daily; the newest run starts about a day and a half before it is out.
                     stale_after_hours=60)
