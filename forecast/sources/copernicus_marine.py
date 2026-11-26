"""Copernicus Marine: surface currents and tides from the global ocean analysis and forecast.

The product is GLOBAL_ANALYSISFORECAST_PHY_001_024 (Mercator Ocean International's GLO12 model at 1/12°,
https://doi.org/10.48670/moi-00016). Two of its hourly datasets are read:

* ``merged-uv``: the total current at the surface, the ocean's circulation plus tidal currents (FES2014)
  plus the drift waves cause (Stokes drift). The model's own currents have no tides, and in tidal straits
  and on shelves tides are most of the current.
* ``merged-sl``: sea level, of which we take ``ocean_tide``, the astronomical tide's height (FES2014)
  around mean sea level. Its ``total_sea_level`` is measured from the geoid, so the ocean's mean shape
  (a metre or so from place to place) would show as much as the tide: not used.

Each is one rolling time series, rewritten daily as a new forecast comes out, that runs about ten days
past its newest forecast's start (the sea level a day further: tides are known ahead); we call the start
of the newest forecast both cover the run (the earlier series end less 239 hours, to 00 UTC) and read its
240 hours. They are read through the official ``copernicusmarine`` toolbox, which needs a (free)
Copernicus Marine account: COPERNICUSMARINE_SERVICE_USERNAME and ..._PASSWORD. Each hour is the global
1/12° field (about 42 MB for both current components, less for the tide); we keep 0.25°, each point the
mean of the 3 x 3 cells around it that are sea, so the coast keeps the values next to it. The hours are
hourly means, so each stored hour is interpolated from those around it (``CUBIC``).

Copernicus Marine's licence allows commercial use; derived products must say "Generated using E.U.
Copernicus Marine Service Information".
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from ..grid import Grid
from .base import ArraySource, StepUnavailable
from .gfs import GRID

log = logging.getLogger(__name__)

PRODUCT_ID = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
CURRENTS_DATASET = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
SEA_LEVEL_DATASET = "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i"
# The dataset each variable is read from.
DATASETS = {"utotal": CURRENTS_DATASET, "vtotal": CURRENTS_DATASET, "ocean_tide": SEA_LEVEL_DATASET}
# Both datasets' hours are hourly means, each labelled with the hour it starts: the value at 03:00 is the
# mean over 03:00-04:00, centred on 03:30. Their titles say "hourly mean" (the ids' "-i" notwithstanding),
# and against NOAA's predictions the tide on open coasts runs 26-30 minutes behind its labels, the current
# in wide tidal entrances 45. We store values at the hour, so each is interpolated from the means around
# it: a cubic through the four centred 1.5 and 0.5 hours either side, or at a series' last hour (no mean
# after it), a parabola through the three before. Both are exact for a straight line; on the main lunar
# tide the cubic keeps 99.8% of the amplitude.
CUBIC = {-2: -1 / 16, -1: 9 / 16, 0: 9 / 16, 1: -1 / 16}
AT_THE_END = {-2: -1 / 8, -1: 3 / 4, 0: 3 / 8}
DOI_URL = "https://doi.org/10.48670/moi-00016"
MAX_FHR = 239
CREDENTIALS = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")
ATTRIBUTION = (f"Generated using E.U. Copernicus Marine Service Information ({DOI_URL}). 4Shadow averages it "
               "onto a 0.25° grid and to the hour, derives current speed and direction, and finds the high "
               "and low waters.")

# The datasets' grid, rows turned to run north -> south: 80° S to 90° N and 180° W to 179.92° E, every 1/12°.
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


def _toolbox():
    """The ``copernicusmarine`` toolbox, imported here: only the ingest needs it. Its import makes every log
    time in the process UTC and gives its logger a handler of its own, so the ingest's log would change
    clock partway and print the toolbox's lines twice. This undoes both."""
    import copernicusmarine

    logging.Formatter.converter = time.localtime
    toolbox_log = logging.getLogger("copernicusmarine")
    for handler in list(toolbox_log.handlers):
        toolbox_log.removeHandler(handler)
    return copernicusmarine


class MarineClient:
    """Reads the datasets through the ``copernicusmarine`` toolbox."""

    def __init__(self) -> None:
        self._run: datetime | None = None
        self._opened: dict[str, object] = {}
        self._hours: dict[tuple[str, int], dict[str, np.ndarray]] = {}  # (dataset, label hour): fields

    def _open(self, dataset_id: str, start: datetime | None = None, end: datetime | None = None):
        missing = [k for k in CREDENTIALS if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"No Copernicus Marine login: set {' and '.join(missing)} (a free account at "
                               "https://data.marine.copernicus.eu/register)")
        copernicusmarine = _toolbox()
        window = {}
        if start is not None:
            window = {"start_datetime": f"{_utc(start):%Y-%m-%dT%H:%M:%S}", "end_datetime": f"{_utc(end):%Y-%m-%dT%H:%M:%S}"}
        # chunk_size_limit=0: no dask chunks of its own. By default the toolbox groups 50 hours into each,
        # so reading one hour downloads 50.
        return copernicusmarine.open_dataset(dataset_id=dataset_id, service="arco-geo-series",
                                             chunk_size_limit=0, **window)

    def latest_run(self, max_fhr: int = MAX_FHR) -> datetime | None:
        """The newest forecast both datasets cover: the sea level runs a day further ahead, so the
        currents' series usually decides."""
        lasts = []
        for dataset_id in sorted(set(DATASETS.values())):
            times = self._open(dataset_id).time.values
            if not len(times):
                return None
            lasts.append(datetime.fromtimestamp(times[-1].astype("datetime64[s]").astype(int), timezone.utc))
        last = min(lasts)
        run = run_for(last)
        return run if run + timedelta(hours=max_fhr) <= last else None

    def _hour(self, dataset_id: str, ds, run: datetime, label: int, names: list[str], grid: Grid):
        """The named fields' hourly means labelled ``label`` hours into the run, on ``grid``; kept for the
        next steps, which read most of the same hours."""
        key = (dataset_id, label)
        held = self._hours.get(key, {})
        if all(name in held for name in names):
            return held
        when = np.datetime64(_utc(run + timedelta(hours=label)), "ns")
        if when not in ds.time.values:
            raise StepUnavailable(f"{dataset_id} has no {when}")
        hour = ds[names].sel(time=when)
        if "depth" in hour.dims:
            hour = hour.isel(depth=0)
        hour = hour.load()
        flip = bool(hour.latitude.values[0] < hour.latitude.values[-1])
        for name in names:
            values = np.asarray(hour[name].values, np.float32)
            if values.shape != NATIVE.shape:
                raise RuntimeError(f"{dataset_id} {name}: shape {values.shape}, expected {NATIVE.shape}")
            if np.isnan(values).all():
                raise StepUnavailable(f"{dataset_id} {name} at {when}: no data yet")
            held[name] = block_mean(values[::-1] if flip else values, NATIVE, grid)
        self._hours[key] = held
        return held

    def step(self, run: datetime, fhr: int, names: list[str], grid: Grid) -> dict[str, np.ndarray]:
        if self._run != run:
            self._opened, self._hours, self._run = {}, {}, run
        out = {}
        for dataset_id in sorted({DATASETS[name] for name in names}):
            group = [name for name in names if DATASETS[name] == dataset_id]
            ds = self._opened.get(dataset_id)
            if ds is None:
                # From two hours before the run (the means its first hour is interpolated from) to its last
                # hour, which the currents' series ends with: that hour is interpolated from those before.
                ds = self._opened[dataset_id] = self._open(dataset_id, run - timedelta(hours=2),
                                                           run + timedelta(hours=MAX_FHR))
            after = np.datetime64(_utc(run + timedelta(hours=fhr + 1)), "ns")
            weights = AT_THE_END if after > ds.time.values[-1] else CUBIC
            parts = [(w, self._hour(dataset_id, ds, run, fhr + offset, group, grid)) for offset, w in weights.items()]
            for name in group:
                out[name] = sum(w * part[name] for w, part in parts).astype(np.float32)
        # Keep only the hours the next step reads (from fhr - 1): a run's hours would fill gigabytes.
        for key in [key for key in self._hours if key[1] < fhr - 1]:
            del self._hours[key]
        return out


def source_info(base_url: str, run: datetime, products=()) -> dict:
    return {
        "provider": ("Copernicus Marine Service global ocean analysis and forecast (Mercator Ocean International "
                     "GLO12, 1/12°): surface currents with tidal currents (FES2014) and wave drift, and tide "
                     "heights (FES2014)"),
        "distribution": "Copernicus Marine Service",
        "url": base_url,
        "datasets": [CURRENTS_DATASET, SEA_LEVEL_DATASET],
        "license": ("Copernicus Marine Service licence, commercial use allowed "
                    "(https://marine.copernicus.eu/user-corner/service-commitments-and-licence)"),
        "attribution": ATTRIBUTION,
    }


SOURCE = ArraySource("ocean", GRID, MAX_FHR, DOI_URL, forecast_hours, source_info, MarineClient,
                     # Updated daily; the newest run starts about a day and a half before it is out.
                     stale_after_hours=60)
