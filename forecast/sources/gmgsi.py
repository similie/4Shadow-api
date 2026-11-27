"""Satellite imagery: NOAA's Global Mosaic of Geostationary Satellite Imagery (GMGSI).

Each hour NOAA blends the newest images of the geostationary weather satellites (GOES-East and -West,
Meteosat at 0° and over the Indian Ocean, Himawari) into one picture of the globe from 72.7° N to 72.7° S,
about 8 km across, and publishes it on NOAA's open-data bucket on AWS some 40 minutes after the hour: a
NetCDF file per band. Three bands are read:

- ``LW``, the infrared window (about 11 µm): cloud tops by their temperature, day and night, and under a
  clear sky the ground or the sea.
- ``WV``, water vapour (about 6.7 µm): the moisture of the middle and upper air.
- ``VIS``, visible: sunlight off clouds and the ground, so nothing at night.

Values are 0-255 brightness on McIDAS's scale. For the infrared bands that's a temperature: 330 - B/2 kelvin
up to 176 (242 K), then 418 - B, so 255 is 163 K; a clear tropical sea reads about 4 K colder than the air
above it, as an 11 µm channel does through moist air. The visible band stays as its 0-255 brightness:
NOAA doesn't document that scale. Pixels the quality flag doesn't mark good are missing: the night side in
the visible band, and seams between satellites. So is what every satellite sees more than 78° from straight
down (towards the poles, and between satellites beyond about 60°): that far round the curve the picture is
smeared and, in places, impossibly cold (down to -108 °C off Antarctica, where real storm tops stop near
-90 °C).

Each hour is a run of one step, its hour the run's time, so the runs are the hours observed; the store
keeps the last day of them. The mosaic is on a Mercator grid: each 0.1° cell of the stored grid is the
mean of the pixels whose centres fall in it.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from collections.abc import Sequence
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from ..grid import Grid
from .base import ArraySource, StepUnavailable

log = logging.getLogger(__name__)

BASE_URL = "https://noaa-gmgsi-pds.s3.amazonaws.com"
REGISTRY_URL = "https://registry.opendata.aws/noaa-gmgsi/"
GRID = Grid(lat0=90.0, dlat=0.1, nlat=1801, lon0=0.0, dlon=0.1, nlon=3600)
BANDS = ("LW", "WV", "VIS")
INFRARED = ("LW", "WV")
KEEP_HOURS = 24
GOOD = 0  # the quality flag of a good pixel: 10 is no data (read as 0), 20 a bad one (read as 255)
# Where the mosaic's satellites sit over the equator (°E): GOES-East and -West, Meteosat 0° and Indian Ocean,
# Himawari. A cell is left out when it's more than MAX_ZENITH from straight down for every one of them.
SATELLITE_LONGITUDES = {"GOES-19": -75.2, "GOES-18": -137.2, "Meteosat-10": 0.0, "Meteosat-9": 45.5, "Himawari-9": 140.7}
MAX_ZENITH = 78.0
EARTH_RADIUS_KM = 6378.137
ORBIT_RADIUS_KM = 42164.0
_HOUR_PATH = re.compile(r"^GMGSI_[A-Z]+/(\d{4})/(\d{2})/(\d{2})/(\d{2})/[^/]+\.nc$")

ATTRIBUTION = ("Satellite: NOAA Global Mosaic of Geostationary Satellite Imagery (GMGSI), from GOES (NOAA), "
               "Meteosat (EUMETSAT) and Himawari (JMA). 4Shadow averages it onto a 0.1° grid.")


def brightness_temperature(counts: np.ndarray) -> np.ndarray:
    """Kelvin from 0-255 infrared brightness on McIDAS's scale (higher is colder)."""
    b = np.asarray(counts, dtype=np.float32)
    return np.where(b > 176, np.float32(418.0) - b, np.float32(330.0) - b / np.float32(2.0))


def mean_on_grid(values: np.ndarray, lats: np.ndarray, lons: np.ndarray, grid: Grid, block: int = 500) -> np.ndarray:
    """``values`` [row, column] at pixel centres on rows of latitude ``lats`` and columns of longitude
    ``lons`` as ``grid``: each cell the mean of the finite pixels nearest it, missing where there are none.
    Taken ``block`` rows at a time, to keep the index arrays small."""
    rows = np.rint((grid.lat0 - np.asarray(lats, np.float64)) / grid.dlat).astype(np.int64)
    cols = np.rint(np.mod(np.asarray(lons, np.float64) - grid.lon0, 360.0) / grid.dlon).astype(np.int64) % grid.nlon
    n = grid.nlat * grid.nlon
    total = np.zeros(n, np.float64)
    count = np.zeros(n, np.int64)
    for start in range(0, len(rows), block):
        r = rows[start:start + block]
        inside = (r >= 0) & (r < grid.nlat)
        v = np.asarray(values[start:start + block], np.float32)[inside].ravel()
        cell = (r[inside, None] * grid.nlon + cols[None, :]).ravel()
        ok = np.isfinite(v)
        total += np.bincount(cell[ok], weights=v[ok], minlength=n)
        count += np.bincount(cell[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out.reshape(grid.shape).astype(np.float32)


@lru_cache(maxsize=4)
def oblique(grid: Grid, max_zenith: float = MAX_ZENITH) -> np.ndarray:
    """Cells every satellite sees more than ``max_zenith`` degrees from straight down (its zenith angle)."""
    lat = np.radians(grid.lat0 - np.arange(grid.nlat) * grid.dlat)[:, None]
    lon = np.radians(grid.lon0 + np.arange(grid.nlon) * grid.dlon)[None, :]
    nearest = np.full(grid.shape, 180.0)
    for sub in SATELLITE_LONGITUDES.values():
        # The angle at the Earth's centre between the cell and the point under the satellite, then the
        # angle at the satellite between that point and the cell.
        centre = np.arccos(np.clip(np.cos(lat) * np.cos(lon - np.radians(sub)), -1.0, 1.0))
        reach = np.sqrt(EARTH_RADIUS_KM**2 + ORBIT_RADIUS_KM**2 - 2 * EARTH_RADIUS_KM * ORBIT_RADIUS_KM * np.cos(centre))
        nadir = np.arcsin(np.clip(EARTH_RADIUS_KM * np.sin(centre) / reach, -1.0, 1.0))
        np.minimum(nearest, np.degrees(centre + nadir), out=nearest)
    return nearest > max_zenith


def hour_of(key: str) -> datetime | None:
    m = _HOUR_PATH.match(key)
    return datetime(*map(int, m.groups()), tzinfo=timezone.utc) if m else None


def read_band(path: Path, band: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mosaic file's values (kelvin for the infrared bands, else 0-255), missing where not good, and
    its rows' latitudes and columns' longitudes."""
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        data = ds["data"][0].values
        flag = ds["dqf"][0].values
        # The grid is regular in latitude by row and longitude by column: one column and one row say it all.
        lats = ds["lat"][:, 0].values.astype(np.float64)
        lons = ds["lon"][0].values.astype(np.float64)
    values = brightness_temperature(data) if band in INFRARED else data.astype(np.float32)
    # The top row has no flag (NaN), and some of it no data: it's left out, as the next row covers its cells.
    values[flag != GOOD] = np.nan
    return values, lats, lons


class MosaicClient:
    """Reads the hourly mosaics from NOAA's bucket: which hours have every band, and an hour's bands."""

    def __init__(self, base_url: str = BASE_URL, session: requests.Session | None = None,
                 now: datetime | None = None, retries: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        if session is None:
            self.session.headers["User-Agent"] = "4shadow-forecast/2 (+https://github.com/similie/4Shadow-api)"
        self.now = now  # None: the clock
        self.retries = retries
        self._keys: dict[tuple[str, datetime], str] = {}

    def _list(self, band: str, day: datetime) -> dict[datetime, str]:
        r = self.session.get(self.base_url + "/", params={"list-type": "2", "prefix": f"GMGSI_{band}/{day:%Y/%m/%d}/"},
                             timeout=60)
        r.raise_for_status()
        found: dict[datetime, str] = {}
        for key in re.findall(r"<Key>([^<]+)</Key>", r.text):
            hour = hour_of(key)
            if hour is not None:
                # A reprocessed hour has a later creation time in its name: the newest wins.
                found[hour] = max(found.get(hour, key), key)
        for hour, key in found.items():
            self._keys[(band, hour)] = key
        return found

    def available_runs(self, hours: int = KEEP_HOURS) -> list[datetime]:
        """The hours of the last ``hours`` with every band published, oldest first."""
        now = self.now or datetime.now(timezone.utc)
        since = now - timedelta(hours=hours)
        days = sorted({since.replace(hour=0, minute=0, second=0, microsecond=0),
                       now.replace(hour=0, minute=0, second=0, microsecond=0)})
        common: set[datetime] | None = None
        for band in BANDS:
            have = {h for day in days for h in self._list(band, day) if since < h <= now}
            common = have if common is None else common & have
        return sorted(common or ())

    def latest_run(self, max_fhr: int = 0) -> datetime | None:
        runs = self.available_runs()
        return runs[-1] if runs else None

    def step(self, run: datetime, fhr: int, names: list[str], grid: Grid) -> dict[str, np.ndarray]:
        out = {}
        with tempfile.TemporaryDirectory(prefix="gmgsi-") as tmp:
            for band in names:
                key = self._keys.get((band, run)) or self._list(band, run).get(run)
                if key is None:
                    raise StepUnavailable(f"GMGSI {band} at {run:%Y-%m-%d %H}Z: not published")
                path = Path(tmp) / f"{band}.nc"
                self._download(key, path)
                values, lats, lons = read_band(path, band)
                cells = mean_on_grid(values, lats, lons, grid)
                cells[oblique(grid)] = np.nan
                out[band] = cells
        return out


    def _download(self, key: str, path: Path) -> None:
        """One file, tried again after a dropped connection or a server error (S3 closes some idle
        connections while a band is being decoded)."""
        for attempt in range(self.retries):
            try:
                with self.session.get(f"{self.base_url}/{key}", timeout=120, stream=True) as r:
                    r.raise_for_status()
                    with open(path, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                return
            except requests.RequestException as exc:
                status = getattr(exc.response, "status_code", None)
                if attempt == self.retries - 1 or (status is not None and status < 500):
                    raise
                log.warning("GMGSI %s: %s; trying again", key, exc)
                time.sleep(min(2**attempt, 20))


def forecast_hours(max_fhr: int | None = None) -> list[int]:
    return [0]


def source_info(base_url: str, run: datetime, products: Sequence[str] = ()) -> dict:
    return {
        "provider": ("NOAA NESDIS Global Mosaic of Geostationary Satellite Imagery (GMGSI): GOES-East and -West "
                     "(NOAA), Meteosat (EUMETSAT) and Himawari (JMA), blended hourly"),
        "distribution": "NOAA Open Data Dissemination on AWS",
        "url": REGISTRY_URL,
        "bands": list(BANDS),
        "license": "NOAA open data: \"open to the public and can be used as desired\"; NOAA asks for credit",
        "attribution": ATTRIBUTION,
    }


SOURCE = ArraySource("satellite", GRID, 0, BASE_URL, forecast_hours, source_info, MosaicClient,
                     cycle_hours=tuple(range(24)), keep_runs=KEEP_HOURS, gfs_static=False,
                     # Published about 40 minutes after the hour; behind by more than three, it's stuck.
                     stale_after_hours=3)
