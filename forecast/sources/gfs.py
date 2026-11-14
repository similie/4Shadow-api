"""NOAA GFS 0.25° source: run discovery, .idx parsing and byte-range downloads.

Every GFS GRIB2 file on NOAA's open-data bucket has a sibling ``.idx`` listing each
message's byte offset. We fetch only the messages in the catalog (~2% of each ~540 MB file)
with HTTP Range requests, which makes ingesting all four daily runs cheap.

Two file families ("products") share the run's 0.25° grid and forecast steps: ``atmos``
(gfs.tCCz.pgrb2.0p25.fFFF) and ``wave`` (GFS-Wave, gfswave.tCCz.global.0p25.fFFF.grib2).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from ..grid import Grid

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GRID = Grid(lat0=90.0, dlat=0.25, nlat=721, lon0=0.0, dlon=0.25, nlon=1440)
CYCLE_HOURS = (0, 6, 12, 18)
MAX_FHR = 384


def forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """GFS 0.25° output steps: hourly to f120, then 3-hourly to f384."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(0, min(max_fhr, 120) + 1)) + list(range(123, max_fhr + 1, 3))


PRODUCTS = ("atmos", "wave")


def file_url(base_url: str, run: datetime, fhr: int, product: str = "atmos") -> str:
    if product == "atmos":
        return f"{base_url}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{fhr:03d}"
    if product == "wave":
        return f"{base_url}/gfs.{run:%Y%m%d}/{run:%H}/wave/gridded/gfswave.t{run:%H}z.global.0p25.f{fhr:03d}.grib2"
    raise ValueError(f"Unknown GFS product {product!r}")


def source_info(base_url: str, run: datetime, products=("atmos",)) -> dict:
    info = {
        "provider": "NOAA/NCEP Global Forecast System (GFS), 0.25°",
        "distribution": "NOAA Open Data Dissemination (AWS bucket noaa-gfs-bdp-pds)",
        "url": f"{base_url}/gfs.{run:%Y%m%d}/{run:%H}/atmos/",
        "license": "Public domain (U.S. Government work)",
    }
    if "wave" in products:
        info["provider"] += ", with GFS-Wave (WAVEWATCH III) sea state"
        info["wave_url"] = f"{base_url}/gfs.{run:%Y%m%d}/{run:%H}/wave/gridded/"
    return info


# --------------------------------------------------------------------------- .idx parsing


@dataclass(frozen=True)
class IdxRecord:
    num: int
    start: int
    end: int | None  # inclusive; None for the last message in the file
    param: str
    level: str
    fcst: str


@dataclass(frozen=True)
class FcstSpec:
    stat: str  # instant | acc | ave | max | min
    start: int | None  # hours; None for instant
    end: int  # hours


_RANGE_RE = re.compile(r"^(\d+)-(\d+) (hour|day) (acc|ave|max|min) fcst$")
_INSTANT_RE = re.compile(r"^(\d+) (hour|day) fcst$")


def parse_idx(text: str) -> list[IdxRecord]:
    rows = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        rows.append((int(parts[0]), int(parts[1]), parts[3], parts[4], parts[5]))
    rows.sort()
    out = []
    for k, (num, start, param, level, fcst) in enumerate(rows):
        end = rows[k + 1][1] - 1 if k + 1 < len(rows) else None
        out.append(IdxRecord(num, start, end, param, level, fcst))
    return out


def parse_fcst(fcst: str) -> FcstSpec | None:
    if fcst == "anl":
        return FcstSpec("instant", None, 0)
    m = _INSTANT_RE.match(fcst)
    if m:
        mult = 24 if m.group(2) == "day" else 1
        return FcstSpec("instant", None, int(m.group(1)) * mult)
    m = _RANGE_RE.match(fcst)
    if m:
        mult = 24 if m.group(3) == "day" else 1
        return FcstSpec(m.group(4), int(m.group(1)) * mult, int(m.group(2)) * mult)
    return None


def select_record(records: list[IdxRecord], param: str, level: str, kind: str, fhr: int) -> tuple[IdxRecord, FcstSpec] | None:
    """Pick the one message that represents ``param@level`` at forecast hour ``fhr``.

    * instant -> the analysis / "N hour fcst" message
    * accum   -> the running total since initialisation ("0-N hour acc" / "0-K day acc")
    * average -> the current averaging bucket ("a-N hour ave"); the shortest if several
    """
    best: tuple[IdxRecord, FcstSpec] | None = None
    for rec in records:
        if rec.param != param or rec.level != level:
            continue
        spec = parse_fcst(rec.fcst)
        if spec is None or spec.end != fhr:
            continue
        if kind == "instant" and spec.stat == "instant":
            return rec, spec
        if kind == "accum" and spec.stat == "acc" and spec.start == 0:
            return rec, spec
        if kind == "average" and spec.stat == "ave":
            if best is None or (spec.start or 0) > (best[1].start or 0):
                best = (rec, spec)
    return best


@dataclass(frozen=True)
class RangeGroup:
    start: int
    end: int | None
    records: tuple[IdxRecord, ...]


def merge_ranges(records: list[IdxRecord], max_gap: int = 512 * 1024) -> list[RangeGroup]:
    """Coalesce nearby messages into single Range requests (S3 allows one range per GET)."""
    groups: list[RangeGroup] = []
    for rec in sorted(set(records), key=lambda r: r.start):
        last = groups[-1] if groups else None
        if last is not None and last.end is not None and rec.start - last.end - 1 <= max_gap:
            groups[-1] = RangeGroup(last.start, rec.end, last.records + (rec,))
        else:
            groups.append(RangeGroup(rec.start, rec.end, (rec,)))
    return groups


# --------------------------------------------------------------------------- HTTP client


class GfsClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0, retries: int = 4,
                 session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=32)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers["User-Agent"] = "foreshadow-forecast/2 (+open-source GFS ingest)"

    def _request(self, method: str, url: str, headers: dict | None = None,
                 not_found_ok: bool = False) -> requests.Response | None:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.request(method, url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
            else:
                if resp.status_code in (200, 206):
                    return resp
                if resp.status_code in (403, 404) and not_found_ok:
                    return None
                if resp.status_code < 500 and resp.status_code != 429:
                    resp.raise_for_status()
                last = requests.HTTPError(f"{resp.status_code} for {url}")
            time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"{method} {url} failed after {self.retries} attempts: {last}")

    def has_idx(self, run: datetime, fhr: int, product: str = "atmos") -> bool:
        url = file_url(self.base_url, run, fhr, product) + ".idx"
        return self._request("HEAD", url, not_found_ok=True) is not None

    def idx(self, run: datetime, fhr: int, product: str = "atmos") -> list[IdxRecord] | None:
        resp = self._request("GET", file_url(self.base_url, run, fhr, product) + ".idx", not_found_ok=True)
        return None if resp is None else parse_idx(resp.text)

    def fetch_records(self, run: datetime, fhr: int, records: list[IdxRecord],
                      product: str = "atmos") -> dict[int, bytes]:
        url = file_url(self.base_url, run, fhr, product)
        blobs: dict[int, bytes] = {}
        for group in merge_ranges(records):
            rng = f"bytes={group.start}-" + ("" if group.end is None else str(group.end))
            resp = self._request("GET", url, headers={"Range": rng})
            assert resp is not None
            if resp.status_code != 206:
                raise RuntimeError(f"Server ignored Range request for {url}")
            data = resp.content
            for rec in group.records:
                lo = rec.start - group.start
                hi = None if rec.end is None else rec.end - group.start + 1
                blob = data[lo:hi]
                if not (blob.startswith(b"GRIB") and blob.endswith(b"7777")):
                    raise RuntimeError(f"Corrupt GRIB message {rec.param}:{rec.level} in {url}")
                blobs[rec.num] = blob
        return blobs

    def latest_complete_run(self, now: datetime | None = None, max_fhr: int = MAX_FHR,
                            lookback_cycles: int = 8) -> datetime | None:
        """Newest cycle whose final required step has been published."""
        now = now or datetime.now(timezone.utc)
        cycle = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0, microsecond=0)
        last_fhr = forecast_hours(max_fhr)[-1]
        for k in range(lookback_cycles):
            run = cycle - timedelta(hours=6 * k)
            if self.has_idx(run, last_fhr):
                return run
        return None


# --------------------------------------------------------------------------- decoding


def decode_message(blob: bytes, grid: Grid = GRID) -> np.ndarray:
    """Decode one GRIB2 message into a float32 array oriented like ``grid`` (north -> south)."""
    import pygrib  # deferred: only the ingest process needs ecCodes

    msg = pygrib.fromstring(blob)
    vals = msg.values
    if np.ma.isMaskedArray(vals):
        vals = vals.astype(np.float64).filled(np.nan)
    arr = np.asarray(vals, dtype=np.float32)
    if arr.shape != grid.shape:
        raise ValueError(f"Unexpected grid shape {arr.shape}, expected {grid.shape}")
    if int(msg["iScansNegatively"]) != 0:
        raise ValueError("Unsupported GRIB scan order (iScansNegatively=1)")
    north = float(msg["latitudeOfFirstGridPointInDegrees"])
    if int(msg["jScansPositively"]) == 1:
        arr = arr[::-1]
        north = float(msg["latitudeOfLastGridPointInDegrees"])
    west = float(msg["longitudeOfFirstGridPointInDegrees"]) % 360.0
    if abs(north - grid.lat0) > 1e-3 or abs(west - grid.lon0 % 360.0) > 1e-3:
        raise ValueError(f"Grid origin mismatch: ({north}, {west}) vs ({grid.lat0}, {grid.lon0})")
    return np.ascontiguousarray(arr)
