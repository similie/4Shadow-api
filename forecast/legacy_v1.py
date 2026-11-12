"""Compatibility layer for clients of the retired v1 forecast service (temporary).

The v1 service (``tile_renderers/``) is being shut down. This module serves v1's response
shapes from the v2 store, so existing v1 clients keep working while they move to ``/v2``.
The API-key router maps the public ``/v1/...`` routes to ``/legacy/v1/...`` here. Delete this
module, its JSON file and its router rows once no client calls ``/v1``.

What v1 clients will notice:

* Values come from the newest complete GFS run. v1 often lagged a cycle or two behind.
* The 5-day, 3-hourly grid has no gaps (v1 sometimes skipped an hour between runs).
* ``precipitation-rate`` is the mean rate over the hour(s) before the time, not an
  instantaneous rate. ``total-precipitation`` accumulates from the run's start, and
  ``daily-total-precipitation`` from 00 UTC of that day, as in v1.
* Tiles use v2's fixed colour scales. ``/parameters`` is v1's static metadata.
* The POST endpoints (multi-parameter and streaming forecasts) are retired and return 410.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from .catalog import derive
from .store import Run, RunStore, decode

PREFIX = "/legacy/v1"
OFFSETS = list(range(0, 121, 3))  # v1: 5 days, 3-hourly, relative to the current hour
DECIMALS = 4
KELVIN = 273.15
LEGACY_MAX_ZOOM = 18
CACHE_CONTROL = "public, max-age=300"
PARAMETERS_FILE = Path(__file__).with_name("legacy_v1_parameters.json")
RETIRED = ("This v1 endpoint has been retired. Use GET /v2/forecast instead "
           "(https://4shadow.io/documentation/api).")

# Pseudo-sources computed from the v2 precipitation rate (mm/h).
TOTAL_PRECIP = "precip:total"
DAILY_PRECIP = "precip:daily"
PRECIP_RATE = "precip:rate"


@dataclass(frozen=True)
class V1Param:
    key: str
    short_name: str
    parameter_name: str
    name: str
    units: str
    type_of_level: str
    level: int
    step_type: str
    source: str  # v2 variable id, or one of the precipitation pseudo-sources
    add: float = 0.0  # v2 value -> v1 native units: value * mul + add
    mul: float = 1.0
    fixed_range: tuple[float, float] | None = None


# v1's default forecast parameters, in the order v1 returned them (sorted by key).
PARAMS: tuple[V1Param, ...] = (
    V1Param("10-metre-u-wind-component", "10u", "u-component of wind", "10 metre U wind component",
            "m s-1", "heightAboveGround", 10, "instant", "u10"),
    V1Param("10-metre-v-wind-component", "10v", "v-component of wind", "10 metre V wind component",
            "m s-1", "heightAboveGround", 10, "instant", "v10"),
    V1Param("10-metre-wind-direction", "wd", "10 Meter Wind Direction", "10 metre derived wind direction",
            "degrees", "heightAboveGround", 10, "instant", "wind_dir", fixed_range=(0.0, 360.0)),
    V1Param("10-metre-wind-speed", "ws", "10 Meter Wind Speed", "10 metre derived wind speed",
            "m s-1", "heightAboveGround", 10, "instant", "wind_speed"),
    V1Param("2-metre-dewpoint-temperature", "2d", "Dew point temperature", "2 metre dewpoint temperature",
            "K", "heightAboveGround", 2, "instant", "d2m", add=KELVIN),
    V1Param("2-metre-relative-humidity", "2r", "Relative humidity", "2 metre relative humidity",
            "%", "heightAboveGround", 2, "instant", "rh2m"),
    V1Param("2-metre-specific-humidity", "2sh", "Specific humidity", "2 metre specific humidity",
            "kg kg-1", "heightAboveGround", 2, "instant", "q2m", mul=0.001),
    V1Param("2-metre-temperature", "2t", "Temperature", "2 metre temperature",
            "K", "heightAboveGround", 2, "instant", "t2m", add=KELVIN),
    V1Param("daily-total-precipitation", "dtp", "Daily Total precipitation", "Daily Derived Total precipitation",
            "kg m-2", "surface", 0, "accum", DAILY_PRECIP),
    V1Param("precipitation-rate", "prate", "Precipitation rate", "Precipitation rate",
            "kg m-2 s-1", "surface", 0, "instant", PRECIP_RATE, mul=1 / 3600),
    V1Param("surface-pressure", "sp", "Pressure", "Surface pressure",
            "Pa", "surface", 0, "instant", "sp", mul=100.0),
    V1Param("total-cloud-cover", "tcc", "Total cloud cover", "Total Cloud Cover",
            "%", "atmosphere", 0, "avg", "tcc"),
    V1Param("total-precipitation", "tp", "Total precipitation", "Total Precipitation",
            "kg m-2", "surface", 0, "accum", TOTAL_PRECIP),
    V1Param("wind-speed-gust", "gust", "Wind speed (gust)", "Wind speed (gust)",
            "m s-1", "surface", 0, "instant", "gust"),
)

# v1 tile parameter keys -> v2 tileable variables.
TILE_VARS = {
    "2-metre-temperature": "t2m",
    "temperature": "t2m",
    "2-metre-dewpoint-temperature": "d2m",
    "2-metre-relative-humidity": "rh2m",
    "relative-humidity": "rh2m",
    "10-metre-wind-speed": "wind_speed",
    "wind-speed-gust": "gust",
    "pressure-reduced-to-msl": "mslp",
    "total-precipitation": "precip",
    "precipitation-rate": "precip",
    "total-cloud-cover": "tcc",
    "convective-available-potential-energy": "cape",
    "precipitable-water": "pwat",
    "visibility": "vis",
    "downward-short-wave-radiation-flux": "dswrf",
    "snow-depth": "snod",
}


def base_time(now: datetime | None = None) -> datetime:
    """v1's reference time: the current UTC hour, or the next one from half past."""
    now = now or datetime.now(timezone.utc)
    if now.minute >= 30:
        now += timedelta(hours=1)
    return now.replace(minute=0, second=0, microsecond=0)


def closest_offset(hour_offset: int) -> int:
    if hour_offset <= 0:
        return 0
    return min(OFFSETS, key=lambda o: abs(o - hour_offset))


def _needed_sources(run: Run) -> list[str]:
    base = {"u10", "v10", "d2m", "rh2m", "q2m", "t2m", "sp", "tcc", "gust", "precip"}
    return sorted(base & set(run.variables))


class PointSeries:
    """All v1 parameters for one location, evaluable at any hour since the run's start."""

    def __init__(self, run: Run, lat: float, lon: float) -> None:
        self.run = run
        self.fh = run.forecast_hours.astype(np.float64)
        self.series = run.point_series(_needed_sources(run), lat, lon)
        if "u10" in self.series and "v10" in self.series:
            for vid in ("wind_speed", "wind_dir"):
                self.series[vid] = derive(vid, self.series)
        if "precip" in self.series:
            rate = np.nan_to_num(self.series["precip"].astype(np.float64), nan=0.0)
            hours = np.diff(self.fh, prepend=self.fh[0])
            self.cumulative = np.cumsum(rate * hours)
        else:
            self.cumulative = None

    def available(self, p: V1Param) -> bool:
        if p.source in (TOTAL_PRECIP, DAILY_PRECIP, PRECIP_RATE):
            return self.cumulative is not None
        return p.source in self.series

    def covers(self, h: float) -> bool:
        return self.fh[0] <= h <= self.fh[-1]

    def _total(self, h: float) -> float:
        # Rates are constant within each interval, so the running total is piecewise linear.
        return float(np.interp(h, self.fh, self.cumulative))

    def value(self, p: V1Param, h: float) -> float | None:
        if not self.covers(h):
            return None
        if p.source == TOTAL_PRECIP:
            v = self._total(h)
        elif p.source == DAILY_PRECIP:
            valid = self.run.init_time + timedelta(hours=h)
            midnight = valid.replace(hour=0, minute=0, second=0, microsecond=0)
            h0 = max((midnight - self.run.init_time).total_seconds() / 3600.0, float(self.fh[0]))
            v = self._total(h) - self._total(h0)
        elif p.source == PRECIP_RATE:
            k = min(bisect.bisect_left(self.fh, h), len(self.fh) - 1)  # interval ending at or after h
            v = float(self.series["precip"][k])
        else:
            v = float(np.interp(h, self.fh, self.series[p.source]))
        if not np.isfinite(v):
            return None
        return round(v * p.mul + p.add, DECIMALS)


def _step_near(run: Run, h: float) -> int:
    return int(np.argmin(np.abs(run.forecast_hours - h)))


def _step_types(p: V1Param, fhr: int) -> tuple[int, int]:
    if p.step_type == "accum":
        return 0, fhr
    if p.step_type == "avg" and fhr > 0:
        return fhr - ((fhr - 1) % 6 + 1), fhr
    return fhr, fhr


def build_router(stores: dict[str, RunStore], tile_bytes: Callable[..., bytes]) -> APIRouter:
    router = APIRouter(prefix=PREFIX, include_in_schema=False)
    store = stores["gfs"]

    def latest() -> Run:
        run = store.get("latest")
        if run is None:
            raise HTTPException(404, "Data is not ready for output.")
        return run

    @lru_cache(maxsize=512)
    def source_range(run_id: str, source: str, step: int) -> tuple[float, float]:
        """Global min/max of one source field at one step, in v2 units (v1 reported these per GRIB message)."""
        run = store.get(run_id)

        def field(vid: str, k: int) -> np.ndarray:
            meta = run.variables[vid]
            return decode(run.array(vid)[k], meta["scale"], meta["offset"])

        if source == TOTAL_PRECIP:
            hours = np.diff(run.forecast_hours, prepend=run.forecast_hours[0])
            values = np.zeros(run.grid.shape, np.float64)
            for k in range(1, step + 1):
                values += np.nan_to_num(field("precip", k), nan=0.0) * hours[k]
        elif source == PRECIP_RATE:
            values = field("precip", step)
        elif source in ("wind_speed", "wind_dir"):
            values = derive(source, {"u10": field("u10", step), "v10": field("v10", step)})
        else:
            values = field(source, step)
        return float(np.nanmin(values)), float(np.nanmax(values))

    def field_range(run: Run, p: V1Param, step: int) -> tuple[float, float]:
        if p.fixed_range:
            return p.fixed_range
        source = TOTAL_PRECIP if p.source == DAILY_PRECIP else p.source  # v1 reused tp's range for dtp
        lo, hi = source_range(run.id, source, step)
        return lo * p.mul + p.add, hi * p.mul + p.add

    def metadata(run: Run, p: V1Param, fhr: int, forecast_time: bool = False) -> dict:
        lo, hi = field_range(run, p, _step_near(run, fhr))
        start, end = _step_types(p, fhr)
        meta = {
            "parameterName": p.parameter_name,
            "parameterUnits": p.units,
            "shortName": p.short_name,
            "typeOfLevel": p.type_of_level,
            "level": p.level,
            "minimum": round(lo, 6),
            "maximum": round(hi, 6),
            "dataDate": int(run.init_time.strftime("%Y%m%d")),
            "dataTime": run.init_time.hour * 100,
        }
        if forecast_time:
            meta["forecastTime"] = fhr
        meta.update({"name": p.name, "stepType": p.step_type, "startStep": start, "endStep": end, "key": p.key})
        return meta

    def hours_since_init(run: Run, valid: datetime) -> int:
        return int(round((valid - run.init_time).total_seconds() / 3600.0))

    @router.get("/point")
    def point(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360),
              hour_offset: int = 0):
        run = latest()
        valid = base_time() + timedelta(hours=closest_offset(hour_offset))
        h = hours_since_init(run, valid)
        ps = PointSeries(run, lat, lon)
        if not ps.covers(h):
            raise HTTPException(404, "No data for that offset.")
        out = []
        for p in PARAMS:
            if not ps.available(p):
                continue
            meta = metadata(run, p, h, forecast_time=True)
            out.append({"unit": p.units, "metadata": meta, "value": ps.value(p, h), "datetime": valid.isoformat()})
        return JSONResponse(out, headers={"Cache-Control": CACHE_CONTROL, "X-Forecast-Run": run.id})

    @router.get("/forecast")
    def forecast(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360),
                 hour_offset: int = 0):
        run = latest()
        base = base_time()
        ps = PointSeries(run, lat, lon)
        first = OFFSETS.index(closest_offset(hour_offset))
        steps = []
        for off in OFFSETS[first:]:
            valid = base + timedelta(hours=off)
            h = hours_since_init(run, valid)
            if ps.covers(h):
                steps.append((off, valid, h))
        if not steps:
            raise HTTPException(404, "No data for that offset.")
        out = []
        for p in PARAMS:
            if not ps.available(p):
                continue
            meta = metadata(run, p, steps[0][2])
            meta["forecastStart"], meta["forecastEnd"] = steps[0][2], steps[-1][2]
            values = [{"value": ps.value(p, h), "datetime": valid.isoformat(), "offset": off,
                       "day": valid.day, "hour": valid.hour} for off, valid, h in steps]
            out.append({"values": values, "metadata": meta})
        return JSONResponse(out, headers={"Cache-Control": CACHE_CONTROL, "X-Forecast-Run": run.id})

    @router.get("/parameters")
    def parameters():
        return Response(_parameters_json(), media_type="application/json",
                        headers={"Cache-Control": "public, max-age=86400"})

    @router.post("/point")
    @router.post("/forecast")
    @router.post("/forecast-stream")
    def retired():
        return JSONResponse({"detail": RETIRED}, status_code=410)

    @router.get("/tiles/{model}/{param_key}/{hour_offset}/{z}/{x}/{y}.png")
    def tile(model: str, param_key: str, hour_offset: int, z: int, x: int, y: int):
        var = TILE_VARS.get(param_key)
        if model != "gfs" or var is None:
            raise HTTPException(404, f"No legacy tiles for {model}/{param_key}. Use /v2/gfs/tiles.")
        if not (0 <= z <= LEGACY_MAX_ZOOM and 0 <= x < 2**z and 0 <= y < 2**z):
            raise HTTPException(404, "Tile outside the world")
        run = latest()
        h = hours_since_init(run, base_time() + timedelta(hours=max(hour_offset, 0)))
        h = min(max(h, int(run.forecast_hours[0])), int(run.forecast_hours[-1]))
        if var in ("precip", "dswrf"):  # interval means: the interval that contains the time
            step = min(bisect.bisect_left(run.forecast_hours.tolist(), h), len(run.forecast_hours) - 1)
        else:
            step = _step_near(run, h)
        png = tile_bytes("gfs", run.id, var, step, z, x, y)
        return Response(png, media_type="image/png", headers={"Cache-Control": CACHE_CONTROL, "X-Forecast-Run": run.id})

    return router


@lru_cache(maxsize=1)
def _parameters_json() -> bytes:
    return json.dumps(json.loads(PARAMETERS_FILE.read_text())).encode()
