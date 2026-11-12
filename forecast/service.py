"""Forecast queries shared by the HTTP API (api.py) and the MCP server (mcp_server.py).

Errors are raised as ForecastError carrying an HTTP status; api.py turns them into JSON
error responses and mcp_server.py into tool errors, so both interfaces report the same
problems in the same words.
"""

from __future__ import annotations

import bisect
import re
from datetime import datetime, timedelta, timezone

import numpy as np

from .catalog import CATALOGS, ModelCatalog, Variable, derive
from .store import RUN_ID_RE, Run, RunStore, iso, parse_run_id
from .timeutil import daily_aggregate, resolve_timezone

_FHR_RE = re.compile(r"^f(\d{1,3})$")


class ForecastError(Exception):
    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers


def clean(arr: np.ndarray, decimals: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), decimals) for v in arr]


def get_catalog(model: str) -> ModelCatalog:
    cat = CATALOGS.get(model)
    if cat is None:
        raise ForecastError(404, f"Unknown model {model!r}. Available: {sorted(CATALOGS)}")
    return cat


def get_run(stores: dict[str, RunStore], model: str, ref: str) -> Run:
    get_catalog(model)
    if ref != "latest" and not RUN_ID_RE.match(ref):
        raise ForecastError(400, "run must be 'latest' or YYYYMMDDHH")
    run = stores[model].get(ref)
    if run is None:
        if ref == "latest":
            raise ForecastError(503, f"No {model} run has been ingested yet", headers={"Retry-After": "600"})
        raise ForecastError(404, f"Run {ref} not available. See /v2/{model}/runs")
    return run


def resolve_vars(cat: ModelCatalog, run: Run, requested: str | None) -> list[Variable]:
    available = {v.id: v for v in cat.available(run.variables)}
    if not requested:
        return list(available.values())
    out = []
    for vid in (s.strip() for s in requested.split(",") if s.strip()):
        if vid not in available:
            raise ForecastError(400, f"Unknown variable {vid!r}. Available: {sorted(available)}")
        out.append(available[vid])
    return out


def step_for(run: Run, time_ref: str) -> int:
    m = _FHR_RE.match(time_ref)
    if m:
        step = run.step_for_hour(int(m.group(1)))
    elif RUN_ID_RE.match(time_ref):
        step = run.step_for_valid(parse_run_id(time_ref))
    else:
        raise ForecastError(400, "time must be fNNN (forecast hour) or YYYYMMDDHH (valid time, UTC)")
    if step is None:
        raise ForecastError(404, f"{time_ref} is not a step of run {run.id}")
    return step


def point_forecast(
    stores: dict[str, RunStore],
    lat: float,
    lon: float,
    *,
    model: str = "gfs",
    run: str = "latest",
    variables: str | None = None,
    hours: int = 384,
    start: str = "now",
    tz: str | None = None,
    daily: bool = True,
    now: datetime | None = None,
) -> tuple[Run, dict]:
    """The /v2/forecast response body for one point, and the run it came from."""
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 360):
        raise ForecastError(400, "lat must be within -90..90 and lon within -180..360")
    cat = get_catalog(model)
    r = get_run(stores, model, run)
    wanted = resolve_vars(cat, r, variables)
    try:
        zone, zone_name = resolve_timezone(lat, lon, tz)
    except ValueError as exc:
        raise ForecastError(400, str(exc)) from exc

    needed: set[str] = set()
    for vid in [v.id for v in wanted] + ([s.source for s in cat.daily] if daily else []):
        v = cat.get(vid)
        if v is not None:
            needed.update(d for d in (v.derived_from or (v.id,)) if d in r.variables)
    base = r.point_series(sorted(needed), lat, lon)

    def with_derived(values: dict[str, np.ndarray], var_list) -> dict[str, np.ndarray]:
        out = dict(values)
        for v in var_list:
            if v.derived_from and all(d in values for d in v.derived_from):
                out[v.id] = derive(v.id, values)
        return out

    series = with_derived(base, cat.variables.values())
    now = now or datetime.now(timezone.utc)
    vt = r.valid_times

    # timeseries window
    first = 0
    if start == "now":
        first = max(0, bisect.bisect_right(vt, now) - 1)
    end_time = vt[first] + timedelta(hours=hours)
    last = bisect.bisect_right(vt, end_time)
    sl = slice(first, last)

    # current conditions: instant vars interpolated in time, interval vars from the containing interval
    current = None
    if vt[0] <= now <= vt[-1]:
        k = min(max(bisect.bisect_right(vt, now), 1), len(vt) - 1)
        frac = (now - vt[k - 1]).total_seconds() / (vt[k] - vt[k - 1]).total_seconds()
        cur_base = {}
        for vid, s in base.items():
            var = cat.get(vid)
            cur_base[vid] = s[k] if (var is not None and var.interval) else s[k - 1] * (1 - frac) + s[k] * frac
        cur = with_derived({k_: np.asarray([v]) for k_, v in cur_base.items()}, wanted)
        current = {"time": iso(now.replace(second=0, microsecond=0))}
        for v in wanted:
            current[v.id] = clean(cur[v.id], v.decimals)[0]

    i, j = r.grid.nearest_index(lat, lon)
    glat, glon = r.grid.latlon_of(i, j)
    body = {
        "model": model,
        "run": r.id,
        "init_time": iso(r.init_time),
        "location": {
            "lat": lat,
            "lon": (lon + 180.0) % 360.0 - 180.0,
            "nearest_grid_point": {"lat": glat, "lon": glon},
            "model_elevation_m": r.point_static("orog", lat, lon),
            "land_fraction": r.point_static("land", lat, lon),
        },
        "timezone": {"name": zone_name, "utc_offset_seconds": int(now.astimezone(zone).utcoffset().total_seconds())},
        "units": {v.id: v.units for v in wanted},
        "current": current,
        "timeseries": {
            "time": [iso(t) for t in vt[sl]],
            "interval_hours": np.diff(r.forecast_hours, prepend=r.forecast_hours[0])[sl].tolist(),
            **{v.id: clean(series[v.id][sl], v.decimals) for v in wanted},
        },
    }
    if daily and cat.daily:
        agg = daily_aggregate(vt, r.forecast_hours, series, {v.id for v in cat.variables.values() if v.interval},
                              cat.daily, zone)
        if start == "now":
            today = now.astimezone(zone).date().isoformat()
            keep = [n for n, d in enumerate(agg["date"]) if d >= today]
            agg = {key: [vals[n] for n in keep] for key, vals in agg.items()}
        body["daily"] = agg
        body["daily_units"] = {d.id: d.units for d in cat.daily if d.id in agg}
    body["source"] = r.manifest.get("source", {})
    return r, body
