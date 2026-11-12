"""Foreshadow forecast API v2.

    uvicorn forecast.api:app --host 0.0.0.0 --port 5003 --workers 2

Stateless: every request reads memory-mapped arrays from the store, so there is no
preloading, no Redis, and no window where a new run is "still processing".
OpenAPI docs at /v2/docs (spec at /v2/openapi.json for generating TypeScript/Node clients).
"""

from __future__ import annotations

import bisect
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import __version__, legacy_v1
from .catalog import CATALOGS, ModelCatalog, Variable, derive
from .config import Settings, get_settings
from .render import render_tile
from .store import RUN_ID_RE, Run, RunStore, iso, parse_run_id
from .timeutil import daily_aggregate, resolve_timezone

MAX_TILE_ZOOM = 12
STALE_AFTER_HOURS = 12
LATEST_MAX_AGE = 300
IMMUTABLE = "public, max-age=31536000, immutable"
_FHR_RE = re.compile(r"^f(\d{1,3})$")


def _clean(arr: np.ndarray, decimals: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), decimals) for v in arr]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    stores = {model: RunStore(settings.data_root, model) for model in CATALOGS}

    app = FastAPI(
        title="Foreshadow Forecast API",
        version=__version__,
        description="Global point forecasts and map tiles from open NWP data. Built by Similie.",
        docs_url="/v2/docs",
        openapi_url="/v2/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    # ----------------------------------------------------------------- helpers

    def get_catalog(model: str) -> ModelCatalog:
        cat = CATALOGS.get(model)
        if cat is None:
            raise HTTPException(404, f"Unknown model {model!r}. Available: {sorted(CATALOGS)}")
        return cat

    def get_run(model: str, ref: str) -> Run:
        get_catalog(model)
        if ref != "latest" and not RUN_ID_RE.match(ref):
            raise HTTPException(400, "run must be 'latest' or YYYYMMDDHH")
        run = stores[model].get(ref)
        if run is None:
            if ref == "latest":
                raise HTTPException(503, f"No {model} run has been ingested yet", headers={"Retry-After": "600"})
            raise HTTPException(404, f"Run {ref} not available. See /v2/{model}/runs")
        return run

    def resolve_vars(cat: ModelCatalog, run: Run, requested: str | None) -> list[Variable]:
        available = {v.id: v for v in cat.available(run.variables)}
        if not requested:
            return list(available.values())
        out = []
        for vid in (s.strip() for s in requested.split(",") if s.strip()):
            if vid not in available:
                raise HTTPException(400, f"Unknown variable {vid!r}. Available: {sorted(available)}")
            out.append(available[vid])
        return out

    def step_for(run: Run, time_ref: str) -> int:
        m = _FHR_RE.match(time_ref)
        if m:
            step = run.step_for_hour(int(m.group(1)))
        elif RUN_ID_RE.match(time_ref):
            step = run.step_for_valid(parse_run_id(time_ref))
        else:
            raise HTTPException(400, "time must be fNNN (forecast hour) or YYYYMMDDHH (valid time, UTC)")
        if step is None:
            raise HTTPException(404, f"{time_ref} is not a step of run {run.id}")
        return step

    @lru_cache(maxsize=settings.tile_cache_size)
    def tile_bytes(model: str, run_id: str, var_id: str, step: int, z: int, x: int, y: int) -> bytes:
        run = stores[model].get(run_id)
        if run is None:
            raise HTTPException(404, f"Run {run_id} not available")
        return render_tile(run, CATALOGS[model].variables[var_id], step, z, x, y)

    # ----------------------------------------------------------------- routes

    @app.get("/v2", tags=["meta"])
    def index():
        return {
            "name": "Foreshadow Forecast API",
            "version": __version__,
            "docs": "/v2/docs",
            "models": {m: c.title for m, c in CATALOGS.items()},
            "endpoints": {
                "forecast": "/v2/forecast?lat=-8.56&lon=125.57",
                "runs": "/v2/{model}/runs",
                "variables": "/v2/{model}/variables",
                "tiles": "/v2/{model}/tiles/{run}/{var}/{time}/{z}/{x}/{y}.png",
            },
        }

    @app.get("/v2/health", tags=["meta"])
    def health():
        now = datetime.now(timezone.utc)
        models, ok = {}, True
        for model, store in stores.items():
            rid = store.latest_id()
            if rid is None:
                ok = False
                models[model] = {"latest_run": None}
                continue
            age = (now - parse_run_id(rid)).total_seconds() / 3600
            models[model] = {"latest_run": rid, "age_hours": round(age, 1), "stale": age > STALE_AFTER_HOURS}
        return JSONResponse({"status": "ok" if ok else "unavailable", "version": __version__, "models": models},
                            status_code=200 if ok else 503)

    @app.get("/v2/{model}/runs", tags=["meta"])
    def list_runs(model: str):
        get_catalog(model)
        store = stores[model]
        runs = []
        for rid in reversed(store.run_ids()):
            run = store.get(rid)
            if run is not None:
                runs.append({
                    "run": run.id,
                    "init_time": iso(run.init_time),
                    "steps": len(run.forecast_hours),
                    "first_valid": iso(run.valid_times[0]),
                    "last_valid": iso(run.valid_times[-1]),
                })
        return {"model": model, "latest": store.latest_id(), "runs": runs}

    @app.get("/v2/{model}/runs/{run}", tags=["meta"])
    def run_detail(model: str, run: str):
        r = get_run(model, run)
        return {
            "model": model,
            "run": r.id,
            "init_time": iso(r.init_time),
            "forecast_hours": r.forecast_hours.tolist(),
            "valid_times": [iso(t) for t in r.valid_times],
            "variables": sorted(v.id for v in get_catalog(model).available(r.variables)),
            "source": r.manifest.get("source", {}),
        }

    @app.get("/v2/{model}/variables", tags=["meta"])
    def variables(model: str):
        cat = get_catalog(model)
        return {
            "model": model,
            "variables": [v.describe() for v in cat.variables.values()],
            "daily": [{"id": d.id, "source": d.source, "aggregation": d.how, "units": d.units} for d in cat.daily],
        }

    @app.get("/v2/forecast", tags=["forecast"])
    def forecast(
        lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
        lon: float = Query(..., ge=-180, le=360, description="Longitude in degrees"),
        model: str = Query("gfs"),
        run: str = Query("latest", description="'latest' or a run id YYYYMMDDHH"),
        variables: str | None = Query(None, alias="vars", description="Comma-separated variable ids; default all"),
        hours: int = Query(384, ge=1, le=384, description="Length of the returned timeseries"),
        start: str = Query("now", pattern="^(now|run)$", description="Start the timeseries at the current hour or at run start"),
        tz: str | None = Query(None, description="IANA time zone for daily aggregation; default from coordinates"),
        daily: bool = Query(True),
    ):
        cat = get_catalog(model)
        r = get_run(model, run)
        wanted = resolve_vars(cat, r, variables)
        try:
            zone, zone_name = resolve_timezone(lat, lon, tz)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

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
        now = datetime.now(timezone.utc)
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
                current[v.id] = _clean(cur[v.id], v.decimals)[0]

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
                **{v.id: _clean(series[v.id][sl], v.decimals) for v in wanted},
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
        headers = {"Cache-Control": f"public, max-age={LATEST_MAX_AGE}", "X-Forecast-Run": r.id}
        return JSONResponse(body, headers=headers)

    @app.get("/v2/{model}/tiles/{run}/{var}/{time}/{z}/{x}/{y}.png", tags=["tiles"],
             response_class=Response, responses={200: {"content": {"image/png": {}}}})
    def tile(
        model: str,
        run: str,
        var: str,
        time: str = Path(..., description="fNNN (forecast hour) or YYYYMMDDHH (valid time, UTC)"),
        z: int = Path(..., ge=0, le=MAX_TILE_ZOOM),
        x: int = Path(..., ge=0),
        y: int = Path(..., ge=0),
    ):
        cat = get_catalog(model)
        r = get_run(model, run)
        if x >= 2**z or y >= 2**z:
            raise HTTPException(404, "Tile outside the world")
        v = next((a for a in cat.available(r.variables) if a.id == var), None)
        if v is None or v.palette is None:
            tileable = sorted(a.id for a in cat.available(r.variables) if a.palette is not None)
            raise HTTPException(404, f"No tiles for {var!r}. Tileable: {tileable}")
        step = step_for(r, time)
        png = tile_bytes(model, r.id, v.id, step, z, x, y)
        cache = IMMUTABLE if run != "latest" else f"public, max-age={LATEST_MAX_AGE}"
        return Response(png, media_type="image/png", headers={"Cache-Control": cache, "X-Forecast-Run": r.id})

    # v1 response shapes for clients still on the retired v1 service (see legacy_v1.py).
    app.include_router(legacy_v1.build_router(stores, tile_bytes))

    return app


app = create_app()
