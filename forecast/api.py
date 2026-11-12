"""Foreshadow forecast API v2.

    uvicorn forecast.api:app --host 0.0.0.0 --port 5003 --workers 2

Stateless: every request reads memory-mapped arrays from the store, so there is no
preloading, no Redis, and no window where a new run is "still processing".
OpenAPI docs at /v2/docs (spec at /v2/openapi.json for generating TypeScript/Node clients).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import __version__, legacy_v1, mcp_server
from .catalog import CATALOGS
from .config import Settings, get_settings
from .render import render_tile
from .service import ForecastError, get_catalog, get_run, point_forecast, step_for
from .store import RunStore, iso, parse_run_id

MAX_TILE_ZOOM = 12
STALE_AFTER_HOURS = 12
LATEST_MAX_AGE = 300
IMMUTABLE = "public, max-age=31536000, immutable"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    stores = {model: RunStore(settings.data_root, model) for model in CATALOGS}
    mcp = mcp_server.build_server(stores)
    mcp_routes = mcp_server.http_routes(mcp)  # creates the session manager the lifespan runs

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(
        title="Foreshadow Forecast API",
        version=__version__,
        description="Global point forecasts and map tiles from open NWP data. Built by Similie.",
        docs_url="/v2/docs",
        openapi_url="/v2/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    @app.exception_handler(ForecastError)
    async def forecast_error(_request, exc: ForecastError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status, headers=exc.headers)

    # ----------------------------------------------------------------- helpers

    @lru_cache(maxsize=settings.tile_cache_size)
    def tile_bytes(model: str, run_id: str, var_id: str, step: int, z: int, x: int, y: int) -> bytes:
        run = stores[model].get(run_id)
        if run is None:
            raise ForecastError(404, f"Run {run_id} not available")
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
        r = get_run(stores, model, run)
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
        r, body = point_forecast(stores, lat, lon, model=model, run=run, variables=variables, hours=hours,
                                 start=start, tz=tz, daily=daily)
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
        r = get_run(stores, model, run)
        if x >= 2**z or y >= 2**z:
            raise ForecastError(404, "Tile outside the world")
        v = next((a for a in cat.available(r.variables) if a.id == var), None)
        if v is None or v.palette is None:
            tileable = sorted(a.id for a in cat.available(r.variables) if a.palette is not None)
            raise ForecastError(404, f"No tiles for {var!r}. Tileable: {tileable}")
        step = step_for(r, time)
        png = tile_bytes(model, r.id, v.id, step, z, x, y)
        cache = IMMUTABLE if run != "latest" else f"public, max-age={LATEST_MAX_AGE}"
        return Response(png, media_type="image/png", headers={"Cache-Control": cache, "X-Forecast-Run": r.id})

    # v1 response shapes for clients still on the retired v1 service (see legacy_v1.py).
    app.include_router(legacy_v1.build_router(stores, tile_bytes))

    # MCP (Model Context Protocol) at /v2/mcp for AI assistants. Added as plain routes rather than
    # a mount so /v2/mcp answers without a trailing-slash redirect, which a proxy would pass on.
    app.router.routes.extend(mcp_routes)

    return app


app = create_app()
