# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

4Shadow is an open-source (GPL-3.0) global weather forecast service in Python. The repository holds:

1. **Forecast service** (`forecast/`, port 5003 on the server): ingests NOAA GFS and GFS-Wave into a compact store and serves point forecasts, map tiles, raw fields, place search, early-warning threshold checks and an MCP server under `/v2`. This is the product; it runs behind the API-key router at api.4shadow.io. See `forecast/README.md`.
2. **Terrain hydrology engine** (`terrain/`, experimental, not deployed): flood and landslide indicators from DEMs and rainfall. See `terrain/README.md`; it has its own requirements (GDAL).

v1 (`tile_renderers/`, `downloaders/`) was retired in November 2026 and removed from the repository; it remains in git history. `/v1` clients are served by `forecast/legacy_v1.py` until they move.

## Forecast v2 (`forecast/`)

```bash
python -m forecast.ingest --max-fhr 24     # dev: ingest hours 0-24 of the newest complete GFS run
python -m forecast.ingest --loop           # prod: keep ingesting new runs (deploy/forecast-ingest.service)
uvicorn forecast.api:app --port 5003       # API; docs at /v2/docs (deploy/forecast-api.service)
pytest                                     # tests (pip install -r requirements-dev.txt); no network needed
```

- **Ingest** (`ingest.py`, `sources/gfs.py`): reads each step's `.idx`, byte-range downloads only catalog messages, decodes with pygrib sequentially (ecCodes is not thread-safe), converts running totals and bucket averages into per-interval mean rates (`IntervalState`), and writes int16 arrays via `store.RunWriter`. A run is published only when complete: `<run>.incoming/` is renamed into place, then `latest.json` is replaced atomically.
- **Store** (`store.py`): `data/forecast/<model>/runs/<YYYYMMDDHH>/<var>.npy`, int16 `[step, lat, lon]` with scale/offset in `manifest.json`. The grid is north→south, lon 0→360.
- **API** (`api.py`): stateless; memory-maps the store. Query logic shared with MCP lives in `service.py` (raises `ForecastError`, which `api.py` turns into JSON errors). Points and tiles are direct bilinear index math (`grid.py`); no Redis, no preloading. Tiles use fixed palettes (`palettes.py`).
- **Catalog** (`catalog.py`): add variables there. The GRIB selectors must match NOAA `.idx` text exactly.
- **Places** (`places.py`): place names to coordinates and coordinates to the nearest town, from GeoNames in a read-only SQLite index (`python -m forecast.places build`; `GET /v2/places`, `q=` on `/v2/forecast`, `location` in the MCP tools).
- **MCP** (`mcp_server.py`): forecast tools for AI assistants at `/v2/mcp` (official `mcp` SDK, stateless streamable HTTP, JSON responses). Tools wrap `service.point_forecast`, the same code as `GET /v2/forecast`. Keep tool parameters plain types with empty defaults (no `X | None`) and results compact Markdown; changing a tool's name, description or schema forces re-approval in Experts.
- **Legacy** (`legacy_v1.py`): temporary v1-shaped endpoints under `/legacy/v1` for clients still calling `/v1` through the API-key router. Don't extend it; delete it when v1 traffic stops.

## Checks

```bash
ruff check .                   # lint (CI runs it; rules in ruff.toml)
pytest                         # tests; no network or data needed
```

CI (`.github/workflows/ci.yml`) runs both (tests on Python 3.12, 3.13 and 3.14), builds the Docker image and checks that it starts. On `main`, daily and by hand, an end-to-end job ingests hours 0-6 of the newest real GFS run and runs `deploy/deploy.sh check` against it. Production runs Python 3.12.

## Deploying

`deploy/deploy.sh` updates a server checkout to the newest `main`: `status`, `update` (fast-forward, install requirements, import check, restart the API, restart ingest only if ingest code changed, smoke tests) and `rollback`. See `docs/deploy.md`. Production is `/root/foreshadow-v2` on 4shadow.io (a shared host: never disturb the other services).
