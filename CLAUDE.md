# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

4Shadow (Foreshadow) is a Python weather forecasting and terrain hydrology API built on FastAPI. It provides these services:

1. **Forecast v2** (`forecast/`, port 5003 on the server) — the replacement for the GFS tile server. Global point forecasts and map tiles from NOAA GFS. New GFS work goes here, not in `tile_renderers/`. See `forecast/README.md`.
2. **GFS Weather Tile Server v1** (`tile_renderers/`, `downloaders/`) — retired on 2026-11-12. Its systemd units and download cron are disabled on the server. Public `/v1/...` routes are now served by `forecast/legacy_v1.py`. Don't change this code; it will be deleted.
3. **Terrain Hydrology Engine** (port 5002) — Processes DEM elevation data to compute flood severity, landslide risk, and hydrological analysis via job-based processing

## Forecast v2 (`forecast/`)

```bash
python -m forecast.ingest --max-fhr 24     # dev: ingest hours 0-24 of the newest complete GFS run
python -m forecast.ingest --loop           # prod: keep ingesting new runs (deploy/forecast-ingest.service)
uvicorn forecast.api:app --port 5003       # API; docs at /v2/docs (deploy/forecast-api.service)
pytest                                     # tests (pip install -r requirements-dev.txt); no network needed
```

- **Ingest** (`ingest.py`, `sources/gfs.py`): reads each step's `.idx`, byte-range downloads only catalog messages, decodes with pygrib sequentially (ecCodes is not thread-safe), converts running totals and bucket averages into per-interval mean rates (`IntervalState`), and writes int16 arrays via `store.RunWriter`. A run is published only when complete: `<run>.incoming/` is renamed into place, then `latest.json` is replaced atomically.
- **Store** (`store.py`): `data/forecast/<model>/runs/<YYYYMMDDHH>/<var>.npy`, int16 `[step, lat, lon]` with scale/offset in `manifest.json`. The grid is north→south, lon 0→360.
- **API** (`api.py`): stateless; memory-maps the store. Points and tiles are direct bilinear index math (`grid.py`); no Redis, no preloading. Tiles use fixed palettes (`palettes.py`).
- **Catalog** (`catalog.py`): add variables there. The GRIB selectors must match NOAA `.idx` text exactly.
- **Legacy** (`legacy_v1.py`): temporary v1-shaped endpoints under `/legacy/v1` for clients still calling `/v1` through the API-key router. Don't extend it; delete it when v1 traffic stops.

## Running the Services

### GFS Weather API
```bash
# From repo root, activate venv then:
cd tile_renderers && uvicorn main:app --host 0.0.0.0 --port 5001 --workers 4
```

### Terrain Hydrology API
```bash
uvicorn terrain.app:app --host 0.0.0.0 --port 5002 --reload
```

### Background Workers
```bash
# GFS data preloader (pre-caches interpolators into Redis)
python -m tile_renderers.gfs_render.run_preloader

# Hydrology job worker (listens on Redis queue, processes jobs)
python terrain/hydrology_worker.py

# Hydrology cleanup worker (deletes expired job directories via Redis key expiry)
python terrain/hydrology_cleanup_worker.py
```

### GFS Data Download
```bash
python downloaders/gfs-hourly-downloader.py [--date-offset N] [--no-cleanup]
```

### Terrain Preprocessing
```bash
# SRTM HGT → DEM + D8 flow directions + accumulation + reprojected rasters
python -m terrain.preprocess --elevation data/elevation --output data/hydrology
```

## Environment Variables

Set in a `.env` file at the repo root (auto-discovered via `python-dotenv`):
- `GRIB_FILES_PATH` — Path to downloaded GRIB files (required for GFS service)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` — Redis connection (defaults: localhost:6379/0)
- `HYDRO_WORKERS` — Number of hydrology worker processes (defaults to CPU count)

## System Dependencies

- Python 3.12+
- GDAL 3.10.x (must be built from source with libeigen3-dev, libboost-all-dev, libpq-dev, cmake)
- Redis (required for both services — caching, job queue, key expiry events)
- `pip install -r requirements.txt` (within a venv at `.venv`)

## Architecture

### GFS Weather Service (`tile_renderers/`)

The entry point is `tile_renderers/main.py` — a FastAPI app. On startup it initializes:
- `RedisCacheBackend` — Redis-backed cache for tiles and interpolators
- `LocalStorage` — In-process L2 cache
- `ModelService` — Central orchestrator for GRIB file access, interpolator building/caching, and point forecasts
- `TileRendering` — Renders 256x256 PNG weather tiles from interpolated GRIB data
- `MemoryLayerCache` — Pre-warmed in-memory layer cache for rapid point forecasts

**Key data flow for tile rendering:**
`ModelService.get_or_build_interpolator()` → reads GRIB via pygrib → builds `LinearNDInterpolator` (scipy/Delaunay) → caches in Redis (pickled) with L2 local cache → `TileRendering.render_tile()` evaluates interpolator on Web Mercator grid points → colorizes via `MapColors` → returns PNG

**Caching strategy:** Two-tier (local dict + Redis) with debounced writes to Redis. Interpolators are serialized with pickle protocol 4. Cache keys incorporate date/hour for time-based invalidation.

**Supported models:** `gfs` (atmospheric) and `gfswave` (ocean waves), configured in `SystemConfig.MODEL_MAP`.

### Terrain Hydrology Service (`terrain/`)

The entry point is `terrain/app.py` — a separate FastAPI app that includes the `terrain.api.hydrology` router.

**Job-based processing flow:**
1. `POST /hydrology/jobs` creates a job → writes `job.json` to `data/hydrology_jobs/<id>/` → pushes job ID to Redis LIST `hydro:queue`
2. `hydrology_worker.py` pops from `hydro:queue` via BLPOP and runs `process_job()`:
   - Loads DEM + accumulation from `data/hydrology_copernicus/`
   - Interpolates rainfall from station data → `rain_mm.tif`
   - Computes runoff → discharge → `discharge_m3s.tif`
   - Computes landslide indices (LSI base/trigger/hazard) → `lsi_*.tif`
   - Computes flood severity index (FSI) → `fsi.tif`
   - Updates job status in both `job.json` and Redis hash
   - Publishes `hydro:events` Redis notification
3. Tile requests (`GET /hydrology/jobs/{id}/tiles/{layer}/...`) warp source EPSG:4326 GeoTIFFs on-the-fly to EPSG:3857 via rasterio `WarpedVRT`, then colorize per layer

**Base layers** (static, in `data/hydrology_copernicus/`): dem, d8_accum, lsi_base, slope_deg, twi, dist_to_channel_m
**Job layers** (per-job, in `data/hydrology_jobs/<id>/`): rain_mm, discharge_m3s, fsi, lsi_trigger, lsi_hazard

**Cleanup:** `hydrology_cleanup_worker.py` subscribes to Redis `__keyevent@0__:expired` events. When a `hydro:job:<id>` key expires (3h TTL), the corresponding job directory is deleted from disk.

### Data Download (`downloaders/`)

`gfs-hourly-downloader.py` pulls GFS GRIB2 files from NOAA's public S3 bucket (`noaa-gfs-bdp-pds`). Downloads run hours 00/06/12/18; hour 00 gets full forecast range (f000–f384), others get f000–f023. Old data is pruned after confirming complete 00-hour coverage.

### Preprocessing Pipeline (`terrain/preprocess*.py`)

Converts SRTM `.hgt` tiles or Copernicus DEM into hydrology-ready GeoTIFFs: DEM mosaic → D8 flow directions → flow accumulation → reprojection to EPSG:3857. Derivative layers (slope, TWI, distance-to-channel) are computed in `preprocess_derivatives.py`.
