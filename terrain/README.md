# Terrain hydrology engine (experimental)

A separate FastAPI service that turns elevation data and rainfall into flood and landslide indicators, as map tiles. It is research code: it isn't part of the hosted 4Shadow API, has no tests yet, and its interfaces may change.

## What it computes

For an area and a set of rain-gauge readings, a job produces:

- **Rain** (`rain_mm`): rainfall interpolated from the stations.
- **Discharge** (`discharge_m3s`): runoff routed down the D8 flow network.
- **Flood severity index** (`fsi`).
- **Landslide indices**: a static base susceptibility (`lsi_base`), a rain trigger (`lsi_trigger`) and the combined hazard (`lsi_hazard`).

Base layers (DEM, flow accumulation, slope, topographic wetness index, distance to channel) are prepared once from SRTM or Copernicus DEM tiles.

## Run it

Needs Python 3.12, GDAL 3.10 built from source (with `libeigen3-dev`, `libboost-all-dev`, `libpq-dev`, `cmake`) and Redis. Use its own virtual environment: its pins differ from the forecast service's.

```bash
pip install -r terrain/requirements.txt

# Base layers: SRTM HGT or Copernicus DEM -> DEM, D8 flow directions, accumulation, derivatives
python -m terrain.preprocess --elevation data/elevation --output data/hydrology

uvicorn terrain.app:app --port 5002          # API
python terrain/hydrology_worker.py           # job worker (Redis queue hydro:queue)
python terrain/hydrology_cleanup_worker.py   # deletes job folders when their Redis key expires (3 h)
```

Settings come from a `.env` at the repository root: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` (default `localhost:6379/0`) and `HYDRO_WORKERS` (default: the number of CPUs).

## How a job runs

1. `POST /hydrology/jobs` writes `data/hydrology_jobs/<id>/job.json` and pushes the id onto the Redis list `hydro:queue`.
2. `hydrology_worker.py` takes it, loads the base layers from `data/hydrology_copernicus/`, interpolates rain, computes runoff and discharge, then the landslide and flood indices, writing a GeoTIFF for each. It updates the job's status in `job.json` and a Redis hash, and publishes on `hydro:events`.
3. `GET /hydrology/jobs/{id}/tiles/{layer}/{z}/{x}/{y}.png` reprojects the GeoTIFFs to web mercator on the fly (rasterio `WarpedVRT`) and colours them per layer.

The cleanup worker listens for Redis key-expiry events (`__keyevent@0__:expired`) and deletes a job's folder when its `hydro:job:<id>` key expires.
