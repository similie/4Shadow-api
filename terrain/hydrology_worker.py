# hydrology_worker.py
from __future__ import annotations

import json
import os
from multiprocessing import Process, current_process
from pathlib import Path
from typing import Optional, Tuple, cast

import numpy as np
import rasterio
from rasterio.crs import CRS
from redis import Redis

from terrain.analysis.flood_index import compute_fsi
from terrain.analysis.landslide_index import (
    compute_lsi_base,
    compute_lsi_hazard,
    compute_lsi_trigger,
)
from terrain.job_engine import HydrologyJobEngine
from terrain.rain_interpolation import RainStation, interpolate_rainfall_to_grid
from terrain.runoff import (
    cell_area_m2_at_lat,
    discharge_from_runoff,
    drainage_area_from_accum,
    runoff_depth_from_rain,
)

BLPOPResult = Optional[Tuple[bytes, bytes]]

BASE_HYDRO_DIR = Path("data/hydrology_copernicus")
JOBS_DIR = Path("data/hydrology_jobs")
REDIS_URL = "redis://localhost:6379/0"

DEFAULT_WORKERS = max(1, (os.cpu_count() or 1))
WORKER_COUNT = int(os.environ.get("HYDRO_WORKERS", DEFAULT_WORKERS))


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _save_geotiff_4326(path: Path, data: np.ndarray, transform, nodata=np.nan) -> None:
    """
    Save a 2D float32 raster in EPSG:4326 with the given transform.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    nodata_val = -9999.0 if np.isnan(nodata) else float(nodata)

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": nodata_val,
        "compress": "LZW",
        # optional speed knobs:
        # "tiled": True,
        # "blockxsize": 256,
        # "blockysize": 256,
    }

    arr = np.where(np.isnan(data), nodata_val, data).astype("float32", copy=False)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def _read_base_layer(
    engine: HydrologyJobEngine, name: str, *, like: np.ndarray
) -> np.ndarray:
    path = engine.base_hydrology_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing base layer: {path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if arr.shape != like.shape:
            raise ValueError(
                f"Layer {name} shape {arr.shape} != DEM shape {like.shape}"
            )
        return arr


def _compute_fsi_vmin_vmax_arrays(
    fsi: np.ndarray,
    rain_mm: np.ndarray,
    dem: np.ndarray,
    *,
    dem_nodata: float | None = None,
) -> tuple[float, float]:
    if dem_nodata is None:
        land = np.isfinite(dem) & (dem > 0)
    else:
        land = np.isfinite(dem) & (dem != dem_nodata) & (dem > 0)

    f_valid = np.isfinite(fsi) & land
    r_valid = np.isfinite(rain_mm) & (rain_mm > 0)

    mask = f_valid & r_valid
    vals = fsi[mask] if np.any(mask) else fsi[f_valid]
    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


# ---------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------
def process_job(
    job_id: str, engine: HydrologyJobEngine, r: Redis, events_channel: str
) -> None:
    job = engine.get_job(job_id)
    params = job.params or {}

    event_duration_hours = float(params["event_duration_hours"])
    runoff_coeff = float(params["runoff_coeff"])
    rain_radius_km = float(params["rain_radius_km"])
    stations_data = params.get("stations", [])

    stations = [
        RainStation(
            lat=float(s["lat"]),
            lon=float(s["lon"]),
            rain_mm=float(s["rain_mm"]),
            wind_speed_ms=float(s["wind_speed_ms"])
            if s.get("wind_speed_ms") is not None
            else None,
            wind_dir_deg=float(s["wind_dir_deg"])
            if s.get("wind_dir_deg") is not None
            else None,
        )
        for s in stations_data
    ]

    total_rain = sum(s.rain_mm for s in stations)
    print(f"[{job_id}] Total rain: {total_rain} mm")
    if total_rain <= 0:
        params["fsi_vmin"] = 0.0
        params["fsi_vmax"] = 1.0
        params["has_runoff"] = False
        params["reason"] = "no_rain"
        params["has_landslide"] = False

        engine.update_job(job_id, status="done", error=None, params=params)
        r.publish(
            events_channel,
            json.dumps({"type": "job_done", "job_id": job_id, "params": params}),
        )
        print(f"[{job_id}] Dry event detected → skipping hydrology.")
        return

    job_dir = engine._job_path(job_id)

    # ------------------------------------------------------------------
    # 1) Load DEM + ACCUM (EPSG:4326)
    # ------------------------------------------------------------------
    dem_path = engine.base_hydrology_dir / "dem.tif"
    accum_path = engine.base_hydrology_dir / "d8_accum.tif"

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype("float32")
        dem_transform = dem_src.transform
        dem_bounds = dem_src.bounds
        dem_nodata = dem_src.nodata

    with rasterio.open(accum_path) as accum_src:
        accum = accum_src.read(1).astype("float32")

    lat_center = 0.5 * (dem_bounds.top + dem_bounds.bottom)
    cellsize_deg = abs(dem_transform.a)
    cell_area_m2 = cell_area_m2_at_lat(lat_center, cellsize_deg)

    # ------------------------------------------------------------------
    # 2) Interpolate rainfall to grid
    # ------------------------------------------------------------------
    rain_mm = interpolate_rainfall_to_grid(
        dem_array=dem,
        transform=dem_transform,
        stations=stations,
        power=2.0,
        min_dist_cells=1.0,
        rain_radius_km=rain_radius_km,
        use_wind=True,
    ).astype("float32")

    rain_path = job_dir / "rain_mm.tif"
    _save_geotiff_4326(rain_path, rain_mm, dem_transform, nodata=0.0)

    # ------------------------------------------------------------------
    # 3) Hydrology chain: rain → runoff → discharge
    # ------------------------------------------------------------------
    duration_seconds = event_duration_hours * 3600.0
    runoff_depth_m = runoff_depth_from_rain(rain_mm, runoff_coeff)
    drainage_area_m2 = drainage_area_from_accum(accum, cell_area_m2)

    discharge_m3s = discharge_from_runoff(
        runoff_depth_m=runoff_depth_m,
        drainage_area_m2=drainage_area_m2,
        duration_seconds=duration_seconds,
    ).astype("float32")

    discharge_path = job_dir / "discharge_m3s.tif"
    _save_geotiff_4326(discharge_path, discharge_m3s, dem_transform, nodata=np.nan)

    # ------------------------------------------------------------------
    # 4) Landslide layers (LSI): base cached, trigger/hazard per job
    # ------------------------------------------------------------------
    lsi_base_path = engine.base_hydrology_dir / "lsi_base.tif"

    if lsi_base_path.exists():
        with rasterio.open(lsi_base_path) as lsi_src:
            lsi_base = lsi_src.read(1).astype("float32")
    else:
        slope = _read_base_layer(engine, "slope_deg.tif", like=dem)
        twi = _read_base_layer(engine, "twi.tif", like=dem)
        dist = _read_base_layer(engine, "dist_to_channel_m.tif", like=dem)

        lsi_base = compute_lsi_base(
            dem=dem,
            accum=accum,
            transform=dem_transform,
            slope_deg=slope,
            twi=twi,
            dist_to_channel_m=dist,
            dem_nodata=dem_nodata,
        ).astype("float32")

        _save_geotiff_4326(lsi_base_path, lsi_base, dem_transform, nodata=np.nan)

    lsi_intensity_mmhr, lsi_trigger = compute_lsi_trigger(
        rain_mm=rain_mm,
        dem=dem,
        event_duration_hours=event_duration_hours,
        dem_nodata=None,
    )

    lsi_hazard = compute_lsi_hazard(lsi_base=lsi_base, lsi_trigger=lsi_trigger).astype(
        "float32"
    )

    lsi_trigger_path = job_dir / "lsi_trigger.tif"
    lsi_hazard_path = job_dir / "lsi_hazard.tif"

    # Visual layer = mm/hr
    _save_geotiff_4326(
        lsi_trigger_path,
        lsi_intensity_mmhr.astype("float32"),
        dem_transform,
        nodata=np.nan,
    )
    # Hazard layer = 0..1
    _save_geotiff_4326(lsi_hazard_path, lsi_hazard, dem_transform, nodata=np.nan)
    # lsi_intensity_mmhr, lsi_trigger = compute_lsi_trigger(
    #     rain_mm=rain_mm,
    #     dem=dem,
    #     event_duration_hours=event_duration_hours,
    #     dem_nodata=dem_nodata,
    # )

    # lsi_hazard = compute_lsi_hazard(lsi_base=lsi_base, lsi_trigger=lsi_trigger).astype(
    #     "float32"
    # )

    # lsi_trigger_path = job_dir / "lsi_trigger.tif"
    # lsi_hazard_path = job_dir / "lsi_hazard.tif"
    # _save_geotiff_4326(lsi_trigger_path, lsi_trigger, dem_transform, nodata=np.nan)
    # _save_geotiff_4326(
    #     lsi_trigger_path, lsi_intensity_mmhr, dem_transform, nodata=np.nan
    # )
    # _save_geotiff_4326(lsi_hazard_path, lsi_hazard, dem_transform, nodata=np.nan)

    params["has_landslide"] = True

    # ------------------------------------------------------------------
    # 5) Flood Severity Index (FSI)
    # ------------------------------------------------------------------
    fsi = compute_fsi(
        dem=dem,
        accum=accum,
        discharge=discharge_m3s,
        rain_mm=rain_mm,
        transform=dem_transform,
    ).astype("float32")

    fsi_path = job_dir / "fsi.tif"
    _save_geotiff_4326(fsi_path, fsi, dem_transform, nodata=np.nan)

    # ------------------------------------------------------------------
    # 6) vmin/vmax + persist job metadata
    # ------------------------------------------------------------------
    fsi_vmin, fsi_vmax = _compute_fsi_vmin_vmax_arrays(
        fsi, rain_mm, dem, dem_nodata=dem_nodata
    )
    params["fsi_vmin"] = fsi_vmin
    params["fsi_vmax"] = fsi_vmax
    params["has_runoff"] = True

    engine.update_job(job_id, status="done", error=None, params=params)

    r.publish(
        events_channel,
        json.dumps({"type": "job_done", "job_id": job_id, "params": params}),
    )


def worker_loop(worker_index: int) -> None:
    proc_name = current_process().name
    print(f"[{proc_name}] Worker #{worker_index} starting…")

    engine = HydrologyJobEngine(
        base_hydrology_dir=BASE_HYDRO_DIR,
        jobs_dir=JOBS_DIR,
        redis_url=REDIS_URL,
        prefix="hydro",
    )

    r: Redis = engine.redis
    queue_key = engine._queue_key()
    events_channel = f"{engine.prefix}:events"

    print(f"[{proc_name}] Listening on Redis list: {queue_key}")

    while True:
        result: BLPOPResult = cast(BLPOPResult, r.blpop([queue_key]))
        if result is None:
            continue

        list_name_raw, job_raw = result

        list_name = (
            list_name_raw.decode("utf-8")
            if isinstance(list_name_raw, (bytes, bytearray))
            else str(list_name_raw)
        )
        job_id = (
            job_raw.decode("utf-8")
            if isinstance(job_raw, (bytes, bytearray))
            else str(job_raw)
        )

        print(f"[{proc_name}] Got job_id={job_id} from {list_name}")

        try:
            engine.update_job(job_id, status="running")
            process_job(job_id, engine, r, events_channel)
            print(f"[{proc_name}] [{job_id}] Done")
        except Exception as e:
            print(f"[{proc_name}] [{job_id}] ERROR:", e)
            engine.update_job(job_id, status="error", error=str(e))


if __name__ == "__main__":
    print(
        f"[hydrology_worker] Starting {WORKER_COUNT} worker process(es) for Redis URL {REDIS_URL}"
    )

    workers: list[Process] = []
    for i in range(WORKER_COUNT):
        p = Process(target=worker_loop, args=(i,), daemon=False)
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
# # hydrology_worker.py
# from __future__ import annotations

# import json
# import os
# from multiprocessing import Process, current_process
# from pathlib import Path
# from typing import Optional, Tuple, cast

# import numpy as np
# import rasterio
# from rasterio.crs import CRS
# from rasterio.enums import Resampling
# from rasterio.warp import calculate_default_transform, reproject
# from redis import Redis

# from terrain.analysis.flood_index import compute_fsi
# from terrain.analysis.landslide_index import (
#     compute_lsi_base,
#     compute_lsi_hazard,
#     compute_lsi_trigger,
# )
# from terrain.job_engine import HydrologyJobEngine
# from terrain.rain_interpolation import RainStation, interpolate_rainfall_to_grid
# from terrain.runoff import (
#     cell_area_m2_at_lat,
#     discharge_from_runoff,
#     drainage_area_from_accum,
#     runoff_depth_from_rain,
# )

# BLPOPResult = Optional[Tuple[bytes, bytes]]

# # ---------------------------------------------------------------------
# # Base dirs / config
# # ---------------------------------------------------------------------
# BASE_HYDRO_DIR = Path("data/hydrology_copernicus")
# JOBS_DIR = Path("data/hydrology_jobs")
# REDIS_URL = "redis://localhost:6379/0"

# DEFAULT_WORKERS = max(1, (os.cpu_count() or 1))
# WORKER_COUNT = int(os.environ.get("HYDRO_WORKERS", DEFAULT_WORKERS))


# # ---------------------------------------------------------------------
# # IO helpers
# # ---------------------------------------------------------------------
# def _save_geotiff_4326(path: Path, data: np.ndarray, transform, nodata=np.nan) -> None:
#     """
#     Save a 2D float32 raster in EPSG:4326 with the given transform.
#     """
#     path.parent.mkdir(parents=True, exist_ok=True)

#     nodata_val = -9999.0 if np.isnan(nodata) else float(nodata)

#     profile = {
#         "driver": "GTiff",
#         "height": data.shape[0],
#         "width": data.shape[1],
#         "count": 1,
#         "dtype": "float32",
#         "crs": CRS.from_epsg(4326),
#         "transform": transform,
#         "nodata": nodata_val,
#         "compress": "LZW",
#     }

#     arr = np.where(np.isnan(data), nodata_val, data).astype("float32")

#     with rasterio.open(path, "w", **profile) as dst:
#         dst.write(arr, 1)


# def _reproject_to_3857(
#     src_path: Path,
#     dst_path: Path,
#     resampling: Resampling = Resampling.nearest,
# ) -> None:
#     """
#     Reproject a raster on disk (assumed EPSG:4326) to EPSG:3857.

#     For rain / discharge we use NEAREST to avoid nodata smearing.
#     For continuous indices (FSI/LSI) use BILINEAR.
#     """
#     dst_path.parent.mkdir(parents=True, exist_ok=True)

#     with rasterio.open(src_path) as src:
#         dst_crs = CRS.from_epsg(3857)

#         transform, width, height = calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds
#         )

#         profile = src.profile.copy()
#         profile.update(
#             {
#                 "crs": dst_crs,
#                 "transform": transform,
#                 "width": width,
#                 "height": height,
#                 "dtype": "float32",
#                 "compress": "LZW",
#             }
#         )

#         with rasterio.open(dst_path, "w", **profile) as dst:
#             reproject(
#                 source=rasterio.band(src, 1),
#                 destination=rasterio.band(dst, 1),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=resampling,
#             )

#     print(f"  ✔ Reprojected → {dst_path}")


# def _read_base_layer(
#     engine: HydrologyJobEngine,
#     name: str,
#     *,
#     like: np.ndarray,
# ) -> np.ndarray:
#     """
#     Read a preprocessed base layer from engine.base_hydrology_dir and
#     verify it matches the DEM grid.
#     """
#     path = engine.base_hydrology_dir / name
#     if not path.exists():
#         raise FileNotFoundError(f"Missing base layer: {path}")

#     with rasterio.open(path) as src:
#         arr = src.read(1).astype("float32")
#         if arr.shape != like.shape:
#             raise ValueError(
#                 f"Layer {name} shape {arr.shape} != DEM shape {like.shape}"
#             )
#         return arr


# # ---------------------------------------------------------------------
# # FSI vmin/vmax, done once per job (for consistent rendering)
# # ---------------------------------------------------------------------
# def _compute_fsi_vmin_vmax(
#     engine: HydrologyJobEngine, job_id: str
# ) -> tuple[float, float]:
#     """
#     Derive vmin/vmax from FSI values inside rain footprints on land.
#     """
#     job_dir = engine._job_path(job_id)
#     fsi_path = job_dir / "fsi.tif"
#     rain_path = job_dir / "rain_mm.tif"
#     dem_path = engine.base_hydrology_dir / "dem.tif"

#     if not (fsi_path.exists() and rain_path.exists() and dem_path.exists()):
#         return 0.0, 1.0

#     with rasterio.open(fsi_path) as fsi_src, rasterio.open(
#         rain_path
#     ) as rain_src, rasterio.open(dem_path) as dem_src:
#         fsi_full = fsi_src.read(1).astype("float32")
#         fsi_nodata = fsi_src.nodata

#         rain_full = rain_src.read(1).astype("float32")
#         rain_nodata = rain_src.nodata

#         dem_full = dem_src.read(1).astype("float32")
#         dem_nodata = dem_src.nodata

#     # Land mask
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem_full) & (dem_full > 0)
#     else:
#         land_mask = (dem_full != dem_nodata) & np.isfinite(dem_full) & (dem_full > 0)

#     # FSI valid + land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi_full) & land_mask
#     else:
#         f_valid = (fsi_full != fsi_nodata) & np.isfinite(fsi_full) & land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_full > 0) & np.isfinite(rain_full)
#     else:
#         r_valid = (rain_full != rain_nodata) & (rain_full > 0) & np.isfinite(rain_full)

#     mask = f_valid & r_valid
#     vals = fsi_full[mask] if np.any(mask) else fsi_full[f_valid]

#     if vals.size == 0:
#         return 0.0, 1.0

#     vmin = float(np.percentile(vals, 5))
#     vmax = float(np.percentile(vals, 95))
#     if vmax <= vmin:
#         vmax = vmin + 1e-6
#     return vmin, vmax


# # ---------------------------------------------------------------------
# # Job processing
# # ---------------------------------------------------------------------
# def process_job(
#     job_id: str, engine: HydrologyJobEngine, r: Redis, events_channel: str
# ) -> None:
#     job = engine.get_job(job_id)
#     params = job.params or {}

#     event_duration_hours = float(params["event_duration_hours"])
#     runoff_coeff = float(params["runoff_coeff"])
#     rain_radius_km = float(params["rain_radius_km"])
#     stations_data = params.get("stations", [])

#     stations = [
#         RainStation(
#             lat=float(s["lat"]),
#             lon=float(s["lon"]),
#             rain_mm=float(s["rain_mm"]),
#             wind_speed_ms=float(s["wind_speed_ms"])
#             if s.get("wind_speed_ms") is not None
#             else None,
#             wind_dir_deg=float(s["wind_dir_deg"])
#             if s.get("wind_dir_deg") is not None
#             else None,
#         )
#         for s in stations_data
#     ]

#     # ------------------------------------------------------------------
#     # Early exit for dry events
#     # ------------------------------------------------------------------
#     total_rain = sum(s.rain_mm for s in stations)
#     print(f"[{job_id}] Total rain: {total_rain} mm")
#     if total_rain <= 0:
#         params["fsi_vmin"] = 0.0
#         params["fsi_vmax"] = 1.0
#         params["has_runoff"] = False
#         params["reason"] = "no_rain"

#         # (Optional) also mark landslide outputs as absent
#         params["has_landslide"] = False

#         engine.update_job(job_id, status="done", error=None, params=params)
#         r.publish(
#             events_channel,
#             json.dumps({"type": "job_done", "job_id": job_id, "params": params}),
#         )
#         print(f"[{job_id}] Dry event detected → skipping hydrology.")
#         return

#     job_dir = engine._job_path(job_id)

#     # ------------------------------------------------------------------
#     # 1) Load DEM + ACCUM (base hydrology products in EPSG:4326)
#     # ------------------------------------------------------------------
#     dem_path = engine.base_hydrology_dir / "dem.tif"
#     accum_path = engine.base_hydrology_dir / "d8_accum.tif"

#     with rasterio.open(dem_path) as dem_src:
#         dem = dem_src.read(1).astype("float32")
#         dem_transform = dem_src.transform
#         dem_bounds = dem_src.bounds

#     with rasterio.open(accum_path) as accum_src:
#         accum = accum_src.read(1).astype("float32")

#     # Representative latitude for area computation (center of DEM)
#     lat_center = 0.5 * (dem_bounds.top + dem_bounds.bottom)
#     cellsize_deg = abs(dem_transform.a)
#     cell_area_m2 = cell_area_m2_at_lat(lat_center, cellsize_deg)

#     # ------------------------------------------------------------------
#     # 2) Interpolate rainfall to grid (with wind-aware anisotropy)
#     # ------------------------------------------------------------------
#     rain_mm = interpolate_rainfall_to_grid(
#         dem_array=dem,
#         transform=dem_transform,
#         stations=stations,
#         power=2.0,
#         min_dist_cells=1.0,
#         rain_radius_km=rain_radius_km,
#         use_wind=True,
#     ).astype("float32")

#     rain_path = job_dir / "rain_mm.tif"
#     _save_geotiff_4326(rain_path, rain_mm, dem_transform, nodata=0.0)
#     _reproject_to_3857(
#         rain_path, job_dir / "rain_mm_3857.tif", resampling=Resampling.nearest
#     )

#     # ------------------------------------------------------------------
#     # 3) Hydrology chain: rain → runoff → discharge
#     # ------------------------------------------------------------------
#     duration_seconds = event_duration_hours * 3600.0
#     runoff_depth_m = runoff_depth_from_rain(rain_mm, runoff_coeff)
#     drainage_area_m2 = drainage_area_from_accum(accum, cell_area_m2)

#     discharge_m3s = discharge_from_runoff(
#         runoff_depth_m=runoff_depth_m,
#         drainage_area_m2=drainage_area_m2,
#         duration_seconds=duration_seconds,
#     )

#     discharge_path = job_dir / "discharge_m3s.tif"
#     _save_geotiff_4326(discharge_path, discharge_m3s, dem_transform, nodata=np.nan)
#     _reproject_to_3857(
#         discharge_path,
#         job_dir / "discharge_m3s_3857.tif",
#         resampling=Resampling.nearest,
#     )

#     # ------------------------------------------------------------------
#     # 4) Landslide layers (LSI)
#     #   - base: static, stored in base_hydrology_dir (and reused across jobs)
#     #   - trigger + hazard: per job (depends on this event's rain/duration)
#     # ------------------------------------------------------------------
#     lsi_base_path = engine.base_hydrology_dir / "lsi_base.tif"

#     if lsi_base_path.exists():
#         with rasterio.open(lsi_base_path) as lsi_src:
#             lsi_base = lsi_src.read(1).astype("float32")
#     else:
#         # These should have been generated by preprocess_derivatives.py
#         slope = _read_base_layer(engine, "slope_deg.tif", like=dem)
#         twi = _read_base_layer(engine, "twi.tif", like=dem)
#         dist = _read_base_layer(engine, "dist_to_channel_m.tif", like=dem)

#         lsi_base = compute_lsi_base(
#             dem=dem,
#             accum=accum,
#             transform=dem_transform,
#             slope_deg=slope,
#             twi=twi,
#             dist_to_channel_m=dist,
#             dem_nodata=None,
#         ).astype("float32")

#         _save_geotiff_4326(lsi_base_path, lsi_base, dem_transform, nodata=np.nan)
#         _reproject_to_3857(
#             lsi_base_path,
#             engine.base_hydrology_dir / "lsi_base_3857.tif",
#             resampling=Resampling.bilinear,
#         )

#     # Trigger + hazard are job-specific (DO NOT COMMENT THESE OUT)
#     lsi_trigger = compute_lsi_trigger(
#         rain_mm=rain_mm,
#         dem=dem,
#         event_duration_hours=event_duration_hours,
#         dem_nodata=None,
#     ).astype("float32")

#     lsi_hazard = compute_lsi_hazard(lsi_base=lsi_base, lsi_trigger=lsi_trigger).astype(
#         "float32"
#     )

#     lsi_trigger_path = job_dir / "lsi_trigger.tif"
#     lsi_hazard_path = job_dir / "lsi_hazard.tif"

#     _save_geotiff_4326(lsi_trigger_path, lsi_trigger, dem_transform, nodata=np.nan)
#     _save_geotiff_4326(lsi_hazard_path, lsi_hazard, dem_transform, nodata=np.nan)

#     _reproject_to_3857(
#         lsi_trigger_path,
#         job_dir / "lsi_trigger_3857.tif",
#         resampling=Resampling.bilinear,
#     )
#     _reproject_to_3857(
#         lsi_hazard_path, job_dir / "lsi_hazard_3857.tif", resampling=Resampling.bilinear
#     )

#     params["has_landslide"] = True

#     # ------------------------------------------------------------------
#     # 5) Flood Severity Index (FSI)
#     # ------------------------------------------------------------------
#     fsi = compute_fsi(
#         dem=dem,
#         accum=accum,
#         discharge=discharge_m3s,
#         rain_mm=rain_mm,
#         transform=dem_transform,
#     ).astype("float32")

#     fsi_path = job_dir / "fsi.tif"
#     _save_geotiff_4326(fsi_path, fsi, dem_transform, nodata=np.nan)
#     _reproject_to_3857(
#         fsi_path, job_dir / "fsi_3857.tif", resampling=Resampling.bilinear
#     )

#     # ------------------------------------------------------------------
#     # 6) Compute vmin/vmax + persist job metadata
#     # ------------------------------------------------------------------
#     fsi_vmin, fsi_vmax = _compute_fsi_vmin_vmax(engine, job_id)
#     params["fsi_vmin"] = fsi_vmin
#     params["fsi_vmax"] = fsi_vmax
#     params["has_runoff"] = True

#     engine.update_job(job_id, status="done", error=None, params=params)

#     # Publish event
#     r.publish(
#         events_channel,
#         json.dumps({"type": "job_done", "job_id": job_id, "params": params}),
#     )


# # ---------------------------------------------------------------------
# # Worker loop (runs in each process)
# # ---------------------------------------------------------------------
# def worker_loop(worker_index: int) -> None:
#     proc_name = current_process().name
#     print(f"[{proc_name}] Worker #{worker_index} starting…")

#     engine = HydrologyJobEngine(
#         base_hydrology_dir=BASE_HYDRO_DIR,
#         jobs_dir=JOBS_DIR,
#         redis_url=REDIS_URL,
#         prefix="hydro",
#     )

#     r: Redis = engine.redis
#     queue_key = engine._queue_key()
#     events_channel = f"{engine.prefix}:events"

#     print(f"[{proc_name}] Listening on Redis list: {queue_key}")

#     while True:
#         result: BLPOPResult = cast(BLPOPResult, r.blpop([queue_key]))
#         if result is None:
#             continue

#         list_name_raw, job_raw = result

#         list_name = (
#             list_name_raw.decode("utf-8")
#             if isinstance(list_name_raw, (bytes, bytearray))
#             else str(list_name_raw)
#         )
#         job_id = (
#             job_raw.decode("utf-8")
#             if isinstance(job_raw, (bytes, bytearray))
#             else str(job_raw)
#         )

#         print(f"[{proc_name}] Got job_id={job_id} from {list_name}")

#         try:
#             engine.update_job(job_id, status="running")
#             process_job(job_id, engine, r, events_channel)
#             print(f"[{proc_name}] [{job_id}] Done")
#         except Exception as e:
#             print(f"[{proc_name}] [{job_id}] ERROR:", e)
#             engine.update_job(job_id, status="error", error=str(e))


# # ---------------------------------------------------------------------
# # Main entry: spawn multiple workers
# # ---------------------------------------------------------------------
# if __name__ == "__main__":
#     print(
#         f"[hydrology_worker] Starting {WORKER_COUNT} worker process(es) for Redis URL {REDIS_URL}"
#     )

#     workers: list[Process] = []
#     for i in range(WORKER_COUNT):
#         p = Process(target=worker_loop, args=(i,), daemon=False)
#         p.start()
#         workers.append(p)

#     for p in workers:
#         p.join()
