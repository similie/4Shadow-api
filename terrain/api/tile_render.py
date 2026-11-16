# terrain/api/tile_render.py
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling

from terrain.api.tile_read import BBox, read_warped_tile
from terrain.colorizers import (
    colorize_accum_rgba,
    colorize_dem_rgba,
    colorize_discharge_rgba,
    colorize_dist_to_channel_rgba,
    colorize_fsi_rgba,
    colorize_lsi_base_rgba,
    colorize_lsi_hazard_rgba,
    colorize_lsi_trigger_rgba,
    colorize_rain_rgba,
    colorize_slope_rgba,
    colorize_twi_rgba,
)


@dataclass(frozen=True)
class RenderContext:
    base_dir: Path  # e.g. data/hydrology_copernicus
    jobs_dir: Path  # e.g. data/hydrology_jobs
    job_id: Optional[str]
    params: dict[str, Any]  # job params from job_engine
    dst_crs: CRS  # from query param srs
    bbox: BBox  # from query param bbox
    width: int
    height: int


BASE_LAYERS = {
    "dem": "dem.tif",
    "accum": "d8_accum.tif",
    "lsi_base": "lsi_base.tif",
    # If you want these exposed too (optional):
    "slope": "slope_deg.tif",
    "twi": "twi.tif",
    "dist_to_channel": "dist_to_channel_m.tif",
}

JOB_LAYERS = {
    "rain": "rain_mm.tif",
    "discharge": "discharge_m3s.tif",
    "fsi": "fsi.tif",
    "lsi_trigger": "lsi_trigger.tif",
    "lsi_hazard": "lsi_hazard.tif",
}


def _job_dir(ctx: RenderContext) -> Path:
    if not ctx.job_id:
        raise HTTPException(400, "This layer requires a job_id")
    return ctx.jobs_dir / ctx.job_id


def _job_path(ctx: RenderContext, name: str) -> Path:
    return _job_dir(ctx) / name


def _base_path(ctx: RenderContext, name: str) -> Path:
    return ctx.base_dir / name


def resolve_base_raster_path(ctx: RenderContext, layer: str) -> Path:
    if layer not in BASE_LAYERS:
        raise HTTPException(400, f"Layer '{layer}' is not a base layer")
    return _base_path(ctx, BASE_LAYERS[layer])


def resolve_job_raster_path(ctx: RenderContext, layer: str) -> Path:
    if layer not in JOB_LAYERS:
        raise HTTPException(400, f"Layer '{layer}' is not a job layer")
    return _job_path(ctx, JOB_LAYERS[layer])


def _resolve_raster_path(ctx: RenderContext, layer: str) -> Path:
    # existing unified resolver (job endpoint can use this)
    if layer in BASE_LAYERS:
        return resolve_base_raster_path(ctx, layer)
    if layer in JOB_LAYERS:
        return resolve_job_raster_path(ctx, layer)
    raise HTTPException(400, f"Unknown layer {layer}")


# def _job_path(ctx: RenderContext, name: str) -> Path:
#     return _job_dir(ctx) / name


# def _resolve_raster_path(ctx: RenderContext, layer: str) -> Path:
#     """
#     Source rasters are all stored as EPSG:4326 GeoTIFFs.
#     We warp per tile request, so we never open *_3857.tif anymore.
#     """
#     if layer == "dem":
#         return _base_path(ctx, "dem.tif")
#     if layer == "accum":
#         return _base_path(ctx, "d8_accum.tif")

#     # LSI base is static (cached in base dir)
#     if layer == "lsi_base":
#         return _base_path(ctx, "lsi_base.tif")

#     # per-job layers
#     if layer == "rain":
#         return _job_path(ctx, "rain_mm.tif")
#     if layer == "discharge":
#         return _job_path(ctx, "discharge_m3s.tif")
#     if layer == "fsi":
#         return _job_path(ctx, "fsi.tif")
#     if layer == "lsi_trigger":
#         return _job_path(ctx, "lsi_trigger.tif")
#     if layer == "lsi_hazard":
#         return _job_path(ctx, "lsi_hazard.tif")

#     raise HTTPException(400, f"Unknown layer {layer}")


def _read_layer_tile(
    ctx: RenderContext,
    layer: str,
    *,
    resampling: Resampling,
) -> tuple[np.ndarray, float | None]:
    path = _resolve_raster_path(ctx, layer)
    if not path.exists():
        raise HTTPException(404, f"Missing raster for layer={layer}: {path}")
    t = read_warped_tile(
        str(path),
        dst_crs=ctx.dst_crs,
        bbox=ctx.bbox,
        width=ctx.width,
        height=ctx.height,
        resampling=resampling,
    )
    return t.data, t.nodata


def render_layer_rgba(ctx: RenderContext, layer: str) -> np.ndarray:
    """
    Central switch for all layers. Returns RGBA uint8 image.
    """

    # Choose resampling per layer
    def rs_for(name: str) -> Resampling:
        # nearest for "categorical-ish" / zero-meaning grids
        if name in ("rain", "discharge"):
            return Resampling.nearest
        # continuous surfaces
        return Resampling.bilinear

    # read main layer tile
    data, nodata = _read_layer_tile(ctx, layer, resampling=rs_for(layer))

    # ----------------------------------------------------------
    # DEM
    # ----------------------------------------------------------
    if layer == "dem":
        return colorize_dem_rgba(data, nodata)

    # ----------------------------------------------------------
    # Accum
    # ----------------------------------------------------------
    if layer == "accum":
        return colorize_accum_rgba(data, nodata)

    # ----------------------------------------------------------
    # LSI Base (static background)
    # ----------------------------------------------------------
    if layer == "lsi_base":
        # you can tune alpha here
        return colorize_lsi_base_rgba(data, nodata, background_alpha=210)

    # ----------------------------------------------------------
    # LSI Trigger
    # ----------------------------------------------------------
    if layer == "lsi_trigger":
        # Make trigger pop: alpha scaled, and boost midrange *display-only*
        # h = np.clip(data.astype("float32"), 0.0, 1.0)
        # h = np.power(h, 0.65)  # stronger pop than hazard; tweak 0.6..0.8
        # return colorize_lsi_hazard_rgba(h, nodata, alpha_min=0, alpha_max=240)
        # return colorize_lsi_trigger_rgba(data, nodata, alpha_min=0, alpha_max=255)
        # d = data.astype("float32")
        # vmax = np.nanmax(d) if np.any(np.isfinite(d)) else 0.0

        # if vmax > 1.5:  # mm/hr stored
        #     return colorize_lsi_hazard_rgba(
        #         d,
        #         nodata,
        #         mode="mmhr",
        #         t0=2.0,
        #         t1=5.0,
        #         t2=10.0,
        #         t3=25.0,
        #         gamma=0.75,
        #         alpha_min=90,
        #         alpha_max=255,
        #     )
        # else:  # old trigger index stored
        #     h = np.clip(d, 0.0, 1.0)
        #     h = np.power(h, 0.55)
        #     return colorize_lsi_hazard_rgba(
        #         h,
        #         nodata,
        #         mode="index",
        #         show_from=0.01,
        #         gamma=0.45,
        #         alpha_min=120,
        #         alpha_max=255,
        #     )
        return colorize_lsi_hazard_rgba(
            data,
            nodata,
            mode="mmhr",
            t0=0.5,
            t3=12.0,  # MUST match compute_lsi_trigger thresholds
            gamma=0.55,  # make the forcing pop
            show_from=0.25,  # hide tiny drizzle noise
            alpha_min=90,
            alpha_max=255,
        )
    # ----------------------------------------------------------
    # LSI Hazard
    # ----------------------------------------------------------
    if layer == "lsi_hazard":
        # h = np.clip(data.astype("float32"), 0.0, 1.0)
        # h = np.power(h, 0.7)  # boost midrange for visibility (display-only)
        # return colorize_lsi_hazard_rgba(h, nodata, alpha_min=0, alpha_max=255)
        # h = np.clip(data.astype("float32"), 0.0, 1.0)
        # h = np.power(h, 0.7)  # display-only boost
        # return colorize_lsi_hazard_rgba(
        #     h,
        #     nodata,
        #     mode="index",
        #     show_from=0.02,
        #     gamma=0.6,
        #     alpha_min=90,
        #     alpha_max=255,
        # )
        h = np.clip(data.astype("float32"), 0.0, 1.0)
        # Boost mid/low hazard so features show up (your “red ridges” come back)
        h = np.power(h, 0.45)
        return colorize_lsi_hazard_rgba(
            h,
            nodata,
            mode="index",
            gamma=0.70,
            show_from=0.02,  # suppress near-zero haze
            alpha_min=80,
            alpha_max=255,
        )
    # ----------------------------------------------------------
    # Rain
    # ----------------------------------------------------------
    if layer == "rain":
        stations_param = ctx.params.get("stations", [])
        rain_radius_km = float(ctx.params.get("rain_radius_km", 10.0))

        # IMPORTANT: Your old rain colorizer builds lat/lon grids from src.transform+window.
        # With warp-per-tile, we no longer have a meaningful 4326 window transform here.
        #
        # Best option: update colorize_rain_rgba to accept bbox in EPSG:3857 + dst_crs,
        # or accept precomputed lat/lon grids from EPSG:4326 by inverse-transforming.
        #
        # For now: keep it simple and pass NO lat/lon grid if your colorizer supports it,
        # otherwise you should refactor rain colorizer separately.
        return colorize_rain_rgba(
            rain_mm=data,
            rain_nodata=nodata,
            lat_grid=None,  # update your colorizer to handle None
            lon_grid=None,  # update your colorizer to handle None
            stations=stations_param,
            rain_radius_km=rain_radius_km,
        )

    # ----------------------------------------------------------
    # Discharge
    # ----------------------------------------------------------
    if layer == "discharge":
        rain_tile, rain_nodata = _read_layer_tile(
            ctx, "rain", resampling=rs_for("rain")
        )
        dem_tile, dem_nodata = _read_layer_tile(ctx, "dem", resampling=rs_for("dem"))

        return colorize_discharge_rgba(
            discharge=data,
            discharge_nodata=nodata,
            rain_mm=rain_tile,
            rain_nodata=rain_nodata,
            dem=dem_tile,
            dem_nodata=dem_nodata,
            background_alpha=166,
        )

    # ----------------------------------------------------------
    # FSI
    # ----------------------------------------------------------
    if layer == "fsi":
        dem_tile, dem_nodata = _read_layer_tile(ctx, "dem", resampling=rs_for("dem"))
        rain_tile, rain_nodata = _read_layer_tile(
            ctx, "rain", resampling=rs_for("rain")
        )

        # Pull job-derived vmin/vmax (your existing helper)
        fsi_vmin = float(ctx.params.get("fsi_vmin", 0.0))
        fsi_vmax = float(ctx.params.get("fsi_vmax", 1.0))

        return colorize_fsi_rgba(
            fsi=data,
            fsi_nodata=nodata,
            dem=dem_tile,
            dem_nodata=dem_nodata,
            rain_mm=rain_tile,
            rain_nodata=rain_nodata,
            vmin=fsi_vmin,
            vmax=fsi_vmax,
        )

    raise HTTPException(400, f"Unknown layer {layer}")


def _read_tile_from_path(
    ctx: RenderContext,
    path: Path,
    *,
    resampling: Resampling,
) -> tuple[np.ndarray, float | None]:
    if not path.exists():
        raise HTTPException(404, f"Missing raster: {path}")
    t = read_warped_tile(
        str(path),
        dst_crs=ctx.dst_crs,
        bbox=ctx.bbox,
        width=ctx.width,
        height=ctx.height,
        resampling=resampling,
    )
    return t.data, t.nodata


def render_base_layer_rgba(ctx: RenderContext, layer: str) -> np.ndarray:
    """
    Base-only switch. No job rasters may be referenced here.
    """
    path = resolve_base_raster_path(ctx, layer)

    # resampling for base layers
    rs = (
        Resampling.bilinear
        if layer in ("dem", "accum", "lsi_base")
        else Resampling.nearest
    )

    data, nodata = _read_tile_from_path(ctx, path, resampling=rs)

    if layer == "dem":
        return colorize_dem_rgba(data, nodata)

    if layer == "accum":
        return colorize_accum_rgba(data, nodata)

    if layer == "lsi_base":
        return colorize_lsi_base_rgba(data, nodata, background_alpha=210)

    if layer == "slope":
        return colorize_slope_rgba(data, nodata, alpha=210)

    if layer == "twi":
        return colorize_twi_rgba(data, nodata, alpha=210)

    if layer == "dist_to_channel":
        return colorize_dist_to_channel_rgba(data, nodata, alpha=220, max_m=3000.0)

    raise HTTPException(400, f"Unknown base layer {layer}")


def _pil_from_rgba(rgba: np.ndarray, width: int, height: int) -> Image.Image:
    img = Image.fromarray(rgba, mode="RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
    return img


def render_data_tile(rgba: np.ndarray, width: int, height: int) -> bytes:
    img = _pil_from_rgba(rgba, width, height)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
