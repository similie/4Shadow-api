from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import from_bounds
from rasterio.warp import reproject

BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


@dataclass(frozen=True)
class RegionalCutoutConfig:
    dem_vrt_path: Path
    out_dir: Path
    bbox: BBox
    # target output resolution in degrees; for Copernicus 30m it's ~ 1/3600 = 0.000277777...
    # If you want to preserve native, we’ll use the finest from the source.
    target_res_deg: Optional[float] = None
    nodata_out: float = -9999.0


def _finest_res_deg(src: DatasetReader) -> float:
    # For VRT, this should still be meaningful (it resolves to sources).
    return float(min(abs(src.transform.a), abs(src.transform.e)))


def cut_dem_from_vrt(cfg: RegionalCutoutConfig) -> Path:
    """
    Warps/samples the VRT into an AOI DEM GeoTIFF in EPSG:4326.
    This reads only what’s needed. Memory use is bounded by AOI size.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "dem.tif"

    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    dst_crs = CRS.from_epsg(4326)

    with rasterio.open(cfg.dem_vrt_path) as src:
        if src.crs is None:
            raise ValueError(f"VRT has no CRS: {cfg.dem_vrt_path}")

        # Decide output resolution
        res = cfg.target_res_deg or _finest_res_deg(src)

        # Compute output shape from bbox + res
        width = int(np.ceil((max_lon - min_lon) / res))
        height = int(np.ceil((max_lat - min_lat) / res))
        width = max(width, 1)
        height = max(height, 1)

        dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Read by warping into a destination array
        dest = np.full((height, width), cfg.nodata_out, dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=cfg.nodata_out,
        )

    # Write GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": cfg.nodata_out,
        "compress": "LZW",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dest, 1)

    return out_path
