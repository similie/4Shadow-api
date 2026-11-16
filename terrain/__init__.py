# terrain/__init__.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


@dataclass
class HydrologyStack:
    dem: np.ndarray  # elevation [m]
    dirs: np.ndarray  # D8 directions (0..7 or -1)
    accum: np.ndarray  # flow accumulation (upstream cell count)
    transform: Affine  # geotransform
    crs: CRS  # typically EPSG:4326 (lat/lon)
    cellsize_deg: float  # angular resolution in degrees


def load_hydrology_stack(
    hydrology_dir: str | Path = "data/hydrology",
) -> HydrologyStack:
    """
    Load DEM, D8 directions, and flow accumulation from GeoTIFFs created
    by terrain.preprocess.run_preprocessing().
    """
    hydrology_dir = Path(hydrology_dir)

    dem_path = hydrology_dir / "dem.tif"
    dirs_path = hydrology_dir / "d8_dirs.tif"
    accum_path = hydrology_dir / "d8_accum.tif"

    if not dem_path.exists():
        raise FileNotFoundError(f"Missing DEM file: {dem_path}")
    if not dirs_path.exists():
        raise FileNotFoundError(f"Missing D8 dirs file: {dirs_path}")
    if not accum_path.exists():
        raise FileNotFoundError(f"Missing accumulation file: {accum_path}")

    with rasterio.open(dem_path) as src_dem:
        dem = src_dem.read(1).astype("float32")
        transform = src_dem.transform
        crs = src_dem.crs

    with rasterio.open(dirs_path) as src_dir:
        dirs = src_dir.read(1).astype("int16")

    with rasterio.open(accum_path) as src_acc:
        accum = src_acc.read(1).astype("float32")

    # cellsize (deg) from transform (assuming north-up, no rotation)
    cellsize_deg = float(abs(transform.a))

    return HydrologyStack(
        dem=dem,
        dirs=dirs,
        accum=accum,
        transform=transform,
        crs=crs,
        cellsize_deg=cellsize_deg,
    )
