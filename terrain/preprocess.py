# terrain/preprocess.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from .flow_directions import FlowResult, d8_flow
from .srtm_hgt import DemMosaic, mosaic_hgt_tiles


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_hgt_files(folder: str | Path) -> List[str]:
    folder = Path(folder)
    return [str(p) for p in folder.glob("*.hgt")]


def save_geotiff(path: str | Path, data: np.ndarray, dem: DemMosaic, nodata=np.nan):
    """
    Save a DEM-aligned raster (2D) to GeoTIFF using the DEM's transform & CRS.
    """
    path = Path(path)
    rows, cols = data.shape

    transform: Affine = dem.transform
    crs: CRS = dem.crs or CRS.from_epsg(4326)

    nodata_val = -9999 if np.isnan(nodata) else float(nodata)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": "LZW",
    }

    arr = np.where(np.isnan(data), nodata_val, data).astype(profile["dtype"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def reproject_to_3857(src_path: Path, dst_path: Path, resampling=Resampling.bilinear):
    """
    Reproject a raster on disk to EPSG:3857 WebMercator.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(3857)

        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32",
                "compress": "LZW",
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )

        print(f"  ✔ Reprojected → {dst_path}")


# ---------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------
def run_preprocessing(
    elevation_dir: str | Path = "data/elevation",
    output_dir: str | Path = "data/hydrology",
):
    """
    Complete hydrology preprocessing pipeline:

      1. Load + mosaic HGT tiles (EPSG:4326)
      2. Compute D8 directions + accumulation
      3. Save these rasters in EPSG:4326
      4. Reproject DEM + accumulation to EPSG:3857 (for tile server)
    """
    elevation_dir = Path(elevation_dir)
    output_dir = Path(output_dir)

    print(f"🌍 Searching for HGT tiles in {elevation_dir} ...")
    paths = find_hgt_files(elevation_dir)
    if not paths:
        raise ValueError(f"No .hgt files found in {elevation_dir}")

    print(f"  Found {len(paths)} tiles:")
    for p in paths:
        print(f"   - {p}")

    print("\n🧩 Building DEM mosaic ...")
    dem_mosaic: DemMosaic = mosaic_hgt_tiles(paths)
    dem = dem_mosaic.data
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM CRS:   {dem_mosaic.crs}")
    print(f"  DEM transform: {dem_mosaic.transform}")

    print("\n📐 Computing D8 flow directions + accumulation ...")
    flow: FlowResult = d8_flow(dem)
    dirs = flow.directions.astype("int16")
    accum = flow.accumulation.astype("float32")

    print("\n💾 Saving WGS84 rasters (EPSG:4326) ...")
    save_geotiff(output_dir / "dem.tif", dem.astype("float32"), dem_mosaic)
    save_geotiff(output_dir / "d8_dirs.tif", dirs, dem_mosaic, nodata=-1)
    save_geotiff(output_dir / "d8_accum.tif", accum, dem_mosaic)

    print("\n🌐 Reprojecting to WebMercator (EPSG:3857) ...")
    reproject_to_3857(
        src_path=output_dir / "dem.tif",
        dst_path=output_dir / "dem_3857.tif",
        resampling=Resampling.bilinear,
    )
    reproject_to_3857(
        src_path=output_dir / "d8_accum.tif",
        dst_path=output_dir / "d8_accum_3857.tif",
        resampling=Resampling.bilinear,
    )

    print("\n✅ Preprocessing complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess SRTM .hgt tiles → DEM + D8 + accumulation + reprojected rasters"
    )
    parser.add_argument(
        "--elevation",
        default="data/elevation",
        help="Folder containing .hgt files",
    )
    parser.add_argument(
        "--output",
        default="data/hydrology",
        help="Output folder for GeoTIFFs",
    )

    args = parser.parse_args()
    run_preprocessing(args.elevation, args.output)
