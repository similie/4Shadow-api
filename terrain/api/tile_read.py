# terrain/api/tile_read.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


@dataclass(frozen=True)
class Tile:
    data: np.ndarray
    nodata: float | None


def parse_bbox(bbox_str: str) -> BBox:
    parts = [float(p) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 4 comma-separated numbers")
    return parts[0], parts[1], parts[2], parts[3]


def parse_srs(srs: str) -> CRS:
    # expects "EPSG:3857" etc.
    return CRS.from_string(srs)


def read_warped_tile(
    src_path: str,
    *,
    dst_crs: CRS,
    bbox: BBox,
    width: int,
    height: int,
    resampling: Resampling,
) -> Tile:
    """
    Read only the requested bbox/size from a source raster (typically EPSG:4326),
    warping on the fly to dst_crs (typically EPSG:3857).
    """
    minx, miny, maxx, maxy = bbox

    with rasterio.open(src_path) as src:
        # Use the raster's own nodata if present; WarpedVRT will mask/propagate.
        with WarpedVRT(
            src,
            crs=dst_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=src.nodata,
        ) as vrt:
            window = from_bounds(minx, miny, maxx, maxy, transform=vrt.transform)

            # out_shape ensures exactly the requested tile size
            arr = vrt.read(
                1,
                window=window,
                out_shape=(height, width),
                masked=False,
            ).astype("float32", copy=False)

            return Tile(data=arr, nodata=vrt.nodata)
