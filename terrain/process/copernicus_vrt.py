from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def _iter_tifs(root: Path) -> Iterable[Path]:
    # Prefer Copernicus naming convention
    for p in root.rglob("*.tif"):
        if p.is_file() and p.name.upper().endswith("_DEM.TIF"):
            yield p


def build_dem_vrt(*, tiles_root: Path, vrt_path: Path) -> Path:
    """
    Build a GDAL VRT that references the individual Copernicus tile GeoTIFFs.
    This is *not* a mosaic raster in RAM. It's just an index.
    """
    tifs: List[str] = [str(p) for p in _iter_tifs(tiles_root)]
    if not tifs:
        raise ValueError(f"No Copernicus *_DEM.tif files found under {tiles_root}")

    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    # Use GDAL Python bindings (recommended). This is not shelling out.
    try:
        from osgeo import gdal  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "GDAL Python bindings are required to build a VRT without shelling out.\n"
            "Install on Linux via system packages (recommended), then pip if needed.\n"
            f"Original import error: {e}"
        )

    # resampleAlg here is mostly irrelevant: real resampling happens when you warp/read.
    opts = gdal.BuildVRTOptions(resampleAlg="nearest")
    vrt_ds = gdal.BuildVRT(str(vrt_path), tifs, options=opts)
    if vrt_ds is None:
        raise RuntimeError("gdal.BuildVRT failed (returned None)")

    vrt_ds.FlushCache()
    vrt_ds = None
    return vrt_path
