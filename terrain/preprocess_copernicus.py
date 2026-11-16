# terrain/preprocess_copernicus.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

# If you put the downloader class elsewhere, update this import.
# (This matches the class I drafted previously.)
from terrain.dem_sources.copernicus_s3 import (
    CopernicusDEMConfig,
    CopernicusDEMS3Downloader,
)
from terrain.flow_directions import FlowResult, d8_flow

reasons = Counter()
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _looks_like_dem(src, path: Path) -> bool:
    if path.name.upper().endswith("_DEM.TIF"):
        return True
    if src.dtypes[0] in ("uint8", "uint16"):
        return False
    return True


def _find_dem_tifs(root: Path) -> list[Path]:
    # Prefer explicit DEM naming convention
    dem = sorted(
        [
            p
            for p in root.rglob("*.tif")
            if p.is_file() and p.name.upper().endswith("_DEM.TIF")
        ]
    )
    if dem:
        return dem

    # Fallback: old heuristic (only if no *_DEM.tif exists)
    out: list[Path] = []
    for p in root.rglob("*.tif"):
        if not p.is_file():
            continue
        try:
            with rasterio.open(p) as src:
                if src.count != 1:
                    continue
                if src.dtypes[0] in ("uint8", "uint16"):
                    continue
                out.append(p)
        except Exception:
            continue
    return sorted(out)


def _native_res_from_sources(
    srcs: list[DatasetReader],
) -> tuple[float, float]:
    """
    Returns the smallest (finest) abs pixel size across all sources.
    This prevents merge() from downsampling and pulling from overviews.
    """
    xres = min(abs(s.transform.a) for s in srcs)
    yres = min(abs(s.transform.e) for s in srcs)
    return float(xres), float(yres)


def _best_nodata(srcs: list[DatasetReader]) -> float:
    """
    Copernicus tiles usually have a nodata value (often -32768 or similar).
    Prefer the first defined nodata, fallback to -9999.
    """
    for s in srcs:
        if s.nodata is not None and np.isfinite(float(s.nodata)):
            return float(s.nodata)
    return -9999.0


def _print_dem_stats(tag: str, dem: np.ndarray, nodata_val: float) -> None:
    valid = np.isfinite(dem) & (dem != nodata_val) & (dem > 0)
    print(f"\n📊 DEM stats ({tag})")
    print("  shape:", dem.shape)
    if np.any(valid):
        vals = dem[valid]
        p = np.percentile(vals, [1, 5, 50, 95, 99])
        print("  min/max:", float(vals.min()), float(vals.max()))
        print("  p01 p05 p50 p95 p99:", [float(x) for x in p])
    else:
        print("  ⚠️ no valid land pixels (check nodata + masking)")


def _save_geotiff(
    path: Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    *,
    nodata: float | int | None = None,
    dtype: str = "float32",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if nodata is None:
        nodata_val = -9999.0
    else:
        nodata_val = float(nodata)

    arr = data.astype(dtype, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isfinite(arr), arr, nodata_val).astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": "LZW",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def _reproject_to_3857(
    src_path: Path, dst_path: Path, *, resampling=Resampling.bilinear
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(3857)

        transform, width, height = calculate_default_transform(
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


def _find_downloaded_tifs(root: Path) -> List[Path]:
    # Copernicus buckets contain nested folders; grab all .tif
    return sorted([p for p in root.rglob("*.tif") if p.is_file()])


def _tile_intersects_bbox(src, bbox_lonlat) -> bool:
    # bbox provided in EPSG:4326 lon/lat
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    bbox_crs = CRS.from_epsg(4326)

    # Convert bbox into the tile CRS so we can compare apples-to-apples
    if src.crs and src.crs != bbox_crs:
        b = transform_bounds(
            bbox_crs, src.crs, min_lon, min_lat, max_lon, max_lat, densify_pts=21
        )
    else:
        b = (min_lon, min_lat, max_lon, max_lat)

    # rasterio bounds: (left, bottom, right, top)
    left, bottom, right, top = src.bounds
    b_left, b_bottom, b_right, b_top = b

    # AABB intersection
    return not (right < b_left or left > b_right or top < b_bottom or bottom > b_top)


#
def _mosaic_geotiffs_to_dem(
    tif_paths: List[Path],
    *,
    bbox: Optional[BBox] = None,
) -> tuple[np.ndarray, Affine, CRS]:
    if not tif_paths:
        raise ValueError("No GeoTIFF tiles found to mosaic.")

    srcs: list[DatasetReader] = []
    skipped = 0
    base_crs: CRS | None = None

    for p in tif_paths:
        try:
            src = rasterio.open(str(p))
        except Exception as e:
            print(f"⚠️  Skipping (open failed): {p.name} ({e})")
            skipped += 1
            continue

        try:
            if not _looks_like_dem(src, p):
                if reasons["not_dem_like"] < 20:
                    print(
                        "REJECT:",
                        p.name,
                        "dtype=",
                        src.dtypes[0],
                        "nodata=",
                        src.nodata,
                    )
                    try:
                        print("  tags:", src.tags())
                        print("  desc:", src.descriptions)
                    except Exception:
                        pass
                reasons["not_dem_like"] += 1
                continue

            if src.count != 1:
                print("⚠️ not_1_band:", p, "bands=", src.count)
                reasons["not_1_band"] += 1
                src.close()
                skipped += 1
                continue

            if src.crs is None:
                reasons["no_crs"] += 1
                src.close()
                skipped += 1
                continue

            if base_crs is None:
                base_crs = src.crs
            elif src.crs != base_crs:
                reasons["crs_mismatch"] += 1
                src.close()
                skipped += 1
                continue

            # read sanity
            try:
                _ = src.read(1, out_shape=(1, 1))
            except Exception as e:
                print(f"DEM File Exception {e}")
                reasons["read_failed"] += 1
                src.close()
                skipped += 1
                continue

            # tile-inclusive bbox filter (keep full tile if intersects)
            if bbox is not None and not _tile_intersects_bbox(src, bbox):
                reasons["outside_bbox"] += 1
                src.close()
                skipped += 1
                continue

            srcs.append(src)

        except Exception:
            try:
                src.close()
            except Exception:
                pass
            skipped += 1

    if not srcs or base_crs is None:
        raise ValueError("No compatible single-band GeoTIFFs survived validation.")

    print(f"🧩 Valid tiles: {len(srcs)} (skipped {skipped})")
    if reasons:
        print("Skip reasons:", dict(reasons))

    # ✅ CRITICAL: force finest source resolution so merge doesn't downsample
    xres, yres = _native_res_from_sources(srcs)

    try:
        mosaic, out_transform = rio_merge(
            srcs,
            res=(xres, yres),  # keep native resolution
            masked=True,  # rely on masks, not nodata metadata
            resampling=Resampling.bilinear,
        )
        print("transform repr:", repr(out_transform))
        print("a,e:", out_transform.a, out_transform.e)
        band0 = mosaic[0]
        # dem = band0.filled(-9999.0).astype("float32")  # choose pipeline nodata
        band0_f32 = band0.astype("float32")

        dem = band0_f32.filled(np.nan).astype("float32")  # keep NaN internally
        nodata_val = -9999.0

        # sanitize voids
        dem[~np.isfinite(dem)] = nodata_val
        dem[dem < -1000] = nodata_val

        _print_dem_stats("post-merge", dem, nodata_val)
        return dem, out_transform, base_crs

    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessCopernicusConfig:
    dataset: str = "30m"  # "30m" or "90m"
    bbox: Optional[BBox] = None  # None => default Timor bbox from downloader config
    include_all_if_bbox_null: bool = (
        False  # if bbox None and True -> download everything (huge)
    )
    download_dir: Path = Path("data/elevation_copernicus")
    output_dir: Path = Path("data/hydrology_copernicus")


def run_preprocess_copernicus(
    cfg: PreprocessCopernicusConfig = PreprocessCopernicusConfig(),
) -> None:
    print("🛰️  Copernicus DEM preprocessing")
    print(f"  dataset: {cfg.dataset}")
    print(f"  download_dir: {cfg.download_dir}")
    print(f"  output_dir:   {cfg.output_dir}")
    print(
        f"  bbox:         {cfg.bbox if cfg.bbox is not None else '(default Timor + buffer)'}"
    )

    # 1) Download tiles
    dl_cfg = CopernicusDEMConfig(dataset=cfg.dataset, out_dir=cfg.download_dir)
    downloader = CopernicusDEMS3Downloader(dl_cfg)

    downloaded = downloader.download(
        bbox=cfg.bbox,
        include_all_if_null=cfg.include_all_if_bbox_null,
    )

    print(f"\n⬇️  Downloaded {len(downloaded)} file(s)")
    if len(downloaded) == 0:
        print(
            "  (If you used 30m, some tiles can be missing; try dataset=90m as fallback.)"
        )
        raise ValueError("No tiles downloaded.")

    # 2) Find all .tif under download_dir (some folders may contain multiple COGs)
    tif_paths = _find_dem_tifs(cfg.download_dir)
    print(f"\n🧩 Found {len(tif_paths)} GeoTIFF(s) under {cfg.download_dir}")
    print("Skip reasons:", dict(reasons))
    # 3) Mosaic into DEM
    print("\n🧩 Mosaicing tiles → DEM ...")
    dem, transform, crs = _mosaic_geotiffs_to_dem(tif_paths, bbox=cfg.bbox)
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM CRS:   {crs}")
    print(f"  DEM transform: {transform}")

    # 4) Save DEM
    dem_path = cfg.output_dir / "dem.tif"
    print("\n💾 Saving DEM (EPSG:4326) ...")
    _save_geotiff(dem_path, dem, transform, crs, nodata=-9999.0, dtype="float32")
    print(f"  ✔ Saved → {dem_path}")
    # 2) Build global DEM VRT (tiny index; no mosaic in RAM)
    # vrt_path = cfg.output_dir / "dem.vrt"
    # print("\n🧩 Building DEM VRT (no mosaic in RAM) ...")
    # build_dem_vrt(tiles_root=cfg.download_dir, vrt_path=vrt_path)
    # print(f"  ✔ Saved → {vrt_path}")

    # # 3) Cut a regional DEM from the VRT (bounded memory)
    # # IMPORTANT: If cfg.bbox is None, do NOT try to cut global DEM — that defeats the purpose.
    # if cfg.bbox is None:
    #     raise ValueError(
    #         "bbox is required for regional preprocessing. "
    #         "Do not cut a global DEM; instead, provide a bbox (min_lon,min_lat,max_lon,max_lat)."
    #     )

    # print("\n✂️  Cutting regional DEM from VRT ...")
    # dem_path = cut_dem_from_vrt(
    #     RegionalCutoutConfig(
    #         dem_vrt_path=vrt_path,
    #         out_dir=cfg.output_dir,
    #         bbox=cfg.bbox,
    #         target_res_deg=None,  # keep finest available
    #         nodata_out=-9999.0,
    #     )
    # )
    # print(f"  ✔ Saved → {dem_path}")

    # Load DEM back for D8 (your d8_flow expects an ndarray)
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype("float32")
        transform = dem_src.transform
        crs = dem_src.crs
    # 5) D8 flow + accumulation (same as your existing preprocessing)
    print("\n📐 Computing D8 flow directions + accumulation ...")
    flow: FlowResult = d8_flow(dem)
    dirs = flow.directions.astype("int16")
    accum = flow.accumulation.astype("float32")

    dirs_path = cfg.output_dir / "d8_dirs.tif"
    accum_path = cfg.output_dir / "d8_accum.tif"

    print("\n💾 Saving D8 outputs (EPSG:4326) ...")
    _save_geotiff(dirs_path, dirs, transform, crs, nodata=-1, dtype="int16")
    _save_geotiff(accum_path, accum, transform, crs, nodata=-9999.0, dtype="float32")
    print(f"  ✔ Saved → {dirs_path}")
    print(f"  ✔ Saved → {accum_path}")

    # 6) Reproject to 3857 for tile serving
    print("\n🌐 Reprojecting to WebMercator (EPSG:3857) ...")
    _reproject_to_3857(
        src_path=dem_path,
        dst_path=cfg.output_dir / "dem_3857.tif",
        resampling=Resampling.bilinear,
    )
    _reproject_to_3857(
        src_path=accum_path,
        dst_path=cfg.output_dir / "d8_accum_3857.tif",
        resampling=Resampling.bilinear,
    )

    print("\n✅ Copernicus preprocessing complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Copernicus DEM tiles (AWS S3) → DEM + D8 + accumulation + 3857 reprojections"
    )
    parser.add_argument("--dataset", default="30m", choices=["30m", "90m"])
    parser.add_argument("--download-dir", default="data/elevation_copernicus")
    parser.add_argument("--output-dir", default="data/hydrology_copernicus")

    # bbox args (optional)
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)

    parser.add_argument(
        "--all",
        action="store_true",
        help="If bbox not provided, download ALL tiles (huge; not recommended).",
    )

    args = parser.parse_args()

    bbox = None
    if None not in (args.min_lon, args.min_lat, args.max_lon, args.max_lat):
        bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)

    cfg = PreprocessCopernicusConfig(
        dataset=args.dataset,
        bbox=bbox,
        include_all_if_bbox_null=args.all,
        download_dir=Path(args.download_dir),
        output_dir=Path(args.output_dir),
    )
    run_preprocess_copernicus(cfg)
