# terrain/preprocess_derivatives.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

# Prefer SciPy for fast EDT distance transform; fallback included.
try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover
    _distance_transform_edt = None

# You already have this helper in terrain.runoff
from terrain.runoff import cell_area_m2_at_lat

BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class DerivativesConfig:
    input_dir: Path = Path("data/hydrology_copernicus")
    output_dir: Path = Path("data/hydrology_copernicus")
    stream_threshold_cells: int = 2000  # tune: depends on DEM resolution + basin size
    eps: float = 1e-6
    reproject_3857: bool = True
    overwrite: bool = False


# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------
def _save_geotiff(
    path: Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    *,
    nodata: float | int | None = None,
    dtype: str = "float32",
    compress: str = "LZW",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return  # caller handles overwrite logic

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
        "compress": compress,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def _reproject_to_3857(
    src_path: Path,
    dst_path: Path,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    if dst_path.exists():
        return

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


def _maybe_overwrite(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink(missing_ok=True)


# ------------------------------------------------------------------------------
# Geo helpers
# ------------------------------------------------------------------------------
def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """
    Approx meters per degree latitude, longitude at latitude.
    Good enough for Timor/regional scales.
    """
    lat = np.deg2rad(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
    m_per_deg_lon = 111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
    return float(m_per_deg_lat), float(m_per_deg_lon)


# ------------------------------------------------------------------------------
# Derivatives
# ------------------------------------------------------------------------------
def compute_slope_aspect(
    dem: np.ndarray,
    transform: Affine,
    *,
    eps: float = 1e-6,
    nodata_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      slope_deg, aspect_deg (0..360, 0=N, 90=E).
    """
    # Use DEM center latitude for meters/deg approximation
    # transform maps pixel -> lon/lat:
    # lon = a*col + c; lat = e*row + f (b,d assumed 0 for north-up rasters)
    rows, cols = dem.shape
    lat_center = transform.f + transform.e * (rows / 2.0)

    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat_center)

    cellsize_lon_deg = float(transform.a)
    cellsize_lat_deg = float(abs(transform.e))

    dx_m = cellsize_lon_deg * m_per_deg_lon
    dy_m = cellsize_lat_deg * m_per_deg_lat

    z = dem.astype("float32", copy=False)

    # Replace nodata with NaN so gradients behave
    if nodata_mask is not None:
        z = np.where(nodata_mask, np.nan, z)

    # Central differences via np.gradient (handles edges)
    dzdy, dzdx = np.gradient(z, dy_m, dx_m)  # note ordering: (y, x)

    # Slope
    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = np.rad2deg(slope_rad).astype("float32")

    # Aspect: convert mathematical to compass (0=N, 90=E)
    # Typical GIS: aspect = atan2(dz/dx, -dz/dy) in radians
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype("float32")

    # mask invalid
    if nodata_mask is not None:
        slope_deg = np.where(nodata_mask, np.nan, slope_deg)
        aspect_deg = np.where(nodata_mask, np.nan, aspect_deg)

    return slope_deg, aspect_deg


def compute_twi_spi(
    accum: np.ndarray,
    slope_deg: np.ndarray,
    transform: Affine,
    *,
    eps: float = 1e-6,
    nodata_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TWI = ln( (a) / tan(slope) )
      a = specific catchment area [m] ~ (accum * cell_area_m2) / cell_width_m
    SPI = ln( a * tan(slope) )  (log form makes nicer ranges)
    """
    rows, cols = accum.shape
    lat_center = transform.f + transform.e * (rows / 2.0)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat_center)

    cellsize_deg = float(abs(transform.a))
    cell_area_m2 = float(cell_area_m2_at_lat(lat_center, cellsize_deg))
    cell_width_m = float(cellsize_deg * m_per_deg_lon)

    a_m2 = accum.astype("float32", copy=False) * cell_area_m2
    a_spec_m = a_m2 / max(cell_width_m, eps)

    slope_rad = np.deg2rad(slope_deg.astype("float32", copy=False))
    tan_slope = np.tan(slope_rad)

    twi = np.log((a_spec_m + eps) / (tan_slope + eps)).astype("float32")
    spi = np.log((a_spec_m + eps) * (tan_slope + eps)).astype("float32")

    if nodata_mask is not None:
        twi = np.where(nodata_mask, np.nan, twi)
        spi = np.where(nodata_mask, np.nan, spi)

    return twi, spi


def compute_streams_mask(
    accum: np.ndarray,
    *,
    threshold_cells: int,
    nodata_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    streams = (accum >= float(threshold_cells)).astype("uint8")
    if nodata_mask is not None:
        streams = np.where(nodata_mask, 0, streams).astype("uint8")
    return streams


def compute_distance_to_stream_m(
    streams_mask: np.ndarray,
    transform: Affine,
    *,
    nodata_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Distance in meters to nearest stream cell.
    Uses SciPy EDT if available; otherwise a coarse fallback.
    """
    rows, cols = streams_mask.shape
    lat_center = transform.f + transform.e * (rows / 2.0)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat_center)

    dx_m = float(abs(transform.a) * m_per_deg_lon)
    dy_m = float(abs(transform.e) * m_per_deg_lat)

    # EDT distance_transform_edt computes distance to nearest "zero" element.
    # We want distance to nearest stream==1, so invert:
    inv = streams_mask == 0

    if _distance_transform_edt is not None:
        dist_px = _distance_transform_edt(inv, sampling=(dy_m, dx_m))
        dist_m = np.asarray(dist_px, dtype="float32")
    else:
        # Fallback: very rough two-pass chamfer-like approximation (not perfect).
        dist_m = np.full((rows, cols), np.inf, dtype="float32")
        dist_m[streams_mask == 1] = 0.0

        # forward pass
        for r in range(rows):
            for c in range(cols):
                d = dist_m[r, c]
                if r > 0:
                    d = min(d, dist_m[r - 1, c] + dy_m)
                if c > 0:
                    d = min(d, dist_m[r, c - 1] + dx_m)
                if r > 0 and c > 0:
                    d = min(
                        d, dist_m[r - 1, c - 1] + (dx_m * dx_m + dy_m * dy_m) ** 0.5
                    )
                dist_m[r, c] = d

        # backward pass
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                d = dist_m[r, c]
                if r < rows - 1:
                    d = min(d, dist_m[r + 1, c] + dy_m)
                if c < cols - 1:
                    d = min(d, dist_m[r, c + 1] + dx_m)
                if r < rows - 1 and c < cols - 1:
                    d = min(
                        d, dist_m[r + 1, c + 1] + (dx_m * dx_m + dy_m * dy_m) ** 0.5
                    )
                dist_m[r, c] = d

    if nodata_mask is not None:
        dist_m = np.where(nodata_mask, np.nan, dist_m).astype("float32")

    return dist_m.astype("float32")


# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
def run_preprocess_derivatives(cfg: DerivativesConfig = DerivativesConfig()) -> None:
    in_dir = cfg.input_dir
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path = in_dir / "dem.tif"
    accum_path = in_dir / "d8_accum.tif"

    if not dem_path.exists():
        raise FileNotFoundError(f"Missing DEM: {dem_path}")
    if not accum_path.exists():
        raise FileNotFoundError(f"Missing accumulation: {accum_path}")

    print("🧠 Preprocessing derivatives")
    print(f"  input_dir:  {in_dir}")
    print(f"  output_dir: {out_dir}")
    print(f"  stream_threshold_cells: {cfg.stream_threshold_cells}")
    print(f"  reproject_3857: {cfg.reproject_3857}")
    print(f"  overwrite: {cfg.overwrite}")

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype("float32")
        transform = dem_src.transform
        crs = dem_src.crs or CRS.from_epsg(4326)
        dem_nodata = dem_src.nodata

    with rasterio.open(accum_path) as acc_src:
        accum = acc_src.read(1).astype("float32")
        # acc_nodata = acc_src.nodata

    # nodata mask: treat sea/invalid as nodata
    nodata_mask = np.zeros_like(dem, dtype=bool)
    if dem_nodata is not None:
        nodata_mask |= dem == float(dem_nodata)
    nodata_mask |= ~np.isfinite(dem)

    # Some DEMs have oceans as <= 0; if you want that convention:
    # (optional) consider oceans nodata for these derivatives
    nodata_mask |= dem <= 0

    # ------------------------
    # Slope + aspect
    # ------------------------
    slope_path = out_dir / "slope_deg.tif"
    aspect_path = out_dir / "aspect_deg.tif"

    _maybe_overwrite(slope_path, cfg.overwrite)
    _maybe_overwrite(aspect_path, cfg.overwrite)

    if not slope_path.exists() or not aspect_path.exists():
        print("\n📐 Computing slope/aspect ...")
        slope_deg, aspect_deg = compute_slope_aspect(
            dem, transform, eps=cfg.eps, nodata_mask=nodata_mask
        )

        if not slope_path.exists():
            _save_geotiff(
                slope_path, slope_deg, transform, crs, nodata=-9999.0, dtype="float32"
            )
            print(f"  ✔ Saved → {slope_path}")
        if not aspect_path.exists():
            _save_geotiff(
                aspect_path, aspect_deg, transform, crs, nodata=-9999.0, dtype="float32"
            )
            print(f"  ✔ Saved → {aspect_path}")
    else:
        print("\n⏩ slope/aspect already exist, skipping")

    # Load slope back (if it existed)
    if slope_path.exists():
        with rasterio.open(slope_path) as ssrc:
            slope_deg = ssrc.read(1).astype("float32")
    else:
        slope_deg, _ = compute_slope_aspect(
            dem, transform, eps=cfg.eps, nodata_mask=nodata_mask
        )

    # ------------------------
    # TWI + SPI
    # ------------------------
    twi_path = out_dir / "twi.tif"
    spi_path = out_dir / "spi.tif"
    _maybe_overwrite(twi_path, cfg.overwrite)
    _maybe_overwrite(spi_path, cfg.overwrite)

    if not twi_path.exists() or not spi_path.exists():
        print("\n💧 Computing TWI/SPI ...")
        twi, spi = compute_twi_spi(
            accum, slope_deg, transform, eps=cfg.eps, nodata_mask=nodata_mask
        )
        if not twi_path.exists():
            _save_geotiff(
                twi_path, twi, transform, crs, nodata=-9999.0, dtype="float32"
            )
            print(f"  ✔ Saved → {twi_path}")
        if not spi_path.exists():
            _save_geotiff(
                spi_path, spi, transform, crs, nodata=-9999.0, dtype="float32"
            )
            print(f"  ✔ Saved → {spi_path}")
    else:
        print("\n⏩ twi/spi already exist, skipping")

    # ------------------------
    # Streams mask
    # ------------------------
    streams_path = out_dir / "streams_mask.tif"
    _maybe_overwrite(streams_path, cfg.overwrite)

    if not streams_path.exists():
        print("\n🌊 Computing streams mask ...")
        streams = compute_streams_mask(
            accum, threshold_cells=cfg.stream_threshold_cells, nodata_mask=nodata_mask
        )
        _save_geotiff(streams_path, streams, transform, crs, nodata=0, dtype="uint8")
        print(f"  ✔ Saved → {streams_path}")
    else:
        print("\n⏩ streams_mask already exists, skipping")

    # ------------------------
    # Distance to stream
    # ------------------------
    dist_path = out_dir / "dist_to_channel_m.tif"
    _maybe_overwrite(dist_path, cfg.overwrite)

    if not dist_path.exists():
        print("\n📏 Computing distance-to-stream (meters) ...")
        if streams_path.exists():
            with rasterio.open(streams_path) as ssrc:
                streams = ssrc.read(1).astype("uint8")
        else:
            streams = compute_streams_mask(
                accum,
                threshold_cells=cfg.stream_threshold_cells,
                nodata_mask=nodata_mask,
            )

        dist_m = compute_distance_to_stream_m(
            streams, transform, nodata_mask=nodata_mask
        )
        _save_geotiff(
            dist_path, dist_m, transform, crs, nodata=-9999.0, dtype="float32"
        )
        print(f"  ✔ Saved → {dist_path}")
    else:
        print("\n⏩ dist_to_stream_m already exists, skipping")

    # ------------------------
    # Reproject to 3857 (optional)
    # ------------------------
    if cfg.reproject_3857:
        print("\n🌐 Reprojecting derivatives to EPSG:3857 ...")
        # continuous rasters: bilinear; masks: nearest
        targets = [
            (slope_path, out_dir / "slope_deg_3857.tif", Resampling.bilinear),
            (aspect_path, out_dir / "aspect_deg_3857.tif", Resampling.bilinear),
            (twi_path, out_dir / "twi_3857.tif", Resampling.bilinear),
            (spi_path, out_dir / "spi_3857.tif", Resampling.bilinear),
            (streams_path, out_dir / "streams_mask_3857.tif", Resampling.nearest),
            (dist_path, out_dir / "dist_to_channel_m_3857.tif", Resampling.bilinear),
        ]

        for src, dst, res in targets:
            if src.exists():
                _reproject_to_3857(src, dst, resampling=res)
                print(f"  ✔ {dst.name}")
    else:
        print("\n⏩ reproject_3857 disabled, skipping")

    print("\n✅ Derivatives preprocessing complete.")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute static DEM derivatives (slope/aspect/twi/spi/streams/dist_to_stream) for terrain layers"
    )
    parser.add_argument("--input-dir", default="data/hydrology_copernicus")
    parser.add_argument("--output-dir", default="data/hydrology_copernicus")
    parser.add_argument("--stream-threshold", type=int, default=2000)
    parser.add_argument(
        "--no-3857", action="store_true", help="Disable 3857 reprojections"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )

    args = parser.parse_args()

    run_preprocess_derivatives(
        DerivativesConfig(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            stream_threshold_cells=args.stream_threshold,
            reproject_3857=(not args.no_3857),
            overwrite=args.overwrite,
        )
    )
