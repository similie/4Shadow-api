# terrain/analysis/flood_index.py
import numpy as np
from scipy.ndimage import sobel


def compute_slope_deg(dem: np.ndarray, cellsize_deg: float) -> np.ndarray:
    """
    Compute slope (degrees) using a Sobel filter on the DEM.
    cellsize_deg ≈ lat/lon spacing.
    """
    dzdx = sobel(dem, axis=1) / (8 * cellsize_deg)
    dzdy = sobel(dem, axis=0) / (8 * cellsize_deg)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad).astype("float32")


def normalize_log(arr, mask):
    """log-normalize array where mask=True."""
    vals = arr[mask]
    logv = np.log10(vals + 1e-6)
    vmin = np.percentile(logv, 5)
    vmax = np.percentile(logv, 95)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    out = np.zeros_like(arr, dtype="float32")
    out[mask] = np.clip((logv - vmin) / (vmax - vmin), 0, 1)
    return out


def compute_fsi(
    dem: np.ndarray,
    accum: np.ndarray,
    discharge: np.ndarray,
    rain_mm: np.ndarray,
    transform,
):
    """
    Compute Flood Susceptibility Index (FSI) on the DEM grid.
    Returns float32 array 0..1 ready for saving as a GeoTIFF.
    """
    h, w = dem.shape
    cellsize_deg = abs(transform.a)

    # -------------------------
    # 1. Masks
    # -------------------------
    d_mask = np.isfinite(discharge) & (discharge > 0)
    a_mask = np.isfinite(accum) & (accum > 0)
    r_mask = np.isfinite(rain_mm) & (rain_mm > 0)

    # -------------------------
    # 2. Components
    # -------------------------
    Q_norm = normalize_log(discharge, d_mask)
    Acc_norm = normalize_log(accum, a_mask)

    slope_deg = compute_slope_deg(dem, cellsize_deg)
    slope_norm = np.clip(slope_deg / 45.0, 0, 1)  # 45° ≈ steep
    Flatness = 1 - slope_norm  # flat = high flood risk

    rain_norm = np.zeros_like(rain_mm, "float32")
    if np.any(r_mask):
        max_r = float(np.nanmax(rain_mm))
        rain_norm[r_mask] = rain_mm[r_mask] / max(max_r, 1e-6)
        rain_norm = np.clip(rain_norm, 0, 1)

    # -------------------------
    # 3. Weighted Combination
    # -------------------------
    FSI = 0.35 * Q_norm + 0.25 * Acc_norm + 0.20 * Flatness + 0.20 * rain_norm

    return np.clip(FSI, 0, 1).astype("float32")
