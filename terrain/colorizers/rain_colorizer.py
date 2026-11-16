# terrain/colorizers/rain_colorizer.py

from __future__ import annotations

import numpy as np

# -------------------------------------------------------
# Helpers: distance + wind-aware effective distance
# -------------------------------------------------------


def _distance_km_latlon(
    lat0: float,
    lon0: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> np.ndarray:
    """
    Equirectangular distance (km) from (lat0, lon0) to each cell.
    Good enough at Timor scales.
    """
    R = 6371.0  # km

    lat0_rad = np.deg2rad(lat0)
    lat_rad = np.deg2rad(lat_grid)

    dlat = lat_rad - lat0_rad
    dlon = np.deg2rad(lon_grid - lon0)

    x = dlon * np.cos((lat_rad + lat0_rad) / 2.0)
    y = dlat

    dist_km = R * np.sqrt(x * x + y * y)
    return dist_km


def _effective_distance_with_wind(
    dist_km: np.ndarray,
    lat0: float,
    lon0: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    wind_speed_ms: float,
    wind_dir_deg_from: float,
    anisotropy_factor: float = 0.08,
) -> np.ndarray:
    """
    Same idea as your interpolation:
    shrink distance downwind, stretch upwind.
    """
    lat0_rad = np.deg2rad(lat0)
    lat_rad = np.deg2rad(lat_grid)

    dlat = lat_rad - lat0_rad
    dlon = np.deg2rad(lon_grid - lon0)

    x = dlon * np.cos((lat_rad + lat0_rad) / 2.0)  # east
    y = dlat  # north

    # Bearing station -> cell (0 north, pi/2 east)
    bearing = np.arctan2(x, y)

    # Wind "to" direction (downwind)
    downwind_deg = (wind_dir_deg_from + 180.0) % 360.0
    downwind_rad = np.deg2rad(downwind_deg)

    delta = bearing - downwind_rad
    cos_delta = np.cos(delta)

    scale = 1.0 + anisotropy_factor * wind_speed_ms * cos_delta
    scale = np.clip(scale, 0.25, 2.5)

    return dist_km / scale


# -------------------------------------------------------
# 1) Existing WMO gradient: unchanged
# -------------------------------------------------------


def _wmo_rain_rgb(
    rain_mm: np.ndarray,
    rain_nodata: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Your original WMO-ish gradient -> returns (r,g,b,valid_mask).
    """
    rain = rain_mm.astype("float32")
    h, w = rain.shape

    if rain_nodata is None:
        valid = (rain > 0) & np.isfinite(rain)
    else:
        valid = (rain != rain_nodata) & (rain > 0) & np.isfinite(rain)

    r = np.zeros((h, w), dtype="float32")
    g = np.zeros((h, w), dtype="float32")
    b = np.zeros((h, w), dtype="float32")

    if not np.any(valid):
        return r.astype("uint8"), g.astype("uint8"), b.astype("uint8"), valid

    # Normalize 0–100 mm, gamma to pop high values
    R_MAX = 100.0
    norm = np.clip(rain / R_MAX, 0.0, 1.0)
    norm = norm**0.6

    # ---- segments ----
    m1 = valid & (norm <= 0.15)
    if np.any(m1):
        t = norm[m1] / 0.15
        r[m1] = 210.0 * (1.0 - t)
        g[m1] = 229.0 * (1.0 - t) + 128.0 * t
        b[m1] = 255.0

    m2 = valid & (norm > 0.15) & (norm <= 0.35)
    if np.any(m2):
        t = (norm[m2] - 0.15) / 0.20
        r[m2] = 0.0
        g[m2] = 128.0 * (1.0 - t) + 255.0 * t
        b[m2] = 255.0

    m3 = valid & (norm > 0.35) & (norm <= 0.60)
    if np.any(m3):
        t = (norm[m3] - 0.35) / 0.25
        r[m3] = 255.0 * t
        g[m3] = 255.0
        b[m3] = 255.0 * (1.0 - t)

    m4 = valid & (norm > 0.60) & (norm <= 0.85)
    if np.any(m4):
        t = (norm[m4] - 0.60) / 0.25
        r[m4] = 255.0
        g[m4] = 255.0 * (1.0 - t)
        b[m4] = 0.0

    m5 = valid & (norm > 0.85)
    if np.any(m5):
        t = np.clip((norm[m5] - 0.85) / 0.15, 0.0, 1.0)
        r[m5] = 255.0
        g[m5] = 0.0
        b[m5] = 255.0 * t

    return (
        r.astype("uint8"),
        g.astype("uint8"),
        b.astype("uint8"),
        valid,
    )


# -------------------------------------------------------
# 2) Final colorizer with per-station radial alpha
# -------------------------------------------------------
def _getattr_or_key(obj, key, default=None):
    """Safely get obj[key] or obj.key or return default."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def colorize_rain_rgba(
    rain_mm: np.ndarray,
    rain_nodata: float | None,
    *,
    lat_grid: np.ndarray | None = None,
    lon_grid: np.ndarray | None = None,
    stations: list[dict] | None = None,
    rain_radius_km: float | None = None,
) -> np.ndarray:
    """
    Rainfall colorizer:

    • RGB = your WMO gradient (unchanged)
    • Alpha:
        - If lat_grid/stations/radius not provided → fully opaque where valid.
        - If provided → for each station:
            α_station = (1 - ρ)^γ,  ρ = r_eff / rain_radius_km
          combine all stations with max() so center is dense,
          edges fade smoothly along the same wind-skewed geometry.
    """
    h, w = rain_mm.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")

    # --- 1) RGB + valid mask from existing logic ---
    r, g, b, valid = _wmo_rain_rgb(rain_mm, rain_nodata)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b

    if not np.any(valid):
        # nothing to draw
        return rgba

    # ------------------------------------------------
    # 2) If we do NOT have geometric info, just full alpha
    # ------------------------------------------------
    if (
        lat_grid is None
        or lon_grid is None
        or stations is None
        or rain_radius_km is None
        or len(stations) == 0
    ):
        rgba[..., 3][valid] = 255
        rgba[..., 3][~valid] = 0
        return rgba

    # ------------------------------------------------
    # 3) Per-station radial alpha in wind-skewed space
    # ------------------------------------------------
    alpha_field = np.zeros((h, w), dtype="float32")

    for s in stations:
        lat0 = float(_getattr_or_key(s, "lat", 0.0))
        lon0 = float(_getattr_or_key(s, "lon", 0.0))

        rain_s = float(_getattr_or_key(s, "rain_mm", 0.0))

        if rain_s <= 0:
            continue

        ws = s.get("wind_speed_ms", None)
        wd = s.get("wind_dir_deg", None)

        # Base distance from this station to all pixels in tile
        dist_km = _distance_km_latlon(lat0, lon0, lat_grid, lon_grid)

        # If we have wind info, warp distance like interpolation
        if ws is not None and wd is not None and ws > 0:
            dist_eff = _effective_distance_with_wind(
                dist_km=dist_km,
                lat0=lat0,
                lon0=lon0,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                wind_speed_ms=float(ws),
                wind_dir_deg_from=float(wd),
                anisotropy_factor=0.08,  # match interpolation
            )
        else:
            dist_eff = dist_km

        # Normalized radius ρ = r_eff / R; only inside footprint matters
        rho = dist_eff / float(rain_radius_km)
        mask = rho <= 1.0
        if not np.any(mask):
            continue

        rho_clipped = np.clip(rho, 0.0, 1.0)

        # Alpha contribution: center = 1, edge = 0
        # gamma < 1 → stronger center, gentle falloff
        gamma = 0.8
        contrib = (1.0 - rho_clipped) ** gamma

        # combine by taking the maximum over all stations
        alpha_field = np.maximum(alpha_field, contrib)

    # restrict to pixels where we actually have rainfall
    alpha_field[~valid] = 0.0

    # scale to 0–255
    rgba[..., 3] = (alpha_field * 255.0).astype("uint8")

    return rgba
