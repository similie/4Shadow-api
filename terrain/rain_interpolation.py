from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rasterio.transform import Affine


@dataclass
class RainStation:
    lat: float
    lon: float
    rain_mm: float  # event rainfall or accumulated rainfall
    wind_speed_ms: float | None = None  # instantaneous wind speed
    wind_dir_deg: float | None = None  # degrees FROM which wind blows (met convention)


def _latlon_grid(
    transform: Affine, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 2D arrays of lat, lon for each cell center.
    """
    r_idx, c_idx = np.indices((rows, cols))

    lon = transform.c + (c_idx + 0.5) * transform.a + (r_idx + 0.5) * transform.b
    lat = transform.f + (c_idx + 0.5) * transform.d + (r_idx + 0.5) * transform.e

    return lat.astype("float32"), lon.astype("float32")


def _distance_km(
    lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate dx, dy and great-circle distance in km from (lat0, lon0)
    to each grid point.
    """
    R_earth = 6371.0  # km

    lat0_rad = np.deg2rad(lat0)
    lat_rad = np.deg2rad(lat)

    dlat = lat_rad - lat0_rad
    dlon = np.deg2rad(lon - lon0)

    # equirectangular projection: x=east, y=north
    x = dlon * np.cos((lat_rad + lat0_rad) / 2.0)
    y = dlat

    dx_km = R_earth * x
    dy_km = R_earth * y

    dist_km = np.sqrt(dx_km**2 + dy_km**2)
    return dx_km, dy_km, dist_km


def _elliptical_wind_distance(
    dx_km: np.ndarray,
    dy_km: np.ndarray,
    wind_speed_ms: float | None,
    wind_dir_deg_from: float | None,
    base_radius_km: float,
    anisotropy_strength: float = 0.03,
    q_max: float = 2.0,
) -> np.ndarray:
    """
    Return an *effective distance* [km] that defines an ellipse aligned
    with the downwind direction, but preserves total area ~ π * R^2.

    - For zero / missing wind: returns isotropic distance (sqrt(dx^2+dy^2)).
    - For stronger winds: stretches downwind, squeezes cross-wind, but the
      radius_km is still the "nominal" footprint size.

    Geometry:
        q = 1 + s * wind_speed
        a = R * q           (semi-axis along downwind)
        b = R / q           (semi-axis cross-wind)
        s^2 = (x_par^2 / a^2) + (y_perp^2 / b^2)
        rho = R * s         (effective distance used for weighting)
    """
    # No wind: just Euclidean distance
    if (
        wind_speed_ms is None
        or wind_dir_deg_from is None
        or wind_speed_ms <= 0.0
        or anisotropy_strength <= 0.0
    ):
        return np.sqrt(dx_km**2 + dy_km**2)

    # Convert "from" direction to downwind "to" direction (deg from north, clockwise)
    theta_from = np.deg2rad(wind_dir_deg_from)
    theta_to = (theta_from + np.pi) % (2.0 * np.pi)

    # Unit vector downwind in our (x=east, y=north) basis
    ux = np.sin(theta_to)  # east component
    uy = np.cos(theta_to)  # north component

    # Rotate coordinates into wind frame
    # x_par  : axis along downwind direction
    # y_perp : axis perpendicular to downwind
    x_par = dx_km * ux + dy_km * uy
    y_perp = -dx_km * uy + dy_km * ux

    # Anisotropy ratio q: how elongated along wind
    q = 1.0 + anisotropy_strength * wind_speed_ms
    q = np.clip(q, 1.0, q_max)

    # Semi-axes of ellipse
    a = base_radius_km * q
    b = base_radius_km / q

    # Elliptical radius (normalized)
    # s=1 defines the ellipse boundary
    a2 = (a * a) + 1e-6
    b2 = (b * b) + 1e-6
    s2 = (x_par**2) / a2 + (y_perp**2) / b2
    s = np.sqrt(s2)

    # Effective distance in km
    rho = base_radius_km * s
    return rho


def interpolate_rainfall_to_grid(
    dem_array: np.ndarray,
    transform: Affine,
    stations: List[RainStation],
    power: float = 2.0,
    min_dist_cells: float = 1.0,
    rain_radius_km: float = 10.0,
    use_wind: bool = True,
) -> np.ndarray:
    """
    Inverse-distance weighted rainfall interpolation with optional
    wind-aware, *area-preserving* elliptical kernel.

    - If wind is missing or disabled → simple circular IDW within rain_radius_km.
    - If wind is present → elliptical footprint aligned with downwind direction,
      stretched by wind_speed, but total "area of influence" stays ~constant.
    """

    rows, cols = dem_array.shape
    lat_grid, lon_grid = _latlon_grid(transform, rows, cols)

    # approximate km per cell in latitude for min_dist
    cell_dlat = abs(transform.e)
    cell_km = 111.0 * cell_dlat if cell_dlat > 0 else 1.0
    min_dist_km = max(min_dist_cells * cell_km, 0.01)

    rain_sum = np.zeros((rows, cols), dtype="float32")
    weight_sum = np.zeros((rows, cols), dtype="float32")

    for st in stations:
        if st.rain_mm <= 0:
            continue

        dx_km, dy_km, dist_km = _distance_km(st.lat, st.lon, lat_grid, lon_grid)

        if use_wind:
            dist_eff_km = _elliptical_wind_distance(
                dx_km=dx_km,
                dy_km=dy_km,
                wind_speed_ms=st.wind_speed_ms,
                wind_dir_deg_from=st.wind_dir_deg,
                base_radius_km=rain_radius_km,
                anisotropy_strength=0.03,  # tweak: smaller → less elongation
                q_max=2.0,  # max elongation factor
            )
        else:
            dist_eff_km = dist_km

        # Final radius mask: effective distance <= rain_radius_km
        mask = dist_eff_km <= rain_radius_km
        if not np.any(mask):
            continue

        # Avoid singularity at station
        dist_eff_km = np.maximum(dist_eff_km, min_dist_km)

        w = np.zeros_like(dist_eff_km, dtype="float32")
        w[mask] = 1.0 / (dist_eff_km[mask] ** power)

        rain_sum[mask] += st.rain_mm * w[mask]
        weight_sum[mask] += w[mask]

    result = np.zeros_like(rain_sum, dtype="float32")
    valid = weight_sum > 0
    result[valid] = rain_sum[valid] / weight_sum[valid]

    return result
