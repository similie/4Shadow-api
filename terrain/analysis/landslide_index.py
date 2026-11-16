# terrain/analysis/landslide_index.py
from __future__ import annotations

import numpy as np


# ----------------------------
# Small helpers
# ----------------------------
def _robust_norm_01(x: np.ndarray, mask: np.ndarray, p_lo=5.0, p_hi=95.0) -> np.ndarray:
    """
    Robust normalize to 0..1 using percentiles over mask.
    Returns float32 array in [0,1], with 0 outside mask.
    """
    out = np.zeros_like(x, dtype="float32")
    if not np.any(mask):
        return out

    vals = x[mask].astype("float32")
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    y = (x.astype("float32") - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    out[mask] = y[mask]
    return out


def _land_mask_from_dem(dem: np.ndarray, dem_nodata: float | None) -> np.ndarray:
    if dem_nodata is None:
        return np.isfinite(dem) & (dem > 0)
    return (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """
    Approx meters per degree latitude / longitude (good enough for Timor scale).
    """
    lat = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
    m_per_deg_lon = 111_412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
    return float(m_per_deg_lat), float(m_per_deg_lon)


def _slope_from_dem(dem: np.ndarray, transform) -> np.ndarray:
    """
    Compute slope (tan(slope)) from DEM using finite differences in meters.

    transform is affine for EPSG:4326 where a is degrees/pixel in lon,
    e is degrees/pixel in lat (negative).
    """
    h, w = dem.shape

    # approximate lat_center from affine (f + e*(row+0.5))
    lat_center = float(transform.f + transform.e * (h * 0.5))
    mlat, mlon = _meters_per_degree(lat_center)

    dx_m = abs(float(transform.a)) * mlon
    dy_m = abs(float(transform.e)) * mlat

    # gradients (dz/dx, dz/dy) in m/m
    dz_dy, dz_dx = np.gradient(dem.astype("float32"), dy_m, dx_m)
    slope_tan = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy).astype("float32")
    return slope_tan


def _curvature_laplacian(dem: np.ndarray, transform) -> np.ndarray:
    """
    Simple curvature proxy: Laplacian of elevation (units ~ 1/m).
    This is not “true” plan/profile curvature but works as a v1 roughness/shape term.
    """
    h, w = dem.shape
    lat_center = float(transform.f + transform.e * (h * 0.5))
    mlat, mlon = _meters_per_degree(lat_center)

    dx_m = abs(float(transform.a)) * mlon
    dy_m = abs(float(transform.e)) * mlat

    z = dem.astype("float32")

    d2x = (np.roll(z, -1, axis=1) - 2.0 * z + np.roll(z, 1, axis=1)) / (dx_m * dx_m)
    d2y = (np.roll(z, -1, axis=0) - 2.0 * z + np.roll(z, 1, axis=0)) / (dy_m * dy_m)
    lap = (d2x + d2y).astype("float32")
    return lap


def _twi_from_accum_and_slope(
    accum: np.ndarray,
    slope_tan: np.ndarray,
    *,
    cell_area_m2: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Classic-ish TWI proxy: ln( a / tan(slope) )

    - accum is flow accumulation in "number of contributing cells"
    - a is specific catchment area approximation:
        a ≈ accum_cells * cell_area_m2 / cell_width_m
      We omit width explicitly (constant factor at this scale) and use:
        a' = accum_cells * cell_area_m2
      TWI will still be relative (then normalized).
    """
    a = np.maximum(accum.astype("float32"), 0.0) * float(cell_area_m2)
    twi = np.log((a + eps) / (slope_tan + eps)).astype("float32")
    return twi


# ----------------------------
# Public API
# ----------------------------
# def compute_lsi_base(
#     *,
#     dem: np.ndarray,
#     accum: np.ndarray,
#     transform,
#     slope_deg: np.ndarray | None = None,
#     twi: np.ndarray | None = None,
#     dist_to_channel_m: np.ndarray | None = None,
#     dem_nodata: float | None = None,
# ) -> np.ndarray:
#     """
#     Intrinsic landslide susceptibility (static):

#     Uses:
#       - slope (dominant)
#       - twi proxy (convergence + wetness potential)
#       - curvature proxy (roughness/shape)

#     Returns float32 in [0..1], masked to land.
#     """
#     land = _land_mask_from_dem(dem, dem_nodata)

#     slope_tan = _slope_from_dem(dem, transform)
#     curv = _curvature_laplacian(dem, transform)

#     # cell area estimate (same idea as your worker)
#     h = dem.shape[0]
#     lat_center = float(transform.f + transform.e * (h * 0.5))
#     mlat, mlon = _meters_per_degree(lat_center)
#     cellsize_deg = abs(float(transform.a))
#     cell_area_m2 = (cellsize_deg * mlon) * (abs(float(transform.e)) * mlat)

#     twi = _twi_from_accum_and_slope(accum, slope_tan, cell_area_m2=float(cell_area_m2))

#     # robust normalize each term
#     slopeN = _robust_norm_01(slope_tan, land, 5, 95)
#     twiN = _robust_norm_01(twi, land, 5, 95)
#     curvN = _robust_norm_01(np.abs(curv), land, 5, 95)

#     # weighted blend (v1)
#     lsi = (0.65 * slopeN + 0.25 * twiN + 0.10 * curvN).astype("float32")
#     lsi = np.clip(lsi, 0.0, 1.0)


#     # hard mask to ocean
#     lsi[~land] = np.nan
#     return lsi
def compute_lsi_base(
    *,
    dem: np.ndarray,
    accum: np.ndarray,
    transform,
    slope_deg: np.ndarray | None = None,
    twi: np.ndarray | None = None,
    dist_to_channel_m: np.ndarray | None = None,
    dem_nodata: float | None = None,
) -> np.ndarray:
    """
    Intrinsic landslide susceptibility (static) in [0..1], masked to land.

    Uses:
      - slope (dominant)
      - twi proxy (convergence + wetness potential)
      - curvature proxy (roughness/shape)
      - OPTIONAL distance-to-channel (stabilizer: farther from channel -> slightly lower susceptibility)
    """
    land = _land_mask_from_dem(dem, dem_nodata)

    # --- slope ---
    if slope_deg is not None:
        # convert degrees -> tan(slope)
        slope_tan = np.tan(np.deg2rad(slope_deg.astype("float32"))).astype("float32")
    else:
        slope_tan = _slope_from_dem(dem, transform)

    # --- curvature (still derived from DEM in v1) ---
    curv = _curvature_laplacian(dem, transform)

    # --- twi ---
    if twi is None:
        # cell area estimate (same idea as your worker)
        h = dem.shape[0]
        lat_center = float(transform.f + transform.e * (h * 0.5))
        mlat, mlon = _meters_per_degree(lat_center)
        cell_area_m2 = (abs(float(transform.a)) * mlon) * (
            abs(float(transform.e)) * mlat
        )

        twi = _twi_from_accum_and_slope(
            accum, slope_tan, cell_area_m2=float(cell_area_m2)
        )

    # robust normalize each term
    slopeN = _robust_norm_01(slope_tan, land, 5, 95)
    twiN = _robust_norm_01(twi.astype("float32"), land, 5, 95)
    curvN = _robust_norm_01(np.abs(curv), land, 5, 95)

    # OPTIONAL stabilizer term: distance to channel (farther -> less susceptible)
    # Turn distance into a 0..1 "near-channel" score.
    distN = 0.0
    if dist_to_channel_m is not None:
        d = dist_to_channel_m.astype("float32")
        valid_d = land & np.isfinite(d) & (d >= 0)
        # normalize distance then invert so near-channel = high
        d_norm = _robust_norm_01(d, valid_d, 5, 95)
        distN = (1.0 - d_norm).astype("float32")

    # weighted blend (v1)
    # If dist term present, steal a little weight from twi/curv to keep total ~1
    if dist_to_channel_m is not None:
        lsi = (0.62 * slopeN + 0.20 * twiN + 0.08 * curvN + 0.10 * distN).astype(
            "float32"
        )
    else:
        lsi = (0.65 * slopeN + 0.25 * twiN + 0.10 * curvN).astype("float32")

    lsi = np.clip(lsi, 0.0, 1.0)
    lsi[~land] = np.nan
    return lsi


def compute_lsi_trigger(
    *,
    rain_mm: np.ndarray,
    dem: np.ndarray,
    event_duration_hours: float,
    dem_nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      intensity_mmhr: mm/hr (FOR RENDERING)
      trigger_01:     0..1  (FOR HAZARD MATH)
    """
    land = _land_mask_from_dem(dem, dem_nodata)

    dur_h = max(float(event_duration_hours), 1e-6)
    intensity_mmhr = (rain_mm.astype("float32") / dur_h).astype("float32")
    intensity_mmhr[~land] = np.nan

    # Map mm/hr -> 0..1 for hazard math.
    # IMPORTANT: These thresholds must match your operational reality.
    # If you routinely see long-duration events, keep t0 low so "big totals" still register.
    t0 = 0.5  # starts to matter
    t3 = 12.0  # very heavy / extreme in Timor context (tune: 10..25)

    trigger_01 = (np.clip(intensity_mmhr, t0, t3) - t0) / max(1e-6, (t3 - t0))
    trigger_01 = np.clip(trigger_01, 0.0, 1.0).astype("float32")

    # mild shaping so moderate-heavy shows up more
    trigger_01 = np.power(trigger_01, 0.85).astype("float32")

    return intensity_mmhr, trigger_01


# def compute_lsi_trigger(
#     *,
#     rain_mm: np.ndarray,
#     dem: np.ndarray,
#     event_duration_hours: float,
#     dem_nodata: float | None = None,
# ) -> np.ndarray:
#     land = _land_mask_from_dem(dem, dem_nodata)

#     dur_h = max(float(event_duration_hours), 1e-6)
#     intensity_mmhr = (rain_mm.astype("float32") / dur_h).astype("float32")

#     # keep only land; ocean/void -> NaN
#     intensity_mmhr[~land] = np.nan

#     # (optional) also remove non-positive rain
#     intensity_mmhr[intensity_mmhr <= 0] = np.nan

#     return intensity_mmhr


def compute_lsi_hazard(*, lsi_base: np.ndarray, lsi_trigger: np.ndarray) -> np.ndarray:
    # lsi_base assumed 0..1 (or close)
    base = lsi_base.astype("float32", copy=False)
    trig = lsi_trigger.astype("float32", copy=False)

    mask = np.isfinite(base) & np.isfinite(trig) & (trig > 0)

    if not np.any(mask):
        out = np.full_like(base, np.nan, dtype="float32")
        return out

    # robust normalize trigger intensity (mm/hr) to 0..1
    v = trig[mask]
    p5 = float(np.percentile(v, 5))
    p95 = float(np.percentile(v, 95))
    if p95 <= p5:
        p95 = p5 + 1e-6

    t = np.clip((trig - p5) / (p95 - p5), 0.0, 1.0)

    # shape it so mid/high rain pops more
    gamma = 0.75  # <1 boosts midrange
    t = np.power(t, gamma).astype("float32")

    hazard = base * t
    hazard[~np.isfinite(hazard)] = np.nan
    return hazard.astype("float32")


# def compute_lsi_trigger(
#     *,
#     rain_mm: np.ndarray,
#     dem: np.ndarray,
#     event_duration_hours: float,
#     twi: np.ndarray | None = None,
#     dist_to_channel_m: np.ndarray | None = None,
#     dem_nodata: float | None = None,
# ) -> np.ndarray:
#     land = _land_mask_from_dem(dem, dem_nodata)

#     dur_h = max(float(event_duration_hours), 1e-6)
#     intensity = rain_mm.astype("float32") / dur_h  # mm/hr

#     # Only consider where rain exists (prevents normalization collapsing to zeros)
#     rain_mask = land & np.isfinite(intensity) & (intensity > 0)

#     # 1) Rain intensity forcing (relative 0..1 inside rainy footprint)
#     I = _robust_norm_01(intensity, rain_mask, 5, 95)

#     # 2) Duration / saturation factor (longer events push trigger up)
#     # 0..1 where ~0 at 0h and ~1 around 12h (tune)
#     D = np.clip(np.log1p(dur_h) / np.log1p(12.0), 0.0, 1.0).astype("float32")

#     # 3) Convergence/wetness predisposition (optional but recommended)
#     if twi is not None:
#         twi_mask = land & np.isfinite(twi)
#         T = _robust_norm_01(twi.astype("float32"), twi_mask, 5, 95)
#     else:
#         T = np.zeros_like(I, dtype="float32")

#     # 4) Near-channel boost (optional: makes “flood-like activation” visible)
#     if dist_to_channel_m is not None:
#         d = dist_to_channel_m.astype("float32")
#         dmask = land & np.isfinite(d) & (d >= 0)
#         # Map 0m->1, 2000m->0 (tune)
#         near = np.zeros_like(I, dtype="float32")
#         near[dmask] = np.clip(1.0 - (d[dmask] / 2000.0), 0.0, 1.0)
#     else:
#         near = np.zeros_like(I, dtype="float32")

#     # Blend: rain dominates, but TWI + near-channel decide where it “activates”
#     trigger = (0.70 * I + 0.20 * T + 0.10 * near).astype("float32")

#     # Apply duration as a global multiplier (keeps spatial pattern from above)
#     trigger *= 0.60 + 0.40 * D  # never goes to zero if raining

#     # Make sure non-rain stays transparent
#     trigger[~rain_mask] = 0.0
#     trigger[~land] = np.nan
#     return np.clip(trigger, 0.0, 1.0)


# def compute_lsi_trigger(
#     *,
#     rain_mm: np.ndarray,
#     dem: np.ndarray,
#     event_duration_hours: float,
#     dem_nodata: float | None = None,
# ) -> np.ndarray:
#     """
#     Rain trigger field in [0..1].

#     Uses intensity (mm/hr) but does NOT rely on a rain sensor in-station.
#     This is purely from your interpolated grid.

#     v1: robust normalize intensity under rain footprint on land,
#         then apply a mild nonlinearity to emphasize heavy bursts.
#     """
#     land = _land_mask_from_dem(dem, dem_nodata)

#     dur_h = max(float(event_duration_hours), 1e-6)
#     intensity = (rain_mm.astype("float32") / dur_h).astype("float32")  # mm/hr

#     # only where rain exists
#     mask = land & np.isfinite(intensity) & (intensity > 0)

#     norm = _robust_norm_01(intensity, mask, 5, 95)

#     # shape it: make high intensity pop without saturating immediately
#     # (gamma < 1 makes highs stronger; >1 makes highs weaker)
#     gamma = 0.8
#     trigger = np.power(norm, gamma).astype("float32")

#     trigger[~land] = np.nan
#     return trigger


# def compute_lsi_hazard(
#     *,
#     lsi_base: np.ndarray,
#     lsi_trigger: np.ndarray,
# ) -> np.ndarray:
#     """
#     Combined hazard: predisposition × trigger.
#     """
#     hazard = (lsi_base.astype("float32") * lsi_trigger.astype("float32")).astype(
#         "float32"
#     )
#     return np.clip(hazard, 0.0, 1.0)
