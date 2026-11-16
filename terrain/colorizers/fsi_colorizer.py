# terrain/fsi_colorizer.py
from __future__ import annotations

import numpy as np


def colorize_fsi_rgba(
    fsi: np.ndarray,
    fsi_nodata: float | None,
    dem: np.ndarray,
    dem_nodata: float | None,
    rain_mm: np.ndarray,
    rain_nodata: float | None,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    FSI Layer (Option B1, full-contrast grayscale):

      • Oceans → fully transparent
      • Land (no rain) → full-contrast grayscale FSI (using global vmin/vmax)
      • Rain-affected land → hazard colors (R → Y → G) based on FSI
      • Rain footprint with no valid FSI → transparent (no solid circles)
      • No alpha tricks on land: land pixels are fully opaque; only oceans
        and non-FSI rain areas are transparent.
    """

    fsi = fsi.astype("float32")
    dem = dem.astype("float32")
    rain_mm = rain_mm.astype("float32")

    h, w = fsi.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")

    # ---------------------------
    # MASKS
    # ---------------------------

    # Land mask from DEM (DEM > 0 and valid)
    if dem_nodata is None:
        land_mask = np.isfinite(dem) & (dem > 0)
    else:
        land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

    # Ocean is everything not land
    ocean_mask = ~land_mask

    # Valid FSI on land
    if fsi_nodata is None:
        f_valid = np.isfinite(fsi) & land_mask
    else:
        f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

    # Rain footprint (regardless of FSI)
    if rain_nodata is None:
        r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
    else:
        r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

    # ---------------------------
    # GLOBAL NORMALIZATION (Option B1)
    # ---------------------------
    # vmin / vmax come from _get_fsi_vmin_vmax(job_id), computed over:
    #   FSI valid + land + rain footprint (or fallback to all land-FSI)

    # Safety
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.zeros_like(fsi, dtype="float32")
    norm[f_valid] = np.clip((fsi[f_valid] - vmin) / (vmax - vmin), 0.0, 1.0)

    # ---------------------------
    # 1. OCEANS → transparent
    # ---------------------------
    rgba[ocean_mask] = [0, 0, 0, 0]

    # ---------------------------
    # 2. GRAYSCALE LAND OUTSIDE RAIN
    # ---------------------------
    land_no_rain = land_mask & (~r_valid) & f_valid
    if np.any(land_no_rain):
        gray = (norm * 255.0).astype("uint8")

        rgba[..., 0][land_no_rain] = gray[land_no_rain]
        rgba[..., 1][land_no_rain] = gray[land_no_rain]
        rgba[..., 2][land_no_rain] = gray[land_no_rain]
        rgba[..., 3][land_no_rain] = 255  # fully opaque

    # ---------------------------
    # 3. TRANSPARENT RAIN FOOTPRINT WITH NO FSI
    # ---------------------------
    # This ensures rain “circles” have no solid fill where we don't
    # have valid FSI underneath.
    circle_bg = r_valid & (~f_valid)
    if np.any(circle_bg):
        rgba[circle_bg] = [0, 0, 0, 0]

    # ---------------------------
    # 4. HAZARD COLORS (FSI + RAIN)
    # ---------------------------
    hazard = f_valid & r_valid
    if np.any(hazard):
        hv = norm[hazard]

        # High FSI → red; medium → yellow; low → green
        # (R,Y,G are all derived from hv)
        r = (255.0 * hv).astype("uint8")  # 0 → 255
        g = (255.0 * (1.0 - hv)).astype("uint8")  # 255 → 0
        b = np.zeros_like(r, dtype="uint8")

        rgba[..., 0][hazard] = r
        rgba[..., 1][hazard] = g
        rgba[..., 2][hazard] = b
        rgba[..., 3][hazard] = 255  # fully opaque hazard

    return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     fsi_vmin: float,
#     fsi_vmax: float,
# ) -> np.ndarray:
#     """
#     FSI visualization (Option B1 FINAL):

#     • Oceans → fully transparent
#     • Land without rain → grayscale = globally normalized FSI
#     • Land with rain → hazard colors (global norm)
#     • Rain circle background → transparent
#     • No opacity layers
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land mask from DEM
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     ocean_mask = ~land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # Valid FSI on land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # ---------------------------
#     # GLOBAL NORMALIZATION (B1)
#     # ---------------------------
#     norm = np.zeros_like(fsi, dtype="float32")
#     if fsi_vmax <= fsi_vmin:
#         fsi_vmax = fsi_vmin + 1e-6

#     norm[f_valid] = np.clip((fsi[f_valid] - fsi_vmin) / (fsi_vmax - fsi_vmin), 0, 1)

#     # ---------------------------
#     # 1. OCEANS → transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GRAYSCALE LAND (no rain)
#     # ---------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid

#     if np.any(land_no_rain):
#         g = (norm * 255).astype("uint8")

#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = 255

#     # ---------------------------
#     # 3. TRANSPARENT rain-circle background
#     # ---------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (rain + FSI)
#     # ---------------------------
#     hazard = f_valid & r_valid
#     if np.any(hazard):
#         hv = norm[hazard]

#         # Smooth red → yellow → green
#         r = (255 * hv).astype("uint8")
#         g = (255 * (1 - hv)).astype("uint8")
#         b = np.zeros_like(r)

#         rgba[..., 0][hazard] = r
#         rgba[..., 1][hazard] = g
#         rgba[..., 2][hazard] = b
#         rgba[..., 3][hazard] = 255

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
# ) -> np.ndarray:
#     """
#     FSI Layer (final, Option A + B1):

#       • Oceans → fully transparent
#       • Land with NO rain → grayscale FSI (high contrast, global per tile)
#       • Land WITH rain   → hazard colors (green → yellow → red) using
#                            percentiles ONLY from FSI under rain (B1)
#       • Rain-circle background (where rain exists but FSI invalid) → transparent
#       • No semi-transparent darkening of land; only oceans / circle BG are alpha=0.
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land / ocean from DEM
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     ocean_mask = ~land_mask

#     # FSI valid on land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # Where FSI + rain both exist → hazard
#     hazard = f_valid & r_valid

#     # Land with FSI but no rain → grayscale context
#     land_no_rain = f_valid & (~r_valid)

#     # Pixels inside rain footprint but without valid FSI → transparent circle background
#     circle_bg = r_valid & (~f_valid)

#     # ---------------------------
#     # 1. OCEANS → transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GLOBAL (per tile) GRAYSCALE FOR LAND (NO RAIN)
#     # ---------------------------
#     # Use ALL valid FSI on land in this tile to define grayscale contrast.
#     if np.any(f_valid):
#         f_vals = fsi[f_valid]
#         gmin = float(np.percentile(f_vals, 5))
#         gmax = float(np.percentile(f_vals, 95))
#         if gmax <= gmin:
#             gmax = gmin + 1e-6

#         norm_global = np.zeros_like(fsi, dtype="float32")
#         norm_global[f_valid] = np.clip(
#             (fsi[f_valid] - gmin) / (gmax - gmin),
#             0.0,
#             1.0,
#         )
#     else:
#         norm_global = np.zeros_like(fsi, dtype="float32")

#     if np.any(land_no_rain):
#         g = (norm_global * 255.0).astype("uint8")

#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = 255  # fully opaque grayscale

#     # ---------------------------
#     # 3. TRANSPARENT RAIN-CIRCLE BACKGROUND
#     # ---------------------------
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (FSI + rain, B1 scaling)
#     # ---------------------------
#     if np.any(hazard):
#         # B1: vmin/vmax from FSI values UNDER rain only (within this tile)
#         vals = fsi[hazard]
#         hmin = float(np.percentile(vals, 5))
#         hmax = float(np.percentile(vals, 95))
#         if hmax <= hmin:
#             hmax = hmin + 1e-6

#         norm_hazard = np.zeros_like(fsi, dtype="float32")
#         norm_hazard[hazard] = np.clip(
#             (fsi[hazard] - hmin) / (hmax - hmin),
#             0.0,
#             1.0,
#         )

#         hv = norm_hazard[hazard]

#         # Green → Yellow → Red:
#         #   hv = 0   → green (0,255,0)
#         #   hv = 0.5 → yellow (255,255,0)
#         #   hv = 1   → red (255,0,0)
#         r = np.zeros_like(hv, dtype="float32")
#         g = np.zeros_like(hv, dtype="float32")
#         b = np.zeros_like(hv, dtype="float32")

#         # 0–0.5: green → yellow
#         m1 = hv <= 0.5
#         t1 = hv[m1] / 0.5
#         r[m1] = 255.0 * t1  # 0 → 255
#         g[m1] = 255.0  # stays 255
#         b[m1] = 0.0

#         # 0.5–1.0: yellow → red
#         m2 = hv > 0.5
#         t2 = (hv[m2] - 0.5) / 0.5
#         r[m2] = 255.0  # stays 255
#         g[m2] = 255.0 * (1.0 - t2)  # 255 → 0
#         b[m2] = 0.0

#         rgba[..., 0][hazard] = r.astype("uint8")
#         rgba[..., 1][hazard] = g.astype("uint8")
#         rgba[..., 2][hazard] = b.astype("uint8")
#         rgba[..., 3][hazard] = 255  # fully opaque hazard

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     vmin: float,
#     vmax: float,
# ) -> np.ndarray:
#     """
#     Final FSI visualization (Option B1):

#     • Oceans → fully transparent
#     • All land with valid FSI → grayscale (using global vmin/vmax)
#     • Land + rain → hazard colors overriding grayscale
#     • Rain-circle background → transparent only where FSI is invalid
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------------------
#     # MASKS
#     # ---------------------------------------

#     # Land mask from DEM
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     ocean_mask = ~land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # FSI valid + land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # ---------------------------------------
#     # NORMALIZATION USING JOB-GLOBAL RANGE
#     # ---------------------------------------
#     norm = np.zeros_like(fsi, dtype="float32")
#     if vmax > vmin:
#         norm[f_valid] = np.clip((fsi[f_valid] - vmin) / (vmax - vmin), 0, 1)

#     # ---------------------------------------
#     # 1. OCEANS → fully transparent
#     # ---------------------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------------------
#     # 2. LAND (no rain) → grayscale
#     # ---------------------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid
#     if np.any(land_no_rain):
#         g = (norm * 255).astype("uint8")
#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = 255

#     # ---------------------------------------
#     # 3. RAIN CIRCLE BACKGROUND → transparent
#     #    BUT ONLY where FSI is invalid
#     # ---------------------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------------------
#     # 4. HAZARD COLORS for land + rain + FSI
#     # ---------------------------------------
#     hazard = f_valid & r_valid
#     if np.any(hazard):
#         hv = norm[hazard]

#         # red → yellow → green
#         r = (255 * hv).astype("uint8")
#         g = (255 * (1 - hv)).astype("uint8")
#         b = np.zeros_like(r, dtype="uint8")

#         rgba[..., 0][hazard] = r
#         rgba[..., 1][hazard] = g
#         rgba[..., 2][hazard] = b
#         rgba[..., 3][hazard] = 255

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     vmin: float,
#     vmax: float,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     FSI Layer (B1, final):

#       • Oceans → fully transparent
#       • Land (no rain) → grayscale FSI (using global vmin/vmax)
#       • Rain-affected land → hazard colors (red→yellow→green)
#       • Transparent rain-circle background where FSI is invalid
#     """
#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land mask from DEM (DEM > 0 and valid)
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     # Ocean is simply not land (or invalid DEM)
#     ocean_mask = ~land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # FSI valid + land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # ---------------------------
#     # NORMALIZATION (global B1)
#     # ---------------------------
#     # vmin / vmax are computed at job-level (B1: FSI under rain on land)
#     if vmax <= vmin:
#         vmax = vmin + 1e-6

#     norm = np.zeros_like(fsi, dtype="float32")
#     norm[f_valid] = np.clip((fsi[f_valid] - vmin) / (vmax - vmin), 0.0, 1.0)

#     # ---------------------------
#     # 1. OCEANS → transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GRAYSCALE LAND (no rain)
#     # ---------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid
#     if np.any(land_no_rain):
#         g = (norm * 255.0).astype("uint8")

#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = background_alpha  # subtle context

#     # ---------------------------
#     # 3. TRANSPARENT RAIN-CIRCLE BACKGROUND (no FSI)
#     # ---------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (FSI + rain)
#     # ---------------------------
#     hazard = f_valid & r_valid
#     if np.any(hazard):
#         hv = norm[hazard]  # 0–1

#         # High FSI = red, medium ≈ yellow, low ≈ green
#         # We invert to get: hv=1 (high)→red, hv=0 (low)→green
#         r = (255.0 * hv).astype("uint8")
#         g = (255.0 * (1.0 - hv)).astype("uint8")
#         b = np.zeros_like(r, dtype="uint8")

#         rgba[..., 0][hazard] = r
#         rgba[..., 1][hazard] = g
#         rgba[..., 2][hazard] = b
#         rgba[..., 3][hazard] = 255  # fully opaque in hazard zones

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     fsi_vmin: float,
#     fsi_vmax: float,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     FSI Layer (B1, grayscale + hazard):

#       • Oceans → fully transparent
#       • Land with valid FSI (no matter rain) → grayscale FSI background
#       • Land with rain + FSI → hazard colors (red→yellow→green)
#       • Pixels inside rain footprint with NO valid FSI → fully transparent
#       • Adjacent tiles keep full grayscale texture.
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land mask from DEM (DEM > 0 and valid)
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     # Ocean = not land
#     ocean_mask = ~land_mask

#     # FSI valid (only on land)
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # Rain footprint (anywhere)
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # ---------------------------
#     # NORMALISE FSI USING B1
#     # ---------------------------
#     norm = np.zeros_like(fsi, dtype="float32")

#     if np.isfinite(fsi_vmin) and np.isfinite(fsi_vmax) and (fsi_vmax > fsi_vmin):
#         norm[f_valid] = np.clip(
#             (fsi[f_valid] - fsi_vmin) / (fsi_vmax - fsi_vmin),
#             0.0,
#             1.0,
#         )
#     else:
#         # Fallback: avoid NaNs if vmin/vmax missing or degenerate
#         if np.any(f_valid):
#             vals = fsi[f_valid]
#             vmin = float(np.percentile(vals, 5))
#             vmax = float(np.percentile(vals, 95))
#             vmax = max(vmax, vmin + 1e-6)
#             norm[f_valid] = np.clip((fsi[f_valid] - vmin) / (vmax - vmin), 0.0, 1.0)

#     # ---------------------------
#     # 1. OCEANS → TRANSPARENT
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. BASE GRAYSCALE FSI (ALL LAND WITH VALID FSI)
#     # ---------------------------
#     base = f_valid  # land + valid FSI, regardless of rain
#     if np.any(base):
#         g = (norm * 255).astype("uint8")

#         rgba[..., 0][base] = g[base]
#         rgba[..., 1][base] = g[base]
#         rgba[..., 2][base] = g[base]
#         rgba[..., 3][base] = background_alpha  # subtle grayscale context

#     # ---------------------------
#     # 3. TRANSPARENT RAIN-CIRCLE BACKGROUND (NO FSI)
#     # ---------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (FSI + RAIN)
#     # ---------------------------
#     hazard = f_valid & r_valid

#     if np.any(hazard):
#         hv = norm[hazard]

#         # Red → Yellow → Green gradient (high → medium → low hazard)
#         #   hv ~ 1.0 → red
#         #   hv ~ 0.5 → yellow
#         #   hv ~ 0.0 → green
#         r_chan = (255 * hv).astype("uint8")
#         g_chan = (255 * (1.0 - hv)).astype("uint8")
#         b_chan = np.zeros_like(r_chan, dtype="uint8")

#         rgba[..., 0][hazard] = r_chan
#         rgba[..., 1][hazard] = g_chan
#         rgba[..., 2][hazard] = b_chan
#         rgba[..., 3][hazard] = 255  # fully opaque hazard

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     fsi_vmin: float,
#     fsi_vmax: float,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     FINAL FSI implementation (Option B1):

#       • Oceans → fully transparent
#       • Land (no rain) → grayscale FSI using GLOBAL vmin/vmax
#       • Rain-affected land → hazard colors (yellow→orange→red→purple)
#       • Rain-circle areas without FSI → transparent
#       • Adjacent tiles remain consistent because vmin/vmax are global.
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land (DEM > 0 and valid)
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     ocean_mask = ~land_mask

#     # FSI valid & land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # ---------------------------
#     # NORMALIZATION (GLOBAL)
#     # ---------------------------
#     fsi_vmin = float(fsi_vmin)
#     fsi_vmax = float(fsi_vmax)
#     spread = max(fsi_vmax - fsi_vmin, 1e-6)

#     norm = np.zeros_like(fsi, dtype="float32")
#     norm[f_valid] = np.clip((fsi[f_valid] - fsi_vmin) / spread, 0, 1)

#     # ---------------------------
#     # 1. OCEANS → fully transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GRAYSCALE FOR LAND WITHOUT RAIN
#     # ---------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid
#     if np.any(land_no_rain):
#         g = (norm * 255).astype("uint8")
#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = background_alpha

#     # ---------------------------
#     # 3. TRANSPARENT RAIN CIRCLE BACKGROUND
#     #    Rain but no valid FSI → transparent
#     # ---------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS FOR FSI × RAIN
#     # ---------------------------
#     hazard = f_valid & r_valid
#     if np.any(hazard):
#         hv = norm[hazard]

#         rgba[..., 0][hazard] = (255 * hv).astype("uint8")  # red
#         rgba[..., 1][hazard] = (255 * (1 - hv)).astype("uint8")  # green→yellow
#         rgba[..., 2][hazard] = 0  # blue = 0
#         rgba[..., 3][hazard] = 255  # opaque

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     vmin: float,
#     vmax: float,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     FSI Layer (Option B1, rain-zone normalization):

#       • Oceans → fully transparent
#       • Land with NO rain → grayscale FSI (context)
#       • Land WITH rain → hazard colors (red ↔ green) using GLOBAL vmin/vmax
#         derived ONLY from FSI inside ALL rain footprints for the job.
#       • Rain-circle background (no FSI) → fully transparent (no rings).

#     Inputs:
#       - fsi:          FSI tile (float32)
#       - fsi_nodata:   nodata for FSI (or None)
#       - dem:          DEM tile (float32)
#       - dem_nodata:   nodata for DEM (or None)
#       - rain_mm:      rain tile in mm (float32)
#       - rain_nodata:  nodata for rain (or None)
#       - vmin/vmax:    GLOBAL job-level bounds (already computed & stored)
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land mask from DEM (DEM > 0 and valid)
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     # Ocean is just "not land"
#     ocean_mask = ~land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # FSI valid + land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # ---------------------------
#     # Normalized FSI (0–1) using GLOBAL vmin/vmax
#     # ---------------------------
#     if vmax <= vmin:
#         vmax = vmin + 1e-6

#     norm = np.zeros_like(fsi, dtype="float32")
#     norm[f_valid] = np.clip((fsi[f_valid] - vmin) / (vmax - vmin), 0.0, 1.0)

#     # ---------------------------
#     # 1. OCEANS → transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GRAYSCALE LAND (no rain)
#     # ---------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid
#     if np.any(land_no_rain):
#         g = (norm * 255).astype("uint8")

#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = background_alpha  # contextual grayscale

#     # ---------------------------
#     # 3. TRANSPARENT RAIN-CIRCLE BACKGROUND (no FSI)
#     # ---------------------------
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (FSI + rain)
#     # ---------------------------
#     hazard = f_valid & r_valid

#     if np.any(hazard):
#         hv = norm[hazard]  # 0..1 using GLOBAL bounds

#         # Red → Yellow → Green gradient (high → medium → low hazard)
#         # hv=1 ⇒ full red; hv=0 ⇒ full green
#         r = (255 * hv).astype("uint8")
#         g = (255 * (1.0 - hv)).astype("uint8")
#         b = np.zeros_like(r, dtype="uint8")

#         rgba[..., 0][hazard] = r
#         rgba[..., 1][hazard] = g
#         rgba[..., 2][hazard] = b
#         rgba[..., 3][hazard] = 255  # fully opaque hazard

#     return rgba


# def colorize_fsi_rgba(
#     fsi: np.ndarray,
#     fsi_nodata: float | None,
#     dem: np.ndarray,
#     dem_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     fsi_vmin: float,
#     fsi_vmax: float,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     FSI Layer (global normalization):

#       • Oceans → fully transparent
#       • Land, no rain → grayscale FSI (global vmin/vmax)
#       • Rain-affected land → hazard colors (R→Y→G based on FSI)
#       • Rain-circle background stays transparent (no solid disks)
#     """

#     fsi = fsi.astype("float32")
#     dem = dem.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = fsi.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # ---------------------------
#     # MASKS
#     # ---------------------------

#     # Land mask from DEM (DEM > 0 and valid)
#     if dem_nodata is None:
#         land_mask = np.isfinite(dem) & (dem > 0)
#     else:
#         land_mask = (dem != dem_nodata) & np.isfinite(dem) & (dem > 0)

#     # Ocean is simply "not land"
#     ocean_mask = ~land_mask

#     # Rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # FSI valid + land
#     if fsi_nodata is None:
#         f_valid = np.isfinite(fsi) & land_mask
#     else:
#         f_valid = (fsi != fsi_nodata) & np.isfinite(fsi) & land_mask

#     # ---------------------------
#     # GLOBAL NORMALIZATION (per job)
#     # ---------------------------
#     # fsi_vmin / fsi_vmax come from _get_fsi_stats(job_id)
#     if fsi_vmax <= fsi_vmin:
#         fsi_vmax = fsi_vmin + 1e-6

#     norm = np.zeros_like(fsi, dtype="float32")
#     norm[f_valid] = np.clip((fsi[f_valid] - fsi_vmin) / (fsi_vmax - fsi_vmin), 0, 1)

#     # ---------------------------
#     # 1. OCEANS → transparent
#     # ---------------------------
#     rgba[ocean_mask] = [0, 0, 0, 0]

#     # ---------------------------
#     # 2. GRAYSCALE LAND (no rain)
#     # ---------------------------
#     land_no_rain = land_mask & (~r_valid) & f_valid
#     if np.any(land_no_rain):
#         g = (norm * 255).astype("uint8")

#         rgba[..., 0][land_no_rain] = g[land_no_rain]
#         rgba[..., 1][land_no_rain] = g[land_no_rain]
#         rgba[..., 2][land_no_rain] = g[land_no_rain]
#         rgba[..., 3][land_no_rain] = background_alpha  # subtle grayscale context

#     # ---------------------------
#     # 3. TRANSPARENT RAIN-CIRCLE BACKGROUND
#     # ---------------------------
#     # Pixels inside rain footprint but with NO valid FSI → transparent circle bg
#     circle_bg = r_valid & (~f_valid)
#     rgba[circle_bg] = [0, 0, 0, 0]

#     # ---------------------------
#     # 4. HAZARD COLORS (FSI + rain)
#     # ---------------------------
#     hazard = f_valid & r_valid

#     if np.any(hazard):
#         hv = norm[hazard]  # 0..1 on valid FSI land

#         # Red → Yellow → Green:
#         # hv = 0   → green   (low FSI)
#         # hv = 0.5 → yellow  (medium)
#         # hv = 1   → red     (high)
#         r = (255 * hv).astype("uint8")
#         g = (255 * (1.0 - hv)).astype("uint8")
#         b = np.zeros_like(r, dtype="uint8")

#         rgba[..., 0][hazard] = r
#         rgba[..., 1][hazard] = g
#         rgba[..., 2][hazard] = b
#         rgba[..., 3][hazard] = 255  # fully opaque for hazard

#     return rgba
