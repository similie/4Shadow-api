# terrain/colormaps_discharge.py
from __future__ import annotations

import numpy as np


def colorize_discharge_rgba(
    discharge,
    discharge_nodata,
    rain_mm,
    rain_nodata,
    dem,
    dem_nodata,
    background_alpha=166,
    river_percentile=80.0,
):
    discharge = discharge.astype("float32")
    rain_mm = rain_mm.astype("float32")
    dem = dem.astype("float32")

    h, w = discharge.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")

    # -----------------------------
    # 1. Base validity masks
    # -----------------------------
    if discharge_nodata is None:
        d_positive = (discharge > 0) & np.isfinite(discharge)
    else:
        d_positive = (
            (discharge != discharge_nodata) & np.isfinite(discharge) & (discharge > 0)
        )

    if rain_nodata is None:
        r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
    else:
        r_valid = (rain_mm != rain_nodata) & np.isfinite(rain_mm) & (rain_mm > 0)

    # -----------------------------
    # 2. Ocean mask (fixes your issue)
    # -----------------------------
    if dem_nodata is None:
        ocean = (~np.isfinite(dem)) | (dem <= 0)
    else:
        ocean = (dem == dem_nodata) | (~np.isfinite(dem)) | (dem <= 0)

    # Discharge cannot exist in ocean
    d_positive = d_positive & (~ocean)

    # -----------------------------
    # 3. No discharge at all case
    # -----------------------------
    if not np.any(d_positive):
        bg = ~r_valid
        rgba[..., 3][bg] = background_alpha
        return rgba

    # -----------------------------
    # 4. River pixels = top X%
    # -----------------------------
    q = discharge[d_positive]
    logq = np.log10(q + 1e-6)

    try:
        thr = np.percentile(logq, river_percentile)
    except Exception:
        thr = logq.min()

    logq_full = np.full_like(discharge, -np.inf, dtype="float32")
    logq_full[d_positive] = logq

    rivers = d_positive & (logq_full >= thr)
    if not np.any(rivers):
        rivers = d_positive

    # Remove from rivers if in ocean (redundant but safe)
    rivers = rivers & (~ocean)

    # -----------------------------
    # 5. Backgrounds
    # -----------------------------
    bg = (~rivers) & (~r_valid)
    rgba[..., 0][bg] = 0
    rgba[..., 1][bg] = 0
    rgba[..., 2][bg] = 0
    rgba[..., 3][bg] = background_alpha

    # Rain circle non-river remains transparent (default)

    # -----------------------------
    # 6. Color the rivers
    # -----------------------------
    q_rivers = discharge[rivers]
    logq_rivers = np.log10(q_rivers + 1e-6)

    qmin = float(np.percentile(logq_rivers, 5))
    qmax = float(np.percentile(logq_rivers, 95))
    if qmax <= qmin:
        qmax = qmin + 1e-6

    norm = np.clip((logq_rivers - qmin) / (qmax - qmin), 0, 1)

    r = np.zeros_like(norm)
    g = np.zeros_like(norm)
    b = np.zeros_like(norm)

    m1 = norm <= 0.5
    t1 = norm[m1] * 2
    r[m1] = 26 * (1 - t1)
    g[m1] = 79 + 176 * t1
    b[m1] = 255

    m2 = norm > 0.5
    t2 = (norm[m2] - 0.5) * 2
    r[m2] = 255 * t2
    g[m2] = 255
    b[m2] = 255

    rgba[..., 0][rivers] = r.astype("uint8")
    rgba[..., 1][rivers] = g.astype("uint8")
    rgba[..., 2][rivers] = b.astype("uint8")
    rgba[..., 3][rivers] = 255

    return rgba


# def colorize_discharge_rgba(
#     discharge,
#     discharge_nodata,
#     rain_mm,
#     rain_nodata,
#     background_alpha=166,
#     min_discharge=1e-4,  # NEW
# ):
#     discharge = discharge.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = discharge.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # --- masks ---
#     if discharge_nodata is None:
#         d_valid_raw = (discharge > 0) & np.isfinite(discharge)
#     else:
#         d_valid_raw = (
#             (discharge != discharge_nodata) & np.isfinite(discharge) & (discharge > 0)
#         )

#     # NEW: remove tiny fake “discharge haze”
#     d_valid = d_valid_raw & (discharge >= min_discharge)

#     # rain circles
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & (rain_mm > 0) & np.isfinite(rain_mm)

#     # --- 1. Dark background ---
#     bg = (~d_valid) & (~r_valid)
#     rgba[..., 3][bg] = background_alpha

#     # --- 2. Transparent rainfall circles ---
#     circle = r_valid & (~d_valid)
#     rgba[circle] = [0, 0, 0, 0]

#     # --- 3. Discharge features ---
#     if np.any(d_valid):
#         q = discharge[d_valid]
#         logq = np.log10(q + 1e-6)
#         qmin = np.percentile(logq, 5)
#         qmax = np.percentile(logq, 95)
#         qmax = max(qmax, qmin + 1e-6)

#         norm = np.clip((logq - qmin) / (qmax - qmin), 0, 1)
#         r = np.zeros_like(norm)
#         g = np.zeros_like(norm)
#         b = np.zeros_like(norm)

#         # 0–0.5 blue → cyan
#         m1 = norm <= 0.5
#         t1 = norm[m1] * 2
#         r[m1] = 26 * (1 - t1)
#         g[m1] = 79 + 176 * t1
#         b[m1] = 255

#         # 0.5–1.0 cyan → white
#         m2 = norm > 0.5
#         t2 = (norm[m2] - 0.5) * 2
#         r[m2] = 255 * t2
#         g[m2] = 255
#         b[m2] = 255

#         rgba[..., 0][d_valid] = r.astype("uint8")
#         rgba[..., 1][d_valid] = g.astype("uint8")
#         rgba[..., 2][d_valid] = b.astype("uint8")
#         rgba[..., 3][d_valid] = 255  # opaque

#     return rgba


# def colorize_discharge_rgba(
#     discharge: np.ndarray,
#     discharge_nodata: float | None,
#     rain_mm: np.ndarray,
#     rain_nodata: float | None,
#     background_alpha: int = 166,
# ) -> np.ndarray:
#     """
#     Discharge layer rendering:

#       • Darkened world background (0,0,0,background_alpha)
#       • Rain footprint (rain > 0) is fully transparent *unless* there is
#         a strong discharge feature.
#       • Strong discharge features (top X% of values) are blue → cyan → white.
#       • No filled blue circles.

#     This is purely a visualization threshold; the underlying discharge raster
#     is unchanged.
#     """
#     discharge = discharge.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = discharge.shape
#     rgba = new_rgba(h, w)

#     # --------------------------
#     # Masks
#     # --------------------------
#     # valid discharge (raw > 0, not nodata)
#     d_raw = mask_valid(discharge, discharge_nodata) & (discharge > 0.0)

#     # rain footprint
#     if rain_nodata is None:
#         r_valid = (rain_mm > 0.0) & np.isfinite(rain_mm)
#     else:
#         r_valid = (rain_mm != rain_nodata) & np.isfinite(rain_mm) & (rain_mm > 0.0)

#     # --------------------------
#     # Threshold discharge so only real channels light up
#     # --------------------------
#     if np.any(d_raw):
#         q = discharge[d_raw]
#         logq = np.log10(q + 1e-6)

#         # Keep only the upper 30% (tuneable) as "features"
#         CHANNEL_QUANTILE = 70.0  # percent; higher = sparser network
#         thr = np.percentile(logq, CHANNEL_QUANTILE)

#         strong = np.log10(discharge + 1e-6) >= thr
#         d_valid = d_raw & strong
#     else:
#         d_valid = np.zeros_like(discharge, dtype=bool)

#     # --------------------------
#     # 1. Darkened background everywhere with no strong discharge
#     # --------------------------
#     # Background = not a strong discharge feature
#     bg_mask = ~d_valid
#     rgba[..., 0][bg_mask] = 0
#     rgba[..., 1][bg_mask] = 0
#     rgba[..., 2][bg_mask] = 0
#     rgba[..., 3][bg_mask] = np.uint8(background_alpha)

#     # --------------------------
#     # 2. Transparent rainfall circles where still no strong discharge
#     # --------------------------
#     # Circle area with no strong discharge should be fully transparent
#     circle_only = r_valid & (~d_valid)
#     rgba[circle_only] = np.array([0, 0, 0, 0], dtype="uint8")

#     # --------------------------
#     # 3. Color strong discharge features
#     # --------------------------
#     if np.any(d_valid):
#         q_feat = discharge[d_valid]
#         logq_feat = np.log10(q_feat + 1e-6)

#         qmin = float(np.percentile(logq_feat, 5))
#         qmax = float(np.percentile(logq_feat, 95))
#         if qmax <= qmin:
#             qmax = qmin + 1e-6

#         norm = np.clip((logq_feat - qmin) / (qmax - qmin), 0.0, 1.0)

#         r = np.zeros_like(norm, dtype="float32")
#         g = np.zeros_like(norm, dtype="float32")
#         b = np.zeros_like(norm, dtype="float32")

#         # 0–0.5: deep blue → cyan
#         m1 = norm <= 0.5
#         t1 = norm[m1] * 2.0
#         r[m1] = 26.0 * (1.0 - t1)  # 26 → 0
#         g[m1] = 79.0 + (176.0 * t1)  # 79 → 255
#         b[m1] = 255.0  # stays 255

#         # 0.5–1.0: cyan → white
#         m2 = norm > 0.5
#         t2 = (norm[m2] - 0.5) * 2.0
#         r[m2] = 255.0 * t2  # 0 → 255
#         g[m2] = 255.0  # stays 255
#         b[m2] = 255.0  # stays 255

#         # Write into RGBA where d_valid
#         rgba_r = rgba[..., 0]
#         rgba_g = rgba[..., 1]
#         rgba_b = rgba[..., 2]
#         rgba_a = rgba[..., 3]

#         rgba_r[d_valid] = r.astype("uint8")
#         rgba_g[d_valid] = g.astype("uint8")
#         rgba_b[d_valid] = b.astype("uint8")
#         rgba_a[d_valid] = 255  # fully opaque for features

#     return rgba


# def colorize_discharge_rgba(
#     discharge, discharge_nodata, rain_mm, rain_nodata, background_alpha=166
# ):
#     discharge = discharge.astype("float32")
#     rain_mm = rain_mm.astype("float32")

#     h, w = discharge.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     # --- masks ---
#     d_valid = (
#         (discharge > 0)
#         & np.isfinite(discharge)
#         & ((discharge_nodata is None) or (discharge != discharge_nodata))
#     )

#     r_valid = (
#         (rain_mm > 0)
#         & np.isfinite(rain_mm)
#         & ((rain_nodata is None) or (rain_mm != rain_nodata))
#     )

#     # --- 1. Dark background ---
#     bg = (~d_valid) & (~r_valid)
#     rgba[..., 3][bg] = background_alpha

#     # --- 2. Transparent circles ---
#     circle = r_valid & (~d_valid)
#     rgba[circle] = [0, 0, 0, 0]

#     # --- 3. Discharge features ---
#     if np.any(d_valid):
#         q = discharge[d_valid]
#         logq = np.log10(q + 1e-6)
#         qmin = np.percentile(logq, 5)
#         qmax = np.percentile(logq, 95)
#         qmax = max(qmax, qmin + 1e-6)

#         norm = np.clip((logq - qmin) / (qmax - qmin), 0, 1)

#         r = np.zeros_like(norm)
#         g = np.zeros_like(norm)
#         b = np.zeros_like(norm)

#         m1 = norm <= 0.5
#         t1 = norm[m1] * 2
#         r[m1] = 26 * (1 - t1)
#         g[m1] = 79 + 176 * t1
#         b[m1] = 255

#         m2 = norm > 0.5
#         t2 = (norm[m2] - 0.5) * 2
#         r[m2] = 255 * t2
#         g[m2] = 255
#         b[m2] = 255

#         rgba[..., 0][d_valid] = r.astype("uint8")
#         rgba[..., 1][d_valid] = g.astype("uint8")
#         rgba[..., 2][d_valid] = b.astype("uint8")
#         rgba[..., 3][d_valid] = 255

#     return rgba
