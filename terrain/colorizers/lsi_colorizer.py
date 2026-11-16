from __future__ import annotations

import numpy as np


def _valid_mask(x: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(x)
    return (x != nodata) & np.isfinite(x)


def _rgba_empty(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype="uint8")


def _lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------
# Palettes
# ------------------------------------------------------------
def _palette_yellow_orange_red(
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    0 -> pale yellow
    0.6 -> orange
    1 -> deep red
    """
    c0 = np.array([255, 245, 200], dtype="float32")  # pale yellow
    c1 = np.array([255, 165, 60], dtype="float32")  # orange
    c2 = np.array([200, 30, 30], dtype="float32")  # deep red

    r = np.zeros_like(t, dtype="float32")
    g = np.zeros_like(t, dtype="float32")
    b = np.zeros_like(t, dtype="float32")

    m = t <= 0.6
    if np.any(m):
        tt = (t[m] / 0.6).astype("float32")
        rgb = _lerp(c0, c1, tt[:, None])
        r[m], g[m], b[m] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    m2 = ~m
    if np.any(m2):
        tt = ((t[m2] - 0.6) / 0.4).astype("float32")
        rgb = _lerp(c1, c2, tt[:, None])
        r[m2], g[m2], b[m2] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    return r, g, b


def _palette_teal_yellow_pink(
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Piecewise palette:
      0.0 -> deep teal
      0.5 -> warm yellow
      1.0 -> hot pink/red
    """
    # anchors (R,G,B)
    c0 = np.array([20, 60, 90], dtype="float32")  # teal
    c1 = np.array([255, 210, 90], dtype="float32")  # yellow
    c2 = np.array([255, 60, 140], dtype="float32")  # pink/red

    r = np.zeros_like(t, dtype="float32")
    g = np.zeros_like(t, dtype="float32")
    b = np.zeros_like(t, dtype="float32")

    m = t <= 0.5
    if np.any(m):
        tt = (t[m] / 0.5).astype("float32")
        rgb = _lerp(c0, c1, tt[:, None])
        r[m], g[m], b[m] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    m2 = ~m
    if np.any(m2):
        tt = ((t[m2] - 0.5) / 0.5).astype("float32")
        rgb = _lerp(c1, c2, tt[:, None])
        r[m2], g[m2], b[m2] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    return r, g, b


# ------------------------------------------------------------
# Base (static predisposition)
# ------------------------------------------------------------
def colorize_lsi_base_rgba(lsi, lsi_nodata, *, background_alpha: int = 210):
    h, w = lsi.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")

    valid = _valid_mask(lsi, lsi_nodata)
    if not np.any(valid):
        return rgba

    t = np.clip(lsi.astype("float32"), 0.0, 1.0)
    r, g, b = _palette_teal_yellow_pink(t[valid])

    rgba[..., 0][valid] = r.astype("uint8")
    rgba[..., 1][valid] = g.astype("uint8")
    rgba[..., 2][valid] = b.astype("uint8")
    rgba[..., 3][valid] = np.uint8(background_alpha)
    return rgba


# ------------------------------------------------------------
# Hazard (base × trigger)
# ------------------------------------------------------------
# def colorize_lsi_hazard_rgba(
#     hazard: np.ndarray,
#     hazard_nodata: float | None,
#     *,
#     show_from: float = 0.12,  # start showing hazard at ~12%
#     gamma: float = 0.6,  # <1 boosts subtle mids
#     alpha_min: int = 70,  # minimum visible once activated
#     alpha_max: int = 255,
# ) -> np.ndarray:
#     h, w = hazard.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(hazard, hazard_nodata)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(hazard.astype("float32"), 0.0, 1.0)

#     # palette
#     r, g, b = _palette_teal_yellow_pink(t[valid])
#     rgba[..., 0][valid] = r.astype("uint8")
#     rgba[..., 1][valid] = g.astype("uint8")
#     rgba[..., 2][valid] = b.astype("uint8")

#     # alpha ramp (thresholded)
#     x = (t - show_from) / max(1e-6, (1.0 - show_from))
#     x = np.clip(x, 0.0, 1.0)
#     x = np.power(x, gamma)

#     # boost low/mid values so it doesn’t disappear
#     a = np.power(t, 0.55)  # 0.5–0.7 range works well
#     alpha = (alpha_min + (alpha_max - alpha_min) * a).astype("float32")

#     # guarantee visibility for anything above a small threshold
#     on = valid & (t > 0.05)
#     alpha[on] = np.maximum(alpha[on], 80.0)


#     rgba[..., 3][valid] = np.clip(alpha[valid], 0, 255).astype("uint8")
#     return rgba
#
def colorize_lsi_hazard_rgba(
    hazard: np.ndarray,
    hazard_nodata: float | None,
    *,
    alpha_min: int = 0,
    alpha_max: int = 255,
    mode: str = "index",  # "index" or "mmhr"
    show_from: float = 0.0,  # values below this become transparent
    gamma: float = 0.65,  # <1 boosts midrange
    # only used when mode="mmhr"
    t0: float = 0.5,  # start showing (mm/hr)
    t3: float = 12.0,  # saturate (mm/hr)
) -> np.ndarray:
    h, w = hazard.shape
    rgba = _rgba_empty(h, w)

    valid = _valid_mask(hazard, hazard_nodata)
    if not np.any(valid):
        return rgba

    v = hazard.astype("float32")

    if mode == "mmhr":
        # map mm/hr -> 0..1
        t = (np.clip(v, t0, t3) - t0) / max(1e-6, (t3 - t0))
        t = np.clip(t, 0.0, 1.0)
    else:
        # already 0..1
        t = np.clip(v, 0.0, 1.0)

    # Visibility shaping
    t = np.power(t, gamma).astype("float32")

    r, g, b = _palette_teal_yellow_pink(t[valid])
    rgba[..., 0][valid] = r.astype("uint8")
    rgba[..., 1][valid] = g.astype("uint8")
    rgba[..., 2][valid] = b.astype("uint8")

    # alpha scaling + cutoff
    a = alpha_min + (alpha_max - alpha_min) * t
    if show_from > 0:
        a = np.where(v >= show_from, a, 0.0)

    rgba[..., 3][valid] = np.clip(a[valid], 0, 255).astype("uint8")
    return rgba


# def colorize_lsi_hazard_rgba(
#     hazard: np.ndarray,
#     hazard_nodata: float | None,
#     *,
#     alpha_min: int = 0,
#     alpha_max: int = 255,
#     # NEW (optional, backwards compatible):
#     mode: str = "index",  # "index" (0..1) or "mmhr"
#     t0: float = 2.0,  # for mmhr: show from
#     t1: float = 5.0,  # for mmhr: low/moderate
#     t2: float = 10.0,  # for mmhr: high
#     t3: float = 25.0,  # for mmhr: extreme/saturate
#     show_from: float = 0.03,  # for index: don’t tint everything
#     gamma: float = 0.65,  # boosts midrange when < 1
# ) -> np.ndarray:
#     h, w = hazard.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(hazard, hazard_nodata)
#     if not np.any(valid):
#         return rgba

#     x = hazard.astype("float32")

#     if mode == "mmhr":
#         x = hazard.astype("float32")

#         tt = np.zeros_like(x, dtype="float32")

#         # piecewise normalization
#         m0 = (x >= t0) & (x < t1)
#         m1 = (x >= t1) & (x < t2)
#         m2 = (x >= t2) & (x < t3)
#         m3 = x >= t3

#         # 0 → 0.33
#         tt[m0] = (x[m0] - t0) / max(1e-6, (t1 - t0)) * 0.33

#         # 0.33 → 0.66
#         tt[m1] = 0.33 + (x[m1] - t1) / max(1e-6, (t2 - t1)) * 0.33

#         # 0.66 → 1.0
#         tt[m2] = 0.66 + (x[m2] - t2) / max(1e-6, (t3 - t2)) * 0.34

#         # saturate
#         tt[m3] = 1.0

#         tt = np.clip(tt, 0.0, 1.0)

#         # color ramp (same palette you already use)
#         r, g, b = _palette_teal_yellow_pink(tt[valid])
#         rgba[..., 0][valid] = r.astype("uint8")
#         rgba[..., 1][valid] = g.astype("uint8")
#         rgba[..., 2][valid] = b.astype("uint8")

#         # alpha ramps hard with intensity
#         a = np.power(tt, gamma)
#         alpha = alpha_min + (alpha_max - alpha_min) * a
#         rgba[..., 3][valid] = np.clip(alpha[valid], 0, 255).astype("uint8")

#         return rgba

#     # Default: mode == "index" (0..1)
#     t = np.clip(x, 0.0, 1.0)

#     # Don’t tint everything: only show above show_from
#     vis = valid & (t >= float(show_from))
#     if not np.any(vis):
#         return rgba

#     r, g, b = _palette_teal_yellow_pink(t[vis])
#     rgba[..., 0][vis] = r.astype("uint8")
#     rgba[..., 1][vis] = g.astype("uint8")
#     rgba[..., 2][vis] = b.astype("uint8")

#     # Alpha proportional to hazard, boosted
#     tt = (t - float(show_from)) / max(1e-6, (1.0 - float(show_from)))
#     tt = np.clip(tt, 0.0, 1.0)
#     a = np.power(tt, float(gamma))
#     alpha = (alpha_min + (alpha_max - alpha_min) * a).astype("float32")
#     rgba[..., 3][vis] = np.clip(alpha[vis], 0, 255).astype("uint8")
#     return rgba


# def colorize_lsi_hazard_rgba(
#     hazard01: np.ndarray,
#     nodata: float | None,
#     *,
#     show_from: float = 0.03,  # show hazard starting at 3%
#     gamma: float = 0.55,  # boosts mids
#     alpha_min: int = 90,  # once “on”, it is visible
#     alpha_max: int = 255,
# ) -> np.ndarray:
#     h, w = hazard01.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(hazard01, nodata)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(hazard01.astype("float32"), 0.0, 1.0)

#     # gate very low hazard so we don’t tint the whole world
#     vis = valid & (t >= float(show_from))
#     if not np.any(vis):
#         return rgba

#     r, g, b = _palette_yellow_orange_red(t[vis])
#     rgba[..., 0][vis] = r.astype("uint8")
#     rgba[..., 1][vis] = g.astype("uint8")
#     rgba[..., 2][vis] = b.astype("uint8")

#     # alpha ramp
#     tt = (t - float(show_from)) / max(1e-6, (1.0 - float(show_from)))
#     tt = np.clip(tt, 0.0, 1.0)
#     tt = np.power(tt, float(gamma))

#     alpha = (alpha_min + (alpha_max - alpha_min) * tt).astype("float32")
#     rgba[..., 3][vis] = np.clip(alpha[vis], 0, 255).astype("uint8")
#     return rgba


# def colorize_lsi_hazard_rgba(
#     hazard: np.ndarray,
#     hazard_nodata: float | None,
#     *,
#     alpha_min: int = 0,
#     alpha_max: int = 255,
#     vis_threshold: float = 0.10,  # <— key: hide “noise”, emphasize real hazard
#     gamma: float = 0.65,  # <— boosts midrange
# ) -> np.ndarray:
#     h, w = hazard.shape
#     rgba = np.zeros((h, w, 4), dtype="uint8")

#     valid = _valid_mask(hazard, hazard_nodata) & (hazard > vis_threshold)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(hazard.astype("float32"), 0.0, 1.0)
#     t = np.power(t, gamma)

#     # hot palette (yellow->orange->red->magenta)
#     c0 = np.array([255, 230, 0], dtype="float32")
#     c1 = np.array([255, 120, 0], dtype="float32")
#     c2 = np.array([255, 40, 40], dtype="float32")
#     c3 = np.array([200, 0, 255], dtype="float32")

#     r = np.zeros_like(t, dtype="float32")
#     g = np.zeros_like(t, dtype="float32")
#     b = np.zeros_like(t, dtype="float32")

#     m1 = t <= 0.33
#     m2 = (t > 0.33) & (t <= 0.66)
#     m3 = t > 0.66

#     if np.any(m1):
#         tt = t[m1] / 0.33
#         rgb = c0 + (c1 - c0) * tt[:, None]
#         r[m1], g[m1], b[m1] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     if np.any(m2):
#         tt = (t[m2] - 0.33) / 0.33
#         rgb = c1 + (c2 - c1) * tt[:, None]
#         r[m2], g[m2], b[m2] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     if np.any(m3):
#         tt = (t[m3] - 0.66) / 0.34
#         rgb = c2 + (c3 - c2) * tt[:, None]
#         r[m3], g[m3], b[m3] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     rgba[..., 0][valid] = r[valid].astype("uint8")
#     rgba[..., 1][valid] = g[valid].astype("uint8")
#     rgba[..., 2][valid] = b[valid].astype("uint8")

#     # alpha: quadratic + threshold makes ridges “snap”
#     a = np.clip((t - vis_threshold) / max(1e-6, (1.0 - vis_threshold)), 0.0, 1.0)
#     a = a * a
#     alpha = alpha_min + (alpha_max - alpha_min) * a
#     rgba[..., 3][valid] = np.clip(alpha[valid], 0, 255).astype("uint8")

#     return rgba


# def colorize_lsi_hazard_rgba(
#     hazard: np.ndarray,
#     hazard_nodata: float | None,
#     *,
#     # These are the knobs that make hotspots pop.
#     show_from: float = 0.30,  # hard threshold: below this => transparent
#     alpha_min: int = 0,
#     alpha_max: int = 255,
#     gamma: float = 2.2,  # strong emphasis of high hazard
# ) -> np.ndarray:
#     h, w = hazard.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(hazard, hazard_nodata)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(hazard.astype("float32"), 0.0, 1.0)

#     # hard gate low hazard to avoid “everything slightly tinted”
#     vis = valid & (t >= float(show_from))
#     if not np.any(vis):
#         return rgba

#     r, g, b = _palette_yellow_orange_red(t[vis])

#     rgba[..., 0][vis] = r.astype("uint8")
#     rgba[..., 1][vis] = g.astype("uint8")
#     rgba[..., 2][vis] = b.astype("uint8")

#     # alpha ramps only over [show_from..1]
#     tt = np.clip((t - float(show_from)) / max(1e-6, (1.0 - float(show_from))), 0.0, 1.0)
#     a = np.power(tt, float(gamma))
#     alpha = (alpha_min + (alpha_max - alpha_min) * a).astype("float32")
#     rgba[..., 3][vis] = np.clip(alpha[vis], 0, 255).astype("uint8")

#     return rgba


# ------------------------------------------------------------
# Trigger (rain intensity forcing)
# ------------------------------------------------------------
# def colorize_lsi_trigger_rgba(
#     intensity_mmhr: np.ndarray,
#     intensity_nodata: float | None,
#     *,
#     # Absolute thresholds (mm/hr) so drizzle doesn’t make a “red circle”
#     t0: float = 5.0,  # below -> transparent
#     t1: float = 15.0,
#     t2: float = 30.0,
#     t3: float = 60.0,  # above -> saturated
#     alpha_max: int = 230,
# ) -> np.ndarray:
#     """
#     Visual meaning:
#       transparent: < t0
#       yellow:      t0..t1
#       orange:      t1..t2
#       red:         t2..t3+
#     """
#     h, w = intensity_mmhr.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(intensity_mmhr, intensity_nodata)
#     if not np.any(valid):
#         return rgba

#     x = intensity_mmhr.astype("float32", copy=False)
#     vis = valid & (x >= float(t0)) & np.isfinite(x)
#     if not np.any(vis):
#         return rgba

#     # Normalize to 0..1 using absolute scale
#     # clamp at t3 so huge values don’t blow the ramp
#     t = (np.clip(x, t0, t3) - float(t0)) / max(1e-6, (float(t3) - float(t0)))
#     t = np.clip(t, 0.0, 1.0)

#     r, g, b = _palette_yellow_orange_red(t[vis])
#     rgba[..., 0][vis] = r.astype("uint8")
#     rgba[..., 1][vis] = g.astype("uint8")
#     rgba[..., 2][vis] = b.astype("uint8")

#     # Alpha: keep lows faint, highs strong
#     a = np.power(t, 1.6)
#     rgba[..., 3][vis] = np.clip(a[vis] * float(alpha_max), 0, 255).astype("uint8")

#     return rgba


def colorize_lsi_trigger_rgba(
    intensity_mmhr: np.ndarray,
    nodata: float | None,
    *,
    # mm/hr thresholds (tweak for your region)
    steps=(0.2, 1, 3, 7, 15, 30, 60),
    # 8 colors = 7 steps + 1
    colors=(
        (0, 0, 0),  # (unused for < first; we keep transparent anyway)
        (120, 200, 255),  # light blue  (drizzle)
        (80, 170, 255),  # blue
        (0, 220, 160),  # green-cyan
        (255, 230, 0),  # yellow
        (255, 140, 0),  # orange
        (255, 60, 60),  # red
        (200, 0, 255),  # magenta (extreme)
    ),
    alpha_min: int = 0,
    alpha_max: int = 255,
) -> np.ndarray:
    h, w = intensity_mmhr.shape
    rgba = _rgba_empty(h, w)

    vmask = _valid_mask(intensity_mmhr, nodata) & (intensity_mmhr > 0)
    if not np.any(vmask):
        return rgba

    v = intensity_mmhr.astype("float32", copy=False)

    # bin indices 1..len(colors)-1
    bins = np.zeros_like(v, dtype=np.int32)
    for i, thr in enumerate(steps, start=1):
        bins += (v >= float(thr)).astype(np.int32)

    bins = np.clip(bins, 1, len(colors) - 1)

    # assign RGB by bin
    rgb = np.array(colors, dtype=np.uint8)
    rgba[..., 0][vmask] = rgb[bins[vmask], 0]
    rgba[..., 1][vmask] = rgb[bins[vmask], 1]
    rgba[..., 2][vmask] = rgb[bins[vmask], 2]

    # alpha: scale smoothly with intensity (log helps “big storms” show structure)
    vv = v.copy()
    vv[~vmask] = 0
    a = np.log1p(vv) / np.log1p(max(float(steps[-1]), 1.0))
    a = np.clip(a, 0.0, 1.0)
    alpha = alpha_min + (alpha_max - alpha_min) * a
    rgba[..., 3][vmask] = np.clip(alpha[vmask], 0, 255).astype(np.uint8)

    return rgba


# # terrain/colorizers/lsi_colorizer.py
# from __future__ import annotations

# import numpy as np


# def _valid_mask(x: np.ndarray, nodata: float | None) -> np.ndarray:
#     if nodata is None:
#         return np.isfinite(x)
#     return (x != nodata) & np.isfinite(x)


# def _rgba_empty(h: int, w: int) -> np.ndarray:
#     return np.zeros((h, w, 4), dtype="uint8")


# def _lerp(a, b, t):
#     return a + (b - a) * t


# def _palette_teal_yellow_pink(
#     t: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Piecewise palette:
#       0.0 -> deep teal
#       0.5 -> warm yellow
#       1.0 -> hot pink/red
#     """
#     # anchors (R,G,B)
#     c0 = np.array([20, 60, 90], dtype="float32")  # teal
#     c1 = np.array([255, 210, 90], dtype="float32")  # yellow
#     c2 = np.array([255, 60, 140], dtype="float32")  # pink/red

#     r = np.zeros_like(t, dtype="float32")
#     g = np.zeros_like(t, dtype="float32")
#     b = np.zeros_like(t, dtype="float32")

#     m = t <= 0.5
#     if np.any(m):
#         tt = (t[m] / 0.5).astype("float32")
#         rgb = _lerp(c0, c1, tt[:, None])
#         r[m], g[m], b[m] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     m2 = ~m
#     if np.any(m2):
#         tt = ((t[m2] - 0.5) / 0.5).astype("float32")
#         rgb = _lerp(c1, c2, tt[:, None])
#         r[m2], g[m2], b[m2] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     return r, g, b


# def colorize_lsi_base_rgba(
#     lsi: np.ndarray,
#     lsi_nodata: float | None,
#     *,
#     background_alpha: int = 190,
# ) -> np.ndarray:
#     """
#     Base susceptibility:
#       - visible over all land (alpha ~ constant)
#       - palette distinct from FSI
#     """
#     h, w = lsi.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(lsi, lsi_nodata)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(lsi.astype("float32"), 0.0, 1.0)
#     r, g, b = _palette_teal_yellow_pink(t[valid])

#     rgba[..., 0][valid] = r.astype("uint8")
#     rgba[..., 1][valid] = g.astype("uint8")
#     rgba[..., 2][valid] = b.astype("uint8")
#     rgba[..., 3][valid] = np.uint8(background_alpha)

#     return rgba


# def colorize_lsi_hazard_rgba(
#     hazard: np.ndarray,
#     hazard_nodata: float | None,
#     *,
#     alpha_min: int = 0,
#     alpha_max: int = 255,
# ) -> np.ndarray:
#     """
#     Hazard:
#       - alpha scales with hazard itself (so only the activated zones pop)
#       - same palette, but visually “hotter” where it matters
#     """
#     h, w = hazard.shape
#     rgba = _rgba_empty(h, w)

#     valid = _valid_mask(hazard, hazard_nodata)
#     if not np.any(valid):
#         return rgba

#     t = np.clip(hazard.astype("float32"), 0.0, 1.0)
#     r, g, b = _palette_teal_yellow_pink(t[valid])

#     rgba[..., 0][valid] = r.astype("uint8")
#     rgba[..., 1][valid] = g.astype("uint8")
#     rgba[..., 2][valid] = b.astype("uint8")

#     # alpha proportional to hazard, with optional curve
#     a = t * t  # emphasize extremes
#     alpha = (alpha_min + (alpha_max - alpha_min) * a).astype("float32")
#     rgba[..., 3][valid] = np.clip(alpha[valid], 0, 255).astype("uint8")

#     return rgba


# # def colorize_lsi_rgba(
# #     *,
# #     lsi: np.ndarray,
# #     lsi_nodata: float | None,
# #     dem: np.ndarray | None = None,
# #     dem_nodata: float | None = None,
# #     background_alpha: int = 0,
# # ) -> np.ndarray:
# #     """
# #     LSI colorizer:
# #       - Transparent where invalid/ocean
# #       - Green → Yellow → Orange → Red as susceptibility rises
# #     """
# #     lsi = lsi.astype("float32", copy=False)
# #     h, w = lsi.shape
# #     rgba = np.zeros((h, w, 4), dtype="uint8")

# #     # valid mask from LSI
# #     if lsi_nodata is None:
# #         valid = np.isfinite(lsi)
# #     else:
# #         valid = np.isfinite(lsi) & (lsi != lsi_nodata)

# #     # optional land mask from DEM
# #     if dem is not None:
# #         dem = dem.astype("float32", copy=False)
# #         if dem_nodata is None:
# #             land = np.isfinite(dem) & (dem > 0)
# #         else:
# #             land = np.isfinite(dem) & (dem != dem_nodata) & (dem > 0)
# #         valid = valid & land

# #     if not np.any(valid):
# #         return rgba

# #     v = np.clip(lsi, 0.0, 1.0)

# #     # Simple 4-stop ramp:
# #     # 0.00 -> green
# #     # 0.40 -> yellow
# #     # 0.70 -> orange
# #     # 1.00 -> red
# #     r = np.zeros((h, w), dtype="float32")
# #     g = np.zeros((h, w), dtype="float32")
# #     b = np.zeros((h, w), dtype="float32")

# #     # segment 1: green -> yellow
# #     m1 = valid & (v <= 0.40)
# #     if np.any(m1):
# #         t = v[m1] / 0.40
# #         r[m1] = 40.0 * (1 - t) + 255.0 * t
# #         g[m1] = 200.0 * (1 - t) + 230.0 * t
# #         b[m1] = 60.0 * (1 - t) + 60.0 * t

# #     # segment 2: yellow -> orange
# #     m2 = valid & (v > 0.40) & (v <= 0.70)
# #     if np.any(m2):
# #         t = (v[m2] - 0.40) / 0.30
# #         r[m2] = 255.0
# #         g[m2] = 230.0 * (1 - t) + 140.0 * t
# #         b[m2] = 60.0 * (1 - t) + 40.0 * t

# #     # segment 3: orange -> red
# #     m3 = valid & (v > 0.70)
# #     if np.any(m3):
# #         t = (v[m3] - 0.70) / 0.30
# #         t = np.clip(t, 0.0, 1.0)
# #         r[m3] = 255.0
# #         g[m3] = 140.0 * (1 - t) + 40.0 * t
# #         b[m3] = 40.0 * (1 - t) + 40.0 * t

# #     rgba[..., 0] = r.astype("uint8")
# #     rgba[..., 1] = g.astype("uint8")
# #     rgba[..., 2] = b.astype("uint8")

# #     # alpha: opaque on valid; optional background alpha elsewhere
# #     rgba[..., 3][valid] = 220
# #     if background_alpha > 0:
# #         rgba[..., 3][~valid] = background_alpha

# #     return rgba
