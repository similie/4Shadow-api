from __future__ import annotations

import numpy as np


def _valid_mask(x: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(x)
    return np.isfinite(x) & (x != nodata)


def _rgba_empty(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype="uint8")


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    # a,b: (3,), t: (N,)
    return a[None, :] + (b - a)[None, :] * t[:, None]


def _linear_ramp_rgb(t: np.ndarray, c0: tuple[int, int, int], c1: tuple[int, int, int]):
    c0a = np.array(c0, dtype="float32")
    c1a = np.array(c1, dtype="float32")
    rgb = _lerp(c0a, c1a, t.astype("float32"))
    return rgb[:, 0], rgb[:, 1], rgb[:, 2]


def _robust_norm(
    x: np.ndarray, valid: np.ndarray, p_lo: float, p_hi: float
) -> np.ndarray:
    """
    Normalize x to [0,1] using percentiles over valid pixels.
    Falls back safely if stats are degenerate.
    """
    out = np.zeros_like(x, dtype="float32")
    if not np.any(valid):
        return out

    vals = x[valid].astype("float32")
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    out = (x.astype("float32") - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------
# SLOPE (degrees)
# --------------------------------------------------------------------
def colorize_slope_rgba(
    slope_deg: np.ndarray,
    slope_nodata: float | None,
    *,
    alpha: int = 210,
) -> np.ndarray:
    """
    Slope degrees: emphasize steep areas.
    Uses robust percentiles and a warm ramp: light -> dark red.
    """
    h, w = slope_deg.shape
    rgba = _rgba_empty(h, w)

    valid = _valid_mask(slope_deg, slope_nodata) & (slope_deg >= 0)
    if not np.any(valid):
        return rgba

    # robust normalize (typical slopes are skewed; 1..99 works well)
    t = _robust_norm(slope_deg, valid, 1, 99)
    t = np.power(t, 0.7)  # boost mid-range so hills show up

    # warm ramp (high contrast over basemaps)
    r, g, b = _linear_ramp_rgb(t[valid], (255, 236, 210), (180, 30, 30))

    rgba[..., 0][valid] = r.astype("uint8")
    rgba[..., 1][valid] = g.astype("uint8")
    rgba[..., 2][valid] = b.astype("uint8")
    rgba[..., 3][valid] = np.uint8(alpha)
    return rgba


# def colorize_slope_rgba(
#     slope_deg: np.ndarray,
#     slope_nodata: float | None,
#     *,
#     vmin: float = 0.0,
#     vmax: float = 60.0,  # matches legend
#     gamma: float = 0.9,  # <1 boosts midrange, >1 suppresses
#     alpha_min: int = 40,  # keep low slopes subtle
#     alpha_max: int = 230,  # steep slopes pop
# ) -> np.ndarray:
#     h, w = slope_deg.shape
#     rgba = _rgba_empty(h, w)

#     valid = (
#         _valid_mask(slope_deg, slope_nodata) & np.isfinite(slope_deg) & (slope_deg >= 0)
#     )
#     if not np.any(valid):
#         return rgba

#     s = slope_deg.astype("float32", copy=False)

#     # Safety: if slope is actually radians (max around 1.57), convert to degrees
#     # (This prevents “everything looks flat/washed” if the upstream layer is radians.)
#     vmax_sample = float(np.nanmax(s[valid]))
#     if vmax_sample <= 3.5:
#         s = s * (180.0 / np.pi)

#     t = (np.clip(s, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
#     t = np.power(t, gamma)

#     # Palette: light -> dark red (high contrast)
#     r, g, b = _linear_ramp_rgb(t[valid], (255, 245, 240), (165, 15, 21))

#     rgba[..., 0][valid] = r.astype("uint8")
#     rgba[..., 1][valid] = g.astype("uint8")
#     rgba[..., 2][valid] = b.astype("uint8")

#     # alpha scaled by slope (so steep areas stand out)
#     a = (alpha_min + (alpha_max - alpha_min) * t).astype("float32")
#     rgba[..., 3][valid] = np.clip(a[valid], 0, 255).astype("uint8")
#     return rgba


# --------------------------------------------------------------------
# TWI (Topographic Wetness Index)
# --------------------------------------------------------------------
def colorize_twi_rgba(
    twi: np.ndarray,
    twi_nodata: float | None,
    *,
    alpha: int = 210,
) -> np.ndarray:
    """
    TWI: highlight wetter/accumulation-prone zones.
    Uses cool ramp: pale -> deep blue.
    """
    h, w = twi.shape
    rgba = _rgba_empty(h, w)

    valid = _valid_mask(twi, twi_nodata)
    if not np.any(valid):
        return rgba

    # TWI can have weird tails; 5..95 is usually stable
    t = _robust_norm(twi, valid, 5, 95)
    t = np.power(t, 0.8)  # keep some subtlety, still readable

    r, g, b = _linear_ramp_rgb(t[valid], (230, 245, 255), (0, 90, 160))

    rgba[..., 0][valid] = r.astype("uint8")
    rgba[..., 1][valid] = g.astype("uint8")
    rgba[..., 2][valid] = b.astype("uint8")
    rgba[..., 3][valid] = np.uint8(alpha)
    return rgba


# --------------------------------------------------------------------
# DISTANCE TO CHANNEL (meters)
# --------------------------------------------------------------------
def colorize_dist_to_channel_rgba(
    dist_m: np.ndarray,
    dist_nodata: float | None,
    *,
    alpha: int = 220,
    max_m: float = 3000.0,
) -> np.ndarray:
    """
    Distance to channel: we want near-channel areas to pop.
    We invert the ramp: near=strong, far=transparent-ish.
    """
    h, w = dist_m.shape
    rgba = _rgba_empty(h, w)

    valid = _valid_mask(dist_m, dist_nodata) & (dist_m >= 0)
    if not np.any(valid):
        return rgba

    d = dist_m.astype("float32")
    d = np.clip(d, 0.0, float(max_m))

    # invert so 0m => 1, max => 0
    t = 1.0 - (d / float(max_m))
    t = np.power(t, 0.6)  # boost midrange (so 0-1000m shows well)

    # green-cyan ramp for “near waterways”
    r, g, b = _linear_ramp_rgb(t[valid], (220, 255, 235), (0, 150, 90))

    rgba[..., 0][valid] = r.astype("uint8")
    rgba[..., 1][valid] = g.astype("uint8")
    rgba[..., 2][valid] = b.astype("uint8")

    # alpha follows “closeness”
    a = (alpha * t).astype("float32")
    rgba[..., 3][valid] = np.clip(a[valid], 0, 255).astype("uint8")
    return rgba
