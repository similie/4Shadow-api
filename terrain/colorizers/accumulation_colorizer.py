import numpy as np

from .colormaps_shared import new_rgba


# ---------------------------------------------------------
# Accumulation (transparent background, gradient features)
# ---------------------------------------------------------
def colorize_accum_rgba(accum: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    Accumulation tile:

      - Fully transparent background
      - Only accumulation features colored
      - Use log scale and a green→yellow→red gradient
      - Small upstream counts are treated as noise and ignored.
    """
    accum = accum.astype("float32")
    h, w = accum.shape
    rgba = new_rgba(h, w)

    if nodata is None:
        nodata_mask = ~np.isfinite(accum)
    else:
        nodata_mask = (accum == nodata) | ~np.isfinite(accum)

    # treat low accumulation as noise (e.g. < 10 cells)
    feature_mask = (~nodata_mask) & (accum > 10.0)
    if not np.any(feature_mask):
        return rgba

    acc = np.zeros_like(accum, dtype="float32")
    acc[feature_mask] = accum[feature_mask]

    log_acc = np.zeros_like(acc, dtype="float32")
    log_acc[feature_mask] = np.log10(acc[feature_mask] + 1.0)

    vals = log_acc[feature_mask]
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.zeros_like(log_acc, dtype="float32")
    norm[feature_mask] = (log_acc[feature_mask] - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    # Green → Yellow → Red
    r = np.zeros_like(norm, dtype="float32")
    g = np.zeros_like(norm, dtype="float32")
    b = np.zeros_like(norm, dtype="float32")

    # 0–0.5: green → yellow
    m1 = feature_mask & (norm <= 0.5)
    t1 = norm[m1] * 2.0  # 0..1
    r[m1] = t1 * 255.0
    g[m1] = 255.0
    b[m1] = 0.0

    # 0.5–1.0: yellow → red
    m2 = feature_mask & (norm > 0.5)
    t2 = (norm[m2] - 0.5) * 2.0  # 0..1
    r[m2] = 255.0
    g[m2] = (1.0 - t2) * 255.0
    b[m2] = 0.0

    rgba[..., 0][feature_mask] = r[feature_mask].astype("uint8")
    rgba[..., 1][feature_mask] = g[feature_mask].astype("uint8")
    rgba[..., 2][feature_mask] = b[feature_mask].astype("uint8")
    rgba[..., 3][feature_mask] = 255  # fully opaque features

    # background remains fully transparent
    return rgba
