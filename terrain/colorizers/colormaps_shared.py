# terrain/colormaps_shared.py
from typing import Tuple

import numpy as np


def rainfall_mask(rain_tile, rain_nodata):
    """
    Returns a boolean mask: TRUE where rain exists (rain > 0)
    Used only for discharge transparency carving.
    NEVER touches rainfall colors.
    """
    if rain_nodata is None:
        return (rain_tile > 0) & np.isfinite(rain_tile)
    return (rain_tile != rain_nodata) & (rain_tile > 0)


def mask_valid(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return boolean mask of valid pixels."""
    if nodata is None:
        return np.isfinite(data)
    return (data != nodata) & np.isfinite(data)


def normalize(data: np.ndarray, mask: np.ndarray, vmin: float, vmax: float):
    """Normalize data -> 0..1 only where mask=True."""
    norm = np.zeros_like(data, dtype="float32")
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm[mask] = np.clip((data[mask] - vmin) / (vmax - vmin), 0.0, 1.0)
    return norm


def new_rgba(h: int, w: int) -> np.ndarray:
    """Create empty RGBA tile."""
    return np.zeros((h, w, 4), dtype="uint8")


def wmo_rain_color(rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized WMO-ish intensity colors for rain rate [mm/hr].

    Buckets:
      <0.25    very light blue
      <2.5     blue
      <7.6     cyan/green
      <25      yellow
      <50      orange
      <100     red
      >=100    magenta
    """
    r = np.zeros_like(rate, dtype="uint8")
    g = np.zeros_like(rate, dtype="uint8")
    b = np.zeros_like(rate, dtype="uint8")

    cond1 = rate < 0.25
    cond2 = (rate >= 0.25) & (rate < 2.5)
    cond3 = (rate >= 2.5) & (rate < 7.6)
    cond4 = (rate >= 7.6) & (rate < 25.0)
    cond5 = (rate >= 25.0) & (rate < 50.0)
    cond6 = (rate >= 50.0) & (rate < 100.0)
    cond7 = rate >= 100.0

    # Very light
    r[cond1] = 204
    g[cond1] = 229
    b[cond1] = 255

    # Light blue
    r[cond2] = 0
    g[cond2] = 128
    b[cond2] = 255

    # Moderate (cyan/green)
    r[cond3] = 0
    g[cond3] = 255
    b[cond3] = 255

    # Heavy (yellow)
    r[cond4] = 255
    g[cond4] = 255
    b[cond4] = 0

    # Very heavy (orange)
    r[cond5] = 255
    g[cond5] = 165
    b[cond5] = 0

    # Intense (red)
    r[cond6] = 255
    g[cond6] = 0
    b[cond6] = 0

    # Violent (magenta)
    r[cond7] = 255
    g[cond7] = 0
    b[cond7] = 255

    return r, g, b
