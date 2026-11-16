import numpy as np

from .colormaps_shared import new_rgba


# ---------------------------------------------------------
# DEM (Sentinel-2 style: greens → browns → grays → white)
# ---------------------------------------------------------
def colorize_dem_rgba(dem: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    DEM colorization:
      - Ocean (<= 0 or nodata): fully transparent
      - 0–300m: lush green
      - 300–900m: yellow-brown
      - 900–2000m: rocky gray
      - >2000m: snow white
    """
    dem = dem.astype("float32")
    h, w = dem.shape
    rgba = new_rgba(h, w)

    if nodata is None:
        nodata_mask = ~np.isfinite(dem)
    else:
        nodata_mask = (dem == nodata) | ~np.isfinite(dem)

    ocean_mask = (dem <= 0) | nodata_mask
    land_mask = ~ocean_mask

    if not np.any(land_mask):
        return rgba  # fully transparent

    z = np.clip(dem, 0, 4000)

    low = land_mask & (z < 300)
    mid = land_mask & (z >= 300) & (z < 900)
    high = land_mask & (z >= 900) & (z < 2000)
    top = land_mask & (z >= 2000)

    # Base alpha for land
    rgba[..., 3][land_mask] = 255

    # 0–300m: lush green
    rgba[..., 0][low] = 50  # R
    rgba[..., 1][low] = 140  # G
    rgba[..., 2][low] = 40  # B

    # 300–900m: yellow-brown
    rgba[..., 0][mid] = 170
    rgba[..., 1][mid] = 140
    rgba[..., 2][mid] = 75

    # 900–2000m: rocky gray
    rgba[..., 0][high] = 160
    rgba[..., 1][high] = 160
    rgba[..., 2][high] = 160

    # >2000m: snow white
    rgba[..., 0][top] = 245
    rgba[..., 1][top] = 245
    rgba[..., 2][top] = 245

    # ocean = fully transparent (already 0 alpha)
    return rgba
