# terrain/runoff.py

from __future__ import annotations

import math
import numpy as np


EARTH_RADIUS_M = 6371000.0


def cell_area_m2_at_lat(lat_deg: float, cellsize_deg: float) -> float:
    """
    Approximate area of a lat/lon grid cell at a given latitude.

    For small regions (like Timor-Leste), using a single representative latitude
    is sufficient. For global work, you'd build a row-wise area grid.
    """
    lat_rad = math.radians(lat_deg)
    dlat = math.radians(cellsize_deg)
    dlon = math.radians(cellsize_deg)

    dy = EARTH_RADIUS_M * dlat
    dx = EARTH_RADIUS_M * math.cos(lat_rad) * dlon

    return dx * dy


def drainage_area_from_accum(
    accum: np.ndarray,
    cell_area_m2: float,
    include_self: bool = True,
) -> np.ndarray:
    """
    Convert flow accumulation (upstream cell count) into drainage area (m²).
    """
    accum = np.asarray(accum, dtype="float32")
    valid = np.isfinite(accum)

    base = accum.copy()
    if include_self:
        base = base + 1.0

    area = np.full_like(base, np.nan, dtype="float32")
    area[valid] = base[valid] * cell_area_m2
    return area


def runoff_depth_from_rain(
    rain_mm: np.ndarray,
    runoff_coeff: float | np.ndarray,
) -> np.ndarray:
    """
    Convert rainfall [mm] and runoff coefficient C (0..1) into
    runoff depth [m]. Supports scalar or grid C.
    """
    rain_mm = np.asarray(rain_mm, dtype="float32")
    C = np.asarray(runoff_coeff, dtype="float32")

    rain_m = rain_mm / 1000.0
    return rain_m * C


def discharge_from_runoff(
    runoff_depth_m: np.ndarray,
    drainage_area_m2: np.ndarray,
    duration_seconds: float,
) -> np.ndarray:
    """
    Q_index = (runoff_depth * area) / duration_seconds

    Units: m³/s if inputs are in m, m², and seconds.
    """
    runoff_depth_m = np.asarray(runoff_depth_m, dtype="float32")
    drainage_area_m2 = np.asarray(drainage_area_m2, dtype="float32")

    valid = np.isfinite(runoff_depth_m) & np.isfinite(drainage_area_m2)
    q = np.full_like(runoff_depth_m, np.nan, dtype="float32")

    volume_m3 = runoff_depth_m[valid] * drainage_area_m2[valid]
    q[valid] = volume_m3 / float(duration_seconds)

    return q


# # terrain/runoff.py
# from __future__ import annotations

# import math
# import numpy as np

# EARTH_RADIUS_M = 6371000.0


# def cell_area_m2_at_lat(lat_deg: float, cellsize_deg: float) -> float:
#     """
#     Approximate area of a lat/lon grid cell at a given latitude.

#     Good approximation for regional-scale work (Timor-Leste).
#     """
#     lat_rad = math.radians(lat_deg)
#     dlat = math.radians(cellsize_deg)
#     dlon = math.radians(cellsize_deg)

#     dy = EARTH_RADIUS_M * dlat
#     dx = EARTH_RADIUS_M * math.cos(lat_rad) * dlon

#     return dx * dy


# def drainage_area_from_accum(
#     accum: np.ndarray,
#     cell_area_m2: float,
#     include_self: bool = True,
# ) -> np.ndarray:
#     """
#     Convert flow accumulation (upstream cell count) into drainage area (m²).
#     """

#     # Safe scalar/array conversion
#     accum = np.asarray(accum, dtype="float32")

#     valid = np.isfinite(accum)

#     if include_self:
#         base = accum + 1.0
#     else:
#         base = accum

#     area = np.full_like(base, np.nan, dtype="float32")
#     area[valid] = base[valid] * float(cell_area_m2)
#     return area


# def runoff_depth_from_rain(
#     rain_mm: np.ndarray,
#     runoff_coeff: float | np.ndarray,
# ) -> np.ndarray:
#     """
#     Convert rainfall [mm] and runoff coefficient C (0..1) into
#     runoff depth [m]. Supports scalar or grid C.
#     """

#     rain_mm = np.asarray(rain_mm, dtype="float32")
#     C = np.asarray(runoff_coeff, dtype="float32")

#     rain_m = rain_mm / 1000.0
#     return rain_m * C


# def discharge_from_runoff(
#     runoff_depth_m: np.ndarray,
#     drainage_area_m2: np.ndarray,
#     duration_seconds: float,
# ) -> np.ndarray:
#     """
#     Q_index = (runoff_depth * area) / duration_seconds

#     Units: m³/s if inputs are in m, m², and seconds.
#     """

#     runoff_depth_m = np.asarray(runoff_depth_m, dtype="float32")
#     drainage_area_m2 = np.asarray(drainage_area_m2, dtype="float32")

#     valid = np.isfinite(runoff_depth_m) & np.isfinite(drainage_area_m2)

#     q = np.full_like(runoff_depth_m, np.nan, dtype="float32")

#     volume_m3 = runoff_depth_m[valid] * drainage_area_m2[valid]
#     q[valid] = volume_m3 / float(duration_seconds)

#     return q
