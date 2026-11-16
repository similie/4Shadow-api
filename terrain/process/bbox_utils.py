from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


def bbox_from_points_with_buffer_km(points: Iterable[LatLon], buffer_km: float) -> BBox:
    pts = list(points)
    if not pts:
        raise ValueError("No points provided")

    lats = [p.lat for p in pts]
    lons = [p.lon for p in pts]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Convert buffer km to degrees
    lat0 = 0.5 * (min_lat + max_lat)
    buf_lat = buffer_km / 111.0
    buf_lon = buffer_km / max(1e-6, (111.0 * math.cos(math.radians(lat0))))

    return (
        min_lon - buf_lon,
        min_lat - buf_lat,
        max_lon + buf_lon,
        max_lat + buf_lat,
    )
