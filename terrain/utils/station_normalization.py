# terrain/utils/station_normalization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict, Union

from terrain.rain_interpolation import RainStation


@dataclass
class StationDict(TypedDict, total=False):
    lat: float
    lon: float
    rain_mm: float
    wind_speed_ms: Optional[float]
    wind_dir_deg: Optional[float]


def normalize_station_dict(raw: Union[StationDict, RainStation]) -> StationDict:
    """
    Accepts either a dict or a RainStation object and returns a clean dict.
    This prevents `'RainStation' object is not subscriptable` errors.
    """

    # Extract attributes safely regardless of input type
    if isinstance(raw, RainStation):
        lat = float(raw.lat)
        lon = float(raw.lon)
        rain_mm = float(raw.rain_mm)

        wind_speed = float(raw.wind_speed_ms) if raw.wind_speed_ms is not None else None
        wind_dir = float(raw.wind_dir_deg) if raw.wind_dir_deg is not None else None

    else:
        # raw is a dict
        lat = float(raw.get("lat", 0.0))
        lon = float(raw.get("lon", 0.0))
        rain_mm = float(raw.get("rain_mm", 0.0))

        ws = raw.get("wind_speed_ms")
        wind_speed = float(ws) if ws is not None else None

        wd = raw.get("wind_dir_deg")
        wind_dir = float(wd) if wd is not None else None

    return {
        "lat": lat,
        "lon": lon,
        "rain_mm": rain_mm,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir,
    }
