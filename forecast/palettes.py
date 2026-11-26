"""Fixed, physically meaningful colour scales.

A palette maps a value in real units straight to RGBA, so the same temperature is always
the same colour: across tiles, forecast hours, runs, and the legend served to clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def _hex_to_rgba(code: str) -> tuple[int, int, int, int]:
    h = code.lstrip("#")
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        raise ValueError(f"Bad colour {code!r}; use #RRGGBB or #RRGGBBAA")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]


@dataclass(frozen=True)
class Palette:
    stops: tuple[tuple[float, str], ...]
    transparent_below: float | None = None
    transparent_above: float | None = None

    def __post_init__(self) -> None:
        values = [v for v, _ in self.stops]
        if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("Palette stops must be strictly increasing and at least two")
        for _, c in self.stops:
            _hex_to_rgba(c)

    @cached_property
    def _tables(self) -> tuple[np.ndarray, np.ndarray]:
        values = np.array([v for v, _ in self.stops], dtype=np.float64)
        colors = np.array([_hex_to_rgba(c) for _, c in self.stops], dtype=np.float64)
        return values, colors

    def colorize(self, data: np.ndarray) -> np.ndarray:
        """Map a 2-D array of values to an (H, W, 4) uint8 RGBA image."""
        values, colors = self._tables
        v = np.asarray(data, dtype=np.float32)
        nan = np.isnan(v)
        vv = np.where(nan, values[0], v)
        out = np.empty(v.shape + (4,), dtype=np.uint8)
        for c in range(4):
            out[..., c] = np.rint(np.interp(vv, values, colors[:, c]))
        hide = nan
        if self.transparent_below is not None:
            hide = hide | (v < self.transparent_below)
        if self.transparent_above is not None:
            hide = hide | (v > self.transparent_above)
        out[hide, 3] = 0
        return out

    def legend(self) -> dict:
        return {
            "stops": [{"value": v, "color": c if len(c) == 9 else c + "ff"} for v, c in self.stops],
            "transparent_below": self.transparent_below,
            "transparent_above": self.transparent_above,
        }


TEMPERATURE_C = Palette(
    (
        (-40, "#f0e6ff"), (-30, "#b18cd9"), (-20, "#6a51a3"), (-10, "#3a6fc9"),
        (0, "#4fb3e8"), (5, "#7fd6c2"), (10, "#a3dc7a"), (15, "#e6e65c"),
        (20, "#f7c843"), (25, "#f39b2f"), (30, "#e3542b"), (35, "#c41f2a"),
        (40, "#8e0f3f"), (50, "#4a0a2e"),
    )
)

RELATIVE_HUMIDITY = Palette(
    ((0, "#a5662a"), (25, "#d9b36b"), (50, "#e8e8a6"), (70, "#7fc97f"), (85, "#3c9ad8"), (100, "#1f3fa3"))
)

WIND_MS = Palette(
    (
        (0, "#5e60ce"), (3, "#4ea8de"), (6, "#56cfa1"), (9, "#a7e05a"), (12, "#f2d743"),
        (16, "#f39c33"), (20, "#e2532d"), (25, "#b5179e"), (32, "#6a0572"), (40, "#2d0136"),
    )
)

PRESSURE_HPA = Palette(
    (
        (960, "#5b1a8f"), (980, "#3f6fd1"), (995, "#5fb8e8"), (1005, "#9ee3c8"),
        (1013, "#f2f2d0"), (1020, "#f7d27a"), (1030, "#f08a4b"), (1045, "#b22a2a"),
    )
)

PRECIP_RATE_MMH = Palette(
    (
        (0.1, "#9fd3ff99"), (0.5, "#5aa9f5"), (1, "#2f6fe0"), (2, "#1aa64a"), (4, "#8fd13f"),
        (6, "#f5e10b"), (10, "#ff9a00"), (16, "#ff3b00"), (25, "#d4007f"), (50, "#7a00c4"),
    ),
    transparent_below=0.1,
)

CLOUD_COVER = Palette(
    ((5, "#ffffff10"), (30, "#e8edf255"), (60, "#d0d7e0aa"), (90, "#b8c2ceee"), (100, "#a0abb8ff")),
    transparent_below=5,
)

CAPE_JKG = Palette(
    (
        (100, "#c7e9c080"), (500, "#74c476"), (1000, "#f2e34c"), (2000, "#f59a2f"),
        (3000, "#e0362a"), (4000, "#a3167a"), (6000, "#5a0a6b"),
    ),
    transparent_below=100,
)

PRECIPITABLE_WATER_MM = Palette(
    (
        (0, "#f3e9d2"), (10, "#d9c27a"), (20, "#a8d58a"), (30, "#5fc1a6"),
        (40, "#3a8fd1"), (50, "#3a4fc1"), (60, "#6a2fb0"), (75, "#3d0f6b"),
    )
)

VISIBILITY_M = Palette(
    ((0, "#6b2f8f"), (1000, "#b54a8acc"), (3000, "#e08a5a99"), (5000, "#e8c46a66"), (10000, "#f0e6a000")),
    transparent_above=10000,
)

SHORTWAVE_WM2 = Palette(
    (
        (0, "#00000000"), (50, "#3b1f5e66"), (200, "#8a3b6b"), (400, "#d9603b"),
        (600, "#f2a33a"), (800, "#f7dd5c"), (1000, "#fff6c2"),
    )
)

SNOW_DEPTH_M = Palette(
    ((0.01, "#dbe9f6aa"), (0.1, "#a6cbe8"), (0.3, "#6fa6d6"), (1, "#4a6fbf"), (3, "#6a4fb0"), (10, "#e6e0f5")),
    transparent_below=0.01,
)

# Sea state. Heights follow the steps sailors use (calm < 0.5 m, moderate 1.25-2.5 m, rough
# 2.5-4 m, very rough 4-6 m, high 6-9 m); land and sea ice are missing, so transparent.
WAVE_HEIGHT_M = Palette(
    (
        (0, "#1d3f7a"), (0.5, "#2a6db5"), (1, "#3a97c9"), (1.5, "#3fbfa8"), (2, "#68cc6a"),
        (3, "#d6d94a"), (4, "#f2a33a"), (5, "#e8622c"), (7, "#c42a4a"), (9, "#8e1f7a"),
        (12, "#5a2a8a"), (15, "#e6d9f2"),
    )
)

WAVE_PERIOD_S = Palette(
    (
        (0, "#2b1d4f"), (4, "#3a4fa8"), (6, "#2f86c4"), (8, "#3fb0a8"), (10, "#7ccc6a"),
        (12, "#d9d94a"), (14, "#f2a33a"), (16, "#e0562c"), (18, "#b3245a"), (22, "#6a1f6b"),
    )
)

# Surface currents: most of the open ocean moves under 0.5 m/s (1 knot); the Gulf Stream, the Agulhas and
# tidal straits run 2 m/s (4 knots) or more. Land is missing, so transparent.
CURRENT_MS = Palette(
    (
        (0, "#1d2f5a"), (0.1, "#24508f"), (0.25, "#2f86c4"), (0.5, "#3fb8a8"), (0.75, "#7ccc6a"),
        (1, "#d6d94a"), (1.5, "#f2a33a"), (2, "#e0562c"), (3, "#b3245a"), (4, "#e6d9f2"),
    )
)

# Tide height above or below mean sea level: blues for low water, warm colours for high, grey at the mean.
# The open ocean's tides are within half a metre; on shelves and in bays they reach 2-4 m and in a few
# places (the Bay of Fundy, Saint-Malo) more, which takes the end colours. Land is missing, so transparent.
TIDE_M = Palette(
    (
        (-4, "#14205a"), (-2.5, "#1f3f8f"), (-1.5, "#2a62b8"), (-0.75, "#3d8fd0"), (-0.25, "#86b8d8"),
        (0, "#b4b8bc"), (0.25, "#e3bd8f"), (0.75, "#f0964a"), (1.5, "#e0602c"), (2.5, "#b8284a"), (4, "#6e1450"),
    )
)

# Simulated radar, in the colours weather radar displays use; below 10 dBZ (drizzle, cloud) is not shown.
REFLECTIVITY_DBZ = Palette(
    (
        (10, "#019ff4"), (15, "#0300f4"), (20, "#02fd02"), (25, "#01c501"), (30, "#008e00"),
        (35, "#fdf802"), (40, "#e5bc00"), (45, "#fd9500"), (50, "#fd0000"), (55, "#d40000"),
        (60, "#bc0000"), (65, "#f800fd"), (70, "#9854c6"),
    ),
    transparent_below=10,
)

FREEZING_LEVEL_M = Palette(
    (
        (0, "#6a51a3"), (500, "#3a6fc9"), (1000, "#4fb3e8"), (2000, "#7fd6c2"), (3000, "#e6e65c"),
        (4000, "#f39b2f"), (5000, "#c41f2a"), (6000, "#6a0f3f"),
    )
)

# Share of precipitation that is frozen: rain blue through mixed violet to snow white; nothing where dry.
FROZEN_PCT = Palette(((0, "#3a97c9"), (40, "#6a6fd1"), (60, "#b18cd9"), (100, "#f0e6ff")))

FREEZING_RAIN = Palette(((0.5, "#d7307dcc"), (1, "#d7307dff")), transparent_below=0.5)

SOIL_MOISTURE = Palette(
    (
        (0.02, "#8a4b1c"), (0.08, "#c47a2c"), (0.15, "#e0b96b"), (0.22, "#e8e8a6"), (0.28, "#a6d88a"),
        (0.34, "#4fb08a"), (0.40, "#3c8ad8"), (0.48, "#1f3fa3"),
    )
)

# Heat index, by the US National Weather Service's categories (caution 27 °C, extreme caution 32,
# danger 39, extreme danger 52). Below caution is not shown.
HEAT_INDEX_C = Palette(
    (
        (27, "#f6e27f"), (32, "#f7b346"), (39, "#e3542b"), (46, "#c41f2a"), (52, "#8e0f3f"), (60, "#4a0a2e"),
    ),
    transparent_below=26,
)

WET_BULB_C = Palette(
    (
        (0, "#3a6fc9"), (10, "#4fb3e8"), (16, "#7fd6c2"), (20, "#a3dc7a"), (24, "#e6e65c"), (26, "#f7c843"),
        (28, "#f39b2f"), (30, "#e3542b"), (31, "#c41f2a"), (33, "#8e0f3f"), (35, "#4a0a2e"),
    )
)

VPD_KPA = Palette(
    (
        (0, "#1f3fa3"), (0.4, "#3c9ad8"), (0.8, "#7fc97f"), (1.2, "#e8e8a6"), (2, "#e0b96b"),
        (3, "#c47a2c"), (4.5, "#8a4b1c"), (6, "#4a2a10"),
    )
)

FIRE_WEATHER = Palette(
    (
        (0, "#1a9850"), (10, "#66bd63"), (20, "#d9ef8b"), (30, "#fee08b"), (40, "#fdae61"),
        (50, "#f46d43"), (65, "#d73027"), (80, "#a50026"), (100, "#4a0010"),
    )
)

WIND_POWER_WM2 = Palette(
    (
        (0, "#2b1d4f"), (100, "#3a4fa8"), (200, "#2f86c4"), (400, "#3fb0a8"), (600, "#7ccc6a"),
        (800, "#d9d94a"), (1200, "#f2a33a"), (2000, "#e0562c"), (4000, "#6a1f6b"),
    )
)

# Air quality: the US EPA's AQI categories (2024) for 24-hour PM2.5 and PM10, in the AQI's own colours
# (good, moderate, unhealthy for sensitive groups, unhealthy, very unhealthy, hazardous). Each colour is
# reached at the top of its category; good air is see-through so the map shows where the air is clean.
PM25_UGM3 = Palette(
    (
        (1, "#00e40018"), (9, "#00e40080"), (35.4, "#ffff00e0"), (55.4, "#ff7e00"), (125.4, "#ff0000"),
        (225.4, "#8f3f97"), (325, "#7e0023"),
    ),
    transparent_below=1,
)

PM10_UGM3 = Palette(
    (
        (5, "#00e40018"), (54, "#00e40080"), (154, "#ffff00e0"), (254, "#ff7e00"), (354, "#ff0000"),
        (424, "#8f3f97"), (604, "#7e0023"),
    ),
    transparent_below=5,
)

# Aerosol optical depth at 550 nm: about 0.1 is clear air, 0.3 visible haze, 1 a thick pall of dust or
# smoke that dims the sun.
AEROSOL_OPTICAL_DEPTH = Palette(
    (
        (0.05, "#fff7bc30"), (0.1, "#fee391"), (0.3, "#fec44f"), (0.5, "#fe9929"), (1, "#d95f0e"),
        (2, "#993404"), (4, "#4a1a02"),
    ),
    transparent_below=0.05,
)
