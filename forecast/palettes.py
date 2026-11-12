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
