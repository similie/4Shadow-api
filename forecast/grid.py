"""Regular latitude/longitude grid geometry and fast sampling.

Everything here is plain index arithmetic: a value at any (lat, lon) is a bilinear blend
of the four surrounding grid cells, so point lookups and tile pixels cost O(1) each.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class Grid:
    lat0: float  # latitude of row 0
    dlat: float  # row spacing; positive means rows run north -> south
    nlat: int
    lon0: float  # longitude of column 0
    dlon: float
    nlon: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)

    @property
    def wraps(self) -> bool:
        """True for global grids whose last column is adjacent to the first."""
        return math.isclose(self.nlon * self.dlon, 360.0, abs_tol=1e-6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(**{k: d[k] for k in ("lat0", "dlat", "nlat", "lon0", "dlon", "nlon")})

    def row_weights(self, lat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows bracketing each latitude and the weight toward the second row."""
        lat = np.asarray(lat, dtype=np.float64)
        fi = np.clip((self.lat0 - lat) / self.dlat, 0.0, self.nlat - 1)
        i0 = np.minimum(np.floor(fi).astype(np.intp), self.nlat - 2)
        return i0, i0 + 1, fi - i0

    def col_weights(self, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Columns bracketing each longitude (wrapping at the antimeridian) and the weight toward the second."""
        lon = np.asarray(lon, dtype=np.float64)
        if self.wraps:
            fj = np.mod(lon - self.lon0, 360.0) / self.dlon
            j0f = np.floor(fj)
            j0 = j0f.astype(np.intp) % self.nlon
            return j0, (j0 + 1) % self.nlon, fj - j0f
        fj = np.clip((lon - self.lon0) / self.dlon, 0.0, self.nlon - 1)
        j0 = np.minimum(np.floor(fj).astype(np.intp), self.nlon - 2)
        return j0, j0 + 1, fj - j0

    def nearest_index(self, lat: float, lon: float) -> tuple[int, int]:
        i0, i1, wi = self.row_weights(lat)
        j0, j1, wj = self.col_weights(lon)
        return int(i1 if wi >= 0.5 else i0), int(j1 if wj >= 0.5 else j0)

    def latlon_of(self, i: int, j: int) -> tuple[float, float]:
        lat = self.lat0 - i * self.dlat
        lon = (self.lon0 + j * self.dlon + 180.0) % 360.0 - 180.0
        return float(lat), float(lon)


def bilinear(v00, v01, v10, v11, wi, wj) -> np.ndarray:
    """NaN-aware bilinear blend.

    ``vab`` is the value at (row a, column b); ``wi``/``wj`` are the weights toward row 1 /
    column 1 and broadcast against the values. Missing corners are dropped and the
    remaining weights renormalised, so coastlines and partial bitmaps degrade gracefully.
    """
    wi = np.asarray(wi, dtype=np.float32)
    wj = np.asarray(wj, dtype=np.float32)
    pairs = (
        (v00, (1 - wi) * (1 - wj)),
        (v01, (1 - wi) * wj),
        (v10, wi * (1 - wj)),
        (v11, wi * wj),
    )
    shape = np.broadcast_shapes(*(np.shape(v) for v, _ in pairs), np.shape(pairs[0][1]))
    num = np.zeros(shape, dtype=np.float32)
    den = np.zeros(shape, dtype=np.float32)
    for v, w in pairs:
        ok = ~np.isnan(v)
        w_ok = np.where(ok, w, np.float32(0))
        num += np.where(ok, v, np.float32(0)) * w_ok
        den += w_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.float32(np.nan)).astype(np.float32, copy=False)


def tile_pixel_centers(z: int, x: int, y: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes (top -> bottom) and longitudes (left -> right) of XYZ tile pixel centres.

    Standard slippy-map convention (Google/OSM/Leaflet/Mapbox): y = 0 is the northern edge.
    """
    n = float(2**z)
    frac = (np.arange(size, dtype=np.float64) + 0.5) / size
    lon = (x + frac) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + frac) / n))))
    return lat, lon
