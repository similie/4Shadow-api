# terrain/watershed.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .srtm_hgt import DemMosaic

# Must match D8_OFFSETS order in flow_directions.py
D8_NEIGHBORS = [
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),  # E
    (1, 1),  # SE
    (1, 0),  # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1),  # NW
]


def downstream_path_indices(
    start_row: int,
    start_col: int,
    dirs: np.ndarray,
    max_steps: int = 100000,
) -> List[Tuple[int, int]]:
    """
    Follow the D8 direction grid downstream from a starting (row, col).

    Stops when:
      - direction is < 0 (sink or nodata)
      - we hit the grid border
      - we exceed max_steps
    """
    rows, cols = dirs.shape
    path: List[Tuple[int, int]] = []

    r, c = start_row, start_col
    for _ in range(max_steps):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            break

        path.append((r, c))
        d = int(dirs[r, c])
        if d < 0 or d > 7:
            break

        dr, dc = D8_NEIGHBORS[d]
        r += dr
        c += dc

    return path


def latlon_to_rowcol(dem_mosaic: DemMosaic, lat: float, lon: float) -> Tuple[int, int]:
    """
    Convenience wrapper around DemMosaic.latlon_to_rowcol().
    """
    return dem_mosaic.latlon_to_rowcol(lat, lon)
