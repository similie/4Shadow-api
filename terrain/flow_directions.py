# terrain/flow_directions.py

from __future__ import annotations

from dataclasses import dataclass


import numpy as np

# D8 neighbour offsets in order:
# 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW
D8_OFFSETS = np.array(
    [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ],
    dtype=np.int16,
)

# distances in cell units (orthogonal = 1, diagonal = sqrt(2))
D8_DISTANCE = np.array(
    [
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
    ],
    dtype=np.float32,
)


@dataclass
class FlowResult:
    directions: np.ndarray  # int8, shape (rows, cols), -1 for sink/nodata
    accumulation: np.ndarray  # float32, number of upstream cells (incl. self)


def compute_d8_flow_directions(dem: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow directions from a DEM (north-up).
    Returns int8 grid with values 0..7 for direction index, or -1 for sink/nodata.
    """
    dem = np.array(dem, copy=False)
    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    rows, cols = dem.shape
    directions = np.full((rows, cols), -1, dtype=np.int8)

    # Mask of valid cells
    valid = np.isfinite(dem)

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue

            z = dem[r, c]
            best_slope = 0.0
            best_dir = -1

            for d, (dr, dc) in enumerate(D8_OFFSETS):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if not valid[nr, nc]:
                    continue

                dz = z - dem[nr, nc]
                if dz <= 0:
                    continue  # can't flow uphill or flat in this simple model

                slope = dz / D8_DISTANCE[d]
                if slope > best_slope:
                    best_slope = slope
                    best_dir = d

            directions[r, c] = best_dir

    return directions


def compute_flow_accumulation(
    dem: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """
    Given a DEM and a D8 direction grid, compute flow accumulation
    (number of upstream cells, including the cell itself).

    Algorithm: process cells in descending elevation order so that when
    a cell is processed, its contribution can be added to its downstream neighbour.
    """
    dem = np.array(dem, copy=False)
    directions = np.array(directions, copy=False)

    if dem.shape != directions.shape:
        raise ValueError("DEM and directions must have same shape")

    rows, cols = dem.shape
    acc = np.ones((rows, cols), dtype=np.float32)  # each cell contributes 1 by default

    valid = np.isfinite(dem)
    flat_dem = dem[valid]

    # Indices of valid cells in descending elevation order
    order = np.argsort(flat_dem)[::-1]  # high → low

    # Map from flat index back to (r, c)
    valid_indices = np.argwhere(valid)  # shape (n_valid, 2)

    for k in order:
        r, c = valid_indices[k]
        d = directions[r, c]
        if d < 0:
            continue  # sink or no flow

        dr, dc = D8_OFFSETS[d]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue

        acc[nr, nc] += acc[r, c]

    acc[~valid] = np.nan
    return acc


def d8_flow(dem: np.ndarray) -> FlowResult:
    """
    Convenience function: compute both D8 directions and accumulation.
    """
    dirs = compute_d8_flow_directions(dem)
    acc = compute_flow_accumulation(dem, dirs)
    return FlowResult(directions=dirs, accumulation=acc)
