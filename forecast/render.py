"""Web-mercator PNG tiles straight from the array store."""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

from .catalog import Variable, derive
from .grid import tile_pixel_centers
from .store import Run

TILE_SIZE = 256


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


@lru_cache(maxsize=1)
def blank_png() -> bytes:
    return encode_png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))


def tile_values(run: Run, var: Variable, step: int, z: int, x: int, y: int) -> np.ndarray:
    lat, lon = tile_pixel_centers(z, x, y, TILE_SIZE)
    rows, cols = run.grid.row_weights(lat), run.grid.col_weights(lon)
    if var.derived_from:
        return derive(var.id, {d: run.sample_field(d, step, rows, cols) for d in var.derived_from})
    return run.sample_field(var.id, step, rows, cols)


def render_tile(run: Run, var: Variable, step: int, z: int, x: int, y: int) -> bytes:
    if var.palette is None:
        raise ValueError(f"{var.id} has no palette")
    rgba = var.palette.colorize(tile_values(run, var, step, z, x, y))
    if not rgba[..., 3].any():
        return blank_png()
    return encode_png(rgba)
