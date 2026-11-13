"""Raw gridded values for client-side rendering: WebGL map layers, wind particles, contours, analysis.

    GET /v2/{model}/fields/{run}/{time}?vars=u10,v10&stride=2&bbox=west,south,east,north

Tiles (render.py) are finished pictures; a field is the numbers behind them, so a client can
colour, animate, contour and sample them itself. The body is a small self-describing binary,
format "4SF1", that browsers read with a DataView and upload straight to GPU textures:

    bytes 0-3   ASCII "4SF1"
    bytes 4-7   uint32 little-endian H: length of the JSON header that follows
    bytes 8..   JSON header (UTF-8), space-padded so the arrays start on an 8-byte boundary
    then        one array per variable, in header order: int16 little-endian, nlat x nlon,
                rows north -> south, columns west -> east

value = raw * scale + offset, and raw -32768 means missing (no data at that cell). These are the
store's own int16 values, so serving a field is slicing, not computing. The header carries the
grid (lat0/lon0 of the first row/column and the spacing), units and semantics of each variable.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np

from .catalog import ModelCatalog, Variable
from .service import ForecastError, step_for
from .store import MISSING, Run, iso

MAGIC = b"4SF1"
MAX_VARS = 4
MAX_STRIDE = 16
MAX_CELLS = 1440 * 721  # per variable: the whole 0.25° globe


def parse_bbox(text: str | None) -> tuple[float, float, float, float]:
    """'west,south,east,north' in degrees -> floats; east < west crosses the antimeridian."""
    if not text:
        return -180.0, -90.0, 180.0, 90.0
    try:
        w, s, e, n = (float(x) for x in text.split(","))
    except ValueError:
        raise ForecastError(400, "bbox must be west,south,east,north in degrees") from None
    if not (-90 <= s < n <= 90) or not (-180 <= w <= 360) or not (-180 <= e <= 360):
        raise ForecastError(400, "bbox needs -90 <= south < north <= 90 and longitudes within -180..360")
    return w, s, e, n


def _rows(run: Run, south: float, north: float, stride: int) -> np.ndarray:
    g = run.grid
    first = max(0, math.floor((g.lat0 - north) / g.dlat + 1e-9))
    last = min(g.nlat - 1, math.ceil((g.lat0 - south) / g.dlat - 1e-9))
    return np.arange(first, last + 1, stride, dtype=np.intp)


def _cols(run: Run, west: float, east: float, stride: int) -> tuple[np.ndarray, bool]:
    """Column indices west -> east (wrapping), and whether they go all the way round."""
    g = run.grid
    span = (east - west) % 360.0 or 360.0
    if span >= 360.0 - g.dlon / 2:  # the whole way round: start at -180 and don't repeat a column
        if not g.wraps:
            return np.arange(0, g.nlon, stride, dtype=np.intp), False
        start = round(((-180.0 - g.lon0) % 360.0) / g.dlon)
        n = g.nlon
        cols = (start + np.arange(0, n, dtype=np.intp)) % n
        return cols[::stride], n % stride == 0
    first = math.floor(((west - g.lon0) % 360.0) / g.dlon + 1e-9)
    count = math.ceil(span / g.dlon - 1e-9) + 1  # include the column at or past the east edge
    cols = first + np.arange(0, count, stride, dtype=np.intp)
    if g.wraps:
        cols %= g.nlon
    else:
        cols = cols[cols < g.nlon]
    return cols, False


def encode_field(cat: ModelCatalog, run: Run, time_ref: str, var_ids: list[str], stride: int = 1,
                 bbox: str | None = None) -> bytes:
    if not var_ids:
        raise ForecastError(400, "Give vars, e.g. vars=u10,v10")
    if len(var_ids) > MAX_VARS:
        raise ForecastError(400, f"At most {MAX_VARS} variables per field request")
    if not 1 <= stride <= MAX_STRIDE:
        raise ForecastError(400, f"stride must be 1..{MAX_STRIDE}")
    stored = {v.id: v for v in cat.stored_variables() if v.id in run.variables}
    variables: list[Variable] = []
    for vid in var_ids:
        if vid not in stored:
            hint = ""
            v = cat.get(vid)
            if v is not None and v.derived_from:
                hint = f" {vid} is computed from {', '.join(v.derived_from)}: request those."
            raise ForecastError(400, f"No field for {vid!r}.{hint} Fields: {sorted(stored)}")
        variables.append(stored[vid])
    step = step_for(run, time_ref)
    west, south, east, north = parse_bbox(bbox)
    rows = _rows(run, south, north, stride)
    cols, wraps = _cols(run, west, east, stride)
    if rows.size * cols.size > MAX_CELLS:
        raise ForecastError(400, f"{rows.size} x {cols.size} cells is too many; use a larger stride or a smaller bbox")

    g = run.grid
    lat0 = g.lat0 - rows[0] * g.dlat
    lon0 = (g.lon0 + cols[0] * g.dlon + 180.0) % 360.0 - 180.0
    fhr = int(run.forecast_hours[step])
    interval = int(fhr - run.forecast_hours[step - 1]) if step > 0 else 0
    header = {
        "format": "4SF1",
        "model": run.model,
        "run": run.id,
        "init_time": iso(run.init_time),
        "valid_time": iso(run.valid_times[step]),
        "forecast_hour": fhr,
        "interval_hours": interval,  # interval variables are mean rates over this many hours before valid_time
        "grid": {"lat0": float(lat0), "dlat": g.dlat * stride, "nlat": int(rows.size),
                 "lon0": float(lon0), "dlon": g.dlon * stride, "nlon": int(cols.size), "wraps": wraps},
        "dtype": "int16",
        "byte_order": "little",
        "missing": MISSING,
        "variables": [
            {"id": v.id, "name": v.name, "units": v.units, "semantics": "interval_mean" if v.interval else "instant",
             "scale": run.variables[v.id]["scale"], "offset": run.variables[v.id]["offset"]}
            for v in variables
        ],
    }
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-(8 + len(text)) % 8)
    parts = [MAGIC, struct.pack("<I", len(text)), text]
    ix = np.ix_(rows, cols)
    for v in variables:
        parts.append(np.ascontiguousarray(run.array(v.id)[step][ix], dtype="<i2").tobytes())
    return b"".join(parts)


def decode_field(body: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Inverse of encode_field (for tests and Python clients): header and float32 arrays, NaN = missing."""
    if body[:4] != MAGIC:
        raise ValueError("not a 4SF1 field")
    (n,) = struct.unpack_from("<I", body, 4)
    header = json.loads(body[8:8 + n])
    g = header["grid"]
    size = g["nlat"] * g["nlon"]
    out, offset = {}, 8 + n
    for v in header["variables"]:
        raw = np.frombuffer(body, dtype="<i2", count=size, offset=offset).reshape(g["nlat"], g["nlon"])
        vals = raw.astype(np.float32) * np.float32(v["scale"]) + np.float32(v["offset"])
        vals[raw == header["missing"]] = np.nan
        out[v["id"]] = vals
        offset += size * 2
    return header, out
