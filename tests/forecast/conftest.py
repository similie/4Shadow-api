from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from forecast.catalog import GFS
from forecast.grid import Grid
from forecast.store import RunWriter

COARSE = Grid(lat0=90.0, dlat=2.5, nlat=73, lon0=0.0, dlon=2.5, nlon=144)
HOURS = list(range(0, 25)) + [27, 30, 33, 36]


def synthetic_value(var_id: str, fhr: int, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray | None:
    shape = lat2d.shape
    if var_id == "t2m":
        return (0.5 * lat2d + 0.1 * fhr).astype(np.float32)
    if var_id == "u10":
        return np.full(shape, 3.0, np.float32)
    if var_id == "v10":
        return np.full(shape, 4.0, np.float32)
    if var_id == "precip":
        return None if fhr == 0 else np.full(shape, 1.0, np.float32)
    if var_id == "dswrf":
        return None if fhr == 0 else np.full(shape, 200.0, np.float32)
    if var_id == "mslp":
        return np.full(shape, 1013.2, np.float32)
    return np.full(shape, 50.0, np.float32)


def write_run(root, init: datetime, hours=HOURS, grid=COARSE, keep_runs=None):
    lat2d, lon2d = np.meshgrid(grid.lat0 - np.arange(grid.nlat) * grid.dlat,
                               grid.lon0 + np.arange(grid.nlon) * grid.dlon, indexing="ij")
    writer = RunWriter(root / "gfs", "gfs", init, grid, hours, GFS.stored_variables(), source={"provider": "test"})
    for k, h in enumerate(hours):
        for v in GFS.stored_variables():
            writer.write(v.id, k, synthetic_value(v.id, h, lat2d, lon2d))
    writer.write_static("orog", (lon2d * 2.0).astype(np.float32), {"name": "orog", "units": "m"})
    writer.write_static("land", np.ones(grid.shape, np.float32), {"name": "land", "units": "1"})
    writer.publish(keep_runs=keep_runs)
    return writer.id


@pytest.fixture
def run_init() -> datetime:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now - timedelta(hours=2)


@pytest.fixture
def store_root(tmp_path, run_init):
    write_run(tmp_path, run_init)
    return tmp_path
