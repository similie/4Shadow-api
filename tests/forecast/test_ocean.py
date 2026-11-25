"""Surface currents with tides from Copernicus Marine: the 1/12° to 0.25° averaging, run dates, the
current's heading, the default model list, and a run ingested and served end to end."""

import dataclasses
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from conftest import COARSE
from fastapi.testclient import TestClient

from forecast import config, ingest
from forecast.api import create_app
from forecast.catalog import OCEAN, derive
from forecast.config import Settings
from forecast.grid import Grid
from forecast.service import point_forecast
from forecast.sources import SOURCES, copernicus_marine
from forecast.sources.base import StepUnavailable
from forecast.sources.copernicus_marine import NATIVE, block_mean, forecast_hours, run_for
from forecast.sources.gfs import GRID
from forecast.store import RunStore

RUN = datetime(2026, 11, 24, 0, tzinfo=timezone.utc)

# A small stand-in for the 1/12° grid: 1° cells from 80° S to 90° N, 180° W eastwards, and a 3° grid on it.
NATIVE_1 = Grid(lat0=90.0, dlat=1.0, nlat=171, lon0=-180.0, dlon=1.0, nlon=360)
TARGET_3 = Grid(lat0=90.0, dlat=3.0, nlat=61, lon0=0.0, dlon=3.0, nlon=120)


def test_the_native_grid_holds_the_stored_grid_points():
    assert NATIVE.wraps and NATIVE.shape == (2041, 4320)
    out = block_mean(np.ones(NATIVE.shape, np.float32), NATIVE, GRID)
    assert out.shape == GRID.shape
    # 80° S is the native grid's last row (row 680 at 0.25°); south of it there is no ocean data.
    assert out[680].min() == 1.0 and np.isnan(out[681:]).all()


def test_block_mean_averages_the_sea_around_each_point():
    lats = NATIVE_1.lat0 - np.arange(NATIVE_1.nlat) * NATIVE_1.dlat
    lons = NATIVE_1.lon0 + np.arange(NATIVE_1.nlon) * NATIVE_1.dlon
    values = np.broadcast_to(lons[None, :], NATIVE_1.shape).astype(np.float32).copy()
    out = block_mean(values, NATIVE_1, TARGET_3)
    # 30° E: the native columns at 29, 30 and 31° E.
    assert out[10, 10] == pytest.approx(30.0)
    # 0°: columns at 1° W, 0° and 1° E. 180°: across the native grid's edge, 179° E, 180° W and 179° W.
    assert out[10, 0] == pytest.approx(0.0) and out[10, 60] == pytest.approx((179 - 180 - 179) / 3)
    # Land (missing) is left out of the mean; a point with only land around it is missing.
    values[np.ix_([59, 60, 61], [209])] = np.nan  # the column at 29° E, around 30° N
    land = block_mean(values, NATIVE_1, TARGET_3)
    assert land[20, 10] == pytest.approx(30.5)
    values[np.ix_([59, 60, 61], [208, 209, 210, 211])] = np.nan
    assert np.isnan(block_mean(values, NATIVE_1, TARGET_3)[20, 10])
    # South of the native grid's last row (80° S) only a point within half a cell of it gets a value.
    assert lats[-1] == -80 and not np.isnan(out[57]).any() and np.isnan(out[58:]).all()  # 81° S, then 84° S


def test_block_mean_wants_matching_grids():
    with pytest.raises(ValueError):
        block_mean(np.zeros(NATIVE_1.shape, np.float32), NATIVE_1, Grid(90.0, 2.5, 73, 0.0, 2.5, 144))
    with pytest.raises(ValueError):
        block_mean(np.zeros(NATIVE_1.shape, np.float32), NATIVE_1, Grid(89.5, 3.0, 60, 0.0, 3.0, 120))


def test_runs_are_the_newest_forecasts_start():
    assert run_for(datetime(2026, 12, 3, 23, tzinfo=timezone.utc)) == RUN
    assert run_for(datetime(2026, 12, 4, 5, tzinfo=timezone.utc)) == RUN
    assert forecast_hours(3) == [0, 1, 2, 3] and len(forecast_hours()) == 240
    with pytest.raises(ValueError):
        forecast_hours(240)


def test_current_direction_is_where_it_flows():
    def heading(u, v):
        return float(derive("current_dir", {"current_u": np.float32(u), "current_v": np.float32(v)}))

    assert heading(1, 0) == pytest.approx(90) and heading(0, -1) == pytest.approx(180)
    assert heading(-1, 0) == pytest.approx(270) and heading(0.5, 0.5) == pytest.approx(45)
    assert float(derive("current_speed", {"current_u": np.float32(3), "current_v": np.float32(4)})) == 5


def test_the_ocean_is_served_by_default_only_with_a_login(monkeypatch):
    for key in copernicus_marine.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FORECAST_MODELS", raising=False)
    assert "ocean" not in config.get_settings().models
    for key in copernicus_marine.CREDENTIALS:
        monkeypatch.setenv(key, "x")
    assert config.get_settings().models[-1] == "ocean"
    monkeypatch.setenv("FORECAST_MODELS", "gfs,aerosols")
    assert "ocean" not in config.get_settings().models


def test_the_client_needs_a_login(monkeypatch):
    for key in copernicus_marine.CREDENTIALS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="COPERNICUSMARINE_SERVICE_USERNAME"):
        copernicus_marine.MarineClient().latest_run()


def test_the_toolbox_leaves_the_log_alone(monkeypatch):
    """Importing the toolbox makes log times UTC and adds a handler of its own; the client undoes both."""
    pytest.importorskip("copernicusmarine")
    monkeypatch.setattr(logging.Formatter, "converter", time.gmtime)  # as the toolbox's import does
    toolbox_log = logging.getLogger("copernicusmarine")
    toolbox_log.addHandler(logging.NullHandler())
    copernicus_marine._toolbox()
    assert logging.Formatter.converter is time.localtime
    assert not toolbox_log.handlers


class FakeOcean:
    """Currents flowing north-east, faster each hour; "land" east of 180°."""

    def __init__(self, missing_from: int | None = None):
        self.missing_from = missing_from
        self.asked: list[int] = []

    def latest_run(self, max_fhr):
        return RUN

    def step(self, run, fhr, names, grid):
        self.asked.append(fhr)
        if self.missing_from is not None and fhr >= self.missing_from:
            raise StepUnavailable(f"no {fhr}")
        lon = grid.lon0 + np.arange(grid.nlon) * grid.dlon
        sea = np.broadcast_to(lon[None, :] < 180, grid.shape)
        speed = 0.1 * (fhr + 1)
        return {name: np.where(sea, np.float32(speed), np.float32(np.nan)) for name in names}


def test_ingest_and_serve_currents(tmp_path, monkeypatch):
    monkeypatch.setitem(ingest.SOURCES, "ocean", dataclasses.replace(SOURCES["ocean"], grid=COARSE))
    settings = Settings(data_root=tmp_path, models=("gfs", "ocean"))
    assert ingest.ingest_latest(settings, client=FakeOcean(), catalog=OCEAN, max_fhr=3) == "2026112400"
    run = RunStore(tmp_path, "ocean").get()
    assert run.forecast_hours.tolist() == [0, 1, 2, 3]
    assert run.manifest["source"]["dataset"] == copernicus_marine.DATASET_ID
    assert "Copernicus Marine Service Information" in run.manifest["source"]["attribution"]

    _, body = point_forecast({"ocean": RunStore(tmp_path, "ocean")}, -9.0, 125.0, model="ocean", start="run", tz="UTC")
    ts = body["timeseries"]
    assert ts["current_u"] == [0.1, 0.2, 0.3, 0.4]
    assert ts["current_speed"] == [pytest.approx(0.14), pytest.approx(0.28), pytest.approx(0.42), pytest.approx(0.57)]
    assert ts["current_dir"] == [45, 45, 45, 45]
    assert body["daily"]["current_speed_max"] == [pytest.approx(0.57)] and body["units"]["current_speed"] == "m/s"
    # On "land" a sea-only variable takes the nearest sea; far inland there is none.
    _, inland = point_forecast({"ocean": RunStore(tmp_path, "ocean")}, 10.0, 260.0, model="ocean", start="run", tz="UTC")
    assert inland["timeseries"]["current_speed"] == [None] * 4

    api = TestClient(create_app(settings))
    r = api.get("/v2/ocean/forecast", params={"lat": -9, "lon": 125, "vars": "current_speed", "start": "run",
                                               "daily": False})
    assert r.status_code == 200 and r.json()["timeseries"]["current_speed"][0] == pytest.approx(0.14)
    v = {x["id"]: x for x in api.get("/v2/ocean/variables").json()["variables"]}
    assert v["current_speed"]["tileable"] and v["current_dir"]["marine"] and v["current_u"]["marine"]
    assert api.get("/v2/ocean/tiles/latest/current_speed/f003/1/1/0.png").status_code == 200
    assert api.get("/v2/health").json()["models"]["ocean"]["latest_run"] == "2026112400"


def test_a_missing_hour_leaves_the_run_unpublished(tmp_path, monkeypatch):
    monkeypatch.setitem(ingest.SOURCES, "ocean", dataclasses.replace(SOURCES["ocean"], grid=COARSE))
    settings = Settings(data_root=tmp_path, models=("gfs", "ocean"))
    with pytest.raises(StepUnavailable):
        ingest.ingest_run(RUN, settings, client=FakeOcean(missing_from=2), catalog=OCEAN, max_fhr=3)
    assert RunStore(tmp_path, "ocean").get() is None
