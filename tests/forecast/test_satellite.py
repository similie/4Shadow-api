"""Satellite imagery from NOAA's hourly mosaic: the brightness scale, the averaging onto the grid, the quality
flags, which hours are there, catching up missed hours, and an hour served end to end."""

import dataclasses
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import requests
from conftest import COARSE
from fastapi.testclient import TestClient

from forecast import ingest
from forecast.api import create_app
from forecast.catalog import SATELLITE
from forecast.config import Settings
from forecast.grid import Grid
from forecast.service import point_forecast
from forecast.sources import SOURCES, gmgsi
from forecast.sources.base import StepUnavailable
from forecast.store import RunStore, published_runs, read_latest

NOW = datetime(2026, 11, 27, 1, 50, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def test_infrared_brightness_is_a_temperature():
    k = gmgsi.brightness_temperature(np.array([0, 32, 176, 177, 255]))
    assert k.tolist() == [330, 314, 242, 241, 163]


def test_each_cell_is_the_mean_of_the_pixels_in_it():
    grid = Grid(lat0=90.0, dlat=1.0, nlat=181, lon0=0.0, dlon=1.0, nlon=360)
    # Pixels every 0.25°, between the grid's points and its cells' edges: 60.125° N to 59.875° S, and
    # from 179.875° W eastwards, each the longitude it's at (0-360).
    lats = 60.125 - 0.25 * np.arange(481)
    lons = -179.875 + 0.25 * np.arange(1440)
    values = np.broadcast_to(np.mod(lons, 360)[None, :], (481, 1440)).astype(np.float32).copy()
    out = gmgsi.mean_on_grid(values, lats, lons, grid, block=7)
    # 10° E is the mean of the pixels at 9.625, 9.875, 10.125 and 10.375° E.
    assert out[30, 10] == pytest.approx(10.0) and out[100, 200] == pytest.approx(200.0)
    # North of 60° N and south of 60° S there are no pixels; between, every cell has some.
    assert np.isnan(out[:30]).all() and np.isnan(out[151:]).all() and not np.isnan(out[30:151]).any()
    # Missing pixels are left out, and a cell with none left is missing.
    values[:, (lons > 9.5) & (lons < 10.25)] = np.nan
    assert gmgsi.mean_on_grid(values, lats, lons, grid)[30, 10] == pytest.approx(10.375)
    values[:, (lons > 9.5) & (lons < 10.5)] = np.nan
    assert np.isnan(gmgsi.mean_on_grid(values, lats, lons, grid)[30, 10])


def test_what_the_satellites_see_too_obliquely_is_left_out():
    grid = Grid(lat0=90.0, dlat=1.0, nlat=181, lon0=0.0, dlon=1.0, nlon=360)
    mask = gmgsi.oblique(grid)
    # Never near the equator; beyond 72° always (the mosaic stops at 72.7°).
    assert not mask[60:121].any() and mask[:18].all() and mask[163:].all()
    # Reykjavik (64° N, 22° W) and Anchorage (61° N, 150° W) are in; Tromsø (70° N, 19° E) is not.
    assert not mask[26, 338] and not mask[29, 210] and mask[20, 19]


def _mosaic(path, data, flag, lats, lons):
    import xarray as xr

    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    xr.Dataset(
        {"data": (("time", "yc", "xc"), data[None].astype(np.float32)),
         "dqf": (("time", "yc", "xc"), flag[None].astype(np.float32))},
        coords={"lat": (("yc", "xc"), lat2.astype(np.float32)), "lon": (("yc", "xc"), lon2.astype(np.float32)),
                "time": [np.datetime64("2026-11-27T00:00", "ns")]},
    ).to_netcdf(path, engine="h5netcdf")


def test_only_pixels_flagged_good_are_kept(tmp_path):
    data = np.array([[64, 200, 255], [0, 100, 30]], np.float32)
    flag = np.array([[np.nan, 0, 20], [10, 0, 0]], np.float32)
    _mosaic(tmp_path / "lw.nc", data, flag, np.array([10.0, 9.9]), np.array([0.0, 0.1, 0.2]))
    values, lats, lons = gmgsi.read_band(tmp_path / "lw.nc", "LW")
    assert np.allclose(lats, [10.0, 9.9]) and np.allclose(lons, [0.0, 0.1, 0.2])
    # The unflagged top row, bad (20) and no-data (10) pixels are missing; the rest are kelvin.
    assert np.isnan(values[0, 0]) and np.isnan(values[0, 2]) and np.isnan(values[1, 0])
    assert values[0, 1] == 218 and values[1, 1] == 280 and values[1, 2] == 315
    vis, _, _ = gmgsi.read_band(tmp_path / "lw.nc", "VIS")
    assert vis[1, 1] == 100  # the visible band stays as its brightness


class FakeBucket:
    """S3's object listing for a few hours of each band; LW at 00Z was reprocessed."""

    def __init__(self, hours: dict[str, list[datetime]]):
        self.hours = hours
        self.listed: list[str] = []

    def get(self, url, params=None, timeout=None, stream=False):
        prefix = params["prefix"]
        self.listed.append(prefix)
        band = prefix.split("/")[0].removeprefix("GMGSI_")
        keys = [f"GMGSI_{band}/{h:%Y/%m/%d/%H}/GLOBCOMP{band}_v3r0_blend_s{h:%Y%m%d%H}00000_c{h:%Y%m%d%H}35000.nc"
                for h in self.hours[band] if f"{h:%Y/%m/%d}/" in prefix]
        if band == "LW" and "2026/11/27/" in prefix:
            keys.append("GMGSI_LW/2026/11/27/00/GLOBCOMPLW_v3r0_blend_s202611270000000_c202611270159000.nc")
        body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
        return type("R", (), {"text": f"<ListBucketResult>{body}</ListBucketResult>", "raise_for_status": lambda self: None})()


def test_the_runs_are_the_last_days_hours_with_every_band():
    every = [NOW.replace(minute=0) - k * HOUR for k in range(30)]  # 01Z back to 20Z two days ago
    bucket = FakeBucket({"LW": every, "WV": every, "VIS": [h for h in every if h.hour != 22]})
    client = gmgsi.MosaicClient(session=bucket, now=NOW)
    runs = client.available_runs()
    # 24 hours back from 01:50 is 01:50 yesterday: 02Z yesterday to 01Z today, but for 22Z without visible.
    assert runs[0] == datetime(2026, 11, 26, 2, tzinfo=timezone.utc) and runs[-1] == datetime(2026, 11, 27, 1, tzinfo=timezone.utc)
    assert len(runs) == 23 and datetime(2026, 11, 26, 22, tzinfo=timezone.utc) not in runs
    assert client.latest_run() == runs[-1]
    # A reprocessed hour's newest file is the one read.
    assert client._keys[("LW", datetime(2026, 11, 27, 0, tzinfo=timezone.utc))].endswith("_c202611270159000.nc")


class FlakyBucket:
    """A file whose first downloads drop the connection, as S3 sometimes does."""

    def __init__(self, drops: int, status: int = 200):
        self.drops = drops
        self.status = status
        self.calls = 0

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls += 1
        if self.calls <= self.drops:
            raise requests.ConnectionError("Remote end closed connection without response")
        status = self.status

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                if status >= 400:
                    raise requests.HTTPError(f"{status}", response=type("R", (), {"status_code": status})())

            def iter_content(self, size):
                yield b"mosaic"

        return Response()


def test_a_dropped_download_is_tried_again(tmp_path, monkeypatch):
    monkeypatch.setattr(gmgsi.time, "sleep", lambda s: None)
    bucket = FlakyBucket(drops=2)
    gmgsi.MosaicClient(session=bucket)._download("GMGSI_LW/x.nc", tmp_path / "lw.nc")
    assert bucket.calls == 3 and (tmp_path / "lw.nc").read_bytes() == b"mosaic"
    # Not forever, and not on an answer that won't change.
    with pytest.raises(requests.ConnectionError):
        gmgsi.MosaicClient(session=FlakyBucket(drops=9), retries=3)._download("k", tmp_path / "a.nc")
    gone = FlakyBucket(drops=0, status=404)
    with pytest.raises(requests.HTTPError):
        gmgsi.MosaicClient(session=gone)._download("k", tmp_path / "b.nc")
    assert gone.calls == 1


class FakeSatellite:
    """A mosaic whose infrared is 20 °C (293.15 K) plus the hour, everywhere; missing one hour if asked."""

    def __init__(self, hours: list[datetime], broken: datetime | None = None):
        self.hours = hours
        self.broken = broken
        self.read: list[datetime] = []

    def available_runs(self):
        return list(self.hours)

    def latest_run(self, max_fhr=0):
        return self.hours[-1]

    def step(self, run, fhr, names, grid):
        if run == self.broken:
            raise StepUnavailable("gone")
        self.read.append(run)
        lat = grid.lat0 - np.arange(grid.nlat) * grid.dlat
        polar = np.broadcast_to((np.abs(lat) > 72.7)[:, None], grid.shape)
        value = {"LW": 293.15 + run.hour, "WV": 240.0, "VIS": 100.0}
        return {name: np.where(polar, np.float32(np.nan), np.float32(value[name])) for name in names}


@pytest.fixture
def small(monkeypatch):
    """The satellite model on a coarse grid, keeping three hours."""
    monkeypatch.setitem(ingest.SOURCES, "satellite", dataclasses.replace(SOURCES["satellite"], grid=COARSE, keep_runs=3))
    return lambda root: Settings(data_root=root, models=("gfs", "satellite"))


def test_missed_hours_are_caught_up_newest_first(tmp_path, small, caplog):
    settings = small(tmp_path)
    hours = [datetime(2026, 11, 26, 20, tzinfo=timezone.utc) + k * HOUR for k in range(5)]  # 20Z to 00Z
    client = FakeSatellite(hours)
    with caplog.at_level(logging.WARNING):
        assert ingest.ingest_latest(settings, client=client, catalog=SATELLITE) == "2026112700"
    # Only the three hours kept, the newest first; no warning about the land mask a satellite grid can't take.
    assert client.read == hours[:1:-1] and published_runs(tmp_path / "satellite") == ["2026112622", "2026112623", "2026112700"]
    assert read_latest(tmp_path / "satellite") == "2026112700" and "land mask" not in caplog.text
    # An hour later: only the new hour is read, and the oldest goes.
    client = FakeSatellite([*hours, hours[-1] + HOUR])
    assert ingest.ingest_latest(settings, client=client, catalog=SATELLITE) == "2026112701"
    assert client.read == [hours[-1] + HOUR] and published_runs(tmp_path / "satellite")[0] == "2026112623"
    # Nothing new: nothing read.
    assert ingest.ingest_latest(settings, client=client, catalog=SATELLITE) is None and len(client.read) == 1


def test_an_hour_that_fails_doesnt_hold_up_the_others(tmp_path, small):
    settings = small(tmp_path)
    hours = [datetime(2026, 11, 26, 22, tzinfo=timezone.utc) + k * HOUR for k in range(3)]
    client = FakeSatellite(hours, broken=hours[-1])
    assert ingest.ingest_latest(settings, client=client, catalog=SATELLITE) == "2026112623"
    assert published_runs(tmp_path / "satellite") == ["2026112622", "2026112623"]
    # Once it's there, it's caught up, and it becomes the latest.
    assert ingest.ingest_latest(settings, client=FakeSatellite(hours), catalog=SATELLITE) == "2026112700"
    assert read_latest(tmp_path / "satellite") == "2026112700"


def test_an_hour_is_served_like_any_run(tmp_path, small):
    settings = small(tmp_path)
    hours = [datetime(2026, 11, 26, 23, tzinfo=timezone.utc) + k * HOUR for k in range(2)]
    ingest.ingest_latest(settings, client=FakeSatellite(hours), catalog=SATELLITE)
    run = RunStore(tmp_path, "satellite").get()
    assert run.forecast_hours.tolist() == [0] and "GMGSI" in run.manifest["source"]["provider"]
    assert run.manifest["static"] == {}
    # EUMETSAT's credit for its Meteosat images, in the wording its data policy asks for.
    assert "Contains modified EUMETSAT Meteosat data 2026." in run.manifest["source"]["attribution"]

    _, body = point_forecast({"satellite": RunStore(tmp_path, "satellite")}, -8.5, 125.5, model="satellite",
                             tz="UTC", now=hours[-1] + timedelta(minutes=50))
    assert body["timeseries"]["sat_ir"] == [20.0] and body["timeseries"]["sat_vis"] == [100]
    assert body["timeseries"]["time"] == ["2026-11-27T00:00Z"]

    api = TestClient(create_app(settings))
    assert api.get("/v2/satellite/runs/latest").json()["grid"] == COARSE.to_dict()
    runs = api.get("/v2/satellite/runs").json()
    assert runs["latest"] == "2026112700" and [r["run"] for r in runs["runs"]] == ["2026112700", "2026112623"]
    meta = api.get("/v2/satellite/variables").json()
    assert meta["observed"] is True and "observed" not in api.get("/v2/gfs/variables").json()
    v = {x["id"]: x for x in meta["variables"]}
    assert v["sat_ir"]["tileable"] and v["sat_ir"]["units"] == "°C" and v["sat_vis"]["legend"]
    assert api.get("/v2/satellite/tiles/2026112623/sat_ir/f000/2/2/1.png").status_code == 200
    assert api.get("/v2/satellite/fields/latest/f000", params={"vars": "sat_ir"}).status_code == 200
