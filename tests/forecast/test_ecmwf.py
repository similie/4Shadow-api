"""ECMWF IFS and AIFS from ECMWF's open data: the index format, file layout, the turn onto GFS's grid,
and a run ingested and served end to end."""

import dataclasses
import json
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from conftest import COARSE
from fastapi.testclient import TestClient

from forecast import ingest
from forecast.api import create_app
from forecast.catalog import AIFS, GFS, IFS
from forecast.config import Settings
from forecast.grid import Grid
from forecast.service import point_forecast
from forecast.sources import SOURCES, ecmwf
from forecast.sources.gfs import GribClient, decode_message, select_record
from forecast.store import RunStore

RUN = datetime(2026, 11, 16, 0, tzinfo=timezone.utc)

# Real lines from 20261116000000-6h-oper-fc.index (IFS 0.25°, 2026-11-16 00 UTC), out of offset order as there.
INDEX_6H = """\
{"domain": "g", "date": "20261116", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "step": "6", "levelist": "1", "levtype": "sol", "param": "vsw", "_offset": 18536762, "_length": 315532}
{"domain": "g", "date": "20261116", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "levtype": "sfc", "step": "6", "param": "tp", "_offset": 13639046, "_length": 316452}
{"domain": "g", "date": "20261116", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "step": "6", "levtype": "sfc", "param": "2t", "_offset": 23750731, "_length": 311449}
{"domain": "g", "date": "20261116", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "step": "6", "levelist": "850", "levtype": "pl", "param": "z", "_offset": 41332202, "_length": 898681}
{"domain": "g", "date": "20261116", "time": "0000", "expver": "0001", "class": "od", "type": "fc", "stream": "oper", "levtype": "sfc", "step": "6", "param": "10fg", "_offset": 8256726, "_length": 644102}
"""


def test_index_format():
    recs = ecmwf.parse_index(INDEX_6H)
    assert [r.param for r in recs] == ["10fg", "tp", "vsw", "2t", "z"]  # by offset
    two_t = recs[3]
    assert (two_t.start, two_t.end, two_t.level) == (23750731, 23750731 + 311449 - 1, "sfc")
    rec, spec = select_record(recs, "tp", "sfc", "accum", 6)  # a total since the start, like GFS's APCP
    assert (spec.start, spec.end) == (0, 6)
    assert select_record(recs, "2t", "sfc", "instant", 6)[0] is two_t
    assert select_record(recs, "vsw", "sol:1", "instant", 6)[0].start == 18536762
    assert select_record(recs, "z", "sfc", "instant", 6) is None  # only on pressure levels at this step


def test_file_layout():
    url = ecmwf.ifs_file_url("https://b", RUN, 6)
    assert url == "https://b/20261116/00z/ifs/0p25/oper/20261116000000-6h-oper-fc.grib2"
    assert ecmwf.IFS.index_url(url) == "https://b/20261116/00z/ifs/0p25/oper/20261116000000-6h-oper-fc.index"
    assert ecmwf.ifs_file_url("https://b", RUN, 6, "wave").endswith("ifs/0p25/wave/20261116000000-6h-wave-fc.grib2")
    assert ecmwf.aifs_file_url("https://b", RUN + timedelta(hours=6), 360).endswith(
        "20261116/06z/aifs-single/0p25/oper/20261116060000-360h-oper-fc.grib2")
    hours = ecmwf.ifs_forecast_hours()
    assert len(hours) == 85 and hours[47:50] == [141, 144, 150] and hours[-1] == 360
    assert ecmwf.aifs_forecast_hours(24) == [0, 6, 12, 18, 24]
    assert SOURCES["ifs"].grid == SOURCES["aifs"].grid == SOURCES["gfs"].grid


def test_latest_ifs_run_is_a_long_one():
    """IFS's 06 and 18 UTC runs stop at 144 h: only 00 and 12 UTC runs are looked for."""
    client = GribClient(source=SOURCES["ifs"])
    asked = []

    def has_idx(run, fhr, product="atmos"):
        asked.append((run.hour, fhr))
        return run.hour == 0

    client.has_idx = has_idx
    run = client.latest_complete_run(now=datetime(2026, 11, 16, 20, 30, tzinfo=timezone.utc))
    assert run == RUN and asked == [(12, 360), (0, 360)]


class FakeMessage(dict):
    def __init__(self, values, west, north=90.0):
        super().__init__(iScansNegatively=0, jScansPositively=0, latitudeOfFirstGridPointInDegrees=north,
                         latitudeOfLastGridPointInDegrees=-north, longitudeOfFirstGridPointInDegrees=west)
        self.values = values


def test_decoding_turns_a_field_to_start_at_0(monkeypatch):
    grid = Grid(lat0=90.0, dlat=45.0, nlat=5, lon0=0.0, dlon=90.0, nlon=4)
    columns = np.array([180.0, 270.0, 0.0, 90.0])  # ECMWF's order: from 180° east
    message = {}
    monkeypatch.setitem(sys.modules, "pygrib", types.SimpleNamespace(fromstring=lambda blob: message["m"]))

    message["m"] = FakeMessage(np.tile(columns, (5, 1)), west=180.0)
    assert decode_message(b"", grid)[0].tolist() == [0.0, 90.0, 180.0, 270.0]
    message["m"] = FakeMessage(np.tile(np.array([-180.0, -90.0, 0.0, 90.0]), (5, 1)), west=-180.0)
    assert decode_message(b"", grid)[0].tolist() == [0.0, 90.0, -180.0, -90.0]
    message["m"] = FakeMessage(np.tile(np.arange(4.0), (5, 1)), west=0.0)  # GFS's: unchanged
    assert decode_message(b"", grid)[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    message["m"] = FakeMessage(np.tile(np.arange(4.0), (5, 1)), west=45.0)  # between grid columns
    with pytest.raises(ValueError, match="origin"):
        decode_message(b"", grid)
    regional = Grid(lat0=90.0, dlat=45.0, nlat=5, lon0=0.0, dlon=10.0, nlon=4)  # doesn't wrap
    message["m"] = FakeMessage(np.zeros((5, 4)), west=180.0)
    with pytest.raises(ValueError, match="origin"):
        decode_message(b"", regional)


def test_ecmwf_variables_match_gfs():
    """Same ids, units and meaning as GFS, so a client can ask any model for t2m; each model's files have
    their own unit factors (IFS: precipitation in m and cloud 0-1; AIFS: kg/m² and %)."""
    for cat in (IFS, AIFS):
        for v in cat.variables.values():
            if v.id in GFS.variables:
                assert v.units == GFS.variables[v.id].units, (cat.model, v.id)
    assert (IFS.variables["precip"].mul, IFS.variables["tcc"].mul) == (1000.0, 100.0)
    assert (AIFS.variables["precip"].mul, AIFS.variables["tcc"].mul) == (1.0, 1.0)
    assert IFS.variables["dswrf"].kind == "accum" and IFS.variables["wave_height"].product == "wave"
    assert "gust" not in AIFS.variables and "wave_height" not in AIFS.variables


# --------------------------------------------------------------------------- ingest, end to end

ATMOS = {  # (param, level) -> value in GRIB units
    ("2t", "sfc"): 300.0, ("2d", "sfc"): 295.0, ("10u", "sfc"): 3.0, ("10v", "sfc"): 4.0,
    ("100u", "sfc"): 6.0, ("100v", "sfc"): 8.0, ("msl", "sfc"): 101300.0, ("sp", "sfc"): 100000.0,
    ("tcc", "sfc"): 0.5, ("lcc", "sfc"): 0.25, ("mcc", "sfc"): 0.0, ("hcc", "sfc"): 1.0,
    ("vsw", "sol:1"): 0.3, ("sot", "sol:1"): 290.0, ("10fg", "sfc"): 12.0, ("mucape", "sfc"): 800.0,
    ("tcwv", "sfc"): 40.0,
}
TOTALS = {("tp", "sfc"): {0: 0.0, 3: 0.003, 6: 0.009},  # metres since the start: 1 mm/h, then 2 mm/h
          ("ssrd", "sfc"): {0: 0.0, 3: 3 * 3600 * 100.0, 6: 3 * 3600 * 300.0}}  # J/m²: 100 W/m², then 200
STATIC = {("z", "sfc"): 500 * 9.80665, ("lsm", "sfc"): 1.0}
WAVE = {("swh", "sfc"): 2.5, ("pp1d", "sfc"): 11.0, ("mwd", "sfc"): 270.0}


class FakeEcmwf:
    base_url = "https://example.test"

    def __init__(self):
        self.fetched = []

    def idx(self, run, fhr, product="atmos"):
        fields = dict(WAVE) if product == "wave" else {**ATMOS, **TOTALS, **(STATIC if fhr == 0 else {})}
        lines = []
        for n, (param, level) in enumerate(fields):
            row = {"param": param, "levtype": level.split(":")[0], "step": str(fhr), "_offset": n * 100, "_length": 100}
            if ":" in level:
                row["levelist"] = level.split(":")[1]
            lines.append(json.dumps(row))
        return ecmwf.parse_index("\n".join(lines))

    def fetch_records(self, run, fhr, records, product="atmos"):
        self.fetched += [(fhr, r.param) for r in records]
        return {r.num: f"GRIB|{product}|{r.param}|{r.level}|{fhr}|7777".encode() for r in records}


def fake_decode(blob: bytes, grid) -> np.ndarray:
    _, product, param, level, fhr, _ = blob.decode().split("|")
    key = (param, level)
    value = (WAVE if product == "wave" else {**ATMOS, **STATIC}).get(key)
    if value is None:
        value = TOTALS[key][int(fhr)]
    return np.full(grid.shape, value, np.float32)


def test_ingest_and_serve_an_ifs_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.gfs, "decode_message", fake_decode)
    monkeypatch.setitem(ingest.SOURCES, "ifs", dataclasses.replace(SOURCES["ifs"], grid=COARSE))
    settings = Settings(data_root=tmp_path)
    client = FakeEcmwf()
    rid = ingest.ingest_run(RUN, settings, client=client, catalog=IFS, max_fhr=6)
    assert (0, "tp") not in client.fetched and (3, "tp") in client.fetched  # no total at the start
    run = RunStore(tmp_path, "ifs").get()
    assert rid == "2026111600" and run.forecast_hours.tolist() == [0, 3, 6]
    assert "CC BY 4.0" in run.manifest["source"]["license"] and "ECMWF" in run.manifest["source"]["attribution"]
    assert float(run.static("orog")[0, 0]) == pytest.approx(500, abs=0.01) and float(run.static("land")[0, 0]) == 1.0
    assert "from" not in run.static_meta["land"]  # its own land mask, not GFS's

    stores = {"ifs": RunStore(tmp_path, "ifs")}
    _, body = point_forecast(stores, -8.56, 125.57, model="ifs", start="run", daily=False,
                             variables="t2m,rh2m,precip,dswrf,tcc,lcc,soil_moisture,soil_temp,gust,cape,pwat,wave_height,mslp")
    ts = body["timeseries"]
    assert ts["interval_hours"] == [0, 3, 3]
    assert ts["precip"] == [None, 1.0, 2.0] and ts["dswrf"] == [None, 100.0, 200.0]
    assert ts["t2m"][0] == 26.9 and ts["rh2m"][0] == pytest.approx(74, abs=1)
    assert (ts["tcc"][0], ts["lcc"][0], ts["mslp"][0]) == (50.0, 25.0, 1013.0)
    assert (ts["soil_moisture"][0], ts["soil_temp"][0], ts["gust"][0], ts["cape"][0]) == (0.3, 16.9, 12.0, 800.0)
    assert ts["wave_height"] == [2.5, 2.5, 2.5]

    api = TestClient(create_app(settings))
    r = api.get("/v2/ifs/forecast", params={"lat": -8.56, "lon": 125.57, "vars": "t2m", "start": "run", "daily": False})
    assert r.status_code == 200 and r.json()["model"] == "ifs" and "ECMWF" in r.json()["source"]["attribution"]
    v = api.get("/v2/ifs/variables").json()
    assert v["title"] == "ECMWF IFS 0.25°" and "CC BY 4.0" in v["description"]


def test_health_allows_each_model_its_own_delay(tmp_path):
    """IFS runs twice a day and its last step is out ~7.5 h after the run: 20 h old is not stale for it."""
    from forecast.store import RunWriter

    old = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=20)
    for model, cat in (("gfs", GFS), ("ifs", IFS)):
        w = RunWriter(tmp_path / model, model, old, COARSE, [0, 3], cat.stored_variables())
        for v in cat.stored_variables():
            for k in (0, 1):
                w.write(v.id, k, None)
        w.publish()
    h = TestClient(create_app(Settings(data_root=tmp_path, models=("gfs", "ifs")))).get("/v2/health").json()
    assert h["models"]["gfs"]["stale"] is True and h["models"]["ifs"]["stale"] is False
