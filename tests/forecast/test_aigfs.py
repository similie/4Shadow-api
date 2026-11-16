"""AIGFS, NOAA's AI model, as a second model: source layout, ingest, and serving next to GFS."""

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest
from conftest import COARSE, write_run
from fastapi.testclient import TestClient

from forecast import ingest
from forecast.api import create_app
from forecast.catalog import AIGFS
from forecast.config import Settings, parse_models
from forecast.service import point_forecast
from forecast.sources import SOURCES, aigfs
from forecast.sources.gfs import parse_idx, select_record
from forecast.store import RunStore

RUN = datetime(2026, 11, 15, 0, tzinfo=timezone.utc)

# Real lines from aigfs.t00z.sfc.f006/f012/f384.grib2.idx. f006 lists the 0-6 h total twice.
F006 = """1:0:d=2026111500:UGRD:100 m above ground:6 hour fcst:
9:6200000:d=2026111500:APCP:surface:0-6 hour acc fcst:
10:6400000:d=2026111500:APCP:surface:0-6 hour acc fcst:
"""
F384 = """9:6200000:d=2026111500:APCP:surface:378-384 hour acc fcst:
10:6400000:d=2026111500:APCP:surface:0-16 day acc fcst:
"""


def test_source_layout():
    assert aigfs.forecast_hours()[:3] == [0, 6, 12] and aigfs.forecast_hours()[-1] == 384
    assert len(aigfs.forecast_hours()) == 65 and aigfs.forecast_hours(24) == [0, 6, 12, 18, 24]
    assert aigfs.file_url("https://b", RUN, 6) == "https://b/aigfs.20261115/00/model/atmos/grib2/aigfs.t00z.sfc.f006.grib2"
    assert aigfs.file_url("https://b", RUN, 6, "pres").endswith("aigfs.t00z.pres.f006.grib2")
    assert SOURCES["aigfs"].grid == SOURCES["gfs"].grid


def test_precipitation_uses_the_total_since_the_start():
    rec, spec = select_record(parse_idx(F006), "APCP", "surface", "accum", 6)
    assert (spec.start, spec.end) == (0, 6)
    rec, spec = select_record(parse_idx(F384), "APCP", "surface", "accum", 384)
    assert (spec.start, spec.end) == (0, 384) and rec.num == 10


def test_parse_models():
    assert parse_models("gfs,aigfs") == ("gfs", "aigfs")
    assert parse_models("aigfs") == ("gfs", "aigfs")  # GFS is always served
    assert parse_models("gfs") == ("gfs",)
    with pytest.raises(ValueError):
        parse_models("gfs,ecmwf")


# --------------------------------------------------------------------------- ingest, end to end

FIELDS = {  # param:level -> value in GRIB units
    "TMP:2 m above ground": 300.0, "DPT:2 m above ground": 295.0,
    "UGRD:10 m above ground": 3.0, "VGRD:10 m above ground": 4.0,
    "UGRD:100 m above ground": 6.0, "VGRD:100 m above ground": 8.0,
    "PRMSL:mean sea level": 101300.0, "PRES:surface": 100000.0,
}
TOTALS = {6: 12.0, 12: 30.0}  # precipitation since the start, mm


def idx_text(fhr: int, drop: str = "") -> str:
    lines, n = [], 1
    for key in FIELDS:
        if key != drop:
            lines.append(f"{n}:{n * 1000}:d=2026111500:{key}:{'anl' if fhr == 0 else f'{fhr} hour fcst'}:")
            n += 1
    if fhr:
        lines.append(f"{n}:{n * 1000}:d=2026111500:APCP:surface:{fhr - 6}-{fhr} hour acc fcst:")
        lines.append(f"{n + 1}:{(n + 1) * 1000}:d=2026111500:APCP:surface:0-{fhr} hour acc fcst:")
    return "\n".join(lines) + "\n"


class FakeAigfs:
    base_url = "https://example.test"

    def idx(self, run, fhr, product="atmos"):
        assert product == "atmos"
        return parse_idx(idx_text(fhr, drop="PRES:surface" if fhr == 12 else ""))

    def fetch_records(self, run, fhr, records, product="atmos"):
        return {r.num: f"GRIB|{r.param}:{r.level}|{r.fcst}|7777".encode() for r in records}


def fake_decode(blob: bytes, grid) -> np.ndarray:
    _, key, fcst, _ = blob.decode().split("|")
    if key == "APCP:surface":
        start, end = fcst.split()[0].split("-")
        value = TOTALS[int(end)] - (TOTALS.get(int(start), 0.0) if start != "0" else 0.0)
    else:
        value = FIELDS[key]
    return np.full(grid.shape, value, np.float32)


def test_ingest_and_serve_an_aigfs_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.gfs, "decode_message", fake_decode)
    monkeypatch.setitem(ingest.SOURCES, "aigfs", dataclasses.replace(SOURCES["aigfs"], grid=COARSE))
    settings = Settings(data_root=tmp_path)
    write_run(tmp_path, RUN)  # a GFS run on the same grid, for the land mask and terrain height
    rid = ingest.ingest_run(RUN, settings, client=FakeAigfs(), catalog=AIGFS, max_fhr=12)
    run = RunStore(tmp_path, "aigfs").get()
    assert run.id == rid == "2026111500" and run.forecast_hours.tolist() == [0, 6, 12]
    assert "GraphCast" in run.manifest["source"]["provider"]
    assert run.static("land") is not None and run.static_meta["land"]["from"] == "gfs run 2026111500"

    stores = {m: RunStore(tmp_path, m) for m in ("gfs", "aigfs")}
    _, body = point_forecast(stores, -8.56, 125.57, model="aigfs", start="run", daily=False,
                             variables="t2m,rh2m,precip,sp,wind_speed_100m,heat_index")
    ts = body["timeseries"]
    assert ts["interval_hours"] == [0, 6, 6]
    assert ts["precip"] == [None, 2.0, 3.0]  # 12 mm in the first 6 h, 18 mm in the next
    assert ts["t2m"][0] == 26.9 and ts["rh2m"][0] == pytest.approx(74, abs=1)
    assert ts["sp"] == [1000.0, 1000.0, None]  # missing from one step's file: empty there, run still published
    assert ts["wind_speed_100m"][0] == 10.0 and ts["heat_index"][0] > ts["t2m"][0]

    api = TestClient(create_app(settings))
    r = api.get("/v2/aigfs/forecast", params={"lat": -8.56, "lon": 125.57, "vars": "t2m", "start": "run", "daily": False})
    assert r.status_code == 200 and r.json()["model"] == "aigfs" and r.json()["timeseries"]["t2m"][0] == 26.9


def test_health_and_routes_with_a_second_model(store_root):
    both = TestClient(create_app(Settings(data_root=store_root)))
    h = both.get("/v2/health")
    assert h.status_code == 200 and h.json()["status"] == "degraded"  # GFS is in; AIGFS not yet
    assert h.json()["models"]["aigfs"] == {"latest_run": None}
    assert both.get("/v2").json()["models"]["aigfs"].startswith("NOAA AIGFS")
    r = both.get("/v2/aigfs/forecast", params={"lat": 0, "lon": 0})
    assert r.status_code == 503 and "aigfs" in r.json()["detail"]
    # Each model has a path of its own, so a gateway can offer models to different plans. /v2/forecast
    # is GFS's, and model=gfs, accepted before models had paths, still works there.
    r = both.get("/v2/forecast", params={"lat": 0, "lon": 0, "model": "aigfs"})
    assert r.status_code == 400 and "/v2/aigfs/forecast" in r.json()["detail"]
    assert both.get("/v2/forecast", params={"lat": 0, "lon": 0, "model": "gfs"}).json()["model"] == "gfs"
    assert both.get("/v2/gfs/forecast", params={"lat": 0, "lon": 0}).json()["model"] == "gfs"
    assert "forecast_by_model" in both.get("/v2").json()["endpoints"]
    v = both.get("/v2/aigfs/variables").json()
    assert "GraphCast" in v["description"] and "rh2m" in {x["id"] for x in v["variables"]}

    gfs_only = TestClient(create_app(Settings(data_root=store_root, models=("gfs",))))
    assert gfs_only.get("/v2/health").json()["status"] == "ok"
    assert gfs_only.get("/v2/aigfs/forecast", params={"lat": 0, "lon": 0}).status_code == 404
    assert gfs_only.get("/v2/aigfs/runs").status_code == 404


def test_a_model_without_a_gfs_run_has_no_land_mask(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ingest.gfs, "decode_message", fake_decode)
    monkeypatch.setitem(ingest.SOURCES, "aigfs", dataclasses.replace(SOURCES["aigfs"], grid=COARSE))
    ingest.ingest_run(RUN, Settings(data_root=tmp_path), client=FakeAigfs(), catalog=AIGFS, max_fhr=6)
    run = RunStore(tmp_path, "aigfs").get()
    assert run.static("land") is None and run.land_mask() is None  # fire weather then shows everywhere
    assert "no land mask" in caplog.text
