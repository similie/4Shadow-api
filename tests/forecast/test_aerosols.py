"""Air quality from NOAA's GEFS-Aerosols: picking one species among messages that share a parameter,
and a run ingested and served end to end."""

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest
from conftest import COARSE
from fastapi.testclient import TestClient

from forecast import ingest
from forecast.api import create_app
from forecast.catalog import AEROSOLS
from forecast.config import Settings
from forecast.service import point_forecast
from forecast.sources import SOURCES, gefs_aerosols
from forecast.sources.gfs import parse_idx, select_record
from forecast.store import RunStore

RUN = datetime(2026, 11, 16, 6, tzinfo=timezone.utc)

# Real lines from gefs.chem.t00z.a2d_0p25.f003.grib2.idx (2026-11-16).
F003 = """\
5:4609256:d=2026111600:AOTK:entire atmosphere:3 hour fcst:aerosol=Total aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07
6:5645216:d=2026111600:SCTAOTK:entire atmosphere:3 hour fcst:aerosol=Total aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07
7:6678657:d=2026111600:AOTK:entire atmosphere:3 hour fcst:aerosol=Dust dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07
13:11132411:d=2026111600:AOTK:entire atmosphere:3 hour fcst:aerosol=Particulate organic matter dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07
15:12830360:d=2026111600:AOTK:entire atmosphere:3 hour fcst:aerosol=Black carbon dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07
17:14475849:d=2026111600:AOTK:entire atmosphere:3 hour fcst:aerosol=Total aerosol:aerosol_size <2e-05:aerosol_wavelength >=6.2e-07,<=6.7e-07
21:18838665:d=2026111600:PMTF:surface:3 hour fcst:aerosol=Dust dry:aerosol_size <2.5e-06:
22:19169400:d=2026111600:PMTC:surface:3 hour fcst:aerosol=Dust dry:aerosol_size <1e-05:
24:20442796:d=2026111600:PMTC:surface:3 hour fcst:aerosol=Total aerosol:aerosol_size <1e-05:
25:21376061:d=2026111600:PMTF:surface:3 hour fcst:aerosol=Total aerosol:aerosol_size <2.5e-06:
26:22225371:d=2026111600:COLMD:entire atmosphere:3 hour fcst:aerosol=Total aerosol:aerosol_size <1e-05:
"""


def test_each_species_is_its_own_message():
    recs = parse_idx(F003)
    assert recs[6].extra == "aerosol=Dust dry:aerosol_size <2.5e-06"  # the trailing ":" dropped
    picked = {v.id: select_record(recs, v.grib_param, v.grib_level, v.kind, 3, v.grib_extra)
              for v in AEROSOLS.stored_variables()}
    assert {vid: hit[0].num for vid, hit in picked.items()} == {
        "pm25": 25, "pm10": 24, "dust": 22, "aod": 5, "dust_aod": 7, "aod_om": 13, "aod_bc": 15}
    # Without the species, the first message with that parameter and level would be taken: dust, not total.
    assert select_record(recs, "PMTF", "surface", "instant", 3)[0].num == 21


def test_file_layout():
    assert gefs_aerosols.file_url("https://b", RUN, 3) == (
        "https://b/gefs.20261116/06/chem/pgrb2ap25/gefs.chem.t06z.a2d_0p25.f003.grib2")
    assert gefs_aerosols.forecast_hours()[-1] == 120 and len(gefs_aerosols.forecast_hours()) == 41
    assert SOURCES["aerosols"].grid == SOURCES["gfs"].grid


VALUES = {  # num in F003 -> value (µg/m³ or optical depth)
    25: 40.0, 24: 90.0, 22: 30.0, 5: 0.5, 7: 0.1, 13: 0.2, 15: 0.05,
}


class FakeAerosols:
    base_url = "https://example.test"

    def idx(self, run, fhr, product="atmos"):
        return parse_idx(F003.replace("3 hour fcst", "anl" if fhr == 0 else f"{fhr} hour fcst"))

    def fetch_records(self, run, fhr, records, product="atmos"):
        return {r.num: f"GRIB|{r.num}|{fhr}|7777".encode() for r in records}


def fake_decode(blob: bytes, grid) -> np.ndarray:
    _, num, fhr, _ = blob.decode().split("|")
    return np.full(grid.shape, VALUES[int(num)] * (1 + int(fhr) / 3), np.float32)


def test_ingest_and_serve_air_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.gfs, "decode_message", fake_decode)
    monkeypatch.setitem(ingest.SOURCES, "aerosols", dataclasses.replace(SOURCES["aerosols"], grid=COARSE))
    settings = Settings(data_root=tmp_path, models=("gfs", "aerosols"))
    ingest.ingest_run(RUN, settings, client=FakeAerosols(), catalog=AEROSOLS, max_fhr=6)
    run = RunStore(tmp_path, "aerosols").get()
    assert run.forecast_hours.tolist() == [0, 3, 6] and "GEFS-Aerosols" in run.manifest["source"]["provider"]

    _, body = point_forecast({"aerosols": RunStore(tmp_path, "aerosols")}, 28.6, 77.2, model="aerosols",
                             start="run", tz="UTC")
    ts = body["timeseries"]
    assert ts["pm25"] == [40.0, 80.0, 120.0] and ts["pm10"][0] == 90.0 and ts["dust"][0] == 30.0
    assert ts["aod"][0] == 0.5 and ts["smoke_aod"] == [pytest.approx(0.25), pytest.approx(0.5), pytest.approx(0.75)]
    assert body["daily"]["pm25_max"] == [120.0] and body["units"]["pm25"] == "µg/m³"

    api = TestClient(create_app(settings))
    r = api.get("/v2/aerosols/forecast", params={"lat": 28.6, "lon": 77.2, "vars": "pm25", "start": "run", "daily": False})
    assert r.status_code == 200 and r.json()["timeseries"]["pm25"][0] == 40.0
    v = {x["id"]: x for x in api.get("/v2/aerosols/variables").json()["variables"]}
    assert v["smoke_aod"]["derived_from"] == ["aod_om", "aod_bc"] and v["pm25"]["tileable"]
    assert api.get("/v2/aerosols/tiles/latest/pm25/f003/1/1/0.png").status_code == 200
