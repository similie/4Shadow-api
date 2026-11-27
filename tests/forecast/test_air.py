"""Air quality from CAMS: the US AQI against EPA's worked examples, its averaging windows, which run the ADS
has, a run asked for on one poll and read on a later one, and a run served end to end."""

import dataclasses
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import requests
from conftest import COARSE
from ecmwf.datastores.processing import ProcessingFailedError
from fastapi.testclient import TestClient

from forecast import aqi, config, ingest
from forecast.api import create_app
from forecast.catalog import AIR
from forecast.config import Settings
from forecast.service import point_forecast
from forecast.sources import SOURCES, cams
from forecast.store import RunStore, read_latest

RUN = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)


def one(**values):
    """The AQI of single values: clean air except what's given."""
    clean = {"pm25_24h": 0, "pm10_24h": 0, "o3_8h": 0, "o3_1h": 0, "co_8h": 0, "so2_1h": 0, "so2_24h": 0, "no2_1h": 0}
    return float(aqi.us_aqi(**{k: np.array([v]) for k, v in {**clean, **values}.items()})[0])


def test_the_aqi_matches_epas_worked_examples():
    # The technical document's examples: an 8-hour ozone of 0.07853333 ppm truncates to 0.078, index 126.
    assert one(o3_8h=0.07853333) == 126
    # Ozone 0.078 ppm (126), PM2.5 35.9 µg/m³ (102) and CO 8.4 ppm (90): the highest, 126.
    assert one(pm25_24h=35.9) == 102 and one(co_8h=8.4) == 90
    assert one(o3_8h=0.078, pm25_24h=35.9, co_8h=8.4) == 126
    # A 1-hour ozone of 0.162 ppm counts when it's the higher; one of 0.104 ppm, below 0.125, doesn't. (The
    # document prints 148 for it, but its Equation 1 gives 49 / 0.039 x 0.037 + 101 = 147.49.)
    assert one(o3_8h=0.078, o3_1h=0.162) == 147 and one(o3_8h=0.078, o3_1h=0.104) == 126
    # The breakpoints' edges, after truncation: 9.05 µg/m³ is 9.0 (good, 50), 9.1 moderate (51).
    assert one(pm25_24h=9.05) == 50 and one(pm25_24h=9.1) == 51 and one(no2_1h=100.9) == 100
    assert one(no2_1h=101) == 101 and one(pm10_24h=154.9) == 100


def test_sulphur_dioxide_over_304_ppb_takes_the_days_mean():
    assert one(so2_1h=304, so2_24h=10) == 200
    # Over 304 ppb in the hour: the 24-hour mean from 305 ppb, or 200 if that's lower.
    assert one(so2_1h=400, so2_24h=200) == 200
    assert one(so2_1h=400, so2_24h=400) == round((300 - 201) / (604 - 305) * (400 - 305) + 201)
    # An 8-hour ozone over 0.200 ppm defines no index of 301 or more: 300, or the 1-hour value's index.
    assert one(o3_8h=0.25) == 300 and one(o3_8h=0.25, o3_1h=0.5) == round(199 / 0.199 * 0.095 + 301)


def test_above_500_the_last_segment_goes_on():
    assert one(pm25_24h=325.4) == 500
    assert one(pm25_24h=425.4) == round((500 - 301) / (325.4 - 225.5) * (425.4 - 225.5) + 301)
    assert one(pm25_24h=np.nan) != one(pm25_24h=np.nan)  # missing stays missing


def test_the_averages_start_with_the_forecast():
    day = aqi.Trailing(3)
    assert [float(day.push(np.array([v]))[0]) for v in (3, 6, 9, 12, 0)] == [3, 4.5, 6, 9, 7]


def test_gases_are_turned_into_volume_mixing_ratios():
    # 100 ppb of nitrogen dioxide (the top of "moderate") as a mass mixing ratio.
    mmr = 100e-9 * aqi.MOLAR_MASS["no2"] / aqi.AIR
    hourly = aqi.HourlyAqi()
    zero = np.zeros(1)
    assert float(hourly.push(pm25=zero, pm10=zero, o3=zero, co=zero, so2=zero, no2=np.array([mmr]))[0]) == 100


# ---------------------------------------------------------------------- the ADS, faked

LEVEL_B = (0.99500251, 0.99763012, 1.0)  # the last three half levels' b, as in CAMS's files
SURFACE_PA, KELVIN = 100000.0, 290.0
MMR = {"no2": 30e-9, "co": 200e-9, "go3": 50e-9, "so2": 5e-9}  # kg/kg
PM = {"pm2p5": 12e-9, "pm10": 20e-9}  # kg/m³


def _message(name: str, step: int) -> bytes:
    """A stand-in message: GRIB2 for the gases, GRIB1 for the surface fields, as in CAMS's files."""
    body = json.dumps({"name": name, "step": step}).encode()
    if name in ("pm2p5", "pm10", "sp"):
        return b"GRIB" + (16 + len(body) + 4).to_bytes(3, "big") + b"\x01" + b"\0" * 8 + body + b"7777"
    return b"GRIB\0\0\0\x02" + (16 + len(body) + 4).to_bytes(8, "big") + body + b"7777"


class FakeMessage(dict):
    def __init__(self, blob: bytes):
        meta = json.loads(blob[16:-4])
        hybrid = meta["name"] in (*MMR, "t")
        super().__init__(pv=[0.0] * 135 + [3.7578, 0.0, 0.0] + [0.9] * 135 + list(LEVEL_B))
        self.shortName, self.step, self.typeOfLevel = meta["name"], meta["step"], "hybrid" if hybrid else "surface"


def _decode(blob: bytes, grid) -> np.ndarray:
    meta = json.loads(blob[16:-4])
    value = {**MMR, **PM, "t": KELVIN, "sp": SURFACE_PA}[meta["name"]]
    if meta["name"] == "no2":
        value *= 1 + meta["step"]  # more each hour
    return np.full(grid.shape, value, np.float32)


class FakeAds:
    def __init__(self, have=(RUN,), status="accepted", drop=None):
        self.have, self.status, self.drop, self.broken = set(have), status, drop, 0
        self.submitted: list[dict] = []
        self.deleted: list[str] = []

    def apply_constraints(self, dataset, req):
        assert dataset == cams.DATASET
        run = datetime.strptime(f"{req['date'][0][:10]} {req['time'][0]}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if run not in self.have:
            return {"time": ["00:00"] if run.hour == 12 else [], "leadtime_hour": []}
        return {"time": ["00:00", "12:00"], "leadtime_hour": [str(h) for h in range(121)]}

    def submit(self, dataset, req):
        self.submitted.append(req)
        return FakeRemote(self, f"job{len(self.submitted)}")

    def get_remote(self, request_id):
        return FakeRemote(self, request_id)


class FakeRemote:
    def __init__(self, ads: FakeAds, request_id: str):
        self.ads, self.request_id = ads, request_id

    @property
    def results_ready(self):
        if self.ads.status == "failed":
            raise ProcessingFailedError("The job failed\nTraceback ...")
        return self.ads.status == "successful"

    def download(self, target):
        if self.ads.broken:
            self.ads.broken -= 1
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
        req = self.ads.submitted[-1]
        with open(target, "wb") as fh:
            for h in map(int, req["leadtime_hour"]):
                for name in cams.FIELDS.values():
                    if (name, h) != self.ads.drop:
                        fh.write(_message(name, h))
        return target

    def delete(self):
        self.ads.deleted.append(self.request_id)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setitem(ingest.SOURCES, "air", dataclasses.replace(SOURCES["air"], grid=COARSE))
    monkeypatch.setitem(__import__("sys").modules, "pygrib", type("pygrib", (), {"fromstring": FakeMessage}))
    monkeypatch.setattr(cams, "decode_message", _decode)
    return lambda root: Settings(data_root=root, models=("gfs", "air"))


def _client(root, ads, now=datetime(2026, 11, 27, 5, tzinfo=timezone.utc)):
    return cams.AtmosphereClient(root / "air", ads=ads, now=lambda: now)


def test_the_newest_run_the_ads_has(tmp_path):
    ads = FakeAds(have={RUN, datetime(2026, 11, 26, 0, tzinfo=timezone.utc)})
    # At 05 UTC today's 00 UTC run isn't out: yesterday's 12 UTC is the newest. Today's isn't looked for.
    assert _client(tmp_path, ads).latest_run() == RUN
    ads.have.add(datetime(2026, 11, 27, 0, tzinfo=timezone.utc))
    assert _client(tmp_path, ads).latest_run() == RUN
    assert _client(tmp_path, ads, now=datetime(2026, 11, 27, 6, tzinfo=timezone.utc)).latest_run().hour == 0
    assert _client(tmp_path, FakeAds(have=())).latest_run() is None
    req = cams.request(RUN, [0, 1])
    assert req["date"] == ["2026-11-26/2026-11-26"] and req["time"] == ["12:00"] and req["leadtime_hour"] == ["0", "1"]
    assert req["model_level"] == ["137"] and "particulate_matter_2.5um" in req["variable"]


def test_a_run_is_asked_for_once_and_read_when_ready(tmp_path, small, caplog):
    settings, ads = small(tmp_path), FakeAds()
    client = _client(tmp_path, ads)
    with caplog.at_level(logging.INFO):
        assert ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=3) is None
        assert ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=3) is None
    # Asked once, then waited for: no error, nothing published, and the other models went on.
    assert len(ads.submitted) == 1 and ads.submitted[0]["leadtime_hour"] == ["0", "1", "2", "3"]
    assert "asked the ADS for CAMS run 2026112612" in caplog.text and "the ADS is preparing" in caplog.text
    assert "ERROR" not in caplog.text and read_latest(tmp_path / "air") is None

    # A restart picks up the same request.
    ads.status = "successful"
    assert ingest.ingest_latest(settings, client=_client(tmp_path, ads), catalog=AIR, max_fhr=3) == "2026112612"
    assert len(ads.submitted) == 1 and ads.deleted == ["job1"]
    assert not list((tmp_path / "air" / ".ads").iterdir())  # the download goes once it's read

    run = RunStore(tmp_path, "air").get()
    assert run.forecast_hours.tolist() == [0, 1, 2, 3] and run.grid == COARSE and run.manifest["static"] == {}
    assert "Contains modified Copernicus Atmosphere Monitoring Service information 2026" in run.manifest["source"]["attribution"]

    # µg/m³ from the air's density at the lowest level: its pressure is the mean of its half levels'.
    density = SURFACE_PA * (LEVEL_B[1] + LEVEL_B[2]) / 2 / (cams.R_DRY * KELVIN)
    _, body = point_forecast({"air": RunStore(tmp_path, "air")}, -8.5, 125.5, model="air", start="run", tz="UTC")
    ts = body["timeseries"]
    assert ts["no2"] == [round(MMR["no2"] * (1 + h) * density * 1e9, 1) for h in range(4)]
    assert ts["co"][0] == round(MMR["co"] * density * 1e9) and ts["pm25"][0] == 12 and ts["pm10"][0] == 20
    # The AQI each hour: PM2.5's (56) until nitrogen dioxide, rising (19, 38, 57 then 76 ppb), passes it.
    ppb = [MMR["no2"] * (1 + h) * aqi.AIR / aqi.MOLAR_MASS["no2"] * 1e9 for h in range(4)]
    assert ts["aqi"] == [one(no2_1h=p, pm25_24h=12, pm10_24h=20) for p in ppb] and ts["aqi"][0] < ts["aqi"][3]
    assert body["daily"]["aqi_max"] == [max(ts["aqi"])] and body["units"]["no2"] == "µg/m³"

    api = TestClient(create_app(settings))
    v = {x["id"]: x for x in api.get("/v2/air/variables").json()["variables"]}
    assert v["aqi"]["tileable"] and v["no2"]["units"] == "µg/m³" and v["aqi"]["legend"]["stops"][-1]["color"] == "#7e0023ff"
    assert api.get("/v2/air/runs/latest").json()["grid"] == COARSE.to_dict()
    assert api.get("/v2/air/tiles/latest/aqi/f003/1/1/0.png").status_code == 200
    assert api.get("/v2/health").json()["models"]["air"]["latest_run"] == "2026112612"
    # Up to date: nothing asked.
    assert ingest.ingest_latest(settings, client=_client(tmp_path, ads), catalog=AIR, max_fhr=3) is None
    assert len(ads.submitted) == 1


def test_a_failed_request_is_asked_again_after_a_wait(tmp_path, small, caplog):
    settings, ads = small(tmp_path), FakeAds(status="failed")
    clock = [datetime(2026, 11, 27, 5, tzinfo=timezone.utc)]
    client = cams.AtmosphereClient(tmp_path / "air", ads=ads, now=lambda: clock[0])

    def poll(minutes=10):
        ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1)
        clock[0] += timedelta(minutes=minutes)

    caplog.set_level(logging.INFO)
    poll()  # asks
    poll()  # failed
    assert "The ADS failed CAMS run 2026112612 (request job1, failure 1): The job failed" in caplog.text
    # The ADS disables accounts that keep sending requests it can't serve: an hour, then two, then no more.
    for _ in range(5):
        poll()
    assert len(ads.submitted) == 1 and "asking again after 06:10 UTC" in caplog.text
    poll(120)  # 06:10: asks again
    poll()  # failed again
    assert len(ads.submitted) == 2
    poll(180)
    poll()
    poll()  # the third failure
    poll(600)
    assert len(ads.submitted) == 3 and "failed CAMS run 2026112612 3 times: waiting for the next run" in caplog.text
    # The next run starts afresh.
    ads.have.add(datetime(2026, 11, 27, 0, tzinfo=timezone.utc))
    ads.status = "successful"
    poll()
    poll()
    assert len(ads.submitted) == 4 and read_latest(tmp_path / "air") == "2026112700"


def test_a_download_without_every_field_is_asked_for_again(tmp_path, small):
    settings, ads = small(tmp_path), FakeAds(status="successful", drop=("so2", 1))
    clock = [datetime(2026, 11, 27, 5, tzinfo=timezone.utc)]
    client = cams.AtmosphereClient(tmp_path / "air", ads=ads, now=lambda: clock[0])
    ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1)  # asks
    with pytest.raises(RuntimeError, match="lacks 1 fields"):
        ingest.ingest_run(RUN, settings, client=client, catalog=AIR, max_fhr=1)
    assert read_latest(tmp_path / "air") is None and not (tmp_path / "air" / ".ads" / "2026112612.grib").exists()
    ads.drop = None
    clock[0] += timedelta(hours=1)
    ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1)  # asks again
    assert ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1) == "2026112612"
    assert len(ads.submitted) == 2


def test_a_broken_download_is_tried_again(tmp_path, small, caplog):
    settings, ads = small(tmp_path), FakeAds(status="successful")
    ads.broken = 1
    client = _client(tmp_path, ads)
    ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1)  # asks
    with caplog.at_level(logging.INFO):
        assert ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1) is None
    assert "the download broke off" in caplog.text and "ERROR" not in caplog.text
    # The same request, downloaded again.
    assert ingest.ingest_latest(settings, client=client, catalog=AIR, max_fhr=1) == "2026112612"
    assert len(ads.submitted) == 1


def test_air_is_served_by_default_only_with_a_login(monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("FORECAST_MODELS", raising=False)
    assert "air" not in config.get_settings().models
    monkeypatch.setenv("CDSAPI_KEY", "x")
    assert "air" in config.get_settings().models


def test_the_client_needs_a_login(monkeypatch, tmp_path):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CDSAPI_KEY"):
        cams.AtmosphereClient(tmp_path).latest_run()


def test_grib_messages_are_found_by_their_lengths(tmp_path):
    path = tmp_path / "x.grib"
    path.write_bytes(_message("no2", 0) + _message("sp", 12))
    first, second = cams.grib_messages(path)
    assert first[0] == 0 and second[0] == first[1] and sum(n for _, n in (first, second)) == path.stat().st_size
    path.write_bytes(_message("no2", 0) + b"junk")
    with pytest.raises(ValueError, match="no GRIB message"):
        cams.grib_messages(path)
