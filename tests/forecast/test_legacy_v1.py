"""The v1 compatibility layer must reproduce v1's shapes and native units from v2 data."""

import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from forecast import legacy_v1
from forecast.api import create_app
from forecast.config import Settings

V1_KEYS = [p.key for p in legacy_v1.PARAMS]


@pytest.fixture
def client(store_root):
    return TestClient(create_app(Settings(data_root=store_root)))


def _hours(run_init: datetime, iso_time: str) -> float:
    return (datetime.fromisoformat(iso_time) - run_init).total_seconds() / 3600


def test_base_time_rounds_to_nearest_hour():
    t = datetime(2026, 1, 1, 10, 29, tzinfo=timezone.utc)
    assert legacy_v1.base_time(t).hour == 10
    assert legacy_v1.base_time(t + timedelta(minutes=1)).hour == 11
    assert legacy_v1.closest_offset(5) == 6 and legacy_v1.closest_offset(-3) == 0


def test_forecast_shape_and_units(client, run_init):
    r = client.get("/legacy/v1/forecast", params={"lat": 10.0, "lon": 20.0})
    assert r.status_code == 200, r.text
    blocks = r.json()
    assert [b["metadata"]["key"] for b in blocks] == V1_KEYS
    lengths = {len(b["values"]) for b in blocks}
    assert len(lengths) == 1  # every parameter shares the same time axis

    by_key = {b["metadata"]["key"]: b for b in blocks}
    t2m = by_key["2-metre-temperature"]
    meta = t2m["metadata"]
    assert meta["shortName"] == "2t" and meta["parameterUnits"] == "K"
    assert meta["dataDate"] == int(run_init.strftime("%Y%m%d")) and meta["dataTime"] == run_init.hour * 100
    first = t2m["values"][0]
    assert set(first) == {"value", "datetime", "offset", "day", "hour"}
    assert first["offset"] == 0 and first["datetime"].endswith("+00:00")
    h = _hours(run_init, first["datetime"])
    assert meta["forecastStart"] == h
    assert first["value"] == pytest.approx(0.5 * 10 + 0.1 * h + 273.15, abs=0.06)
    offsets = [v["offset"] for v in t2m["values"]]
    assert offsets == list(range(0, 3 * len(offsets), 3))
    assert _hours(run_init, t2m["values"][-1]["datetime"]) <= 36  # synthetic run ends at +36 h

    ws = by_key["10-metre-wind-speed"]["values"][0]["value"]
    wd = by_key["10-metre-wind-direction"]["values"][0]["value"]
    assert ws == pytest.approx(5.0) and wd == pytest.approx(216.8699, abs=1e-3)
    assert by_key["precipitation-rate"]["values"][1]["value"] == pytest.approx(1 / 3600, abs=1e-4)


def test_precipitation_totals(client, run_init):
    blocks = {b["metadata"]["key"]: b for b in client.get("/legacy/v1/forecast", params={"lat": 0, "lon": 0}).json()}
    for tp, dtp in zip(blocks["total-precipitation"]["values"], blocks["daily-total-precipitation"]["values"]):
        valid = datetime.fromisoformat(tp["datetime"])
        h = _hours(run_init, tp["datetime"])
        midnight = valid.replace(hour=0)
        since_midnight = h - max((midnight - run_init).total_seconds() / 3600, 0)
        assert tp["value"] == pytest.approx(h, abs=1e-3)  # 1 mm/h since the run started
        assert dtp["value"] == pytest.approx(since_midnight, abs=1e-3)


def test_hour_offset_and_point(client, run_init):
    fc = client.get("/legacy/v1/forecast", params={"lat": 10, "lon": 20, "hour_offset": 5}).json()
    assert fc[0]["values"][0]["offset"] == 6

    r = client.get("/legacy/v1/point", params={"lat": 10, "lon": 20, "hour_offset": 3})
    assert r.status_code == 200
    items = r.json()
    assert [i["metadata"]["key"] for i in items] == V1_KEYS
    t2m = next(i for i in items if i["metadata"]["key"] == "2-metre-temperature")
    assert set(t2m) == {"unit", "metadata", "value", "datetime"} and t2m["unit"] == "K"
    h = _hours(run_init, t2m["datetime"])
    assert t2m["metadata"]["forecastTime"] == h
    assert legacy_v1.base_time() + timedelta(hours=3) == datetime.fromisoformat(t2m["datetime"])
    assert t2m["value"] == pytest.approx(0.5 * 10 + 0.1 * h + 273.15, abs=0.06)


def test_retired_posts_parameters_and_tiles(client):
    assert client.post("/legacy/v1/forecast", json={}).status_code == 410
    assert client.post("/legacy/v1/point", json={}).status_code == 410
    assert client.post("/legacy/v1/forecast-stream", json={}).status_code == 410
    assert client.get("/legacy/v1/parameters").json()["models"][0]["model"] == "gfs"

    r = client.get("/legacy/v1/tiles/gfs/2-metre-temperature/3/2/1/1.png",
                   params={"typeOfLevel": "heightAboveGround", "level": 2, "stepType": "instant"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (256, 256)
    assert client.get("/legacy/v1/tiles/gfs/cloud-mixing-ratio/0/0/0/0.png").status_code == 404
    assert client.get("/legacy/v1/tiles/gfswave/total-precipitation/0/0/0/0.png").status_code == 404


def test_legacy_routes_stay_out_of_v2_docs(client):
    paths = client.get("/v2/openapi.json").json()["paths"]
    assert not any(p.startswith("/legacy") for p in paths)
