import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from forecast.api import create_app
from forecast.config import Settings


@pytest.fixture
def client(store_root):
    return TestClient(create_app(Settings(data_root=store_root)))


def test_health_and_runs(client, run_init):
    r = client.get("/v2/health")
    assert r.status_code == 200 and r.json()["models"]["gfs"]["stale"] is False
    runs = client.get("/v2/gfs/runs").json()
    assert runs["latest"] == runs["runs"][0]["run"] and runs["runs"][0]["steps"] == 29


def test_health_reports_unavailable_without_data(tmp_path):
    c = TestClient(create_app(Settings(data_root=tmp_path)))
    assert c.get("/v2/health").status_code == 503
    r = c.get("/v2/forecast", params={"lat": 0, "lon": 0})
    assert r.status_code == 503 and r.headers["retry-after"] == "600"


def test_forecast_values(client):
    r = client.get("/v2/forecast", params={"lat": 10.3, "lon": 20.7, "start": "run", "tz": "UTC",
                                           "vars": "t2m,wind_speed,wind_dir,precip"})
    assert r.status_code == 200, r.text
    body = r.json()
    ts = body["timeseries"]
    assert ts["t2m"][:2] == [pytest.approx(5.15, abs=0.06), pytest.approx(5.25, abs=0.06)]
    assert ts["t2m"][24] == pytest.approx(0.5 * 10.3 + 2.4, abs=0.06)
    assert set(ts["wind_speed"]) == {5.0}
    assert set(ts["wind_dir"]) == {217.0}  # u=3 (eastward), v=4 (northward): wind from the south-west
    assert ts["precip"][0] is None and ts["precip"][1] == 1.0
    assert ts["interval_hours"][:3] == [0, 1, 1] and ts["interval_hours"][-1] == 3
    assert body["units"]["t2m"] == "°C" and body["timezone"]["name"] == "UTC"
    assert body["location"]["model_elevation_m"] == pytest.approx(41.4, abs=0.01)
    assert body["current"]["wind_speed"] == 5.0
    assert r.headers["x-forecast-run"] == body["run"]


def test_forecast_daily_precip_is_rate_times_hours(client):
    body = client.get("/v2/forecast", params={"lat": 0, "lon": 0, "start": "run", "tz": "UTC"}).json()
    daily = body["daily"]
    for covered, total in zip(daily["hours_covered"], daily["precip_sum"]):
        assert total == pytest.approx(covered * 1.0)
    assert sum(daily["hours_covered"]) == 36
    assert daily["dswrf_sum"][0] == pytest.approx(daily["hours_covered"][0] * 200 * 0.0036, abs=0.06)


def test_forecast_start_now_trims_past_steps(client, run_init):
    body = client.get("/v2/forecast", params={"lat": 0, "lon": 0, "vars": "t2m", "hours": 6}).json()
    times = body["timeseries"]["time"]
    assert len(times) == 7 and times[0].endswith(":00Z")
    assert body["timeseries"]["t2m"][0] == pytest.approx(0.2, abs=0.01)  # run_init + 2 h


def test_forecast_validation(client):
    assert client.get("/v2/forecast", params={"lat": 95, "lon": 0}).status_code == 422
    assert client.get("/v2/forecast", params={"lat": 0, "lon": 0, "vars": "nope"}).status_code == 400
    assert client.get("/v2/forecast", params={"lat": 0, "lon": 0, "tz": "Mars/Base"}).status_code == 400
    assert client.get("/v2/forecast", params={"lat": 0, "lon": 0, "run": "../../etc"}).status_code == 400


def test_variables_lists_legends(client):
    body = client.get("/v2/gfs/variables").json()
    t2m = next(v for v in body["variables"] if v["id"] == "t2m")
    assert t2m["tileable"] and t2m["legend"]["stops"][0]["color"].startswith("#")


def test_tiles(client):
    r = client.get("/v2/gfs/tiles/latest/t2m/f003/1/0/0.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "public, max-age=300"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (256, 256) and img.mode == "RGBA"

    run_id = r.headers["x-forecast-run"]
    pinned = client.get(f"/v2/gfs/tiles/{run_id}/wind_speed/f003/1/0/0.png")
    assert pinned.status_code == 200 and "immutable" in pinned.headers["cache-control"]


def test_tile_errors(client):
    assert client.get("/v2/gfs/tiles/latest/u10/f003/1/0/0.png").status_code == 404  # not tileable
    assert client.get("/v2/gfs/tiles/latest/t2m/f999/1/0/0.png").status_code == 404
    assert client.get("/v2/gfs/tiles/latest/t2m/f003/1/2/0.png").status_code == 404
    assert client.get("/v2/gfs/tiles/latest/t2m/f003/13/0/0.png").status_code == 422
    assert client.get("/v2/ecmwf/tiles/latest/t2m/f003/1/0/0.png").status_code == 404
