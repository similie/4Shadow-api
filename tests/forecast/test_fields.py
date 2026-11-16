import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecast.api import create_app
from forecast.config import Settings
from forecast.fields import decode_field


@pytest.fixture
def client(store_root):
    return TestClient(create_app(Settings(data_root=store_root)))


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    return decode_field(r.content)


def test_global_field_layout(client):
    header, data = get(client, "/v2/gfs/fields/latest/f003", vars="u10,v10,t2m")
    assert header["format"] == "4SF1" and header["forecast_hour"] == 3 and header["interval_hours"] == 1
    g = header["grid"]  # the test grid is 2.5°: 73 x 144, re-centred to start at -180
    assert (g["lat0"], g["dlat"], g["nlat"], g["lon0"], g["dlon"], g["nlon"], g["wraps"]) == (90, 2.5, 73, -180, 2.5, 144, True)
    assert [v["id"] for v in header["variables"]] == ["u10", "v10", "t2m"]
    assert np.allclose(data["u10"], 3.0) and np.allclose(data["v10"], 4.0)
    lat = 90 - 2.5 * np.arange(73)
    assert np.allclose(data["t2m"][:, 0], 0.5 * lat + 0.1 * 3, atol=0.01)  # synthetic t2m = 0.5 lat + 0.1 fhr


def test_binary_framing(client):
    body = client.get("/v2/gfs/fields/latest/f000", params={"vars": "mslp"}).content
    assert body[:4] == b"4SF1"
    (n,) = struct.unpack_from("<I", body, 4)
    assert (8 + n) % 8 == 0 and len(body) == 8 + n + 73 * 144 * 2  # arrays start 8-byte aligned


def test_stride_and_bbox(client):
    header, data = get(client, "/v2/gfs/fields/latest/f000", vars="t2m", stride=2)
    assert (header["grid"]["nlat"], header["grid"]["nlon"], header["grid"]["dlat"]) == (37, 72, 5.0)
    assert header["grid"]["wraps"] is True

    header, data = get(client, "/v2/gfs/fields/latest/f000", vars="t2m", bbox="100,-20,140,10")
    g = header["grid"]
    assert (g["lat0"], g["lon0"], g["nlat"], g["nlon"], g["wraps"]) == (10, 100, 13, 17, False)
    assert data["t2m"].shape == (13, 17)
    assert np.allclose(data["t2m"][-1], 0.5 * -20, atol=0.01)

    header, _ = get(client, "/v2/gfs/fields/latest/f000", vars="t2m", bbox="170,-10,-170,10")  # across the antimeridian
    assert (header["grid"]["lon0"], header["grid"]["nlon"]) == (170, 9)


def test_field_errors(client):
    def status(**params):
        return client.get("/v2/gfs/fields/latest/f000", params=params)

    r = status(vars="wind_speed")
    assert r.status_code == 400 and "computed from u10, v10" in r.json()["detail"]
    assert status(vars="u10,v10,t2m,mslp,gust").status_code == 400
    assert status(vars="t2m", bbox="1,2,3").status_code == 400
    assert status(vars="t2m", stride=0).status_code == 422
    assert client.get("/v2/gfs/fields/latest/f999", params={"vars": "t2m"}).status_code == 404
    pinned = client.get("/v2/gfs/fields/latest/f000", params={"vars": "t2m"}).headers["x-forecast-run"]
    r = client.get(f"/v2/gfs/fields/{pinned}/f000", params={"vars": "t2m"})
    assert "immutable" in r.headers["cache-control"]
