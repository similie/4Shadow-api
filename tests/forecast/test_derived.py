"""Derived variables: heat, wet-bulb, fire weather and wind power, checked against published values."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecast.api import create_app
from forecast.catalog import AIGFS, GFS, derive, fosberg, heat_index, wet_bulb, wind_chill
from forecast.config import Settings


def a(*values):
    return np.asarray(values, dtype=np.float32)


def test_heat_index_matches_the_nws_table():
    # NWS heat index chart: 90 °F at 70% humidity feels like 106 °F (41.1 °C).
    assert heat_index(a((90 - 32) / 1.8), a(70))[0] == pytest.approx(41.1, abs=0.3)
    # 100 °F at 40%: 109 °F (42.8 °C).
    assert heat_index(a((100 - 32) / 1.8), a(40))[0] == pytest.approx(42.8, abs=0.3)
    # In mild weather it stays close to the air temperature.
    assert heat_index(a(20), a(50))[0] == pytest.approx(20, abs=1)


def test_wind_chill_matches_environment_canada():
    # -10 °C in a 20 km/h wind feels like -18 °C (Environment Canada's table: -17.9).
    assert wind_chill(a(-10), a(20 / 3.6))[0] == pytest.approx(-17.9, abs=0.2)
    # No chill in calm air or above 10 °C.
    assert wind_chill(a(-10, 15), a(0.5, 10))[:].tolist() == [-10, 15]


def test_wet_bulb_matches_stull_2011():
    assert wet_bulb(a(20), a(50))[0] == pytest.approx(13.7, abs=0.1)  # the paper's worked example
    saturated = wet_bulb(a(30), a(100))[0]
    assert 29 < saturated <= 30  # never above the air temperature


def test_fosberg_index():
    # 90 °F, 20% humidity, 10 mph: equilibrium moisture 4.1%, index 25.
    assert fosberg(a((90 - 32) / 1.8), a(20), a(10 * 0.44704))[0] == pytest.approx(25.2, abs=0.2)
    # Bone dry and very windy is capped at 100; humid and calm is low.
    hot, damp = fosberg(a(40, 15), a(5, 95), a(25, 1))
    assert hot == 100 and damp < 5


def test_humidity_from_the_dew_point_matches_stored_humidity():
    t, td = a(20, 30, 10), a(10, 30, -5)
    rh = derive("rh2m", {"t2m": t, "d2m": td})
    assert rh.tolist() == pytest.approx([52.5, 100.0, 34.2], abs=0.2)
    # A model with only the dew point (AIGFS) gets the same indices as one that stores humidity (GFS).
    for vid in ("heat_index", "wet_bulb", "vpd"):
        assert derive(vid, {"t2m": t, "d2m": td}) == pytest.approx(derive(vid, {"t2m": t, "rh2m": rh}), abs=1e-4)


def test_vpd_and_wind_power():
    assert derive("vpd", {"t2m": a(25), "rh2m": a(50)})[0] == pytest.approx(1.58, abs=0.01)
    # 8 m/s at standard sea-level density (1.225 kg/m³): 1/2 x 1.225 x 8³ = 314 W/m².
    p = derive("wind_power_100m", {"u100": a(8), "v100": a(0), "sp": a(1013.25), "t2m": a(15)})[0]
    assert p == pytest.approx(313.6, abs=0.5)
    assert derive("wind_dir_100m", {"u100": a(0), "v100": a(-5)})[0] == pytest.approx(0)  # from the north


def test_feels_like_picks_heat_index_wind_chill_or_air_temperature():
    inputs = {"t2m": a(35, -10, 18), "rh2m": a(60, 80, 60), "u10": a(1, 20 / 3.6, 5), "v10": a(0, 0, 0)}
    hot, cold, mild = derive("feels_like", inputs)
    assert hot > 40 and cold == pytest.approx(-17.9, abs=0.2) and mild == 18


def test_nan_stays_missing():
    out = derive("wet_bulb", {"t2m": a(np.nan, 20), "rh2m": a(50, np.nan)})
    assert np.isnan(out).all()


def test_catalogs_declare_inputs_they_can_serve():
    for cat in (GFS, AIGFS):
        stored = {v.id for v in cat.stored_variables()}
        for v in cat.variables.values():
            assert set(v.derived_from) <= stored, (cat.model, v.id)
            if v.derived_from:  # every derived variable computes from its declared inputs alone
                inputs = {d: a(20.0, 1.0) for d in v.derived_from}
                assert derive(v.id, inputs).shape == (2,)


def test_new_variables_in_forecasts_and_tiles(store_root):
    c = TestClient(create_app(Settings(data_root=store_root, models=("gfs",))))
    body = c.get("/v2/forecast", params={"lat": 60, "lon": 10, "vars": "t2m,heat_index,wet_bulb,fire_weather,feels_like",
                                         "start": "run"}).json()
    ts = body["timeseries"]
    assert ts["wet_bulb"][0] < ts["t2m"][0] and ts["fire_weather"][0] is not None
    assert {"feels_like_max", "wet_bulb_max", "fire_weather_max"} <= set(body["daily"])
    for var in ("wet_bulb", "fire_weather", "heat_index", "refc", "soil_moisture"):
        assert c.get(f"/v2/gfs/tiles/latest/{var}/f000/1/1/0.png").status_code == 200, var
    described = {v["id"]: v for v in c.get("/v2/gfs/variables").json()["variables"]}
    assert described["wet_bulb"]["derived_from"] == ["t2m", "rh2m"] and described["refc"]["units"] == "dBZ"


def test_computed_variables_as_fields(store_root):
    from forecast.fields import decode_field

    c = TestClient(create_app(Settings(data_root=store_root, models=("gfs",))))
    r = c.get("/v2/gfs/fields/latest/f003", params={"vars": "wet_bulb,wind_dir,t2m", "stride": 4})
    assert r.status_code == 200
    header, data = decode_field(r.content)
    meta = {v["id"]: v for v in header["variables"]}
    assert meta["wet_bulb"]["derived_from"] == ["t2m", "rh2m"] and "derived_from" not in meta["t2m"]
    assert np.allclose(data["wind_dir"], 216.9, atol=0.1)  # u=3, v=4: from the south-west, quantised to 0.1°
    expected = wet_bulb(data["t2m"], np.full_like(data["t2m"], 50.0))
    assert np.allclose(data["wet_bulb"], expected, atol=0.02)


def test_fire_weather_is_land_only(store_root):
    """Over open water the index means nothing: maps and area checks leave it out; a point keeps it."""
    import io

    from PIL import Image

    from forecast.fields import decode_field
    from forecast.store import RunStore
    from forecast.thresholds import _values, build_area

    run = RunStore(store_root, "gfs").get()
    land = np.ones(run.grid.shape, np.float32)
    land[:, run.grid.nlon // 2:] = 0  # 180°E to 360°: water
    np.save(run.path / "static" / "land.npy", land)

    c = TestClient(create_app(Settings(data_root=store_root, models=("gfs",))))
    header, data = decode_field(c.get("/v2/gfs/fields/latest/f000", params={"vars": "fire_weather,t2m"}).content)
    g = header["grid"]
    lons = (g["lon0"] + np.arange(g["nlon"]) * g["dlon"] + 180) % 360 - 180
    fw = data["fire_weather"]
    assert np.isnan(fw[:, lons < 0]).all() and np.isfinite(fw[:, lons >= 0]).all()
    assert np.isfinite(data["t2m"]).all()  # other variables are untouched

    def alpha(x):
        png = c.get(f"/v2/gfs/tiles/latest/fire_weather/f000/1/{x}/0.png")
        return np.asarray(Image.open(io.BytesIO(png.content)).convert("RGBA"))[..., 3]
    # Tile x=0 is 180°W-0° (water), x=1 is 0°-180°E (land); the last pixels of each blend into the other.
    assert alpha(0)[:, :240].max() == 0 and alpha(1)[:, :240].min() > 0

    sea = build_area(run.grid, {"type": "Polygon", "coordinates": [[[-150, 0], [-140, 0], [-140, 10], [-150, 10]]]})
    ashore = build_area(run.grid, {"type": "Polygon", "coordinates": [[[10, 0], [20, 0], [20, 10], [10, 10]]]})
    assert np.isnan(_values(run, GFS, sea, GFS.variables["fire_weather"])).all()
    assert np.isfinite(_values(run, GFS, ashore, GFS.variables["fire_weather"])).all()
    point = c.get("/v2/forecast", params={"lat": 5, "lon": -145, "vars": "fire_weather", "start": "run"}).json()
    assert point["timeseries"]["fire_weather"][0] is not None
