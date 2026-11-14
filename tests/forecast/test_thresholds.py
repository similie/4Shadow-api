"""Threshold checks (forecast/thresholds.py) on the synthetic run from conftest.py.

The run's grid is 2.5°; t2m = 0.5 * lat + 0.1 * hour, precip 1 mm/h from hour 1, wind 5 m/s,
waves 2 + 0.01 * hour m over "sea" (lon < 180) and missing over "land".
"""

from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecast import thresholds
from forecast.api import create_app
from forecast.catalog import GFS
from forecast.config import Settings
from forecast.service import ForecastError
from forecast.store import RunStore, iso
from forecast.thresholds import ThresholdRequest, build_area, check_thresholds, rolling_total


@pytest.fixture
def run(store_root):
    return RunStore(store_root, "gfs").get("latest")


def check(run, run_init, **kw):
    kw.setdefault("lead_days", 1.5)
    return check_thresholds(run, GFS, ThresholdRequest(at=run_init, **kw))


def point(lat, lon):
    return {"type": "Point", "coordinates": [lon, lat]}


def box(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}


def test_point_temperature(run, run_init):
    body = check(run, run_init, area=point(20, 15), lead_days=1,
                 conditions=[{"variable": "t2m", "operator": ">=", "threshold": 11}])
    assert body["window"] == {"from": iso(run_init), "to": iso(run_init + timedelta(hours=24))}
    assert body["area"]["type"] == "point" and body["area"]["cells"] == 1
    [ev] = body["events"]
    assert ev["start"] == iso(run_init + timedelta(hours=10))  # 10 + 0.1 * 10 = 11
    assert ev["end"] == iso(run_init + timedelta(hours=24))
    assert ev["peak"] == {"time": iso(run_init + timedelta(hours=24)), "variable": "t2m", "value": 12.4, "units": "°C",
                          "lat": 20.0, "lon": 15.0, "coverage_percent": 100.0}
    assert ev["ongoing"] is False and ev["duration_hours"] == 14


def test_no_event_below_threshold(run, run_init):
    body = check(run, run_init, area=point(20, 15), conditions=[{"variable": "t2m", "operator": ">", "threshold": 40}])
    assert body["events"] == [] and body["warnings"] == []


def test_window_beyond_the_run(run, run_init):
    body = check(run, run_init, area=point(20, 15), lead_days=16, conditions=[{"variable": "t2m", "operator": ">", "threshold": 40}])
    assert body["window"]["to"] == iso(run.valid_times[-1])
    assert body["warnings"] == [f"Run {run.id} ends at {iso(run.valid_times[-1])}, before the end of the 16-day window"]


def test_accumulated_rain(run, run_init):
    body = check(run, run_init, area=point(-8.5, 125.5),
                 conditions=[{"variable": "precip", "operator": ">=", "threshold": 20, "accumulate_hours": 24}])
    [ev] = body["events"]
    assert ev["start"] == iso(run_init + timedelta(hours=20))  # 1 mm/h since the run started
    assert ev["end"] == iso(run_init + timedelta(hours=36))
    assert ev["peak"]["value"] == 24.0 and ev["peak"]["units"] == "mm"
    assert body["conditions"][0]["statistic"] == "24 h total, area maximum"
    assert any("count only from the run's start" in w for w in body["warnings"])


def test_rolling_total_on_three_hourly_steps():
    hours = np.array([0, 1, 2, 3, 6, 9, 12])
    rate = np.ones((len(hours), 2), np.float32)
    rate[0] = np.nan
    rate[:, 1] = np.nan
    total = rolling_total(rate, hours, 6)
    assert total[:, 0].tolist() == [0, 1, 2, 3, 6, 6, 6]
    assert np.isnan(total[:, 1]).all()


def test_derived_wind_speed_holds_throughout(run, run_init):
    body = check(run, run_init, area=point(0, 0), lead_days=1,
                 conditions=[{"variable": "wind_speed", "operator": ">=", "threshold": 5}])
    [ev] = body["events"]
    assert ev["start"] == iso(run_init) and ev["ongoing"] is True


def test_area_modes(run, run_init):
    area = box(9, 9, 21, 31)  # rows at 10, 12.5, ... 30 (9) x columns at 10 ... 20 (5)
    cond = [{"variable": "t2m", "operator": ">=", "threshold": 15}]
    anywhere = check(run, run_init, area=area, conditions=cond)
    assert anywhere["area"]["cells"] == 45 and anywhere["area"]["type"] == "polygon"
    [ev] = anywhere["events"]
    assert ev["start"] == iso(run_init) and ev["peak"]["lat"] == 30.0  # the warmest edge

    assert check(run, run_init, area=area, area_mode="mean", conditions=cond)["events"] == []  # mean reaches 15 at hour 50

    share = check(run, run_init, area=area, area_mode="coverage", coverage_percent=20, conditions=cond)
    [ev] = share["events"]
    assert ev["start"] == iso(run_init + timedelta(hours=13))  # rows 27.5 and 30 of 9 reach 15
    assert ev["peak"]["coverage_percent"] >= 20


def test_match_any_and_all(run, run_init):
    conds = [{"variable": "t2m", "operator": ">=", "threshold": 11}, {"variable": "wind_speed", "operator": ">", "threshold": 9}]
    assert check(run, run_init, area=point(20, 15), conditions=conds, match="all")["events"] == []
    [ev] = check(run, run_init, area=point(20, 15), conditions=conds, match="any")["events"]
    assert ev["start"] == iso(run_init + timedelta(hours=10))
    assert [c["variable"] for c in ev["conditions"]] == ["t2m", "wind_speed"]


def test_min_duration(run, run_init):
    cond = [{"variable": "t2m", "operator": ">=", "threshold": 13}]  # from hour 30 to 36
    assert len(check(run, run_init, area=point(20, 15), conditions=cond)["events"]) == 1
    assert check(run, run_init, area=point(20, 15), conditions=cond, min_duration_hours=12)["events"] == []


def test_radius_and_small_polygon(run):
    g = run.grid
    assert build_area(g, point(20, 15), radius_km=300).size == 5  # the cell and its four neighbours
    tiny = build_area(g, box(15.1, 20.1, 15.3, 20.3))
    assert tiny.size == 1 and (tiny.lats[0], tiny.lons[0]) == (20.0, 15.0)


def test_antimeridian(run):
    g = run.grid
    across = build_area(g, box(171, -9, 189, 9))  # columns 172.5 ... 187.5, rows -7.5 ... 7.5
    split = build_area(g, {"type": "MultiPolygon", "coordinates": [
        box(171, -9, 180, 9)["coordinates"], box(-180, -9, -171, 9)["coordinates"]]})
    assert across.size == split.size == 49
    assert sorted(zip(across.rows, across.cols)) == sorted(zip(split.rows, split.cols))


def test_sea_variable(run, run_init):
    cond = [{"variable": "wave_height", "operator": ">=", "threshold": 2.1}]
    coast = check(run, run_init, area=box(170, -10, 190, 10), conditions=cond, area_mode="coverage", coverage_percent=100)
    [ev] = coast["events"]  # land cells (lon >= 180) don't count against the coverage
    assert ev["start"] == iso(run_init + timedelta(hours=10))
    inland = check(run, run_init, area=point(0, 200), conditions=cond)
    assert inland["events"] == [] and "open sea only" in inland["warnings"][0]


@pytest.mark.parametrize("kw, message", [
    ({"conditions": [{"variable": "nope", "operator": ">", "threshold": 1}]}, "Unknown variable"),
    ({"conditions": [{"variable": "wind_dir", "operator": ">", "threshold": 1}]}, "direction"),
    ({"conditions": [{"variable": "t2m", "operator": ">", "threshold": 1, "accumulate_hours": 24}]}, "accumulate_hours"),
    ({"area": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}, "Point, Polygon or MultiPolygon"),
    ({"area": box(0, 0, 10, 10), "radius_km": 10}, "radius_km"),
    ({"area": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 0]]]}}, "at least 3"),
])
def test_rejects(run, run_init, kw, message):
    kw = {"area": point(0, 0), "conditions": [{"variable": "t2m", "operator": ">", "threshold": 1}], **kw}
    with pytest.raises(ForecastError) as e:
        check(run, run_init, **kw)
    assert e.value.status == 422 and message in e.value.message


def test_area_too_large(run, monkeypatch):
    monkeypatch.setattr(thresholds, "MAX_CELLS", 10)
    with pytest.raises(ForecastError, match="Area too large"):
        build_area(run.grid, box(0, 0, 20, 20))


def test_endpoint(store_root, run_init):
    client = TestClient(create_app(Settings(data_root=store_root)))
    body = {"area": point(20, 15), "conditions": [{"variable": "t2m", "operator": ">=", "threshold": 11}],
            "lead_days": 1, "at": iso(run_init), "timeline": True}
    r = client.post("/v2/gfs/thresholds", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert r.headers["x-forecast-run"] == out["run"] and len(out["events"]) == 1
    assert len(out["timeline"]) == 25 and out["timeline"][10]["met"] and not out["timeline"][9]["met"]
    bad = client.post("/v2/gfs/thresholds", json={**body, "area_mode": "coverage"})
    assert bad.status_code == 422  # coverage needs coverage_percent
    unknown = client.post("/v2/gfs/thresholds", json={**body, "conditions": [{"variable": "x", "operator": ">", "threshold": 1}]})
    assert unknown.status_code == 422 and "Unknown variable" in unknown.json()["detail"]
