"""Sea state from GFS-Wave: catalog, ingest of a second file family, coastal and angle handling."""

from datetime import datetime, timezone

import pytest

from forecast import ingest
from forecast.catalog import GFS
from forecast.service import point_forecast
from forecast.sources.gfs import file_url, parse_idx
from forecast.store import RunStore

WAVE_VARS = ("wave_height", "wave_period", "wave_dir", "wind_wave_height", "swell_height", "swell_period", "swell_dir")

# Real lines from gfs.t00z.pgrb2.0p25.f006.idx / gfswave.t00z.global.0p25.f006.grib2.idx (trimmed).
# Message numbers overlap between the two files, as they do on NOAA's bucket.
ATMOS_IDX = """1:0:d=2026111300:PRMSL:mean sea level:6 hour fcst:
2:1009487:d=2026111300:TMP:2 m above ground:6 hour fcst:
3:2310871:d=2026111300:UGRD:10 m above ground:6 hour fcst:
4:3329021:d=2026111300:VGRD:10 m above ground:6 hour fcst:
"""
WAVE_IDX = """1:0:d=2026111300:WIND:surface:6 hour fcst:
2:676523:d=2026111300:WDIR:surface:6 hour fcst:
3:3000528:d=2026111300:HTSGW:surface:6 hour fcst:
4:3422616:d=2026111300:PERPW:surface:6 hour fcst:
5:3926729:d=2026111300:DIRPW:surface:6 hour fcst:
6:4817621:d=2026111300:WVHGT:surface:6 hour fcst:
7:5194224:d=2026111300:SWELL:1 in sequence:6 hour fcst:
8:5661506:d=2026111300:SWELL:2 in sequence:6 hour fcst:
9:6815857:d=2026111300:SWPER:1 in sequence:6 hour fcst:
10:9067783:d=2026111300:SWDIR:1 in sequence:6 hour fcst:
11:9964882:d=2026111300:SWDIR:2 in sequence:6 hour fcst:
"""


class FakeClient:
    def __init__(self, idx: dict[str, str | None]):
        self.idx_text = idx
        self.fetched: list[tuple[str, list[str]]] = []

    def idx(self, run, fhr, product="atmos"):
        text = self.idx_text.get(product)
        return None if text is None else parse_idx(text)

    def fetch_records(self, run, fhr, records, product="atmos"):
        self.fetched.append((product, sorted(f"{r.param}:{r.level}" for r in records)))
        return {r.num: f"{product}:{r.param}:{r.level}".encode() for r in records}


RUN = datetime(2026, 11, 13, 0, tzinfo=timezone.utc)


def test_catalog_lists_wave_variables_as_a_second_product():
    assert GFS.products() == ["atmos", "wave"]
    for vid in WAVE_VARS:
        v = GFS.get(vid)
        assert v.product == "wave" and v.marine and v.stored
        assert v.describe()["marine"] is True
    assert GFS.get("wave_dir").circular and GFS.get("swell_dir").circular
    assert GFS.get("wave_height").describe()["tileable"] and not GFS.get("wave_dir").describe()["tileable"]
    assert "wave_height_max" in {d.id for d in GFS.daily}


def test_wave_file_url():
    assert file_url("https://b", RUN, 6, "wave") == "https://b/gfs.20261113/00/wave/gridded/gfswave.t00z.global.0p25.f006.grib2"
    assert file_url("https://b", RUN, 6) == "https://b/gfs.20261113/00/atmos/gfs.t00z.pgrb2.0p25.f006"


def test_download_step_reads_each_product_from_its_own_file():
    client = FakeClient({"atmos": ATMOS_IDX, "wave": WAVE_IDX})
    payload = ingest.download_step(client, GFS, RUN, 6, include_static=False)
    products = [p for p, _ in client.fetched]
    assert products == ["atmos", "wave"]
    # Same message numbers in both files must not overwrite each other.
    assert payload.messages["t2m"][1] == b"atmos:TMP:2 m above ground"
    assert payload.messages["wave_height"][1] == b"wave:HTSGW:surface"
    assert payload.messages["swell_height"][1] == b"wave:SWELL:1 in sequence"  # not the 2nd swell partition
    assert payload.messages["swell_dir"][1] == b"wave:SWDIR:1 in sequence"
    assert "wind_wave_height" in payload.messages and "wave_dir" in payload.messages


def test_missing_wave_file_leaves_waves_empty_but_missing_atmos_file_fails():
    payload = ingest.download_step(FakeClient({"atmos": ATMOS_IDX, "wave": None}), GFS, RUN, 6, include_static=False)
    assert "t2m" in payload.messages and not any(v in payload.messages for v in WAVE_VARS)
    with pytest.raises(ingest.StepUnavailable):
        ingest.download_step(FakeClient({"atmos": None, "wave": WAVE_IDX}), GFS, RUN, 6, include_static=False)


def test_directions_are_blended_as_angles(store_root):
    run = RunStore(store_root, "gfs").get()
    # Columns at 0° and 2.5° hold 350° and 10°; halfway between, the wave comes from the north.
    s = run.point_series(["wave_dir"], 10.0, 1.25, circular={"wave_dir"})["wave_dir"]
    assert min(s[0], 360 - s[0]) < 0.5
    linear = run.point_series(["wave_dir"], 10.0, 1.25)["wave_dir"]
    assert abs(linear[0] - 180.0) < 0.5  # what plain bilinear would have said


def test_sea_point_finds_the_nearest_sea_cell(store_root):
    run = RunStore(store_root, "gfs").get()
    assert run.sea_point("wave_height", 0.0, 170.0) is None  # at sea: use the point itself
    lat, lon, km = run.sea_point("wave_height", 0.0, 181.0, max_km=600)
    assert (lat, lon) == (0.0, 177.5) and 380 < km < 400
    assert run.sea_point("wave_height", 0.0, 181.0) is None  # nothing within 50 km


def test_point_forecast_has_waves_at_sea_and_nulls_inland(store_root):
    stores = {"gfs": RunStore(store_root, "gfs")}
    _, sea = point_forecast(stores, 10.0, 120.0, variables="wave_height,wave_dir,swell_period,t2m")
    assert sea["units"]["wave_height"] == "m"
    assert sea["timeseries"]["wave_height"][0] is not None and sea["current"]["wave_height"] is not None
    assert sea["daily"]["wave_height_max"][0] is not None
    assert "sea_point" not in sea["location"]
    _, land = point_forecast(stores, 10.0, 250.0, variables="wave_height,t2m")
    assert land["timeseries"]["wave_height"][0] is None and land["timeseries"]["t2m"][0] is not None
    assert land["daily"]["wave_height_max"][0] is None


def test_partitions_are_zero_at_sea_where_the_model_found_none():
    import numpy as np

    total = np.array([2.0, 3.0, np.nan], np.float32)  # sea, sea, land
    swell = np.array([1.0, np.nan, np.nan], np.float32)  # swell, none, land
    out = ingest.zero_where_absent(swell, total)
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])
    order = [v.id for v in GFS.stored_variables()]
    for v in GFS.stored_variables():
        if v.zero_within:  # the reference must be decoded first
            assert order.index(v.zero_within) < order.index(v.id)
