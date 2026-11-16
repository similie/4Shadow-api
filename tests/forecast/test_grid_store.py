import json

import numpy as np
import pytest

from forecast.grid import Grid, bilinear, tile_pixel_centers
from forecast.ingest import IntervalState
from forecast.sources.gfs import FcstSpec
from forecast.store import MISSING, RunStore, decode, encode, published_runs, read_latest

from conftest import COARSE, write_run


def test_encode_decode_roundtrip_and_missing():
    v = np.array([-12.345, 0.0, 25.678, np.nan], dtype=np.float32)
    raw = encode(v, 0.01, 0.0)
    assert raw[3] == MISSING
    back = decode(raw, 0.01, 0.0)
    assert np.allclose(back[:3], v[:3], atol=0.005) and np.isnan(back[3])


def test_encode_clips_instead_of_overflowing():
    assert encode(np.array([1e9, -1e9]), 0.01, 0.0).tolist() == [32767, -32767]


def test_bilinear_wraps_across_antimeridian():
    g = Grid(lat0=10, dlat=10, nlat=2, lon0=0, dlon=90, nlon=4)
    j0, j1, wj = g.col_weights(-45.0)
    assert (int(j0), int(j1), float(wj)) == (3, 0, 0.5)
    field = np.array([[0, 10, 20, 30], [0, 10, 20, 30]], dtype=np.float32)
    i0, i1, wi = g.row_weights(5.0)
    v = bilinear(field[i0, j0], field[i0, j1], field[i1, j0], field[i1, j1], wi, wj)
    assert float(v) == pytest.approx(15.0)


def test_bilinear_ignores_missing_corners():
    v = bilinear(np.float32(1), np.float32(np.nan), np.float32(3), np.float32(np.nan), 0.5, 0.5)
    assert float(v) == pytest.approx(2.0)
    assert np.isnan(bilinear(*(np.float32(np.nan),) * 4, 0.5, 0.5))


def test_tile_pixel_centers_follow_xyz_convention():
    lat, lon = tile_pixel_centers(1, 0, 0)
    assert (lat > 0).all() and np.all(np.diff(lat) < 0)  # y=0 is the north, rows run southward
    assert (lon < 0).all() and np.all(np.diff(lon) > 0)
    lat, _ = tile_pixel_centers(0, 0, 0)
    assert lat[0] == pytest.approx(85.05, abs=0.4) and lat[-1] == pytest.approx(-85.05, abs=0.4)


def test_point_series_is_exact_for_linear_fields(store_root):
    run = RunStore(store_root, "gfs").get()
    s = run.point_series(["t2m"], 10.3, 20.7)["t2m"]
    expected = 0.5 * 10.3 + 0.1 * run.forecast_hours
    assert np.allclose(s, expected, atol=0.006)


def test_tile_sampling_orientation(store_root):
    run = RunStore(store_root, "gfs").get()
    lat, lon = tile_pixel_centers(2, 1, 1)
    vals = run.sample_field("t2m", 0, run.grid.row_weights(lat), run.grid.col_weights(lon))
    assert np.allclose(vals[:, 0], 0.5 * lat, atol=0.01)


def test_interval_state_recovers_hourly_rates():
    rates = np.arange(1, 13, dtype=np.float32)  # true mean rate in hour (h-1, h]
    shape = (1, 1)
    acc, ave = IntervalState(shape), IntervalState(shape)
    for h in range(1, 13):
        total = rates[:h].sum()
        start = 6 * ((h - 1) // 6)
        bucket_mean = rates[start:h].mean()
        got_acc = acc.accum(h, np.full(shape, total, np.float32))
        got_ave = ave.average(h, FcstSpec("ave", start, h), np.full(shape, bucket_mean, np.float32))
        assert float(got_acc[0, 0]) == pytest.approx(rates[h - 1], rel=1e-5)
        assert float(got_ave[0, 0]) == pytest.approx(rates[h - 1], rel=1e-5)


def test_interval_state_three_hourly_buckets():
    # Beyond f120 GFS steps every 3 h with 6 h buckets: 120-123, 120-126, 126-129 ...
    st = IntervalState((1, 1))
    st.prev_fhr = 120
    a = st.average(123, FcstSpec("ave", 120, 123), np.full((1, 1), 10.0, np.float32))
    b = st.average(126, FcstSpec("ave", 120, 126), np.full((1, 1), 15.0, np.float32))  # second half averaged 20
    c = st.average(129, FcstSpec("ave", 126, 129), np.full((1, 1), 7.0, np.float32))
    assert [float(x[0, 0]) for x in (a, b, c)] == pytest.approx([10.0, 20.0, 7.0])


def test_publish_prunes_and_never_moves_latest_backwards(tmp_path, run_init):
    from datetime import timedelta

    ids = [write_run(tmp_path, run_init - timedelta(hours=6 * k), hours=[0, 1], keep_runs=2) for k in range(3)]
    root = tmp_path / "gfs"
    assert read_latest(root) == ids[0]  # older runs published later don't steal "latest"
    assert published_runs(root) == sorted(ids)[1:]
    manifest = json.loads((root / "runs" / ids[0] / "manifest.json").read_text())
    assert manifest["grid"] == COARSE.to_dict() and manifest["forecast_hours"] == [0, 1]


def test_republishing_a_run_replaces_it_in_place(tmp_path, run_init):
    rid = write_run(tmp_path, run_init)
    store = RunStore(tmp_path, "gfs")
    before = store.get(rid).manifest["created_at"]
    write_run(tmp_path, run_init)  # e.g. re-ingest after adding a variable
    runs_dir = tmp_path / "gfs" / "runs"
    assert published_runs(tmp_path / "gfs") == [rid] and read_latest(tmp_path / "gfs") == rid
    assert not any(p.name.endswith(".replaced") for p in runs_dir.iterdir())
    assert store.get(rid).manifest["created_at"] >= before
