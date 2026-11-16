"""Per-run array store.

Layout (one directory per model under the data root)::

    gfs/
      latest.json                 {"run": "2026111206"}  (atomically replaced)
      runs/2026111206/
        manifest.json             grid, steps, valid times, per-variable scale/offset/units
        t2m.npy ...               int16 [step, lat, lon], standard .npy (np.load(mmap_mode="r"))
        static/orog.npy ...       float32 [lat, lon]
      runs/2026111212.incoming/   being written; renamed into place when complete

Values are stored as int16 (``value = raw * scale + offset``, -32768 = missing). The API
memory-maps these files, so every worker process shares one copy through the OS page cache
and a run switch is a pointer flip, with no warm-up and no window where data is missing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .grid import Grid, bilinear

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
MISSING = -32768
RUN_ID_FORMAT = "%Y%m%d%H"
RUN_ID_RE = re.compile(r"^\d{10}$")
INCOMING_SUFFIX = ".incoming"


def run_id(init: datetime) -> str:
    return init.astimezone(timezone.utc).strftime(RUN_ID_FORMAT)


def parse_run_id(value: str) -> datetime:
    if not RUN_ID_RE.match(value):
        raise ValueError(f"Run id must look like YYYYMMDDHH, got {value!r}")
    return datetime.strptime(value, RUN_ID_FORMAT).replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def encode(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    q = np.rint((v - offset) / scale)
    np.clip(q, -32767, 32767, out=q)
    q[np.isnan(v)] = MISSING
    return q.astype(np.int16)


def decode(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    raw = np.asarray(raw)
    out = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
    return np.where(raw == MISSING, np.float32(np.nan), out)


# --------------------------------------------------------------------------- pointers / housekeeping


def read_latest(model_root: Path) -> str | None:
    try:
        return json.loads((model_root / "latest.json").read_text())["run"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def write_latest(model_root: Path, rid: str) -> None:
    tmp = model_root / "latest.json.tmp"
    tmp.write_text(json.dumps({"run": rid, "updated_at": iso(datetime.now(timezone.utc))}))
    os.replace(tmp, model_root / "latest.json")


def published_runs(model_root: Path) -> list[str]:
    runs = model_root / "runs"
    if not runs.is_dir():
        return []
    return sorted(p.name for p in runs.iterdir() if RUN_ID_RE.match(p.name) and (p / "manifest.json").is_file())


def prune_runs(model_root: Path, keep: int, stale_incoming_hours: float = 12.0) -> list[str]:
    """Delete all but the newest ``keep`` published runs (never the one ``latest`` points at)."""
    removed = []
    latest = read_latest(model_root)
    runs = published_runs(model_root)
    for rid in runs[: max(0, len(runs) - keep)]:
        if rid != latest:
            shutil.rmtree(model_root / "runs" / rid, ignore_errors=True)
            removed.append(rid)
    cutoff = time.time() - stale_incoming_hours * 3600
    runs_dir = model_root / "runs"
    if runs_dir.is_dir():
        for p in runs_dir.glob(f"*{INCOMING_SUFFIX}"):
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p.name)
    return removed


# --------------------------------------------------------------------------- writing


class RunWriter:
    """Writes one model run into ``runs/<id>.incoming`` and publishes it atomically."""

    def __init__(self, model_root: Path, model: str, init_time: datetime, grid: Grid,
                 forecast_hours: list[int], variables: list, source: dict | None = None) -> None:
        self.model_root = model_root
        self.model = model
        self.init_time = init_time
        self.id = run_id(init_time)
        self.grid = grid
        self.forecast_hours = list(forecast_hours)
        self.variables = {v.id: v for v in variables}
        self.source = source or {}
        self.final_dir = model_root / "runs" / self.id
        self.dir = self.final_dir.with_name(self.id + INCOMING_SUFFIX)
        if self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "static").mkdir(parents=True)
        shape = (len(self.forecast_hours), *grid.shape)
        self._arrays = {
            vid: np.lib.format.open_memmap(self.dir / f"{vid}.npy", mode="w+", dtype=np.int16, shape=shape)
            for vid in self.variables
        }
        self._written = {vid: np.zeros(len(self.forecast_hours), dtype=bool) for vid in self.variables}
        self._static: dict[str, dict] = {}

    def write(self, var_id: str, step: int, values: np.ndarray | None) -> None:
        var = self.variables[var_id]
        if values is None:
            self._arrays[var_id][step] = MISSING
        else:
            if values.shape != self.grid.shape:
                raise ValueError(f"{var_id}: shape {values.shape} != grid {self.grid.shape}")
            self._arrays[var_id][step] = encode(values, var.scale, var.offset)
        self._written[var_id][step] = True

    def write_static(self, name: str, values: np.ndarray, meta: dict) -> None:
        np.save(self.dir / "static" / f"{name}.npy", np.asarray(values, dtype=np.float32))
        self._static[name] = {**meta, "file": f"static/{name}.npy"}

    def _close(self) -> None:
        for arr in self._arrays.values():
            arr.flush()
        self._arrays.clear()

    def publish(self, keep_runs: int | None = None, extra: dict | None = None) -> Path:
        unwritten = {vid: int((~w).sum()) for vid, w in self._written.items() if not w.all()}
        if unwritten:
            raise RuntimeError(f"Run {self.id} incomplete, unwritten steps: {unwritten}")
        self._close()
        manifest = {
            "format_version": FORMAT_VERSION,
            "model": self.model,
            "run": self.id,
            "init_time": iso(self.init_time),
            "grid": self.grid.to_dict(),
            "forecast_hours": self.forecast_hours,
            "valid_times": [iso(self.init_time + timedelta(hours=h)) for h in self.forecast_hours],
            "variables": {
                vid: {
                    "name": v.name, "units": v.units, "kind": v.kind,
                    "scale": v.scale, "offset": v.offset,
                    "source": f"{v.grib_param}:{v.grib_level}" if v.grib_param else None,
                    "file": f"{vid}.npy",
                }
                for vid, v in self.variables.items()
            },
            "static": self._static,
            "source": self.source,
            "created_at": iso(datetime.now(timezone.utc)),
            **(extra or {}),
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if self.final_dir.exists():
            # Re-ingest of a published run: swap with two renames so readers never see it missing.
            replaced = self.final_dir.with_name(self.final_dir.name + ".replaced")
            shutil.rmtree(replaced, ignore_errors=True)
            os.replace(self.final_dir, replaced)
            os.replace(self.dir, self.final_dir)
            shutil.rmtree(replaced, ignore_errors=True)
        else:
            os.replace(self.dir, self.final_dir)
        current = read_latest(self.model_root)
        if current is None or self.id >= current:
            write_latest(self.model_root, self.id)
        if keep_runs is not None:
            prune_runs(self.model_root, keep_runs)
        log.info("Published %s run %s", self.model, self.id)
        return self.final_dir

    def abort(self) -> None:
        self._arrays.clear()
        shutil.rmtree(self.dir, ignore_errors=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


# --------------------------------------------------------------------------- reading


class Run:
    """Read-only view of a published run backed by memory-mapped arrays."""

    def __init__(self, path: Path) -> None:
        self.path = path
        m = json.loads((path / "manifest.json").read_text())
        if m.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported store format {m.get('format_version')} in {path}")
        self.manifest = m
        self.id: str = m["run"]
        self.model: str = m["model"]
        self.init_time = parse_run_id(self.id)
        self.grid = Grid.from_dict(m["grid"])
        self.forecast_hours = np.asarray(m["forecast_hours"], dtype=np.int32)
        self.valid_times = [self.init_time + timedelta(hours=int(h)) for h in self.forecast_hours]
        self.variables: dict[str, dict] = m["variables"]
        self.static_meta: dict[str, dict] = m.get("static", {})
        self._arrays: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _load(self, key: str, rel: str) -> np.ndarray:
        arr = self._arrays.get(key)
        if arr is None:
            with self._lock:
                arr = self._arrays.get(key)
                if arr is None:
                    arr = np.load(self.path / rel, mmap_mode="r")
                    self._arrays[key] = arr
        return arr

    def array(self, var_id: str) -> np.ndarray:
        meta = self.variables[var_id]
        return self._load(var_id, meta.get("file", f"{var_id}.npy"))

    def static(self, name: str) -> np.ndarray | None:
        meta = self.static_meta.get(name)
        return None if meta is None else self._load("static:" + name, meta["file"])

    def step_for_hour(self, fhr: int) -> int | None:
        hits = np.nonzero(self.forecast_hours == fhr)[0]
        return int(hits[0]) if hits.size else None

    def step_for_valid(self, valid: datetime) -> int | None:
        delta = (valid - self.init_time).total_seconds() / 3600.0
        return self.step_for_hour(int(delta)) if delta == int(delta) else None

    def point_series(self, var_ids, lat: float, lon: float, circular=()) -> dict[str, np.ndarray]:
        """Bilinear-interpolated values over every step for each variable.

        Variables in ``circular`` are directions in degrees: they are blended as unit vectors,
        so 350° and 10° give 0°, not 180°.
        """
        i0, i1, wi = (np.asarray(a).item() for a in self.grid.row_weights(lat))
        j0, j1, wj = (np.asarray(a).item() for a in self.grid.col_weights(lon))
        out = {}
        for vid in var_ids:
            arr, meta = self.array(vid), self.variables[vid]
            corners = [decode(arr[:, a, b], meta["scale"], meta["offset"]) for a, b in ((i0, j0), (i0, j1), (i1, j0), (i1, j1))]
            if vid in circular:
                rad = [np.radians(c) for c in corners]
                sin = bilinear(*(np.sin(r) for r in rad), wi, wj)
                cos = bilinear(*(np.cos(r) for r in rad), wi, wj)
                out[vid] = np.mod(np.degrees(np.arctan2(sin, cos)), 360.0).astype(np.float32)
            else:
                out[vid] = bilinear(*corners, wi, wj)
        return out

    def sea_point(self, var_id: str, lat: float, lon: float, max_km: float = 50.0) -> tuple[float, float, float] | None:
        """For a sea-only variable at a point with no sea around it (on land or a coast the
        model grid treats as land): the nearest grid point with data within ``max_km``, as
        (lat, lon, distance_km). None when the point's own cells have data, or none is near."""
        g = self.grid
        arr = self.array(var_id)
        i0, i1, _ = (int(np.asarray(a).item()) for a in g.row_weights(lat))
        j0, j1, _ = (int(np.asarray(a).item()) for a in g.col_weights(lon))
        if any((arr[:, a, b] != MISSING).any() for a, b in ((i0, j0), (i0, j1), (i1, j0), (i1, j1))):
            return None
        reach = max(1, math.ceil(max_km / (111.2 * g.dlat)))
        rows = np.arange(max(0, i0 - reach), min(g.nlat, i1 + reach + 1))
        cols = np.arange(j0 - reach, j1 + reach + 1)
        cols = cols % g.nlon if g.wraps else cols[(cols >= 0) & (cols < g.nlon)]
        has = (arr[:, rows[:, None], cols[None, :]] != MISSING).any(axis=0)
        if not has.any():
            return None
        plat = g.lat0 - rows[:, None] * g.dlat
        plon = g.lon0 + cols[None, :] * g.dlon
        dist = haversine_km(lat, lon, plat, plon)
        dist = np.where(has, dist, np.inf)
        a, b = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[a, b]) or dist[a, b] > max_km:
            return None
        glat, glon = g.latlon_of(int(rows[a]), int(cols[b]))
        return glat, glon, round(float(dist[a, b]), 1)

    def point_static(self, name: str, lat: float, lon: float) -> float | None:
        field = self.static(name)
        if field is None:
            return None
        i0, i1, wi = (np.asarray(a).item() for a in self.grid.row_weights(lat))
        j0, j1, wj = (np.asarray(a).item() for a in self.grid.col_weights(lon))
        v = bilinear(*(np.float32(field[a, b]) for a, b in ((i0, j0), (i0, j1), (i1, j0), (i1, j1))), wi, wj)
        return None if np.isnan(v) else float(v)

    def sample_field(self, var_id: str, step: int, rows, cols) -> np.ndarray:
        """Bilinear sample of one step on a separable pixel grid (rows x cols)."""
        i0, i1, wi = rows
        j0, j1, wj = cols
        field, meta = self.array(var_id)[step], self.variables[var_id]
        s, o = meta["scale"], meta["offset"]
        r0, r1, c0, c1 = i0[:, None], i1[:, None], j0[None, :], j1[None, :]
        return bilinear(
            decode(field[r0, c0], s, o), decode(field[r0, c1], s, o),
            decode(field[r1, c0], s, o), decode(field[r1, c1], s, o),
            wi[:, None], wj[None, :],
        )


class RunStore:
    """Resolves run references ("latest" or YYYYMMDDHH) to cached ``Run`` objects."""

    def __init__(self, data_root: Path, model: str, max_cached: int = 4) -> None:
        self.model = model
        self.root = data_root / model
        self.max_cached = max_cached
        self._cache: dict[str, tuple[int, Run]] = {}
        self._latest: tuple[int, str | None] = (-1, None)
        self._lock = threading.Lock()

    def latest_id(self) -> str | None:
        try:
            mtime = (self.root / "latest.json").stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if mtime != self._latest[0]:
            self._latest = (mtime, read_latest(self.root))
        return self._latest[1]

    def run_ids(self) -> list[str]:
        return published_runs(self.root)

    def get(self, ref: str = "latest") -> Run | None:
        rid = self.latest_id() if ref == "latest" else ref
        if rid is None or not RUN_ID_RE.match(rid):
            return None
        manifest = self.root / "runs" / rid / "manifest.json"
        try:
            mtime = manifest.stat().st_mtime_ns
        except FileNotFoundError:
            with self._lock:
                self._cache.pop(rid, None)
            return None
        with self._lock:
            hit = self._cache.get(rid)
            if hit is not None and hit[0] == mtime:
                return hit[1]
            run = Run(manifest.parent)
            self._cache[rid] = (mtime, run)
            while len(self._cache) > self.max_cached:
                oldest = min(k for k in self._cache if k != rid)
                self._cache.pop(oldest)
            return run
