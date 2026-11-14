"""Threshold checks: when and where does the forecast cross a set of thresholds in an area?

The stateless engine behind early warnings. A client (the 4Shadow site's alert worker, or a map
previewing an alert while it is drawn) sends an area and conditions, such as "24-hour rain of
50 mm or more anywhere in this catchment", and gets back the periods within a lead time when the
forecast meets them. Storing alerts, matching events from one run to the next and sending
webhooks happen in the client, not here.

Areas are a point (interpolated like a point forecast), a point with a radius, or a polygon. Over
an area, a condition holds somewhere (``any``), on average (``mean``) or over a share of it
(``coverage``), judged on the model's grid cells whose centres fall inside.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .catalog import ModelCatalog, Variable, derive
from .grid import Grid, bilinear
from .service import ForecastError
from .store import Run, decode, haversine_km, iso

MAX_CELLS = 20_000  # about 15 million km² at 0.25°: a large country
MAX_VERTICES = 10_000
MAX_CONDITIONS = 5
MAX_LEAD_DAYS = 16
MERGE_GAP_HOURS = 6  # crossings closer together than this are one event
KM_PER_DEG = 111.32

Operator = Literal[">", ">=", "<", "<="]
OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


class Condition(BaseModel):
    variable: str = Field(..., description=(
        "A variable id from /v2/{model}/variables: precip, t2m, wind_speed, gust, wave_height, ... "
        "Directions (wind_dir, wave_dir, swell_dir) can't be thresholds."))
    operator: Operator
    threshold: float = Field(..., description="In the variable's units (or the accumulated units, e.g. mm)")
    accumulate_hours: int | None = Field(None, ge=1, le=240, description=(
        "For rates such as precip (mm/h): compare the total over this many hours instead, e.g. 24 for "
        "mm per 24 h. Totals are rounded to the model's steps (hourly to 5 days, then 3-hourly)."))


class ThresholdRequest(BaseModel):
    area: dict[str, Any] = Field(..., description="GeoJSON geometry: Point, Polygon or MultiPolygon, positions as [lon, lat]")
    radius_km: float = Field(0.0, ge=0, le=500, description=(
        "With a Point: check every grid cell within this distance. 0 checks the point itself, interpolated "
        "like a point forecast"))
    conditions: list[Condition] = Field(..., min_length=1, max_length=MAX_CONDITIONS)
    match: Literal["all", "any"] = Field("all", description=(
        "all: every condition at the same place and time; any: at least one of them"))
    area_mode: Literal["any", "mean", "coverage"] = Field("any", description=(
        "any: somewhere in the area; mean: the area average; coverage: at least coverage_percent of the area"))
    coverage_percent: float | None = Field(None, gt=0, le=100)
    lead_days: float = Field(3, gt=0, le=MAX_LEAD_DAYS, description="How far ahead to look, in days from now")
    min_duration_hours: float = Field(0, ge=0, le=240, description="Report only crossings lasting at least this long")
    run: str = Field("latest", description="'latest' or a run id YYYYMMDDHH")
    at: datetime | None = Field(None, description="Evaluate as if it were this time (UTC). Default: now")
    timeline: bool = Field(False, description="Also return the area's values at every step of the window")

    @model_validator(mode="after")
    def _coverage(self):
        if self.area_mode == "coverage" and self.coverage_percent is None:
            raise ValueError("area_mode 'coverage' needs coverage_percent")
        return self


# --------------------------------------------------------------------------- areas


@dataclass
class Area:
    """Where to look: grid cells, or (for a point) the four cells around it and their weights."""

    kind: str  # point | radius | polygon
    rows: np.ndarray
    cols: np.ndarray
    lats: np.ndarray  # per cell (the point itself for a point)
    lons: np.ndarray
    bbox: list[float]
    center: tuple[float, float]  # lat, lon
    area_km2: float | None = None
    weights: tuple[float, float] | None = None  # point: (wi, wj) toward the second row / column
    sea_point: dict | None = field(default=None)

    @property
    def size(self) -> int:
        return 1 if self.kind == "point" else len(self.rows)

    def sample(self, run: Run, var_id: str) -> np.ndarray:
        """A stored variable's values, [steps, cells]; NaN where missing."""
        meta = run.variables[var_id]
        raw = run.array(var_id)[:, self.rows, self.cols]
        vals = decode(raw, meta["scale"], meta["offset"])
        if self.kind == "point":
            wi, wj = self.weights
            vals = bilinear(vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], wi, wj)[:, None]
        return vals


def _positions(coords, depth: int) -> list:
    """Validate nested GeoJSON coordinates (depth 1 = a ring, 2 = a polygon, 3 = a multipolygon)."""
    if not isinstance(coords, list) or not coords:
        raise ForecastError(422, "Empty or malformed coordinates in area")
    if depth == 1:
        ring = []
        for p in coords:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(c, (int, float)) for c in p[:2])):
                raise ForecastError(422, "Area positions must be [lon, lat] numbers")
            lon, lat = float(p[0]), float(p[1])
            if not (-90 <= lat <= 90 and -540 <= lon <= 540 and math.isfinite(lon)):
                raise ForecastError(422, f"Position out of range: [{lon}, {lat}]")
            ring.append((lon, lat))
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            raise ForecastError(422, "A polygon ring needs at least 3 distinct positions")
        return ring
    return [_positions(c, depth - 1) for c in coords]


def _cell_area_km2(grid: Grid, lats: np.ndarray) -> float:
    return float(np.sum(grid.dlat * grid.dlon * KM_PER_DEG**2 * np.cos(np.radians(lats))))


def _cells(grid: Grid, rows: np.ndarray, cols: np.ndarray, kind: str, bbox, center) -> Area:
    if rows.size > MAX_CELLS:
        raise ForecastError(422, f"Area too large: {rows.size} grid cells; at most {MAX_CELLS} "
                                 f"(about {MAX_CELLS * 0.0625 * KM_PER_DEG**2 / 1e6:.0f} million km² at the equator)")
    lats = grid.lat0 - rows * grid.dlat
    lons = (grid.lon0 + cols * grid.dlon + 180.0) % 360.0 - 180.0
    return Area(kind, rows.astype(np.intp), cols.astype(np.intp), lats, lons, bbox, center,
                area_km2=round(_cell_area_km2(grid, lats), 1))


def _candidate_rows(grid: Grid, south: float, north: float) -> np.ndarray:
    i0 = max(0, math.ceil((grid.lat0 - north) / grid.dlat - 1e-9))
    i1 = min(grid.nlat - 1, math.floor((grid.lat0 - south) / grid.dlat + 1e-9))
    return np.arange(i0, i1 + 1)


def _candidate_cols(grid: Grid, west: float, east: float) -> tuple[np.ndarray, np.ndarray]:
    """Columns whose centre lies in [west, east], with their longitudes shifted into that range."""
    cols = np.arange(grid.nlon)
    lon = grid.lon0 + cols * grid.dlon
    shifted = west + np.mod(lon - west, 360.0)
    keep = shifted <= east + 1e-9
    return cols[keep], shifted[keep]


def _polygon_area(grid: Grid, polygons: list[list[list[tuple[float, float]]]]) -> Area:
    nverts = sum(len(r) for poly in polygons for r in poly)
    if nverts > MAX_VERTICES:
        raise ForecastError(422, f"Area has {nverts} positions; at most {MAX_VERTICES}")
    pts = np.array([p for poly in polygons for ring in poly for p in ring], dtype=np.float64)
    west, east = float(pts[:, 0].min()), float(pts[:, 0].max())
    south, north = float(pts[:, 1].min()), float(pts[:, 1].max())
    if east - west > 360:
        raise ForecastError(422, "Area spans more than 360° of longitude")
    # Every edge of every ring; even-odd filling handles holes and the parts of a MultiPolygon.
    x1, y1, x2, y2 = [], [], [], []
    for poly in polygons:
        for ring in poly:
            r = np.asarray(ring, dtype=np.float64)
            x1.append(r[:-1, 0]), y1.append(r[:-1, 1]), x2.append(r[1:, 0]), y2.append(r[1:, 1])
    x1, y1, x2, y2 = (np.concatenate(a) for a in (x1, y1, x2, y2))
    rows_c = _candidate_rows(grid, south, north)
    cols_c, lon_c = _candidate_cols(grid, west, east)
    rows, cols = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in rows_c:  # scanline: the edge crossings on each row's latitude, left to right
            y = grid.lat0 - i * grid.dlat
            m = (y1 > y) != (y2 > y)
            if not m.any():
                continue
            xs = np.sort(x1[m] + (y - y1[m]) * (x2[m] - x1[m]) / (y2[m] - y1[m]))
            inside = np.searchsorted(xs, lon_c, side="right") % 2 == 1
            rows.append(np.full(int(inside.sum()), i)), cols.append(cols_c[inside])
    center_lon = float(pts[:, 0].mean())
    center = (float(pts[:, 1].mean()), (center_lon + 180.0) % 360.0 - 180.0)
    bbox = [round(west, 4), round(south, 4), round(east, 4), round(north, 4)]
    r = np.concatenate(rows) if rows else np.array([], dtype=np.intp)
    c = np.concatenate(cols) if cols else np.array([], dtype=np.intp)
    if r.size == 0:  # smaller than a grid cell: the cell nearest its middle
        i, j = grid.nearest_index(*center)
        r, c = np.array([i]), np.array([j])
    return _cells(grid, r, c, "polygon", bbox, center)


def build_area(grid: Grid, geometry: dict, radius_km: float = 0.0) -> Area:
    gtype = geometry.get("type") if isinstance(geometry, dict) else None
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if gtype == "Point":
        if not (isinstance(coords, list) and len(coords) >= 2 and all(isinstance(c, (int, float)) for c in coords[:2])):
            raise ForecastError(422, "A Point needs coordinates [lon, lat]")
        lon, lat = float(coords[0]), float(coords[1])
        if not (-90 <= lat <= 90 and -540 <= lon <= 540):
            raise ForecastError(422, f"Point out of range: [{lon}, {lat}]")
        lon = (lon + 180.0) % 360.0 - 180.0
        if radius_km > 0:
            dlat = radius_km / KM_PER_DEG
            dlon = min(180.0, radius_km / (KM_PER_DEG * max(math.cos(math.radians(min(89.0, abs(lat) + dlat))), 0.01)))
            rows_c = _candidate_rows(grid, lat - dlat, lat + dlat)
            cols_c, _ = _candidate_cols(grid, lon - dlon, lon + dlon)
            rr, cc = np.meshgrid(rows_c, cols_c, indexing="ij")
            rr, cc = rr.ravel(), cc.ravel()
            d = haversine_km(lat, lon, grid.lat0 - rr * grid.dlat, grid.lon0 + cc * grid.dlon)
            near = d <= radius_km
            i, j = grid.nearest_index(lat, lon)  # always at least the cell the point is in
            rows = np.concatenate([rr[near], [i]])
            cols = np.concatenate([cc[near], [j]])
            uniq = np.unique(np.stack([rows, cols]), axis=1)
            bbox = [round(lon - dlon, 4), round(lat - dlat, 4), round(lon + dlon, 4), round(lat + dlat, 4)]
            return _cells(grid, uniq[0], uniq[1], "radius", bbox, (lat, lon))
        i0, i1, wi = (np.asarray(a).item() for a in grid.row_weights(lat))
        j0, j1, wj = (np.asarray(a).item() for a in grid.col_weights(lon))
        return Area("point", np.array([i0, i0, i1, i1]), np.array([j0, j1, j0, j1]), np.array([lat]), np.array([lon]),
                    [lon, lat, lon, lat], (lat, lon), weights=(float(wi), float(wj)))
    if radius_km > 0:
        raise ForecastError(422, "radius_km goes with a Point area")
    if gtype == "Polygon":
        return _polygon_area(grid, [_positions(coords, 2)])
    if gtype == "MultiPolygon":
        return _polygon_area(grid, _positions(coords, 3))
    raise ForecastError(422, "area must be a GeoJSON Point, Polygon or MultiPolygon geometry")


# --------------------------------------------------------------------------- evaluation


def _variable(cat: ModelCatalog, run: Run, c: Condition) -> Variable:
    available = {v.id: v for v in cat.available(run.variables)}
    v = available.get(c.variable)
    if v is None:
        raise ForecastError(422, f"Unknown variable {c.variable!r}. Available: {sorted(available)}")
    if v.circular or v.units == "°":
        raise ForecastError(422, f"{v.id} is a direction; it can't be a threshold")
    if c.accumulate_hours and not (v.interval and v.units.endswith("/h")):
        raise ForecastError(422, f"accumulate_hours works with rates in units per hour (such as precip), not {v.id}")
    return v


def condition_units(v: Variable, c: Condition) -> str:
    return v.units[:-2] if c.accumulate_hours else v.units


def rolling_total(rate: np.ndarray, hours: np.ndarray, window: int) -> np.ndarray:
    """Totals over the ``window`` hours ending at each step, from rates per hour over each step's
    interval ([steps, cells]). Windows reaching back before the run start count from the start."""
    dt = np.diff(hours, prepend=hours[0]).astype(np.float32)
    cum = np.cumsum(np.nan_to_num(rate, nan=0.0) * dt[:, None], axis=0)
    j = np.searchsorted(hours, hours - window, side="right") - 1
    before = np.where(j[:, None] >= 0, cum[np.maximum(j, 0)], np.float32(0))
    total = (cum - before).astype(np.float32)
    total[:, np.isnan(rate).all(axis=0)] = np.nan  # cells with no data at all stay missing
    return total


def _values(run: Run, cat: ModelCatalog, area: Area, v: Variable) -> np.ndarray:
    if v.derived_from:
        return derive(v.id, {d: area.sample(run, d) for d in v.derived_from}).astype(np.float32)
    vals = area.sample(run, v.id)
    if area.kind == "point" and v.marine and np.isnan(vals).all():
        # On a coast the model's cells are land: use the nearest open sea, as point forecasts do.
        lat, lon = area.center
        sea = run.sea_point(v.id, lat, lon)
        if sea is not None:
            sea_area = build_area(run.grid, {"type": "Point", "coordinates": [sea[1], sea[0]]})
            area.sea_point = {"lat": sea[0], "lon": sea[1], "distance_km": sea[2]}
            vals = sea_area.sample(run, v.id)
    return vals


def _extreme(op: str):
    return np.nanmax if op in (">", ">=") else np.nanmin


def _arg_extreme(op: str):
    return np.nanargmax if op in (">", ">=") else np.nanargmin


def _round(x: float | None, decimals: int) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), decimals)


def check_thresholds(run: Run, cat: ModelCatalog, req: ThresholdRequest, now: datetime | None = None) -> dict:
    now = req.at or now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    variables = [_variable(cat, run, c) for c in req.conditions]
    area = build_area(run.grid, req.area, req.radius_km)
    vt, hours = run.valid_times, run.forecast_hours
    warnings: list[str] = []

    # The window: from the step in effect now to the last step within the lead time.
    k_from = max(0, bisect.bisect_right(vt, now) - 1)
    k_to = bisect.bisect_right(vt, now + timedelta(days=req.lead_days)) - 1
    if k_to < k_from or vt[-1] < now:
        k_to = k_from - 1
        warnings.append(f"Run {run.id} doesn't reach {iso(now)}; nothing to check")
    elif vt[-1] < now + timedelta(days=req.lead_days):
        warnings.append(f"Run {run.id} ends at {iso(vt[-1])}, before the end of the {req.lead_days:g}-day window")

    series, stats_fn = [], []
    for c, v in zip(req.conditions, variables):
        vals = _values(run, cat, area, v)
        if c.accumulate_hours:
            vals = rolling_total(vals, hours, c.accumulate_hours)
            if k_to >= k_from and hours[k_from] < c.accumulate_hours:
                warnings.append(f"{c.accumulate_hours} h totals of {v.id} before {iso(run.init_time + timedelta(hours=c.accumulate_hours))} "
                                f"count only from the run's start, {iso(run.init_time)}")
        if np.isnan(vals[k_from:k_to + 1]).all() and k_to >= k_from:
            warnings.append(f"{v.id} has no data in this area" + (" (it is defined over open sea only)" if v.marine else ""))
        series.append(vals)

    with np.errstate(invalid="ignore"), _quiet():
        if req.area_mode == "mean":
            stats = [np.nanmean(s, axis=1) for s in series]
            mets = [OPS[c.operator](st, c.threshold) for c, st in zip(req.conditions, stats)]
            coverage = None
            where = None
        else:
            cell_mets = [OPS[c.operator](s, c.threshold) for c, s in zip(req.conditions, series)]
            stats = [_extreme(c.operator)(s, axis=1) for c, s in zip(req.conditions, series)]
            cell_met = np.logical_and.reduce(cell_mets) if req.match == "all" else np.logical_or.reduce(cell_mets)
            valid = (~np.isnan(series[0])).sum(axis=1)
            coverage = 100.0 * cell_met.sum(axis=1) / np.maximum(valid, 1)
            primary = series[0]
            filled = np.where(np.isnan(primary), -np.inf if req.conditions[0].operator in (">", ">=") else np.inf, primary)
            where = _arg_extreme(req.conditions[0].operator)(filled, axis=1)
        if req.area_mode == "mean":
            met = np.logical_and.reduce(mets) if req.match == "all" else np.logical_or.reduce(mets)
        elif req.area_mode == "coverage":
            met = coverage >= req.coverage_percent
        else:
            met = coverage > 0

    # Runs of steps that meet the conditions, merging short breaks.
    spans: list[list[int]] = []
    for k in range(k_from, k_to + 1):
        if not met[k]:
            continue
        if spans and (vt[k] - vt[spans[-1][1]]) <= timedelta(hours=MERGE_GAP_HOURS):
            spans[-1][1] = k
        else:
            spans.append([k, k])

    c0, v0 = req.conditions[0], variables[0]
    events = []
    for a, b in spans:
        duration = (vt[b] - vt[a]).total_seconds() / 3600
        if duration < req.min_duration_hours:
            continue
        steps = np.arange(a, b + 1)
        cond_info = []
        for c, v, st in zip(req.conditions, variables, stats):
            seg = st[a:b + 1]
            if np.isnan(seg).all():
                cond_info.append({"variable": v.id, "peak": None, "peak_time": None})
                continue
            k = int(steps[_arg_extreme(c.operator)(seg)])
            cond_info.append({"variable": v.id, "units": condition_units(v, c), "peak": _round(st[k], v.decimals),
                              "peak_time": iso(vt[k])})
        seg0 = stats[0][a:b + 1]
        kp = int(steps[_arg_extreme(c0.operator)(seg0)]) if not np.isnan(seg0).all() else a
        if where is not None and area.kind != "point":
            cell = int(where[kp])
            plat, plon = float(area.lats[cell]), float(area.lons[cell])
        else:
            plat, plon = area.center if area.sea_point is None else (area.sea_point["lat"], area.sea_point["lon"])
        events.append({
            "start": iso(vt[a]),
            "end": iso(vt[b]),
            "duration_hours": round(duration, 1),
            "ongoing": bool(a == k_from and vt[a] <= now),
            "peak": {
                "time": iso(vt[kp]),
                "variable": v0.id,
                "value": _round(stats[0][kp], v0.decimals),
                "units": condition_units(v0, c0),
                "lat": round(plat, 4),
                "lon": round(plon, 4),
                "coverage_percent": None if coverage is None else round(float(coverage[kp]), 1),
            },
            "max_coverage_percent": None if coverage is None else round(float(coverage[a:b + 1].max()), 1),
            "conditions": cond_info,
        })

    body: dict[str, Any] = {
        "model": run.model,
        "run": run.id,
        "init_time": iso(run.init_time),
        "evaluated_at": iso(now),
        "window": {"from": iso(vt[k_from]), "to": iso(vt[k_to])} if k_to >= k_from else None,
        "area": {
            "type": area.kind,
            "cells": area.size,
            "area_km2": area.area_km2,
            "bbox": area.bbox,
            "center": [round(area.center[1], 4), round(area.center[0], 4)],
            **({"sea_point": area.sea_point} if area.sea_point else {}),
        },
        "conditions": [{"variable": v.id, "name": v.name, "operator": c.operator, "threshold": c.threshold,
                        "units": condition_units(v, c), "accumulate_hours": c.accumulate_hours,
                        "statistic": _statistic(req, c)}
                       for c, v in zip(req.conditions, variables)],
        "match": req.match,
        "area_mode": req.area_mode,
        "events": events,
        "warnings": warnings,
    }
    if req.timeline:
        body["timeline"] = [{
            "time": iso(vt[k]),
            "met": bool(met[k]),
            "values": [_round(st[k], v.decimals) for st, v in zip(stats, variables)],
            "coverage_percent": None if coverage is None else round(float(coverage[k]), 1),
        } for k in range(k_from, k_to + 1)]
    return body


def _statistic(req: ThresholdRequest, c: Condition) -> str:
    total = f"{c.accumulate_hours} h total, " if c.accumulate_hours else ""
    if req.area_mode == "mean":
        return total + "area mean"
    return total + ("area maximum" if c.operator in (">", ">=") else "area minimum")


class _quiet:
    """Silence numpy's warnings about all-NaN slices (cells over land for sea variables)."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
