"""Time zones and local-day aggregation."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from .catalog import DailySpec

_finder = None
_finder_lock = threading.Lock()


def _timezone_finder():
    global _finder
    if _finder is None:
        with _finder_lock:
            if _finder is None:
                try:
                    from timezonefinder import TimezoneFinder
                except ImportError:  # optional dependency
                    _finder = False
                else:
                    _finder = TimezoneFinder()
    return _finder or None


def resolve_timezone(lat: float, lon: float, name: str | None = None) -> tuple[tzinfo, str]:
    """IANA zone for a request: explicit ``name``, else looked up from the coordinates.

    Falls back to a whole-hour offset from longitude if ``timezonefinder`` is unavailable.
    Raises ValueError for an unknown explicit name.
    """
    if name:
        if name.upper() in ("UTC", "Z", "GMT"):
            return timezone.utc, "UTC"
        try:
            return ZoneInfo(name), name
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone {name!r}") from exc
    lon = (lon + 180.0) % 360.0 - 180.0
    finder = _timezone_finder()
    if finder is not None:
        zone = finder.timezone_at(lng=lon, lat=lat)
        if zone:
            try:
                return ZoneInfo(zone), zone
            except ZoneInfoNotFoundError:
                pass
    hours = int(round(lon / 15.0))
    return timezone(timedelta(hours=hours)), f"UTC{hours:+d}"


def daily_aggregate(valid_times: list[datetime], forecast_hours: np.ndarray, series: dict[str, np.ndarray],
                    interval_vars: set[str], specs: tuple[DailySpec, ...], tz: tzinfo) -> dict:
    """Aggregate a forecast into local calendar days.

    Instant values belong to the local date of their timestamp. Interval values (rates over
    the interval ending at a step) belong to the local date of the interval's midpoint and
    are weighted by interval length, so 1-hourly and 3-hourly steps mix correctly.
    """
    fh = np.asarray(forecast_hours, dtype=np.float64)
    dt = np.diff(fh, prepend=fh[0])  # hours in the interval ending at each step; 0 for the first
    inst_day = np.array([vt.astimezone(tz).date().toordinal() for vt in valid_times])
    mid_day = np.array([(vt - timedelta(hours=h / 2)).astimezone(tz).date().toordinal() for vt, h in zip(valid_times, dt)])
    has_interval = dt > 0

    days = sorted(set(inst_day.tolist()) | set(mid_day[has_interval].tolist()))
    hours_covered = [float(dt[has_interval & (mid_day == d)].sum()) for d in days]
    out: dict[str, list] = {
        "date": [datetime.fromordinal(d).date().isoformat() for d in days],
        "hours_covered": hours_covered,
    }
    for spec in specs:
        src = series.get(spec.source)
        if src is None:
            continue
        is_interval = spec.source in interval_vars
        values: list[float | None] = []
        for d in days:
            if is_interval:
                mask = has_interval & (mid_day == d)
            else:
                mask = inst_day == d
            x, w = src[mask].astype(np.float64), dt[mask]
            ok = ~np.isnan(x)
            if not ok.any():
                values.append(None)
                continue
            x, w = x[ok], w[ok]
            if spec.how == "max":
                v = x.max()
            elif spec.how == "min":
                v = x.min()
            elif spec.how == "sum":
                v = (x * w).sum() * spec.factor
            elif spec.how == "mean":
                v = (x * w).sum() / w.sum() if is_interval and w.sum() > 0 else x.mean()
            else:
                raise ValueError(spec.how)
            values.append(round(float(v), spec.decimals))
        out[spec.id] = values
    return out
