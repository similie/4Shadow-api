"""High and low water from an hourly tide-height series.

A high water is where the series stops rising and then falls by at least ``min_range``; a low water the
reverse. That threshold keeps the source's rounding (whole centimetres) from making turns of its own: a
slow turn can span two equal hours, or wiggle by a centimetre, and it counts once. Each turn is then
placed between the hours by a least-squares parabola through the hours within ``min_range`` of it and
one hour either side: on a semidiurnal tide sampled hourly that puts the time within a few minutes and
the height within a centimetre or so. The turn is within half an hour of a sampled hour, so at the
fitted curvature ``a`` its height is at most ``|a| / 4`` beyond the extreme hour's; that bound keeps a
parabola fitted over a flat-topped turn from overshooting.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

MIN_RANGE_M = 0.03


def _pivots(values: np.ndarray, min_range: float) -> list[tuple[int, str]]:
    """The turns, alternating high and low: each the extreme value the series then leaves by min_range."""
    out: list[tuple[int, str]] = []
    hi = lo = 0
    trend = 0  # +1 rising towards a high, -1 falling towards a low, 0 not yet known
    for k in range(1, len(values)):
        v = values[k]
        if trend >= 0:
            if v > values[hi]:
                hi = k
            if values[hi] - v >= min_range:
                out.append((hi, "high"))
                trend, lo = -1, k
                continue
        if trend <= 0:
            if v < values[lo]:
                lo = k
            if v - values[lo] >= min_range:
                out.append((lo, "low"))
                trend, hi = 1, k
    return out


def _place(values: np.ndarray, k: int, kind: str, min_range: float) -> tuple[float, float] | None:
    """The turn's position (in steps from index 0) and height, or None when it is too near either end of
    the series to tell."""
    first = last = k
    while first > 0 and abs(values[first - 1] - values[k]) < min_range:
        first -= 1
    while last < len(values) - 1 and abs(values[last + 1] - values[k]) < min_range:
        last += 1
    if first == 0 or last == len(values) - 1:
        return None
    idx = np.arange(first - 1, last + 2)
    mid = (first + last) / 2.0
    a, b, c = np.polyfit(idx - mid, values[idx], 2)
    if (a >= 0) if kind == "high" else (a <= 0):  # not shaped as a turn: the middle of the flat part
        return mid, float(values[k])
    # A turn lies within its flat part's half-steps either side.
    half = (last - first) / 2.0 + 0.5
    x = float(np.clip(-b / (2 * a), -half, half))
    height = float(a * x * x + b * x + c)
    beyond = abs(a) / 4 if kind == "high" else -abs(a) / 4
    return mid + x, float(np.clip(height, *sorted((values[k], values[k] + beyond))))


def extremes(times: Sequence[datetime], heights: np.ndarray, min_range: float = MIN_RANGE_M) -> list[dict]:
    """The high and low waters within an hourly series, in order: ``{"time", "type", "height"}`` with
    ``time`` a datetime and ``height`` in the series' units. Missing values end the series' usable part
    (a point with no sea has none). Turns at either end of the series are not known, so not given."""
    values = np.asarray(heights, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        values = values[: int(np.argmin(finite))]
    events = []
    for k, kind in _pivots(values, min_range):
        placed = _place(values, k, kind, min_range)
        if placed is None:
            continue
        pos, height = placed
        whole = min(int(np.floor(pos)), len(times) - 2)
        events.append({"time": times[whole] + (times[whole + 1] - times[whole]) * (pos - whole), "type": kind,
                       "height": height})
    return events
