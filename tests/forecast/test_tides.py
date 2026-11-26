"""High and low waters from an hourly tide series: placed between the hours, rounding's flat turns and
wiggles, and the series' ends and gaps."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from forecast.tides import extremes

T0 = datetime(2026, 11, 24, tzinfo=timezone.utc)
M2 = 12.42  # hours: the main lunar tide's period


def hourly(n):
    return [T0 + timedelta(hours=k) for k in range(n)]


def check(events, first_high, amplitude, minutes, metres):
    """Alternating highs and lows every half period from first_high, within the given errors."""
    for n, event in enumerate(events):
        hour = first_high + n * M2 / 2
        assert event["type"] == ("low" if n % 2 else "high")
        assert abs((event["time"] - (T0 + timedelta(hours=hour))).total_seconds()) <= minutes * 60
        assert event["height"] == pytest.approx(amplitude if n % 2 == 0 else -amplitude, abs=metres)


@pytest.mark.parametrize("first_high", [2.0, 2.3, 2.5, 2.8])
def test_turns_fall_between_the_hours(first_high):
    hours = np.arange(49)
    events = extremes(hourly(49), 1.5 * np.cos(2 * np.pi * (hours - first_high) / M2))
    assert len(events) == 8
    check(events, first_high, 1.5, minutes=3, metres=0.01)


@pytest.mark.parametrize("first_high", [2.0, 2.3, 2.5, 2.8])
def test_heights_rounded_to_centimetres(first_high):
    """Copernicus Marine's heights are whole centimetres, so a slow turn can span equal hours."""
    hours = np.arange(49)
    small = np.round(0.12 * np.cos(2 * np.pi * (hours - first_high) / M2), 2)  # a Mediterranean tide
    events = extremes(hourly(49), small)
    assert len(events) == 8
    check(events, first_high, 0.12, minutes=25, metres=0.01)


def test_a_wiggle_from_rounding_is_not_a_tide():
    values = np.array([0.0, 0.5, 0.8, 0.81, 0.80, 0.81, 0.6, 0.2, -0.3])
    events = extremes(hourly(len(values)), values)
    assert [e["type"] for e in events] == ["high"]
    assert 3 <= (events[0]["time"] - T0).total_seconds() / 3600 <= 5
    assert events[0]["height"] == pytest.approx(0.81, abs=0.02)
    assert extremes(hourly(5), np.array([0.0, 0.01, 0.0, 0.01, 0.0])) == []


def test_ends_and_gaps():
    assert extremes(hourly(6), np.arange(6.0)) == []  # still rising: the next turn is after the series
    assert extremes(hourly(2), np.array([0.0, 1.0])) == []
    assert extremes(hourly(5), np.full(5, np.nan)) == []  # no sea
    # A gap ends the usable part: the low after it is not given.
    values = np.array([0.0, 1.0, 0.0, -1.0, np.nan, -1.0, 0.0])
    assert [e["type"] for e in extremes(hourly(len(values)), values)] == ["high"]
