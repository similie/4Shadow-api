"""The US EPA's Air Quality Index (AQI), from pollutant concentrations.

As EPA's Technical Assistance Document for the Reporting of Daily Air Quality (EPA-403/B-26-003, May 2026)
sets it out: each pollutant's concentration, averaged over its own window and truncated, falls between two
breakpoints of Table 6 and is scaled in a straight line to the index between theirs (Equation 1), rounded
to a whole number; the AQI is the highest of the pollutants' indices. The windows are 24 hours for PM2.5
and PM10, 8 hours for ozone and carbon monoxide (ozone's 1-hour value counts too when it is the higher),
and 1 hour for nitrogen dioxide and sulphur dioxide, whose 24-hour mean takes over above 304 ppb.

Categories: 0-50 good, 51-100 moderate, 101-150 unhealthy for sensitive groups, 151-200 unhealthy, 201-300
very unhealthy, 301 and up hazardous. Above 500 the last segment goes on (the document's footnote 4).
"""

from __future__ import annotations

from collections import deque

import numpy as np

# (low, high) concentration and the (low, high) index they map to, category by category.
Table = tuple[tuple[float, float, int, int], ...]

PM25_24H: Table = ((0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200),
                   (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500))  # µg/m³
PM10_24H: Table = ((0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
                   (355, 424, 201, 300), (425, 604, 301, 500))  # µg/m³
O3_8H: Table = ((0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150), (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300))  # ppm; 8-hour values don't define an AQI of 301 or more
O3_1H: Table = ((0.125, 0.164, 101, 150), (0.165, 0.204, 151, 200), (0.205, 0.404, 201, 300),
                (0.405, 0.604, 301, 500))  # ppm; none below 0.125
CO_8H: Table = ((0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500))  # ppm
SO2_1H: Table = ((0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200))  # ppb
SO2_24H: Table = ((305, 604, 201, 300), (605, 1004, 301, 500))  # ppb; only from 305
NO2_1H: Table = ((0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200),
                 (650, 1249, 201, 300), (1250, 2049, 301, 500))  # ppb

# Decimal places each concentration is truncated to before it is placed in the table.
DIGITS = {"pm25": 1, "pm10": 0, "o3": 3, "co": 1, "so2": 0, "no2": 0}


def truncate(values, digits: int) -> np.ndarray:
    """Cut to ``digits`` decimal places (not rounded), as the AQI wants; negatives (a model's noise) are 0."""
    scale = 10.0**digits
    # The nudge keeps a value stored as 35.39999... at 35.4.
    return np.floor(np.maximum(np.asarray(values, np.float64), 0.0) * scale + 1e-6) / scale


def sub_index(values, table: Table, digits: int) -> np.ndarray:
    """A pollutant's index (not yet rounded): NaN below the table's first breakpoint."""
    c = truncate(values, digits)
    out = np.full(c.shape, np.nan)
    # Truncated, every concentration from the first breakpoint up is in one segment; the last goes on.
    for lo, hi, i_lo, i_hi in table:
        inside = c >= lo
        out[inside] = (i_hi - i_lo) / (hi - lo) * (c[inside] - lo) + i_lo
    return out


def us_aqi(*, pm25_24h, pm10_24h, o3_8h, o3_1h, co_8h, so2_1h, so2_24h, no2_1h) -> np.ndarray:
    """The AQI from each pollutant's averages: PM in µg/m³, ozone and carbon monoxide in ppm, sulphur and
    nitrogen dioxide in ppb. NaN where any is missing."""
    o3 = sub_index(o3_8h, O3_8H, DIGITS["o3"])
    o3 = np.where(truncate(o3_8h, DIGITS["o3"]) > O3_8H[-1][1], O3_8H[-1][3], o3)
    o3 = np.fmax(o3, sub_index(o3_1h, O3_1H, DIGITS["o3"]))
    # Sulphur dioxide: the hour's value up to 304 ppb; above, the 24-hour mean from 305 ppb, or 200 if it is
    # lower (the document's "use 200 for the lower and upper AQI breakpoints").
    high = sub_index(so2_24h, SO2_24H, DIGITS["so2"])
    so2 = np.where(truncate(so2_1h, DIGITS["so2"]) <= SO2_1H[-1][1], sub_index(so2_1h, SO2_1H, DIGITS["so2"]),
                   np.where(np.isnan(high), SO2_1H[-1][3], high))
    indices = (sub_index(pm25_24h, PM25_24H, DIGITS["pm25"]), sub_index(pm10_24h, PM10_24H, DIGITS["pm10"]),
               o3, sub_index(co_8h, CO_8H, DIGITS["co"]), so2, sub_index(no2_1h, NO2_1H, DIGITS["no2"]))
    aqi = np.max(np.stack(indices), axis=0)  # NaN wherever any is NaN
    # Rounded half up, as the document's examples are (125.5 is 126).
    return np.floor(aqi + 0.5).astype(np.float32)


class Trailing:
    """The mean of the last ``hours`` values pushed, one an hour: fewer at the start of a forecast, which
    has no hours before its first."""

    def __init__(self, hours: int) -> None:
        self.hours = hours
        self._held: deque[np.ndarray] = deque()
        self._total: np.ndarray | None = None

    def push(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, np.float64)
        self._held.append(values)
        self._total = values.copy() if self._total is None else self._total + values
        if len(self._held) > self.hours:
            self._total -= self._held.popleft()
        return (self._total / len(self._held)).astype(np.float32)


# Molar masses (g/mol), to turn a mass mixing ratio into a volume one (ppb): dry air and each gas.
AIR = 28.9644
MOLAR_MASS = {"no2": 46.0055, "co": 28.0101, "o3": 47.9982, "so2": 64.064}


class HourlyAqi:
    """The AQI hour by hour through a forecast, from each hour's concentrations: keeps the windows."""

    def __init__(self) -> None:
        self.pm25, self.pm10, self.so2 = Trailing(24), Trailing(24), Trailing(24)
        self.o3, self.co = Trailing(8), Trailing(8)

    def push(self, *, pm25, pm10, o3, co, so2, no2) -> np.ndarray:
        """The next hour: PM in µg/m³, gases as mass mixing ratios (kg/kg)."""
        ppb = {gas: np.asarray(mmr, np.float64) * (AIR / MOLAR_MASS[gas]) * 1e9
               for gas, mmr in (("o3", o3), ("co", co), ("so2", so2), ("no2", no2))}
        return us_aqi(pm25_24h=self.pm25.push(pm25), pm10_24h=self.pm10.push(pm10),
                      o3_8h=self.o3.push(ppb["o3"] / 1000), o3_1h=ppb["o3"] / 1000,
                      co_8h=self.co.push(ppb["co"] / 1000), so2_1h=ppb["so2"], so2_24h=self.so2.push(ppb["so2"]),
                      no2_1h=ppb["no2"])
