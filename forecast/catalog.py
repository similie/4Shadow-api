"""Variable catalog: what we ingest, how it is stored, and how it is presented.

Adding a variable is a catalog entry, not code: give it a GRIB selector (the abbreviation
and level text exactly as they appear in NOAA's .idx files), a storage quantum and a palette.

Series semantics:
  * ``instant`` values are valid at the step's time.
  * ``accum`` / ``average`` GRIB fields are converted at ingest into the mean *rate* over the
    interval ending at the step (e.g. precipitation in mm/h), so 1-hourly and 3-hourly steps
    are directly comparable and daily totals are sum(rate x interval hours).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import palettes
from .palettes import Palette


@dataclass(frozen=True)
class Variable:
    id: str
    name: str
    units: str
    description: str
    kind: str = "instant"  # instant | accum | average  (how the source stores it)
    grib_param: str | None = None
    grib_level: str | None = None
    mul: float = 1.0  # unit conversion applied at ingest: value * mul + add
    add: float = 0.0
    clip: tuple[float | None, float | None] = (None, None)
    scale: float = 0.01  # int16 storage quantum
    offset: float = 0.0
    decimals: int = 1
    palette: Palette | None = None
    derived_from: tuple[str, ...] = ()  # non-empty => computed at read time from these

    @property
    def stored(self) -> bool:
        return not self.derived_from

    @property
    def interval(self) -> bool:
        """True when each value describes the interval ending at its step, not an instant."""
        return self.kind in ("accum", "average")

    def describe(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "units": self.units,
            "description": self.description,
            "semantics": "interval_mean" if self.interval else "instant",
            "tileable": self.palette is not None,
        }
        if self.derived_from:
            out["derived_from"] = list(self.derived_from)
        if self.palette is not None:
            out["legend"] = self.palette.legend()
        return out


@dataclass(frozen=True)
class StaticField:
    id: str
    name: str
    units: str
    grib_param: str
    grib_level: str


@dataclass(frozen=True)
class DailySpec:
    id: str
    source: str
    how: str  # min | max | mean | sum
    units: str
    factor: float = 1.0  # for sums: rate * hours * factor
    decimals: int = 1


def derive(var_id: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    if var_id == "wind_speed":
        return np.hypot(inputs["u10"], inputs["v10"])
    if var_id == "wind_dir":
        # Meteorological convention: direction the wind blows FROM, clockwise from north.
        return np.mod(270.0 - np.degrees(np.arctan2(inputs["v10"], inputs["u10"])), 360.0)
    raise KeyError(var_id)


@dataclass(frozen=True)
class ModelCatalog:
    model: str
    title: str
    variables: dict[str, Variable]
    static: dict[str, StaticField] = field(default_factory=dict)
    daily: tuple[DailySpec, ...] = ()

    def stored_variables(self) -> list[Variable]:
        return [v for v in self.variables.values() if v.stored]

    def get(self, var_id: str) -> Variable | None:
        return self.variables.get(var_id)

    def available(self, run_variables) -> list[Variable]:
        """Catalog variables a given run can serve (stored ones present + derivable ones)."""
        present = set(run_variables)
        out = []
        for v in self.variables.values():
            if v.stored and v.id in present:
                out.append(v)
            elif v.derived_from and all(d in present for d in v.derived_from):
                out.append(v)
        return out


def _vars(*items: Variable) -> dict[str, Variable]:
    return {v.id: v for v in items}


K_TO_C = -273.15

GFS = ModelCatalog(
    model="gfs",
    title="NOAA GFS 0.25°",
    variables=_vars(
        Variable("t2m", "Air temperature (2 m)", "°C", "Air temperature 2 m above ground.",
                 grib_param="TMP", grib_level="2 m above ground", add=K_TO_C,
                 palette=palettes.TEMPERATURE_C),
        Variable("d2m", "Dew point (2 m)", "°C", "Dew point temperature 2 m above ground.",
                 grib_param="DPT", grib_level="2 m above ground", add=K_TO_C,
                 palette=palettes.TEMPERATURE_C),
        Variable("rh2m", "Relative humidity (2 m)", "%", "Relative humidity 2 m above ground.",
                 grib_param="RH", grib_level="2 m above ground", clip=(0, 100), decimals=0,
                 palette=palettes.RELATIVE_HUMIDITY),
        Variable("u10", "Wind u-component (10 m)", "m/s", "Eastward wind 10 m above ground.",
                 grib_param="UGRD", grib_level="10 m above ground"),
        Variable("v10", "Wind v-component (10 m)", "m/s", "Northward wind 10 m above ground.",
                 grib_param="VGRD", grib_level="10 m above ground"),
        Variable("wind_speed", "Wind speed (10 m)", "m/s", "Wind speed 10 m above ground.",
                 derived_from=("u10", "v10"), palette=palettes.WIND_MS),
        Variable("wind_dir", "Wind direction (10 m)", "°",
                 "Direction the 10 m wind blows from, clockwise from north.",
                 derived_from=("u10", "v10"), decimals=0),
        Variable("gust", "Wind gust", "m/s", "Surface wind gust.",
                 grib_param="GUST", grib_level="surface", clip=(0, None), palette=palettes.WIND_MS),
        Variable("mslp", "Mean sea-level pressure", "hPa", "Pressure reduced to mean sea level.",
                 grib_param="PRMSL", grib_level="mean sea level", mul=0.01, scale=0.1,
                 palette=palettes.PRESSURE_HPA),
        Variable("sp", "Surface pressure", "hPa", "Air pressure at the model surface (not reduced to sea level).",
                 grib_param="PRES", grib_level="surface", mul=0.01, scale=0.1),
        Variable("q2m", "Specific humidity (2 m)", "g/kg", "Mass of water vapour per mass of air, 2 m above ground.",
                 grib_param="SPFH", grib_level="2 m above ground", mul=1000.0, clip=(0, 32), scale=0.001,
                 decimals=2),
        Variable("precip", "Precipitation", "mm/h",
                 "Mean precipitation rate over the interval ending at this step.",
                 kind="accum", grib_param="APCP", grib_level="surface", clip=(0, None), decimals=2,
                 palette=palettes.PRECIP_RATE_MMH),
        Variable("tcc", "Total cloud cover", "%", "Total cloud cover of the whole atmosphere.",
                 grib_param="TCDC", grib_level="entire atmosphere", clip=(0, 100), decimals=0,
                 palette=palettes.CLOUD_COVER),
        Variable("cape", "CAPE", "J/kg", "Surface-based convective available potential energy.",
                 grib_param="CAPE", grib_level="surface", clip=(0, None), scale=1.0, decimals=0,
                 palette=palettes.CAPE_JKG),
        Variable("pwat", "Precipitable water", "mm", "Total column precipitable water.",
                 grib_param="PWAT", grib_level="entire atmosphere (considered as a single layer)",
                 clip=(0, None), palette=palettes.PRECIPITABLE_WATER_MM),
        Variable("vis", "Visibility", "m", "Surface visibility.",
                 grib_param="VIS", grib_level="surface", clip=(0, None), scale=1.0, decimals=0,
                 palette=palettes.VISIBILITY_M),
        Variable("dswrf", "Solar radiation", "W/m²",
                 "Mean downward short-wave radiation at the surface over the interval ending at this step.",
                 kind="average", grib_param="DSWRF", grib_level="surface", clip=(0, None), scale=0.1,
                 decimals=0, palette=palettes.SHORTWAVE_WM2),
        Variable("snod", "Snow depth", "m", "Snow depth on the ground.",
                 grib_param="SNOD", grib_level="surface", clip=(0, None), decimals=2,
                 palette=palettes.SNOW_DEPTH_M),
    ),
    static={
        "orog": StaticField("orog", "Model surface elevation", "m", "HGT", "surface"),
        "land": StaticField("land", "Land fraction", "1", "LAND", "surface"),
    },
    daily=(
        DailySpec("t2m_max", "t2m", "max", "°C"),
        DailySpec("t2m_min", "t2m", "min", "°C"),
        DailySpec("precip_sum", "precip", "sum", "mm", decimals=1),
        DailySpec("wind_speed_max", "wind_speed", "max", "m/s"),
        DailySpec("gust_max", "gust", "max", "m/s"),
        DailySpec("rh2m_mean", "rh2m", "mean", "%", decimals=0),
        DailySpec("tcc_mean", "tcc", "mean", "%", decimals=0),
        DailySpec("dswrf_sum", "dswrf", "sum", "MJ/m²", factor=0.0036, decimals=1),
    ),
)

CATALOGS: dict[str, ModelCatalog] = {GFS.model: GFS}
