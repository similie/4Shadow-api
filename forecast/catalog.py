"""Variable catalog: what we ingest, how it is stored, and how it is presented.

Adding a variable is a catalog entry, not code: give it a GRIB selector (the abbreviation
and level text exactly as they appear in NOAA's .idx files), a storage quantum and a palette.

Series semantics:
  * ``instant`` values are valid at the step's time.
  * ``accum`` / ``average`` GRIB fields are converted at ingest into the mean *rate* over the
    interval ending at the step (e.g. precipitation in mm/h), so 1-hourly and 3-hourly steps
    are directly comparable and daily totals are sum(rate x interval hours).

A variable's ``product`` names the GRIB file family it comes from: ``atmos`` (the GFS
pgrb2.0p25 files) or ``wave`` (GFS-Wave, WAVEWATCH III, on the same 0.25° grid and steps).
``marine`` variables exist only over open sea (land and sea ice are missing); ``circular``
ones are directions in degrees and are interpolated as angles.

Derived variables (heat index, wet-bulb temperature, fire weather, wind power...) are computed when
they are read, from stored ones, so they cost no storage and work everywhere a stored variable does:
point forecasts, tiles and threshold checks. Their formulas are in ``derive`` below.

Each model has its own catalog: GFS; AIGFS (NOAA's AI model, fewer variables, 6-hourly); IFS and AIFS
(ECMWF's physics and AI models, from its open data); GEFS-Aerosols (air quality); and the ocean (surface
currents and tides, from Copernicus Marine, whose "GRIB selector" is the variable's name in its dataset). A variable keeps
the same id, units and meaning in every model that has it, so clients can ask any model for ``t2m``.
For ECMWF's files the GRIB selector is its short name (``2t``) and level type (``sfc``, ``sol:1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    # The rest of the inventory line, when several messages share a param and level (an aerosol species).
    grib_extra: str | None = None
    mul: float = 1.0  # unit conversion applied at ingest: value * mul + add
    add: float = 0.0
    clip: tuple[float | None, float | None] = (None, None)
    valid: tuple[float | None, float | None] = (None, None)  # values outside are stored as missing
    scale: float = 0.01  # int16 storage quantum
    offset: float = 0.0
    decimals: int = 1
    palette: Palette | None = None
    derived_from: tuple[str, ...] = ()  # non-empty => computed at read time from these
    product: str = "atmos"  # GRIB file family: atmos | wave
    circular: bool = False  # a direction in degrees (0-360): interpolate as an angle
    marine: bool = False  # defined over open sea only
    # Meaningful on land only (fire weather): maps (tiles, fields) and area threshold checks leave water
    # cells empty, by the run's land mask. Point forecasts keep the value, so a coastal town gets one.
    land_only: bool = False
    # Wave partitions (wind sea, swell) are missing where the model finds no such waves. Where
    # this is missing but the named variable (an earlier catalog entry) has data, store 0.
    zero_within: str | None = None

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
        if self.marine:
            out["marine"] = True
        if self.land_only:
            out["land_only"] = True
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
    mul: float = 1.0  # unit conversion at ingest, e.g. geopotential to height


@dataclass(frozen=True)
class DailySpec:
    id: str
    source: str
    how: str  # min | max | mean | sum
    units: str
    factor: float = 1.0  # for sums: rate * hours * factor
    decimals: int = 1


# --------------------------------------------------------------------------- derived variables
#
# Inputs are float32 arrays of any shape in catalog units (°C, %, m/s, hPa); NaN stays NaN.
# Humidity comes from rh2m where a model stores it, else from the dew point (d2m).


def saturation_vapour_pressure(t_c):
    """hPa over water (Bolton 1980)."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def _rh(inputs: dict[str, np.ndarray]) -> np.ndarray:
    if "rh2m" in inputs:
        return inputs["rh2m"]
    t, td = inputs["t2m"], inputs["d2m"]
    return np.clip(100.0 * saturation_vapour_pressure(td) / saturation_vapour_pressure(t), 0.0, 100.0)


def heat_index(t_c, rh):
    """US National Weather Service heat index (Rothfusz regression with its adjustments), °C.
    Below about 27 °C it is close to the air temperature."""
    t = t_c * 1.8 + 32.0
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 0.00683783 * t * t
            - 0.05481717 * rh * rh + 0.00122874 * t * t * rh + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh)
    dry = (rh < 13) & (t >= 80) & (t <= 112)
    full = np.where(dry, full - (13 - rh) / 4 * np.sqrt(np.clip(17 - np.abs(t - 95), 0, None) / 17), full)
    humid = (rh > 85) & (t >= 80) & (t <= 87)
    full = np.where(humid, full + (rh - 85) / 10 * (87 - t) / 5, full)
    hi = np.where((simple + t) / 2 >= 80, full, simple)
    return ((hi - 32.0) / 1.8).astype(np.float32)


def wind_chill(t_c, wind_ms):
    """US/Canadian wind chill (2001), °C, for 10 m wind; equal to the air temperature where it doesn't apply."""
    v = np.maximum(wind_ms * 3.6, 0.0) ** 0.16  # km/h
    wc = 13.12 + 0.6215 * t_c - 11.37 * v + 0.3965 * t_c * v
    return np.where((t_c <= 10.0) & (wind_ms > 1.34), wc, t_c).astype(np.float32)


def wet_bulb(t_c, rh):
    """Wet-bulb temperature at sea-level pressure (Stull 2011), °C; within about 1 °C for
    relative humidity 5-99% and air temperatures -20 to 50 °C."""
    rh = np.clip(rh, 1.0, 100.0)
    tw = (t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t_c + rh) - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return np.minimum(tw, t_c).astype(np.float32)


def fosberg(t_c, rh, wind_ms):
    """Fosberg fire weather index (0-100): how fast fine fuels dry and fire spreads, from air temperature,
    humidity and wind alone. Uses the 10 m wind; capped at 100."""
    t, h, u = t_c * 1.8 + 32.0, np.clip(rh, 0.0, 100.0), wind_ms * 2.23694  # °F, %, mph
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t,
                 np.where(h <= 50, 2.22749 + 0.160107 * h - 0.01478 * t,
                          21.0606 + 0.005565 * h * h - 0.00035 * h * t - 0.483199 * h))
    x = np.clip(m, 0.0, 30.0) / 30.0
    eta = 1 - 2 * x + 1.5 * x * x - 0.5 * x ** 3
    return np.clip(eta * np.sqrt(1 + u * u) / 0.3002, 0.0, 100.0).astype(np.float32)


def _speed(u, v):
    return np.hypot(u, v)


def _direction(u, v):
    # Meteorological convention: direction the wind blows FROM, clockwise from north.
    return np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)


def _heading(u, v):
    # Oceanographic convention: direction a current flows TOWARDS, clockwise from north.
    return np.mod(np.degrees(np.arctan2(u, v)), 360.0)


def derive(var_id: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    if var_id == "wind_speed":
        return _speed(inputs["u10"], inputs["v10"])
    if var_id == "wind_dir":
        return _direction(inputs["u10"], inputs["v10"])
    if var_id == "wind_speed_100m":
        return _speed(inputs["u100"], inputs["v100"])
    if var_id == "wind_dir_100m":
        return _direction(inputs["u100"], inputs["v100"])
    if var_id == "wind_power_100m":
        # 1/2 rho v^3, with the air density at the surface (a few % above the density at 100 m).
        rho = inputs["sp"] * 100.0 / (287.05 * (inputs["t2m"] + 273.15))
        return (0.5 * rho * _speed(inputs["u100"], inputs["v100"]) ** 3).astype(np.float32)
    if var_id == "rh2m":
        return _rh(inputs)
    if var_id == "heat_index":
        return heat_index(inputs["t2m"], _rh(inputs))
    if var_id == "feels_like":
        t = inputs["t2m"]
        wind = _speed(inputs["u10"], inputs["v10"])
        return np.where(t >= 26.7, heat_index(t, _rh(inputs)), wind_chill(t, wind)).astype(np.float32)
    if var_id == "wet_bulb":
        return wet_bulb(inputs["t2m"], _rh(inputs))
    if var_id == "vpd":
        t = inputs["t2m"]
        return (saturation_vapour_pressure(t) * (1.0 - _rh(inputs) / 100.0) / 10.0).astype(np.float32)
    if var_id == "fire_weather":
        return fosberg(inputs["t2m"], _rh(inputs), _speed(inputs["u10"], inputs["v10"]))
    if var_id == "current_speed":
        return _speed(inputs["current_u"], inputs["current_v"])
    if var_id == "current_dir":
        return _heading(inputs["current_u"], inputs["current_v"])
    if var_id == "smoke_aod":
        return (inputs["aod_om"] + inputs["aod_bc"]).astype(np.float32)
    raise KeyError(var_id)


@dataclass(frozen=True)
class ModelCatalog:
    model: str
    title: str
    variables: dict[str, Variable]
    description: str = ""
    static: dict[str, StaticField] = field(default_factory=dict)
    daily: tuple[DailySpec, ...] = ()
    # Observations, not a forecast (satellite imagery): each run is one hour observed, its only step.
    observed: bool = False

    def stored_variables(self) -> list[Variable]:
        return [v for v in self.variables.values() if v.stored]

    def products(self) -> list[str]:
        """GRIB file families the stored variables come from, the primary (atmos) first."""
        found = {v.product for v in self.stored_variables()}
        return sorted(found, key=lambda p: (p != "atmos", p))

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


def _wind_100m() -> list[Variable]:
    return [
        Variable("u100", "Wind u-component (100 m)", "m/s", "Eastward wind 100 m above ground.",
                 grib_param="UGRD", grib_level="100 m above ground"),
        Variable("v100", "Wind v-component (100 m)", "m/s", "Northward wind 100 m above ground.",
                 grib_param="VGRD", grib_level="100 m above ground"),
        Variable("wind_speed_100m", "Wind speed (100 m)", "m/s",
                 "Wind speed 100 m above ground, about the hub height of a large wind turbine.",
                 derived_from=("u100", "v100"), palette=palettes.WIND_MS),
        Variable("wind_dir_100m", "Wind direction (100 m)", "°",
                 "Direction the 100 m wind blows from, clockwise from north.",
                 derived_from=("u100", "v100"), scale=0.1, decimals=0, circular=True),
        Variable("wind_power_100m", "Wind power density (100 m)", "W/m²",
                 "Power in the wind crossing each square metre at 100 m (½ρv³). A turbine can capture at most "
                 "59% of it (the Betz limit).",
                 derived_from=("u100", "v100", "sp", "t2m"), scale=1.0, decimals=0, palette=palettes.WIND_POWER_WM2),
    ]


def _comfort_and_fire(humidity: str) -> list[Variable]:
    """Heat, dryness and fire-weather indices. ``humidity`` is the model's stored humidity input:
    rh2m, or the dew point d2m."""
    h, hw = ("t2m", humidity), ("t2m", humidity, "u10", "v10")
    return [
        Variable("feels_like", "Feels-like temperature", "°C",
                 "How warm or cold it feels: the heat index from about 27 °C, the wind chill at 10 °C and below in "
                 "a wind, otherwise the air temperature.",
                 derived_from=hw, palette=palettes.TEMPERATURE_C),
        Variable("heat_index", "Heat index", "°C",
                 "How hot it feels in the shade when humidity slows cooling by sweat (US National Weather Service "
                 "formula). Caution from 27 °C, extreme caution from 32 °C, danger from 39 °C, extreme danger from 52 °C.",
                 derived_from=h, palette=palettes.HEAT_INDEX_C),
        Variable("wet_bulb", "Wet-bulb temperature", "°C",
                 "The temperature a wet surface cools to by evaporation: the limit of cooling by sweat. Around 31 °C "
                 "is dangerous even at rest for many people; 35 °C is the theoretical limit of survival. Estimated "
                 "from temperature and humidity (Stull 2011), to within about 1 °C.",
                 derived_from=h, palette=palettes.WET_BULB_C),
        Variable("vpd", "Vapour pressure deficit", "kPa",
                 "How far the air is from saturation: how hard it pulls water from plants, soil and dead fuels. "
                 "Plants are stressed above about 1.5 kPa, and fire danger rises with it.",
                 derived_from=h, decimals=2, palette=palettes.VPD_KPA),
        Variable("fire_weather", "Fire weather index (Fosberg)", "0-100",
                 "How much the weather alone favours wildfire spread, from temperature, humidity and wind (Fosberg "
                 "index, 0 to 100). Higher is worse; values above about 50 are commonly treated as critical. It says "
                 "nothing about fuels or about fires already burning.",
                 derived_from=hw, decimals=0, land_only=True, palette=palettes.FIRE_WEATHER),
    ]


def _daily_comfort_and_fire() -> tuple[DailySpec, ...]:
    return (
        DailySpec("feels_like_max", "feels_like", "max", "°C"),
        DailySpec("feels_like_min", "feels_like", "min", "°C"),
        DailySpec("wet_bulb_max", "wet_bulb", "max", "°C"),
        DailySpec("fire_weather_max", "fire_weather", "max", "0-100", decimals=0),
    )


GFS = ModelCatalog(
    model="gfs",
    title="NOAA GFS 0.25°",
    description=("NOAA's Global Forecast System, a physics-based model run four times a day: hourly steps to 5 days, "
                 "then 3-hourly to 16 days, with sea state from GFS-Wave."),
    variables=_vars(
        Variable("t2m", "Air temperature (2 m)", "°C", "Air temperature 2 m above the ground, where people feel it: the temperature in a weather report.",
                 grib_param="TMP", grib_level="2 m above ground", add=K_TO_C,
                 palette=palettes.TEMPERATURE_C),
        Variable("d2m", "Dew point (2 m)", "°C", "The temperature at which the air would be saturated and dew would form. The closer it is to the air "
                 "temperature, the more humid; above about 20 °C the air feels muggy.",
                 grib_param="DPT", grib_level="2 m above ground", add=K_TO_C,
                 palette=palettes.TEMPERATURE_C),
        Variable("rh2m", "Relative humidity (2 m)", "%", "How much water vapour the air holds, as a share of the most it could hold at its temperature. Near "
                 "100% fog, dew or rain is likely.",
                 grib_param="RH", grib_level="2 m above ground", clip=(0, 100), decimals=0,
                 palette=palettes.RELATIVE_HUMIDITY),
        Variable("u10", "Wind u-component (10 m)", "m/s", "Eastward wind 10 m above ground.",
                 grib_param="UGRD", grib_level="10 m above ground"),
        Variable("v10", "Wind v-component (10 m)", "m/s", "Northward wind 10 m above ground.",
                 grib_param="VGRD", grib_level="10 m above ground"),
        Variable("wind_speed", "Wind speed (10 m)", "m/s", "Mean wind speed 10 m above the ground, the standard height for wind reports. About 10 m/s is a fresh "
                 "breeze, 17 m/s a gale and 33 m/s hurricane force.",
                 derived_from=("u10", "v10"), palette=palettes.WIND_MS),
        Variable("wind_dir", "Wind direction (10 m)", "°",
                 "Direction the 10 m wind blows from, clockwise from north.",
                 derived_from=("u10", "v10"), scale=0.1, decimals=0, circular=True),
        Variable("gust", "Wind gust", "m/s", "The strongest few-second burst of wind expected near the ground: what breaks branches and lifts roofs. "
                 "Often half as strong again as the mean wind, or more.",
                 grib_param="GUST", grib_level="surface", clip=(0, None), palette=palettes.WIND_MS),
        Variable("mslp", "Mean sea-level pressure", "hPa", "Air pressure, adjusted to sea level so places at different heights compare. Low pressure (below "
                 "about 1000 hPa) brings cloud, rain and wind; high pressure brings settled weather.",
                 grib_param="PRMSL", grib_level="mean sea level", mul=0.01, scale=0.1,
                 palette=palettes.PRESSURE_HPA),
        Variable("sp", "Surface pressure", "hPa", "Air pressure at the model surface (not reduced to sea level).",
                 grib_param="PRES", grib_level="surface", mul=0.01, scale=0.1),
        Variable("q2m", "Specific humidity (2 m)", "g/kg", "Mass of water vapour per mass of air, 2 m above ground.",
                 grib_param="SPFH", grib_level="2 m above ground", mul=1000.0, clip=(0, 32), scale=0.001,
                 decimals=2),
        Variable("precip", "Precipitation", "mm/h",
                 "Rain, snow and other precipitation as a rate, averaged over the forecast step ending at this time. "
                 "Under 0.5 mm/h is light, 0.5-4 mm/h moderate and over 4 mm/h heavy.",
                 kind="accum", grib_param="APCP", grib_level="surface", clip=(0, None), decimals=2,
                 palette=palettes.PRECIP_RATE_MMH),
        Variable("tcc", "Total cloud cover", "%", "Share of the sky covered by cloud at any height, as seen from the ground.",
                 grib_param="TCDC", grib_level="entire atmosphere", clip=(0, 100), decimals=0,
                 palette=palettes.CLOUD_COVER),
        Variable("cape", "CAPE", "J/kg", "Energy available to rising air: the fuel for thunderstorms. Above about 1000 J/kg storms can be "
                 "strong if something sets them off, above about 2500 J/kg severe; near zero the air is stable.",
                 grib_param="CAPE", grib_level="surface", clip=(0, None), scale=1.0, decimals=0,
                 palette=palettes.CAPE_JKG),
        Variable("pwat", "Precipitable water", "mm", "All the water vapour above a point, as the depth of water it would make if it all fell. Above about "
                 "50 mm, as in the tropics, very heavy rain is possible.",
                 grib_param="PWAT", grib_level="entire atmosphere (considered as a single layer)",
                 clip=(0, None), palette=palettes.PRECIPITABLE_WATER_MM),
        Variable("vis", "Visibility", "m", "How far one can see near the ground. Under 1000 m is fog; under 5000 m mist or haze.",
                 grib_param="VIS", grib_level="surface", clip=(0, None), scale=1.0, decimals=0,
                 palette=palettes.VISIBILITY_M),
        Variable("dswrf", "Solar radiation", "W/m²",
                 "Sunlight reaching the ground, direct and diffuse, averaged over the forecast step ending at this time: "
                 "what solar panels and crops receive. About 1000 W/m² is full midday sun.",
                 kind="average", grib_param="DSWRF", grib_level="surface", clip=(0, None), scale=0.1,
                 decimals=0, palette=palettes.SHORTWAVE_WM2),
        Variable("snod", "Snow depth", "m", "Depth of snow lying on the ground.",
                 grib_param="SNOD", grib_level="surface", clip=(0, None), decimals=2,
                 palette=palettes.SNOW_DEPTH_M),
        Variable("refc", "Radar reflectivity (simulated)", "dBZ",
                 "The strongest echo weather radar would see in the column, from the model's rain, snow and hail: "
                 "about 20 dBZ is light rain, 40 heavy rain and 50 or more a strong thunderstorm. -20 where there is none.",
                 grib_param="REFC", grib_level="entire atmosphere", clip=(-20, 80), scale=0.1, decimals=0,
                 palette=palettes.REFLECTIVITY_DBZ),
        Variable("lcc", "Low cloud cover", "%", "Cloud cover in the model's low layer, roughly the lowest 3-4 km.",
                 grib_param="LCDC", grib_level="low cloud layer", clip=(0, 100), decimals=0,
                 palette=palettes.CLOUD_COVER),
        Variable("mcc", "Middle cloud cover", "%", "Cloud cover in the model's middle layer, roughly 3-4 km to 8 km.",
                 grib_param="MCDC", grib_level="middle cloud layer", clip=(0, 100), decimals=0,
                 palette=palettes.CLOUD_COVER),
        Variable("hcc", "High cloud cover", "%", "Cloud cover in the model's high layer, roughly above 8 km.",
                 grib_param="HCDC", grib_level="high cloud layer", clip=(0, 100), decimals=0,
                 palette=palettes.CLOUD_COVER),
        Variable("freezing_level", "Freezing level", "m",
                 "Height above sea level of the lowest level where the air is at 0 °C. Falling snow usually turns "
                 "to rain within a few hundred metres below it.",
                 grib_param="HGT", grib_level="0C isotherm", clip=(0, None), scale=1.0, decimals=0,
                 palette=palettes.FREEZING_LEVEL_M),
        Variable("frozen_precip", "Frozen precipitation", "%",
                 "Share of the precipitation falling as snow, sleet or ice; empty where none is falling.",
                 grib_param="CPOFP", grib_level="surface", valid=(0, 100), decimals=0, palette=palettes.FROZEN_PCT),
        Variable("freezing_rain", "Freezing rain", "0/1",
                 "1 where the model expects freezing rain (rain that freezes on the ground, roads and power lines), "
                 "else 0.",
                 grib_param="CFRZR", grib_level="surface", clip=(0, 1), decimals=0, palette=palettes.FREEZING_RAIN),
        *_wind_100m(),
        Variable("soil_moisture", "Soil moisture (0-10 cm)", "m³/m³",
                 "Water in the top 10 cm of soil as a fraction of its volume: about 0.05 is very dry and 0.4-0.5 "
                 "saturated for most soils. Land only (empty on water and ice sheets).",
                 grib_param="SOILW", grib_level="0-0.1 m below ground", valid=(0, 0.99), scale=0.001, decimals=3,
                 land_only=True, palette=palettes.SOIL_MOISTURE),
        Variable("soil_moisture_10_40", "Soil moisture (10-40 cm)", "m³/m³",
                 "Water in the soil 10 to 40 cm down, where most crop roots are, as a fraction of its volume. "
                 "Land only (empty on water and ice sheets).",
                 grib_param="SOILW", grib_level="0.1-0.4 m below ground", valid=(0, 0.99), scale=0.001, decimals=3,
                 land_only=True, palette=palettes.SOIL_MOISTURE),
        Variable("soil_temp", "Soil temperature (0-10 cm)", "°C",
                 "Temperature of the top 10 cm of soil; on ice sheets, of the ice. Land only.",
                 grib_param="TSOIL", grib_level="0-0.1 m below ground", add=K_TO_C, land_only=True,
                 palette=palettes.TEMPERATURE_C),
        # Sea state, from GFS-Wave. Directions are where the waves come FROM, like wind.
        Variable("wave_height", "Significant wave height", "m",
                 "Significant height of wind waves and swell combined: about the mean height of the "
                 "highest third of the waves. Individual waves can reach about twice this. Open sea only.",
                 grib_param="HTSGW", grib_level="surface", clip=(0, None), product="wave", marine=True,
                 palette=palettes.WAVE_HEIGHT_M),
        Variable("wave_period", "Wave period", "s", "Time between the crests of the dominant waves. 12 s or more is powerful swell from far away; under "
                 "8 s choppy local wind waves. Open sea only.",
                 grib_param="PERPW", grib_level="surface", clip=(0, None), product="wave", marine=True,
                 palette=palettes.WAVE_PERIOD_S),
        Variable("wave_dir", "Wave direction", "°",
                 "Direction the dominant waves come from, clockwise from north. Open sea only.",
                 grib_param="DIRPW", grib_level="surface", scale=0.1, decimals=0, product="wave",
                 circular=True, marine=True),
        Variable("wind_wave_height", "Wind-wave height", "m",
                 "Significant height of the waves raised by the local wind (the wind sea); 0 when there is none. Open sea only.",
                 grib_param="WVHGT", grib_level="surface", clip=(0, None), product="wave", marine=True,
                 zero_within="wave_height", palette=palettes.WAVE_HEIGHT_M),
        Variable("swell_height", "Swell height", "m",
                 "Significant height of the main swell: waves raised by distant winds; 0 when there is none. Open sea only.",
                 grib_param="SWELL", grib_level="1 in sequence", clip=(0, None), product="wave", marine=True,
                 zero_within="wave_height", palette=palettes.WAVE_HEIGHT_M),
        Variable("swell_period", "Swell period", "s", "Period of the main swell; empty when there is no swell. Open sea only.",
                 grib_param="SWPER", grib_level="1 in sequence", clip=(0, None), product="wave", marine=True,
                 palette=palettes.WAVE_PERIOD_S),
        Variable("swell_dir", "Swell direction", "°",
                 "Direction the main swell comes from, clockwise from north; empty when there is no swell. Open sea only.",
                 grib_param="SWDIR", grib_level="1 in sequence", scale=0.1, decimals=0, product="wave",
                 circular=True, marine=True),
        *_comfort_and_fire("rh2m"),
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
        DailySpec("wave_height_max", "wave_height", "max", "m"),
        *_daily_comfort_and_fire(),
    ),
)

# Relative humidity for models that store the dew point instead.
AIGFS_RH = Variable("rh2m", "Relative humidity (2 m)", "%", "Relative humidity 2 m above ground, from the dew point.",
                    derived_from=("t2m", "d2m"), decimals=0, palette=palettes.RELATIVE_HUMIDITY)

AIGFS = ModelCatalog(
    model="aigfs",
    title="NOAA AIGFS 0.25° (AI)",
    description=("NOAA's operational AI global model, built on Google DeepMind's GraphCast and run by NCEP four times a "
                 "day from NOAA's analyses. Steps are 6-hourly to 16 days, so daily highs and lows come from four values "
                 "a day and can miss the peak. Fewer variables than GFS: no cloud, radiation, gusts or sea state."),
    variables=_vars(
        GFS.variables["t2m"],
        GFS.variables["d2m"],
        AIGFS_RH,
        *(GFS.variables[v] for v in ("u10", "v10", "wind_speed", "wind_dir")),
        *_wind_100m(),
        *(GFS.variables[v] for v in ("mslp", "sp", "precip")),
        *_comfort_and_fire("d2m"),
    ),
    daily=(
        DailySpec("t2m_max", "t2m", "max", "°C"),
        DailySpec("t2m_min", "t2m", "min", "°C"),
        DailySpec("precip_sum", "precip", "sum", "mm", decimals=1),
        DailySpec("wind_speed_max", "wind_speed", "max", "m/s"),
        DailySpec("rh2m_mean", "rh2m", "mean", "%", decimals=0),
        *_daily_comfort_and_fire(),
    ),
)

# --------------------------------------------------------------------------- ECMWF IFS and AIFS
#
# The same variables as GFS where ECMWF publishes them, read from its short names. Totals since the run
# started (precipitation, sunshine) become rates over each step like GFS's. Units differ between the two
# ECMWF models' files: IFS gives precipitation in metres and cloud as 0-1, AIFS in kg/m² (mm) and %.

G = 9.80665  # m/s², to turn geopotential into height


def _ecmwf(var_id: str, param: str, level: str = "sfc", **changes) -> Variable:
    """A GFS variable read from ECMWF's files: same id, units and palette."""
    return replace(GFS.variables[var_id], grib_param=param, grib_level=level, **changes)


_ECMWF_SOIL = (
    _ecmwf("soil_moisture", "vsw", "sol:1", name="Soil moisture (0-7 cm)", valid=(0.001, 0.99),
           description="Water in the top 7 cm of soil (ECMWF's first soil layer) as a fraction of its volume: about "
                       "0.05 is very dry and 0.4-0.5 saturated for most soils. Land only (empty on water and ice sheets)."),
    _ecmwf("soil_temp", "sot", "sol:1", name="Soil temperature (0-7 cm)",
           description="Temperature of the top 7 cm of soil (ECMWF's first soil layer). Land only."),
)


def _ecmwf_common(precip_mul: float, cloud_mul: float) -> tuple[Variable, ...]:
    """What both ECMWF models publish, with each model's unit factors for precipitation and cloud."""
    return (
        _ecmwf("t2m", "2t"),
        _ecmwf("d2m", "2d"),
        AIGFS_RH,
        _ecmwf("u10", "10u"),
        _ecmwf("v10", "10v"),
        GFS.variables["wind_speed"],
        GFS.variables["wind_dir"],
        _ecmwf("u100", "100u"),
        _ecmwf("v100", "100v"),
        *(GFS.variables[v] for v in ("wind_speed_100m", "wind_dir_100m", "wind_power_100m")),
        _ecmwf("mslp", "msl"),
        _ecmwf("sp", "sp"),
        _ecmwf("precip", "tp", mul=precip_mul),  # total since the start -> mm, then mm/h over each step
        *(_ecmwf(c, c, mul=cloud_mul) for c in ("tcc", "lcc", "mcc", "hcc")),
        # J/m² since the start -> mean W/m² over each step.
        _ecmwf("dswrf", "ssrd", kind="accum", mul=1.0 / 3600.0),
        *_ECMWF_SOIL,
    )


_ECMWF_STATIC = {
    "orog": StaticField("orog", "Model surface elevation", "m", "z", "sfc", mul=1.0 / G),
    "land": StaticField("land", "Land fraction", "1", "lsm", "sfc"),
}

IFS = ModelCatalog(
    model="ifs",
    title="ECMWF IFS 0.25°",
    description=("ECMWF's Integrated Forecasting System, the European physics model, from its 00 and 12 UTC runs: "
                 "3-hourly steps to 6 days, then 6-hourly to 15 days, with sea state from ECMWF's wave model. "
                 "ECMWF open data, CC BY 4.0."),
    variables=_vars(
        *_ecmwf_common(precip_mul=1000.0, cloud_mul=100.0),  # m, 0-1
        _ecmwf("gust", "10fg",
               description="The strongest few-second gust near the ground in the hour before this time: what breaks "
                           "branches and lifts roofs. Often half as strong again as the mean wind, or more."),
        _ecmwf("cape", "mucape",
               description="Energy available to the most unstable rising air: the fuel for thunderstorms. Above about "
                           "1000 J/kg storms can be strong if something sets them off, above about 2500 J/kg severe; near "
                           "zero the air is stable."),
        _ecmwf("pwat", "tcwv"),
        _ecmwf("wave_height", "swh", product="wave"),
        _ecmwf("wave_period", "pp1d", product="wave",
               description="Time between the crests of the most energetic waves (the peak period). 12 s or more is "
                           "powerful swell from far away; under 8 s choppy local wind waves. Open sea only."),
        _ecmwf("wave_dir", "mwd", product="wave",
               description="Mean direction the waves come from, clockwise from north. Open sea only."),
        *_comfort_and_fire("d2m"),
    ),
    static=_ECMWF_STATIC,
    daily=(
        DailySpec("t2m_max", "t2m", "max", "°C"),
        DailySpec("t2m_min", "t2m", "min", "°C"),
        DailySpec("precip_sum", "precip", "sum", "mm", decimals=1),
        DailySpec("wind_speed_max", "wind_speed", "max", "m/s"),
        DailySpec("gust_max", "gust", "max", "m/s"),
        DailySpec("rh2m_mean", "rh2m", "mean", "%", decimals=0),
        DailySpec("tcc_mean", "tcc", "mean", "%", decimals=0),
        DailySpec("dswrf_sum", "dswrf", "sum", "MJ/m²", factor=0.0036, decimals=1),
        DailySpec("wave_height_max", "wave_height", "max", "m"),
        *_daily_comfort_and_fire(),
    ),
)

AIFS = ModelCatalog(
    model="aifs",
    title="ECMWF AIFS 0.25° (AI)",
    description=("ECMWF's AI model (AIFS Single), trained on ECMWF's reanalysis and started from its analyses, run four "
                 "times a day: 6-hourly steps to 15 days, so daily highs and lows come from four values a day and can miss "
                 "the peak. No gusts, CAPE or sea state. ECMWF open data, CC BY 4.0."),
    variables=_vars(*_ecmwf_common(precip_mul=1.0, cloud_mul=1.0), *_comfort_and_fire("d2m")),  # kg/m², %
    static=_ECMWF_STATIC,
    daily=(
        DailySpec("t2m_max", "t2m", "max", "°C"),
        DailySpec("t2m_min", "t2m", "min", "°C"),
        DailySpec("precip_sum", "precip", "sum", "mm", decimals=1),
        DailySpec("wind_speed_max", "wind_speed", "max", "m/s"),
        DailySpec("rh2m_mean", "rh2m", "mean", "%", decimals=0),
        DailySpec("tcc_mean", "tcc", "mean", "%", decimals=0),
        DailySpec("dswrf_sum", "dswrf", "sum", "MJ/m²", factor=0.0036, decimals=1),
        *_daily_comfort_and_fire(),
    ),
)


# --------------------------------------------------------------------------- GEFS-Aerosols (air quality)

_FINE, _COARSE = "aerosol_size <2.5e-06", "aerosol_size <1e-05"
_AOD_550 = "aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07"
_AQ_NOTE = (" A global model's average over a 25 km cell: it misses sources a city block away, such as traffic and "
            "cooking fires, and has no nitrate or ozone.")

AEROSOLS = ModelCatalog(
    model="aerosols",
    title="NOAA GEFS-Aerosols 0.25° (air quality)",
    description=("Air quality from NOAA's global aerosol model: particles near the ground (PM2.5, PM10) and haze from "
                 "dust, sea salt, sulphate and smoke, every 3 hours to 5 days, four runs a day. Fires come from "
                 "satellite fire detections."),
    variables=_vars(
        Variable("pm25", "PM2.5", "µg/m³",
                 "Fine particles under 2.5 µm in the air near the ground: smoke, dust, sea salt and sulphate haze, the "
                 "particles that harm health most. The WHO's 24-hour guideline is 15 µg/m³; on the US Air Quality Index "
                 "over 35 is unhealthy for sensitive groups and over 55 unhealthy." + _AQ_NOTE,
                 grib_param="PMTF", grib_level="surface", grib_extra=f"aerosol=Total aerosol:{_FINE}",
                 clip=(0, None), scale=0.1, decimals=0, palette=palettes.PM25_UGM3),
        Variable("pm10", "PM10", "µg/m³",
                 "Particles under 10 µm in the air near the ground, including coarse dust and sea salt. The WHO's 24-hour "
                 "guideline is 45 µg/m³; on the US Air Quality Index over 154 is unhealthy for sensitive groups." + _AQ_NOTE,
                 grib_param="PMTC", grib_level="surface", grib_extra=f"aerosol=Total aerosol:{_COARSE}",
                 clip=(0, None), scale=0.5, decimals=0, palette=palettes.PM10_UGM3),
        Variable("dust", "Dust (PM10)", "µg/m³",
                 "Mineral dust under 10 µm in the air near the ground, as in a desert dust storm or the Harmattan.",
                 grib_param="PMTC", grib_level="surface", grib_extra=f"aerosol=Dust dry:{_COARSE}",
                 clip=(0, None), scale=0.5, decimals=0, palette=palettes.PM10_UGM3),
        Variable("aod", "Haze (aerosol optical depth)", "1",
                 "How much the particles in the whole air column dim sunlight (aerosol optical depth at 550 nm): about 0.1 "
                 "is clear air, 0.3 visible haze, 1 or more a thick pall of dust or smoke.",
                 grib_param="AOTK", grib_level="entire atmosphere", grib_extra=f"aerosol=Total aerosol:{_AOD_550}",
                 clip=(0, None), scale=0.001, decimals=2, palette=palettes.AEROSOL_OPTICAL_DEPTH),
        Variable("dust_aod", "Dust haze", "1", "The part of the haze (aerosol optical depth at 550 nm) made by dust.",
                 grib_param="AOTK", grib_level="entire atmosphere", grib_extra=f"aerosol=Dust dry:{_AOD_550}",
                 clip=(0, None), scale=0.001, decimals=2, palette=palettes.AEROSOL_OPTICAL_DEPTH),
        Variable("aod_om", "Organic matter haze", "1",
                 "Aerosol optical depth at 550 nm of organic matter, mostly smoke from fires.",
                 grib_param="AOTK", grib_level="entire atmosphere",
                 grib_extra=f"aerosol=Particulate organic matter dry:{_AOD_550}", clip=(0, None), scale=0.001, decimals=2),
        Variable("aod_bc", "Black carbon haze", "1",
                 "Aerosol optical depth at 550 nm of black carbon (soot), from fires and burning fuel.",
                 grib_param="AOTK", grib_level="entire atmosphere", grib_extra=f"aerosol=Black carbon dry:{_AOD_550}",
                 clip=(0, None), scale=0.001, decimals=2),
        Variable("smoke_aod", "Smoke haze", "1",
                 "The part of the haze (aerosol optical depth at 550 nm) made by smoke: organic matter and black carbon, "
                 "mostly from fires.",
                 derived_from=("aod_om", "aod_bc"), decimals=2, palette=palettes.AEROSOL_OPTICAL_DEPTH),
    ),
    daily=(
        DailySpec("pm25_mean", "pm25", "mean", "µg/m³", decimals=0),
        DailySpec("pm25_max", "pm25", "max", "µg/m³", decimals=0),
        DailySpec("pm10_mean", "pm10", "mean", "µg/m³", decimals=0),
        DailySpec("aod_max", "aod", "max", "1", decimals=2),
    ),
)


_TOTAL = "the ocean's circulation, tidal currents and the drift waves cause, combined"

OCEAN = ModelCatalog(
    model="ocean",
    title="Copernicus Marine global ocean 1/12° (surface currents and tides)",
    description=("Surface currents from Mercator Ocean's global ocean model (GLO12) for the Copernicus Marine "
                 "Service, with tidal currents (FES2014) and the drift waves cause, and tide heights (FES2014): "
                 "hourly to about 10 days, updated daily. Stored at 0.25°, each point the mean of the sea around "
                 "it at 1/12°. Point forecasts also list the high and low waters."),
    variables=_vars(
        Variable("current_u", "Current u-component", "m/s", f"Eastward surface current: {_TOTAL}. Open sea only.",
                 grib_param="utotal", valid=(-10, 10), scale=0.001, decimals=2, marine=True),
        Variable("current_v", "Current v-component", "m/s", f"Northward surface current: {_TOTAL}. Open sea only.",
                 grib_param="vtotal", valid=(-10, 10), scale=0.001, decimals=2, marine=True),
        Variable("current_speed", "Current speed", "m/s",
                 f"Speed of the water at the sea surface: {_TOTAL}. 0.5 m/s (1 knot) is a current a swimmer or a "
                 "small boat feels; tidal straits and the strongest ocean currents run 2 m/s (4 knots) or more. "
                 "Open sea only.",
                 derived_from=("current_u", "current_v"), decimals=2, marine=True, palette=palettes.CURRENT_MS),
        Variable("current_dir", "Current direction", "°",
                 "Direction the surface current flows towards, clockwise from north: the opposite of wind and "
                 "wave directions, which say where those come from. Open sea only.",
                 derived_from=("current_u", "current_v"), scale=0.1, decimals=0, circular=True, marine=True),
        Variable("tide_height", "Tide height", "m",
                 "Height of the tide above (or below) mean sea level: the astronomical tide alone (FES2014), "
                 "without the rise or fall that wind and air pressure add. Heights are from mean sea level, not "
                 "from the chart datum (the lowest tide) that tide tables use, so they differ from a tide table "
                 "by a constant there; times and ranges compare. A global model: less exact in harbours, "
                 "estuaries and narrow bays, and not for navigation. Open sea only.",
                 grib_param="ocean_tide", valid=(-20, 20), scale=0.001, decimals=2, marine=True,
                 palette=palettes.TIDE_M),
    ),
    # No daily tide summary: the hourly values miss a high or low water by up to 10 cm on a large tide, so
    # the point forecast's ``tides`` (placed between the hours) are the day's highs and lows.
    daily=(DailySpec("current_speed_max", "current_speed", "max", "m/s", decimals=2),),
)


_SAT = (" From NOAA's hourly mosaic of the geostationary weather satellites (GOES, Meteosat, Himawari), 72.7° N to "
        "72.7° S: each 0.1° cell is the mean of the 8 km pixels in it.")

SATELLITE = ModelCatalog(
    model="satellite",
    title="NOAA GMGSI satellite imagery (hourly global mosaic)",
    description=("Observed, not forecast: the geostationary weather satellites' images of the globe (GOES-East and "
                 "-West, Meteosat, Himawari), blended each hour by NOAA (GMGSI), from 72.7° N to 72.7° S, stored at "
                 "0.1°. Each hour is a run of one step at its own time, about an hour after it; the last 24 are "
                 "kept. Infrared shows clouds day and night by how cold their tops are, water vapour the moisture "
                 "of the upper air, visible the daylit side."),
    observed=True,
    variables=_vars(
        Variable("sat_ir", "Infrared (cloud-top temperature)", "°C",
                 "Temperature the 11 µm infrared window sees: the tops of clouds, colder the higher they reach "
                 "(below -50 °C, the tops of thunderstorms and deep storms), and under a clear sky the ground or "
                 "the sea, a few degrees colder than it is through moist air. Day and night." + _SAT,
                 grib_param="LW", add=K_TO_C, valid=(-130, 70), decimals=1, palette=palettes.INFRARED_C),
        Variable("sat_wv", "Water vapour", "°C",
                 "Temperature the 6.7 µm water vapour channel sees: the moisture of the middle and upper air "
                 "(about 3 to 10 km up). Cold, below -40 °C, is moist air high up; warm, above -25 °C, is dry air "
                 "the channel sees down through. It shows jet streams, dry intrusions and storms where there is "
                 "no cloud." + _SAT,
                 grib_param="WV", add=K_TO_C, valid=(-130, 70), decimals=1, palette=palettes.WATER_VAPOUR_C),
        Variable("sat_vis", "Visible", "1",
                 "Brightness of the sunlight clouds and the ground reflect, 0 (dark) to 255 (brightest) as the "
                 "mosaic gives it: thick cloud and snow are bright, the sea dark. Missing where it is night." + _SAT,
                 grib_param="VIS", valid=(0, 255), decimals=0, palette=palettes.VISIBLE),
    ),
)


CATALOGS: dict[str, ModelCatalog] = {c.model: c for c in (GFS, AIGFS, IFS, AIFS, AEROSOLS, OCEAN,
                                                            SATELLITE)}

