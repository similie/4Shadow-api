# Foreshadow Forecast v2

A forecast for any point on Earth, plus map tiles, from open numerical weather prediction
data. Six models today: NOAA GFS 0.25° with sea state from GFS-Wave, NOAA's AI model AIGFS, ECMWF's
IFS (with sea state) and AI model AIFS, NOAA GEFS-Aerosols for air quality, and surface currents with
tides from Copernicus Marine. The store and API are model-agnostic, and every model is stored on the same
0.25° grid.

```
NOAA S3 (.idx + byte ranges) ──► ingest ──► data/forecast/gfs/runs/<run>/*.npy ──► API (/v2)
         ~12 MB per step            decode,      int16 [step, lat, lon] arrays,       mmap reads,
         instead of ~540 MB         de-bucket,   one consistent run, atomic           no warm-up
                                    quantise     "latest" pointer
```

## Quick start

```bash
pip install -r requirements-dev.txt
python -m forecast.ingest --max-fhr 24          # dev slice: newest run, hours 0-24 (~300 MB download)
uvicorn forecast.api:app --port 5003            # docs at http://localhost:5003/v2/docs
curl 'localhost:5003/v2/forecast?lat=-8.556&lon=125.560&hours=48'
pytest                                          # unit + API tests, no network needed
```

Production: `python -m forecast.ingest --loop` (see `deploy/forecast-ingest.service`) plus the
API service (`deploy/forecast-api.service`); `deploy/deploy.sh` updates a server (see
`docs/deploy.md`). The service's dependencies are `forecast/requirements.txt` (what the root
`requirements.txt` installs); keep them in their own venv, apart from the experimental terrain
engine's, which pins an older numpy for GDAL.

## API

| Endpoint | Purpose |
|---|---|
| `GET /v2/forecast?lat&lon` or `?q=` | Point forecast: `current`, `timeseries`, local-day `daily` (GFS) |
| `GET /v2/{model}/forecast?lat&lon` or `?q=` | The same from another model: `/v2/aigfs/forecast` |
| `GET /v2/places?q=` or `?lat&lon` | Place search: names to coordinates, or the nearest towns to a point |
| `GET /v2/{model}/tiles/{run}/{var}/{time}/{z}/{x}/{y}.png` | XYZ map tiles |
| `GET /v2/{model}/fields/{run}/{time}?vars=u10,v10` | Raw gridded values (binary) for client-side rendering |
| `GET /v2/{model}/variables` | Variable catalog with units and legend colour stops |
| `GET /v2/{model}/runs`, `/v2/{model}/runs/{run}` | Available runs and their valid times |
| `POST /v2/{model}/thresholds` | Early-warning check: when an area meets conditions in a run (internal; see below) |
| `GET /v2/health` | Latest run per model and its age: `ok`, `degraded` (GFS is in, another model isn't yet), or `unavailable` with 503 (no GFS run) |
| `POST /v2/mcp` | MCP server for AI assistants (see below) |

**Forecast parameters**: `lat` and `lon`, or `q` (a place name such as `Paris, France`; the best
match is used), `model` (`gfs`, the default, or `aigfs`), `vars` (comma list, default all), `hours` (default 384),
`start=now|run`, `tz` (IANA name; default looked up from the coordinates), `run`
(`latest` or `YYYYMMDDHH`), `daily=true|false`.

```jsonc
{
  "run": "2026111200", "init_time": "2026-11-12T00:00Z",
  "location": {"lat": -8.556, "lon": 125.56, "nearest_grid_point": {...}, "model_elevation_m": 319.7, "land_fraction": 0.23},
  "timezone": {"name": "Asia/Dili", "utc_offset_seconds": 32400},
  "units": {"t2m": "°C", "precip": "mm/h", ...},
  "current": {"time": "...", "t2m": 27.5, ...},              // interpolated to now
  "timeseries": {"time": [...], "interval_hours": [...], "t2m": [...], ...},  // columnar
  "daily": {"date": [...], "hours_covered": [...], "t2m_max": [...], "precip_sum": [...], ...},
  "source": {"provider": "NOAA/NCEP Global Forecast System (GFS), 0.25°", ...}
}
```

Semantics worth knowing:

- Steps are hourly to +120 h, then 3-hourly to +384 h (`interval_hours` says which).
- `precip` and `dswrf` are **mean rates over the interval ending at the step** (mm/h, W/m²),
  derived from GFS running totals / bucket averages, so they are comparable across 1 h and
  3 h steps. Daily sums are `rate x interval_hours`.
- Daily values use **local calendar days** (`hours_covered` < 24 marks a partial day).
- Values are bilinear between the four surrounding 0.25° cells. `model_elevation_m` is
  the model's terrain height there, useful for elevation corrections.
- With `q`, `location` adds `query`, `place` (the match) and `alternatives` (other places with the
  same name). With `lat`/`lon` it adds `nearest_place` (the closest town or village within 50 km).
- **Sea state** (`wave_height`, `wave_period`, `wave_dir`, `wind_wave_height`, `swell_height`,
  `swell_period`, `swell_dir`, daily `wave_height_max`) comes from GFS-Wave (WAVEWATCH III) on the
  same grid and steps, and exists over open sea only: `null` over land, and over the Black and
  Caspian Seas, which the global wave grid leaves out. On a coast whose grid cells are all land
  for the wave model, it comes from the nearest sea cell within 50 km, given as `location.sea_point`
  (`lat`, `lon`, `distance_km`); sheltered water there is usually calmer. Directions are where the
  waves come from, clockwise from north, and are interpolated as angles. `wind_wave_height` and
  `swell_height` are 0 at sea when the model finds no wind sea or swell (their period and
  direction are then `null`).

## Models

| Model | What it is | Steps | Variables |
|---|---|---|---|
| `gfs` | NOAA's Global Forecast System, physics-based, with GFS-Wave sea state | Hourly to 5 days, then 3-hourly to 16 days | All 45 below |
| `aigfs` | NOAA's operational AI model (since December 2025), built on Google DeepMind's GraphCast and started from NOAA's analyses | 6-hourly to 16 days | Temperature, dew point and humidity, 10 m and 100 m wind (from the first forecast step; the analysis has none), pressure, precipitation, and the indices computed from them. Its land mask and terrain height come from GFS |
| `ifs` | ECMWF's Integrated Forecasting System, the European physics model, from its 00 and 12 UTC runs, with sea state from ECMWF's wave model | 3-hourly to 6 days, then 6-hourly to 15 days | Temperature, dew point and humidity, 10 m and 100 m wind, the 10 m gust (strongest in the hour before), pressure, precipitation, total/low/middle/high cloud, CAPE (most unstable), precipitable water, solar radiation, soil moisture and temperature (0-7 cm), wave height, period and direction, and the indices. Its own land mask and terrain height |
| `aifs` | ECMWF's AI model, AIFS Single, four runs a day | 6-hourly to 15 days | As IFS without gusts, CAPE, precipitable water and sea state |
| `aerosols` | NOAA GEFS-Aerosols: air quality from NOAA's global aerosol model (GEFS v12 with GOCART), with fires from satellite detections | 3-hourly to 5 days, four runs a day | PM2.5, PM10, dust, and haze (aerosol optical depth) in total and from dust and smoke. See *Air quality* below |
| `ocean` | Copernicus Marine's global ocean analysis and forecast (Mercator Ocean's GLO12 model, 1/12°), with tidal currents and wave drift. Needs a Copernicus Marine account | Hourly to 10 days, from a run once a day | Surface current, its speed and the direction it flows towards. See *Ocean currents* below |

GFS arrives about 5 hours after each run starts, AIGFS on NOAA's AWS bucket about 8 hours after, IFS's
last step about 7.5 hours after (twice a day), AIFS and GEFS-Aerosols about 6 hours after. ECMWF also
publishes IFS runs at 06 and 18 UTC, but only to 6 days, so the newest IFS run is always a 15-day one.
With 6-hourly steps, AIGFS's and AIFS's daily highs and lows are of four values a day and can miss the
peak. AIGFS's source files list fewer fields than GFS's and NCEP says their contents may change; a field
missing from a step's file is left empty there. A variable has the same id, units and meaning in every
model that has it, so `t2m` or `precip` can be compared across models.

ECMWF's files start at 180° and list each message in a JSON-lines `.index`
(`forecast/sources/ecmwf.py`); the ingest turns each field to start at 0° like GFS's. Their unit
factors differ by model: IFS gives precipitation in metres and cloud as 0-1, AIFS in kg/m² and %.

## Variables

`GET /v2/{model}/variables` lists each model's variables with units, a description and the legend.
Besides the basics (temperature, humidity, wind, gusts, pressure, precipitation, cloud, CAPE,
precipitable water, visibility, solar radiation, snow depth and sea state), GFS has:

| Id | | Source |
|---|---|---|
| `refc` | Simulated radar reflectivity, dBZ: the strongest echo in the column | GFS `REFC` |
| `lcc`, `mcc`, `hcc` | Low, middle and high cloud cover, % | GFS `LCDC`, `MCDC`, `HCDC` |
| `freezing_level` | Height of the 0 °C level, m above sea level | GFS `HGT` at the 0C isotherm |
| `frozen_precip` | Share of precipitation falling frozen, %; `null` where none falls | GFS `CPOFP` (-50, "none", stored as missing) |
| `freezing_rain` | 1 where freezing rain is expected, else 0 | GFS `CFRZR` |
| `u100`, `v100` | Wind at 100 m, m/s | GFS `UGRD`/`VGRD` at 100 m |
| `soil_moisture`, `soil_moisture_10_40` | Volumetric soil water at 0-10 and 10-40 cm, m³/m³; land only | GFS `SOILW` (1.0 on ice sheets is stored as missing) |
| `soil_temp` | Soil temperature at 0-10 cm, °C; the ice's temperature on ice sheets. Land only | GFS `TSOIL` |

Computed when read (no storage, and available in tiles and threshold checks like any other):

| Id | | Formula |
|---|---|---|
| `wind_speed_100m`, `wind_dir_100m` | Wind speed and direction at 100 m | From `u100`, `v100` |
| `wind_power_100m` | Wind power density at 100 m, W/m² | ½ρv³, with the air density from surface pressure and temperature |
| `heat_index` | Heat index, °C | US National Weather Service (Rothfusz regression and its adjustments) |
| `feels_like` | Feels-like temperature, °C | Heat index from 26.7 °C, wind chill (Environment Canada/NWS 2001, 10 m wind) at 10 °C and below, else the air temperature |
| `wet_bulb` | Wet-bulb temperature, °C | Stull (2011), at sea-level pressure; within about 1 °C |
| `vpd` | Vapour pressure deficit, kPa | Saturation vapour pressure (Bolton 1980) times (1 - RH) |
| `fire_weather` | Fosberg fire weather index, 0-100. Land only: tiles, fields and area checks leave water empty | Fosberg (1978), from temperature, humidity and the 10 m wind, capped at 100 |

AIGFS, IFS and AIFS have no stored humidity, so their `rh2m` and the indices use the dew point. Daily
summaries add `feels_like_max`, `feels_like_min`, `wet_bulb_max` and `fire_weather_max`.

### Air quality (`aerosols`)

| Id | | Source (GEFS-Aerosols `a2d_0p25`) |
|---|---|---|
| `pm25` | PM2.5 near the ground, µg/m³. Tiles use the US AQI's categories and colours | `PMTF` surface, total aerosol |
| `pm10` | PM10 near the ground, µg/m³ | `PMTC` surface, total aerosol |
| `dust` | Dust under 10 µm near the ground, µg/m³ | `PMTC` surface, dust |
| `aod` | Haze: aerosol optical depth at 550 nm | `AOTK` total aerosol at 545-565 nm |
| `dust_aod` | The part of the haze from dust | `AOTK` dust |
| `smoke_aod` | The part from smoke (computed: organic matter + black carbon, `aod_om` + `aod_bc`, both stored) | `AOTK` organic matter, black carbon |

Several messages in these files share a parameter and level and differ only in the species and size
the inventory line names, so these variables also match it (`Variable.grib_extra`). The model has no
nitrate or ozone, and as a 25 km average it misses sources a street away; the variable descriptions say
so. Daily summaries: `pm25_mean`, `pm25_max`, `pm10_mean`, `aod_max`.

### Ocean currents (`ocean`)

| Id | | Source (`cmems_mod_glo_phy_anfc_merged-uv_PT1H-i`) |
|---|---|---|
| `current_u`, `current_v` | Eastward and northward surface current, m/s | `utotal`, `vtotal` |
| `current_speed` | Its speed, m/s (computed) | |
| `current_dir` | The direction it flows towards, clockwise from north (computed). Wind and wave directions say where those come from; currents, where they go | |

The current is the total at the surface: the model's circulation, tidal currents (FES2014) and the drift
waves cause (Stokes drift). The model's own currents have no tides, and in tidal straits and on shelves
tides are most of the current, so this is the dataset to use. It is one rolling time series that
Copernicus Marine rewrites daily, running about ten days past the newest forecast's start; that start is
the run (the series' last hour less 239 hours, to 00 UTC), so the newest run begins about a day before it
is ingested. Each hour is read through the `copernicusmarine` toolbox as the global 1/12° field (about
42 MB, so about 10 GB a day) and kept at 0.25°, each point the mean of the 3 x 3 cells around it that are
sea; the grid stops at 80° S. Like sea state, currents are sea only: a point forecast on a coast takes the
nearest sea cell. The model is in the default `FORECAST_MODELS` only when `COPERNICUSMARINE_SERVICE_USERNAME`
and `COPERNICUSMARINE_SERVICE_PASSWORD` are set. Daily summary: `current_speed_max`.

**Fields** are the numbers behind the tiles, for clients that render, animate or analyse the data
themselves (WebGL colour layers, wind particles, isobars). `vars` takes up to 4 variables, stored
(`u10,v10`) or computed (`wet_bulb`, `fire_weather`: worked out on the requested cells), `stride` thins the grid (1 = 0.25°, 2 = 0.5°, 4 = 1°) and
`bbox=west,south,east,north` crops it (east < west crosses the antimeridian); the default is the
whole globe, starting at -180. The body is format "4SF1": `"4SF1"`, a uint32 LE header length, a
JSON header (grid, valid time, per-variable units/scale/offset), then one int16 LE array per
variable, rows north to south, starting on an 8-byte boundary. `value = raw * scale + offset`,
`-32768` = missing. A global 0.5° wind field (u10 + v10) is about 1 MB; `forecast.fields.decode_field`
reads it in Python, and the JS SDK in the browser.

## Thresholds (early warnings)

`forecast/thresholds.py` answers one question, statelessly: *in this run, when does this area meet
these conditions?* The 4Shadow site stores users' alerts, calls it for each new run, turns the answers
into events and sends the webhooks (`foreshadow-site/src/lib/server/alerts`). It isn't on the public
router; users reach early warnings through the site's `/v2/alerts` API.

```jsonc
// POST /v2/gfs/thresholds
{
  "area": {"type": "Polygon", "coordinates": [[[124, -9.5], [127.3, -9.5], [127.3, -8.1], [124, -8.1], [124, -9.5]]]},
  "conditions": [{"variable": "precip", "operator": ">=", "threshold": 50, "accumulate_hours": 24}],
  "match": "all",            // all: every condition in the same cell at once; any: one is enough
  "area_mode": "coverage",   // any (max/min over the cells) | mean | coverage (share of cells)
  "coverage_percent": 25,
  "lead_days": 3, "min_duration_hours": 0,
  "run": "latest", "at": null, "timeline": false   // `at` overrides "now"; timeline adds per-step values
}
```

- Areas: a `Point` is bilinear (with the `sea_point` fallback for sea variables); with `radius_km` it
  is every cell within the radius. Polygons take the cells whose centres fall inside (even-odd, holes
  and the antimeridian handled), or the nearest cell if the polygon is smaller than a cell. Up to
  20,000 cells and 10,000 vertices.
- `accumulate_hours` turns an hourly rate (`precip`) into a running total over that many hours.
  Directions can't be thresholds.
- Events are runs of steps that meet the conditions between the step in effect `at` and
  `at + lead_days`. Gaps of up to 6 h are merged, and events shorter than `min_duration_hours` are
  dropped. Each event has its peak (value, time, place, coverage) and each condition's peak.
- `warnings` explain partial answers: totals that start at the run's start, a variable with no data
  in the area, or a run that ends before the window does.

## Places

`forecast/places.py` turns place names into coordinates and coordinates into the nearest town,
from [GeoNames](https://www.geonames.org) data held in one read-only SQLite file
(`data/places/places.sqlite`). Lookups never leave the server.

```bash
python -m forecast.places build --dump allCountries   # production: ~2-3 GB index, every village plus
                                                      # countries, regions, seas, bays, islands, ports
python -m forecast.places build                       # dev: cities500 (~240k places, ~70 MB, seconds)
python -m forecast.places search "Same, Timor-Leste"
python -m forecast.places nearest -8.556 125.560
```

Names match regardless of case, accents and punctuation (`Díli` = `dili`), in any language
GeoNames records (`Lisboa`, `Москва`). Text after a comma narrows by region or country
(`Paris, Texas`); without a comma, trailing words are tried as the country (`paris france`).
Ranking: a place's own name, then alternate names, then names that only start with the query;
within each, larger and more important places (capitals, countries) first. A rebuild writes a new
file and renames it into place; running APIs pick it up on their next lookup. Rebuild now and then
(GeoNames changes slowly); there is no need to restart anything.

**Tiles**: `time` is `fNNN` (forecast hour) or `YYYYMMDDHH` (valid time, UTC). Use
`run=latest` for "whatever is newest" (cached 5 min), or pin a run id from `/v2/{model}/runs`
for immutable, CDN-cacheable URLs. Colour scales are fixed per variable, so tiles never
seam or flicker, and the legend in `/v2/{model}/variables` is the exact scale used.

## MCP (AI assistants)

`forecast/mcp_server.py` serves the forecast as [Model Context Protocol](https://modelcontextprotocol.io)
tools at `/v2/mcp`, built on the official `mcp` SDK. It speaks streamable HTTP, stateless, with
plain JSON responses: any worker can answer any request, and it handles both the 2025 handshake
protocols and the 2026-07-28 stateless revision. `GET` and `DELETE` return 405 (there is no
server-to-client stream and no session).

| Tool | Returns |
|---|---|
| `get_forecast(location, days=7, timezone="")` | Current conditions and a daily table (min/max temperature, rain, wind, gusts, humidity, cloud, solar; at sea, waves and swell now and the daily highest waves) |
| `get_hourly_forecast(location, hours=24, variables="", timezone="")` | A time-step table: hourly to 48 h, then 3-hourly to 7 days, then 6-hourly. Rain is the total since the previous row |
| `find_place(query, limit=5)` | Places matching a name (or nearest to coordinates), with region, country, kind, population |
| `list_variables()` | Variable ids, units and meaning |

`location` is a place name (`Paris`, `Same, Timor-Leste`) or `latitude, longitude`. Each forecast
starts by saying which place it used and naming other places with the same name, so the model
can catch a wrong match and ask the user.

All tools are read-only and return compact Markdown, which keeps results inside hosts' size
limits (Experts truncates at 8000 characters). Public access goes through the API-key router
(`https://api.4shadow.io/v2/mcp`, header `api-key: <key>`). The router meters MCP routes
separately from REST calls. Experts asks an admin to re-approve a connection whenever a tool's
name, description or parameters change, so change them deliberately.

```bash
# Smoke test (2025 handshake; a tools/call needs the initialize step first)
curl -s -X POST localhost:5003/v2/mcp -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

## Store format

`data/forecast/<model>/runs/<YYYYMMDDHH>/` holds one standard `.npy` file per variable,
int16 `[step, lat, lon]` with `value = raw * scale + offset` (`-32768` = missing), plus a
`manifest.json` describing the grid, steps, valid times, units and scale/offset. Anything
can read it: `np.load("t2m.npy", mmap_mode="r")`. Runs are written to `<run>.incoming/`
and renamed into place, then `latest.json` is replaced atomically, so readers never see a
partial run. The newest `FORECAST_KEEP_RUNS` runs are kept (default 2). A full GFS run with sea
state is about 15 GB, an IFS run about 3.8 GB, an AIFS run 2 GB, an AIGFS run 1.2 GB, an ocean run 1 GB and
a GEFS-Aerosols run 0.6 GB (about 2 MB per variable and step).

## Adding a variable

Add a `Variable` to `forecast/catalog.py` with the GRIB abbreviation and level text exactly
as they appear in the NOAA `.idx` files (e.g. `TMP` / `2 m above ground`; for ECMWF, the short name and
level type from its `.index`, e.g. `2t` / `sfc` or `vsw` / `sol:1`), its `kind`
(`instant`, `accum` for running totals, `average` for bucket means), unit conversion,
storage quantum and an optional palette. The next ingested run includes it. Set `product="wave"`
for a GFS-Wave field (`gfswave.tCCz.global.0p25.fFFF.grib2`); a missing wave file leaves its
variables empty for that step instead of failing the run. Wave files are JPEG 2000 packed, so
they decode at about 0.14 s per field (about 3 minutes per full run for the seven wave fields).

## Adding a model

A model needs a source (`forecast/sources/`: its grid, steps, file names and attribution) and a
catalog (`forecast/catalog.py`), registered in `sources.SOURCES` and `catalog.CATALOGS` under the same
id, and added to `config.KNOWN_MODELS`. Sources that publish NCEP-style GRIB2 with `.idx` files use
the existing client unchanged; `forecast/sources/aigfs.py` is the example. Another inventory format
sets `index_url` and `parse_index` on the source (`forecast/sources/ecmwf.py`), and a global field on
the same grid spacing that starts at another longitude is turned at decoding. A model read through a
data service's own client rather than as GRIB files is an `ArraySource` (`forecast/sources/copernicus_marine.py`):
its client finds the newest run and returns each step's arrays on the source's grid. The API serves every model
in `FORECAST_MODELS` under `/v2/{model}/...` (its point forecast at `/v2/{model}/forecast`), and the ingest reads them one after the other.

## Configuration

| Env var | Default | |
|---|---|---|
| `FORECAST_DATA_ROOT` | `<repo>/data/forecast` | Store location |
| `FORECAST_MODELS` | every model; `ocean` only with a Copernicus Marine login | Models to serve and ingest. GFS is always included |
| `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD` | none | A Copernicus Marine account (free, at data.marine.copernicus.eu), for the ocean model |
| `GFS_BASE_URL` | NOAA's bucket on AWS | Where GFS comes from (a mirror, for example) |
| `AIGFS_BASE_URL` | NOAA's bucket on AWS | Where AIGFS comes from; NOMADS is earlier but rate-limited |
| `IFS_BASE_URL`, `AIFS_BASE_URL` | ECMWF's Google Cloud mirror | Where ECMWF's open data comes from; also on AWS (`https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com`) and at `https://data.ecmwf.int/forecasts` |
| `AEROSOLS_BASE_URL` | NOAA's bucket on AWS | Where GEFS-Aerosols comes from |
| `FORECAST_KEEP_RUNS` | `2` | Published runs to keep |
| `FORECAST_MAX_FHR` | `384` | Last forecast hour to ingest |
| `FORECAST_DOWNLOAD_WORKERS` | `6` | Concurrent step downloads |
| `FORECAST_TILE_CACHE_SIZE` | `4096` | In-process rendered-tile LRU per worker |
| `FORECAST_PLACES_DB` | `<repo>/data/places/places.sqlite` | Place index (`python -m forecast.places build`) |

## Legacy v1 compatibility (temporary)

`forecast/legacy_v1.py` serves the retired v1 service's response shapes (`/forecast`, `/point`,
`/parameters`, `/tiles`, in v1 units such as Kelvin and Pa) from the v2 store under
`/legacy/v1/...`, so existing v1 clients keep working after v1 is shut down. The API-key router
maps public `/v1/...` routes there. It is excluded from the v2 OpenAPI docs. Delete the module, its
JSON file and the router rows once no client calls `/v1`.

## Data attribution

GFS and GFS-Wave data: NOAA/NCEP, public domain, distributed through the NOAA Open Data
Dissemination program on AWS (`noaa-gfs-bdp-pds`). AIGFS data: NOAA/NCEP, based on Google
DeepMind's GraphCast; NOAA releases its products under CC0 (`noaa-nws-graphcastgfs-pds`). GEFS-Aerosols
data: NOAA/NCEP, public domain (`noaa-gefs-pds`). IFS and AIFS data: © European Centre for
Medium-Range Weather Forecasts (ECMWF), open data under CC BY 4.0; 4Shadow converts units, turns
totals into rates and derives further variables, and every answer from these models carries the
attribution in `source.attribution`. Ocean currents: generated using E.U. Copernicus Marine Service
Information (https://doi.org/10.48670/moi-00016), under the Copernicus Marine Service licence, which
allows commercial use; 4Shadow averages them onto 0.25° and derives speed and direction, and the
attribution is in `source.attribution`. Place names: GeoNames (geonames.org), licensed under
CC BY 4.0; responses that use them say so.
