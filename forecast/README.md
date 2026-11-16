# Foreshadow Forecast v2

A forecast for any point on Earth, plus map tiles, from open numerical weather prediction
data. Two models today: NOAA GFS 0.25°, with sea state from GFS-Wave, and NOAA AIGFS, NOAA's AI
model. The store and API are model-agnostic.

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

Both are on the same 0.25° grid and arrive about four times a day: GFS about 5 hours after each run
starts, AIGFS on NOAA's AWS bucket about 8 hours after. With 6-hourly steps, AIGFS's daily highs and
lows are of four values a day and can miss the peak. Its source files list fewer fields than GFS's
and NCEP says their contents may change; a field missing from a step's file is left empty there.

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

AIGFS has no stored humidity, so its `rh2m` and the indices use the dew point. Daily summaries add
`feels_like_max`, `feels_like_min`, `wet_bulb_max` and `fire_weather_max`.

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
state is about 15 GB, and an AIGFS run about 1.2 GB.

## Adding a variable

Add a `Variable` to `forecast/catalog.py` with the GRIB abbreviation and level text exactly
as they appear in the NOAA `.idx` files (e.g. `TMP` / `2 m above ground`), its `kind`
(`instant`, `accum` for running totals, `average` for bucket means), unit conversion,
storage quantum and an optional palette. The next ingested run includes it. Set `product="wave"`
for a GFS-Wave field (`gfswave.tCCz.global.0p25.fFFF.grib2`); a missing wave file leaves its
variables empty for that step instead of failing the run. Wave files are JPEG 2000 packed, so
they decode at about 0.14 s per field (about 3 minutes per full run for the seven wave fields).

## Adding a model

A model needs a source (`forecast/sources/`: its grid, steps, file names and attribution) and a
catalog (`forecast/catalog.py`), registered in `sources.SOURCES` and `catalog.CATALOGS` under the same
id, and added to `config.KNOWN_MODELS`. Sources that publish NCEP-style GRIB2 with `.idx` files use
the existing client unchanged; `forecast/sources/aigfs.py` is the example. The API serves every model
in `FORECAST_MODELS` under `/v2/{model}/...` (its point forecast at `/v2/{model}/forecast`), and the ingest reads them one after the other.

## Configuration

| Env var | Default | |
|---|---|---|
| `FORECAST_DATA_ROOT` | `<repo>/data/forecast` | Store location |
| `FORECAST_MODELS` | `gfs,aigfs` | Models to serve and ingest. GFS is always included |
| `GFS_BASE_URL` | NOAA's bucket on AWS | Where GFS comes from (a mirror, for example) |
| `AIGFS_BASE_URL` | NOAA's bucket on AWS | Where AIGFS comes from; NOMADS is earlier but rate-limited |
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
DeepMind's GraphCast; NOAA releases its products under CC0 (`noaa-nws-graphcastgfs-pds`). Place names: GeoNames (geonames.org), licensed under
CC BY 4.0; responses that use them say so.
