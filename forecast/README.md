# Foreshadow Forecast v2

A forecast for any point on Earth, plus map tiles, from open numerical weather prediction
data. Today the source is NOAA GFS 0.25°; the store and API are model-agnostic.

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
API service (`deploy/forecast-api.service`). v2 has its own dependency set
(`forecast/requirements.txt`); give it its own venv rather than sharing the v1/terrain one,
which pins an older numpy for GDAL.

## API

| Endpoint | Purpose |
|---|---|
| `GET /v2/forecast?lat&lon` or `?q=` | Point forecast: `current`, `timeseries`, local-day `daily` |
| `GET /v2/places?q=` or `?lat&lon` | Place search: names to coordinates, or the nearest towns to a point |
| `GET /v2/{model}/tiles/{run}/{var}/{time}/{z}/{x}/{y}.png` | XYZ map tiles |
| `GET /v2/{model}/variables` | Variable catalog with units and legend colour stops |
| `GET /v2/{model}/runs`, `/v2/{model}/runs/{run}` | Available runs and their valid times |
| `GET /v2/health` | Latest run per model and its age (503 if none) |
| `POST /v2/mcp` | MCP server for AI assistants (see below) |

**Forecast parameters**: `lat` and `lon`, or `q` (a place name such as `Paris, France`; the best
match is used), `vars` (comma list, default all), `hours` (default 384),
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
| `get_forecast(location, days=7, timezone="")` | Current conditions and a daily table (min/max temperature, rain, wind, gusts, humidity, cloud, solar) |
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
partial run. The newest `FORECAST_KEEP_RUNS` runs are kept (default 2, about 6 GB each for GFS).

## Adding a variable

Add a `Variable` to `forecast/catalog.py` with the GRIB abbreviation and level text exactly
as they appear in the NOAA `.idx` files (e.g. `TMP` / `2 m above ground`), its `kind`
(`instant`, `accum` for running totals, `average` for bucket means), unit conversion,
storage quantum and an optional palette. The next ingested run includes it.

## Configuration

| Env var | Default | |
|---|---|---|
| `FORECAST_DATA_ROOT` | `<repo>/data/forecast` | Store location |
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

GFS data: NOAA/NCEP, public domain, distributed through the NOAA Open Data Dissemination
program on AWS (`noaa-gfs-bdp-pds`). Place names: GeoNames (geonames.org), licensed under
CC BY 4.0; responses that use them say so.
