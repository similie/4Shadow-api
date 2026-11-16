# Changelog

Notable changes to the forecast service. API answers only gain fields within a major version; anything that could break a client is listed here.

## Unreleased

## 2.0.0 (2026-11-16)

The service rebuilt from scratch as 4Shadow v2, now on `main`. It has served api.4shadow.io since 2026-11-12.

### Added

- `/v2/forecast`: a forecast for any coordinates or place name, with current conditions, a time series (hourly to 5 days, 3-hourly to 16 days) and local-day summaries, in the place's time zone.
- Sea state from GFS-Wave: wave height, period and direction, wind sea and swell, with the nearest sea cell used on coasts.
- `/v2/places`: place search and nearest towns from GeoNames, in a local SQLite index.
- `/v2/{model}/tiles/...`: XYZ PNG tiles with fixed colour scales, and `/v2/{model}/variables` with the legends.
- `/v2/{model}/fields/...`: raw gridded values in the `4SF1` binary format, for client-side maps.
- `/v2/{model}/thresholds`: early-warning checks for an area and a set of conditions.
- `/v2/mcp`: an MCP server with forecast, hourly forecast, place search and variable tools.
- Ingest by byte range from NOAA's open data on AWS, with each run published atomically.
- `deploy/deploy.sh` to update a server (status, update, check, rollback), a Dockerfile and a Compose file.
- CI: lint, tests on Python 3.12 to 3.14, the Docker image, and a daily end-to-end run on real GFS data.

### Changed

- pygrib 2.1.8 (from 2.1.6), which has wheels for Linux on ARM and for Python 3.14. It writes the same store, byte for byte.
- Pillow 12.3.0 (from 11.0.0), which has wheels for Python 3.14. Tiles have the same pixels and are about 3% smaller.
- `requirements.txt` installs the forecast service. The experimental terrain engine has its own, in `terrain/requirements.txt`.

### Fixed

- Coordinates typed as a place (`q=`, the MCP `location`) are parsed in linear time. The old pattern backtracked polynomially on long runs of spaces, and the MCP `location` has no length limit, so one request could hold a worker for minutes.
- The experimental terrain engine accepts only real job ids (uuid4 hex) in paths, so a crafted id can't name a folder outside its jobs folder.

### Removed

- v1 (`tile_renderers/`, `downloaders/` and their systemd units), retired on 2026-11-12. It remains in git history. Clients of v1's response shapes are served by `forecast/legacy_v1.py` under `/legacy/v1` until they move to `/v2`.
