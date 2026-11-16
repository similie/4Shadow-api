# Deploying the forecast service

Two ways to run a server that keeps itself current: **systemd** on a Linux host (what api.4shadow.io runs), or **Docker Compose**. Both run the same two processes:

- **The API** (`uvicorn forecast.api:app`), stateless; it reads the store.
- **The ingest** (`python -m forecast.ingest --loop`), which checks for a new GFS run every 10 minutes and publishes each complete run.

## What it needs

| | Full service (every step to 16 days, sea state, worldwide places) | Trying it out (`--max-fhr 24`, cities500) |
| --- | --- | --- |
| Disk | 40 GB: two published runs of about 9.5 GB, one run being written, and a 1.3 GB place index | 3 GB |
| Memory | 8 GB. The workers memory-map the store, so more memory means more of it stays in the page cache | 2 GB |
| CPU | 2 cores or more. Ingest uses one, for several minutes per run | 1 core |
| Download | About 2.4 GB per run, four runs a day, from NOAA's open data on AWS (free) | 300 MB once |
| Python | 3.12, 3.13 or 3.14. pygrib's wheels include ecCodes, so there is nothing else to install | |

The API has no authentication of its own. Keep its port private and put something in front that decides who may call it: a reverse proxy for your own apps, or an API-key gateway for a public service.

## systemd

### Install

As root, or adjust the paths and `User=` in the units:

```sh
git clone https://github.com/similie/4Shadow-api.git /root/foreshadow-v2
cd /root/foreshadow-v2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional settings (see "Settings" below); the units read this file.
touch .env

# The units expect the checkout at /root/foreshadow-v2; change WorkingDirectory, EnvironmentFile,
# ExecStart and the log paths if yours is elsewhere.
cp deploy/forecast-api.service deploy/forecast-ingest.service /etc/systemd/system/
cp deploy/forecast.logrotate /etc/logrotate.d/foreshadow-v2
systemctl daemon-reload

# The first run: hours 0-24 in a few minutes, so the API has something to serve...
.venv/bin/python -m forecast.ingest --max-fhr 24
# ...and the worldwide place index (1.3 GB; it takes a while). `build` alone makes the small cities500 index.
.venv/bin/python -m forecast.places build --dump allCountries

systemctl enable --now forecast-api forecast-ingest
deploy/deploy.sh check
```

The ingest service then replaces the short first run with the next full one. The units cap the services' resources (`CPUQuota`, `MemoryMax`, and idle IO priority for the ingest), so they share a host politely; raise the limits on a dedicated machine. Logs go to `forecast-api.log` and `forecast-ingest.log` in the checkout and are rotated daily.

### Update

`deploy/deploy.sh` moves a checkout to the newest commit of a branch (`main` by default) and checks the result:

```sh
deploy/deploy.sh status      # what's running, and what's new on origin/main
deploy/deploy.sh update      # deploy it
deploy/deploy.sh check       # smoke tests only
deploy/deploy.sh rollback    # back to the commit before the last update
```

`update` does this, and stops at the first problem:

1. Refuses to run if tracked files were changed by hand.
2. Fetches and fast-forwards. It never merges or rewrites history; a checkout that has diverged from the branch is left as it is.
3. Installs `forecast/requirements.txt` and checks that the new code imports. If it doesn't, the checkout goes back to the old commit and nothing is restarted.
4. Restarts the API and waits for it to answer.
5. Restarts the ingest **only if the ingest code changed**, because a restart begins any run being downloaded again. `INGEST=skip` or `INGEST=restart` overrides this.
6. Runs the smoke tests: health, a forecast by coordinates and by name, place search, a tile, a field (decoded), the MCP handshake and the legacy `/v1` shim.

The commit before each update is kept in `.deploy/previous` and every update and rollback is logged in `.deploy/history`. A rollback goes back one step only; run `update` to move forward again.

A checkout on another branch is switched to `main` by `update`, but only when that is a pure move forward: `main` must contain the running commit, and a local `main` must not have commits of its own. Otherwise it stops and says so.

Settings come from the environment: `BRANCH`, `API` (default `http://127.0.0.1:5003`), `API_UNIT`, `INGEST_UNIT`, `INGEST`, `PYTHON` and `SYSTEMCTL`. The script's header describes each.

## Docker Compose

```sh
git clone https://github.com/similie/4Shadow-api.git && cd 4Shadow-api
docker compose up -d                                          # the API on :5003 and the ingest
docker compose run --rm api python -m forecast.places build   # place search (cities500)
```

`FORECAST_MAX_FHR=24 docker compose up -d` keeps the store small for a try-out. Data lives in the `forecast-data` volume; to update, `git pull && docker compose up -d --build`. The image runs as an unprivileged user, and its health check calls `/v2` (which answers before the first run is in; `/v2/health` returns 503 until then).

## Behind a reverse proxy

A minimal nginx site for your own apps, with a cache for tiles and fields. Tiles and fields of a pinned run (`/v2/gfs/tiles/2026111600/...`) never change, and the API marks them `immutable`; `latest` URLs are cached for 5 minutes.

```nginx
proxy_cache_path /var/cache/nginx/forecast levels=1:2 keys_zone=forecast:50m max_size=5g inactive=12h;

server {
    listen 443 ssl;
    server_name forecast.example.org;
    # ssl_certificate ...;

    location /v2/ {
        proxy_pass http://127.0.0.1:5003;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }

    location ~ ^/v2/[a-z0-9_]+/(tiles|fields)/ {
        proxy_pass http://127.0.0.1:5003;
        proxy_cache forecast;
        proxy_cache_valid 200 12h;
        proxy_cache_use_stale updating error timeout;
        add_header X-Cache $upstream_cache_status;
    }
}
```

`POST /v2/mcp` and `POST /v2/{model}/thresholds` must reach the API unbuffered and uncached, which the `/v2/` block does. On the hosted service, thresholds aren't public: the 4Shadow site calls them for users' alerts.

## Monitoring

`GET /v2/health` returns 200 with each model's latest run and its age, `stale: true` when the newest run is more than 12 hours old, and 503 when no run is published. GFS runs start every 6 hours (00, 06, 12 and 18 UTC) and are published a few hours later, once every step is in. If runs stop arriving, `forecast-ingest.log` says why (NOAA delays, a missing file, disk space).

## Settings

Environment variables, from the unit's `EnvironmentFile` (`.env` in the checkout) or the Compose file. All are optional.

| Variable | Default | |
| --- | --- | --- |
| `FORECAST_DATA_ROOT` | `<checkout>/data/forecast` | Where runs are stored |
| `FORECAST_KEEP_RUNS` | `2` | Published runs to keep |
| `FORECAST_MAX_FHR` | `384` | Last forecast hour to ingest |
| `FORECAST_DOWNLOAD_WORKERS` | `6` | Steps downloaded at once |
| `FORECAST_STEP_WAIT_SECONDS` | `0` | Pause between steps, to spread the load |
| `GFS_BASE_URL` | `https://noaa-gfs-bdp-pds.s3.amazonaws.com` | Where GFS comes from (a mirror, for example) |
| `FORECAST_TILE_CACHE_SIZE` | `4096` | Rendered tiles kept in memory per worker |
| `FORECAST_PLACES_DB` | `<checkout>/data/places/places.sqlite` | The place index |
