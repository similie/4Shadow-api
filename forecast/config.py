"""Runtime settings, read from the environment (a repo-root .env is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_MODELS = ("gfs", "aigfs", "ifs", "aifs", "aerosols", "air", "ocean", "satellite")
# Models read with an account, by the environment variables that hold its login. Without them a model is
# left out of the default FORECAST_MODELS (it couldn't be ingested); listing it there turns it on anyway.
ACCOUNT_MODELS = {"air": ("CDSAPI_KEY",),
                  "ocean": ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")}


def default_models() -> str:
    """Every model, except one whose account login isn't set."""
    return ",".join(m for m in KNOWN_MODELS if all(os.getenv(k) for k in ACCOUNT_MODELS.get(m, ())))


def parse_models(value: str) -> tuple[str, ...]:
    """FORECAST_MODELS, e.g. "gfs,aigfs,ifs". GFS is always included and comes first: the MCP server and the
    /v1 compatibility routes use it, and the service counts as up when it has a GFS run."""
    models = [m.strip().lower() for m in value.split(",") if m.strip()]
    unknown = sorted(set(models) - set(KNOWN_MODELS))
    if unknown:
        raise ValueError(f"FORECAST_MODELS: unknown model(s) {unknown}; known: {list(KNOWN_MODELS)}")
    return ("gfs", *dict.fromkeys(m for m in models if m != "gfs"))


@dataclass(frozen=True)
class Settings:
    data_root: Path
    keep_runs: int = 2
    max_fhr: int = 384
    download_workers: int = 6
    step_wait_seconds: int = 0
    gfs_base_url: str = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    aigfs_base_url: str | None = None  # None: the model's default (NOAA's open-data bucket on AWS)
    # Other models' servers by model id (IFS_BASE_URL, AIFS_BASE_URL, AEROSOLS_BASE_URL); absent: the default.
    base_urls: tuple[tuple[str, str], ...] = ()
    models: tuple[str, ...] = KNOWN_MODELS  # models to serve and ingest; GFS first
    tile_cache_size: int = 4096
    places_db: Path | None = None  # place search index (python -m forecast.places build); None turns it off

    def base_url(self, model: str) -> str | None:
        return {**dict(self.base_urls), "gfs": self.gfs_base_url, "aigfs": self.aigfs_base_url}.get(model)


def get_settings() -> Settings:
    root = os.getenv("FORECAST_DATA_ROOT")
    return Settings(
        data_root=Path(root).expanduser().resolve() if root else REPO_ROOT / "data" / "forecast",
        keep_runs=int(os.getenv("FORECAST_KEEP_RUNS", "2")),
        max_fhr=int(os.getenv("FORECAST_MAX_FHR", "384")),
        download_workers=int(os.getenv("FORECAST_DOWNLOAD_WORKERS", "6")),
        step_wait_seconds=int(os.getenv("FORECAST_STEP_WAIT_SECONDS", "0")),
        gfs_base_url=os.getenv("GFS_BASE_URL", "https://noaa-gfs-bdp-pds.s3.amazonaws.com"),
        aigfs_base_url=os.getenv("AIGFS_BASE_URL") or None,
        base_urls=tuple((m, url) for m in KNOWN_MODELS if m not in ("gfs", "aigfs")
                        if (url := os.getenv(f"{m.upper()}_BASE_URL"))),
        models=parse_models(os.getenv("FORECAST_MODELS") or default_models()),
        tile_cache_size=int(os.getenv("FORECAST_TILE_CACHE_SIZE", "4096")),
        places_db=Path(os.getenv("FORECAST_PLACES_DB") or REPO_ROOT / "data" / "places" / "places.sqlite").expanduser(),
    )
