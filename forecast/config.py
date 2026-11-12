"""Runtime settings, read from the environment (a repo-root .env is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_root: Path
    keep_runs: int = 2
    max_fhr: int = 384
    download_workers: int = 6
    step_wait_seconds: int = 0
    gfs_base_url: str = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    tile_cache_size: int = 4096


def get_settings() -> Settings:
    root = os.getenv("FORECAST_DATA_ROOT")
    return Settings(
        data_root=Path(root).expanduser().resolve() if root else REPO_ROOT / "data" / "forecast",
        keep_runs=int(os.getenv("FORECAST_KEEP_RUNS", "2")),
        max_fhr=int(os.getenv("FORECAST_MAX_FHR", "384")),
        download_workers=int(os.getenv("FORECAST_DOWNLOAD_WORKERS", "6")),
        step_wait_seconds=int(os.getenv("FORECAST_STEP_WAIT_SECONDS", "0")),
        gfs_base_url=os.getenv("GFS_BASE_URL", "https://noaa-gfs-bdp-pds.s3.amazonaws.com"),
        tile_cache_size=int(os.getenv("FORECAST_TILE_CACHE_SIZE", "4096")),
    )
