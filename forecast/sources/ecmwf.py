"""ECMWF open data: IFS, ECMWF's physics model ("the European model"), and AIFS Single, its AI model.

ECMWF publishes both under CC BY 4.0 with commercial use allowed, as GRIB2 files with a JSON-lines
``.index`` per file that gives each message's byte offset and length. Its 0.25° grid has the same
points as GFS's but starts at 180°; decoding turns each field to start at 0°, so all models share one
stored grid (tiles, fields, land masks and point lookups don't care which model a run came from).

* IFS 0.25°: 3-hourly to 144 h, then 6-hourly to 15 days, from the 00 and 12 UTC runs. ECMWF also
  publishes 06 and 18 UTC runs, but only to 144 h; ingesting them would make the newest run's range
  jump between 6 and 15 days, so this reads the long runs only. The last step is out about 7.5 h after
  the run starts. Sea state comes from the ``wave`` stream (ECWAM) on the same grid and steps.
* AIFS Single 0.25°: 6-hourly to 15 days, four runs a day, out about 6 h after the run starts.

The files are mirrored on Google Cloud, AWS and Azure and served from data.ecmwf.int. The Google Cloud
mirror is the default here; IFS_BASE_URL / AIFS_BASE_URL choose another, e.g.
https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com or https://data.ecmwf.int/forecasts.
ECMWF keeps the newest 12 runs.
"""

from __future__ import annotations

import json
from datetime import datetime

from .base import Source
from .gfs import GRID, IdxRecord

DEFAULT_BASE_URL = "https://storage.googleapis.com/ecmwf-open-data"
MAX_FHR = 360
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
ATTRIBUTION = ("Contains ECMWF open data, © European Centre for Medium-Range Weather Forecasts (ECMWF), "
               "licensed under CC BY 4.0. 4Shadow converts units, turns totals into rates and derives "
               "further variables from it.")

# Parameters ECMWF stores as totals since the run started; the rest are values at the step (or, like
# the 10 m gust and the 3-hourly temperature extremes, a statistic over the time just before it).
ACCUMULATED = frozenset({"tp", "cp", "lsp", "sf", "ssrd", "strd", "ssr", "str", "ttr", "tsr", "ro", "e", "pev",
                         "ewss", "nsss", "rowe"})


def parse_index(text: str) -> list[IdxRecord]:
    """ECMWF's ``.index``: one JSON object per message with ``param``, ``levtype``, ``levelist``,
    ``step`` and the exact ``_offset`` / ``_length``. Levels read ``sfc``, ``sol:1`` (soil layer 1) or
    ``pl:850``; totals since the start get NCEP's "0-N hour acc fcst" so selection works as for GFS."""
    out = []
    rows = [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
    for num, r in enumerate(sorted(rows, key=lambda r: int(r["_offset"])), start=1):
        step = int(str(r["step"]).split("-")[-1])
        level = r["levtype"] + (f":{r['levelist']}" if r.get("levelist") else "")
        fcst = f"0-{step} hour acc fcst" if r["param"] in ACCUMULATED else f"{step} hour fcst"
        start = int(r["_offset"])
        out.append(IdxRecord(num, start, start + int(r["_length"]) - 1, r["param"], level, fcst))
    return out


def index_url(file_url: str) -> str:
    return file_url.removesuffix(".grib2") + ".index"


def _stem(base_url: str, run: datetime, path: str) -> str:
    return f"{base_url}/{run:%Y%m%d}/{run:%H}z/{path}/{run:%Y%m%d%H}0000"


# --------------------------------------------------------------------------- IFS

IFS_STREAMS = {"atmos": ("ifs/0p25/oper", "oper"), "wave": ("ifs/0p25/wave", "wave")}


def ifs_forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """IFS 00/12 UTC steps: 3-hourly to 144 h, then 6-hourly to 360 h."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(0, min(max_fhr, 144) + 1, 3)) + list(range(150, max_fhr + 1, 6))


def ifs_file_url(base_url: str, run: datetime, fhr: int, product: str = "atmos") -> str:
    if product not in IFS_STREAMS:
        raise ValueError(f"Unknown IFS product {product!r}")
    path, stream = IFS_STREAMS[product]
    return f"{_stem(base_url, run, path)}-{fhr}h-{stream}-fc.grib2"


def ifs_source_info(base_url: str, run: datetime, products=("atmos",)) -> dict:
    info = {
        "provider": "ECMWF Integrated Forecasting System (IFS), 0.25° open data",
        "distribution": "ECMWF open data",
        "url": f"{_stem(base_url, run, 'ifs/0p25/oper').rsplit('/', 1)[0]}/",
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }
    if "wave" in products:
        info["provider"] += ", with sea state from ECMWF's wave model (ECWAM)"
        info["wave_url"] = f"{_stem(base_url, run, 'ifs/0p25/wave').rsplit('/', 1)[0]}/"
    return info


IFS = Source("ifs", GRID, MAX_FHR, DEFAULT_BASE_URL, ifs_forecast_hours, ifs_file_url, ifs_source_info,
             cycle_hours=(0, 12), index_url=index_url, parse_index=parse_index,
             stale_after_hours=24)  # twice a day, last step out ~7.5 h after the run starts


# --------------------------------------------------------------------------- AIFS Single

def aifs_forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """AIFS Single steps: every 6 hours to 360 h."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(0, max_fhr + 1, 6))


def aifs_file_url(base_url: str, run: datetime, fhr: int, product: str = "atmos") -> str:
    if product != "atmos":
        raise ValueError(f"Unknown AIFS product {product!r}")
    return f"{_stem(base_url, run, 'aifs-single/0p25/oper')}-{fhr}h-oper-fc.grib2"


def aifs_source_info(base_url: str, run: datetime, products=("atmos",)) -> dict:
    return {
        "provider": "ECMWF Artificial Intelligence Forecasting System (AIFS Single), 0.25° open data",
        "distribution": "ECMWF open data",
        "url": f"{_stem(base_url, run, 'aifs-single/0p25/oper').rsplit('/', 1)[0]}/",
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


AIFS = Source("aifs", GRID, MAX_FHR, DEFAULT_BASE_URL, aifs_forecast_hours, aifs_file_url, aifs_source_info,
              index_url=index_url, parse_index=parse_index,
              stale_after_hours=15)  # 6-hourly, last step out ~6 h after the run starts
