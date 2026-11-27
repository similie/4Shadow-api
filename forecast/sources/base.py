"""What the ingest needs to know about a model: GRIB files on an open-data server (``Source``), or arrays
read through a data service's own client (``ArraySource``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..grid import Grid


class StepUnavailable(RuntimeError):
    """A step a source hasn't published yet: the run isn't ingested this time."""


def _idx_url(file_url: str) -> str:
    return file_url + ".idx"


@dataclass(frozen=True)
class Source:
    """A model published as GRIB2 files with an inventory per file, one file per step and product.

    ``file_url(base_url, run, fhr, product)`` names a step's file and ``index_url`` its inventory: NCEP's
    ``.idx`` text by default, read by ``gfs.parse_idx``; ECMWF's JSON-lines ``.index`` sets both
    ``index_url`` and ``parse_index``. The first product in a catalog is required at every step, others
    optional. Files on another 0.25° global grid (ECMWF's starts at 180°) are turned at decoding onto
    ``grid``, so every model is stored on the same grid.
    """

    model: str
    grid: Grid
    max_fhr: int
    default_base_url: str
    forecast_hours: Callable[[int], list[int]]
    file_url: Callable[[str, datetime, int, str], str]
    source_info: Callable[[str, datetime, Sequence[str]], dict]
    cycle_hours: tuple[int, ...] = (0, 6, 12, 18)
    index_url: Callable[[str], str] = _idx_url
    parse_index: Callable[[str], list] | None = None  # None: NCEP .idx (gfs.parse_idx)
    # /v2/health calls a model's newest run stale after this many hours: its cycle, the hours until its
    # last step is published, and the ingest.
    stale_after_hours: float = 12.0
    keep_runs: int | None = None  # published runs kept; None: FORECAST_KEEP_RUNS
    gfs_static: bool = True  # take GFS's land mask and terrain height when it has none of its own


@dataclass(frozen=True)
class ArraySource:
    """A model read as arrays through a data service's client rather than as GRIB files (Copernicus Marine).

    ``client()`` makes the reader: an object with ``latest_run(max_fhr)``, the newest run with that many
    hours (or None), and ``step(run, fhr, names, grid)``, the named variables at that step on ``grid`` (a
    dict of float32 arrays, NaN where missing), raising ``StepUnavailable`` when the step isn't there yet.
    ``base_url`` is where a run's ``source`` record says the data came from. A client that also has
    ``available_runs()`` (the runs it can read, oldest first) is caught up on every one of those the store
    lacks, not only the newest: observations, an hour a run, keep each hour.
    """

    model: str
    grid: Grid
    max_fhr: int
    base_url: str
    forecast_hours: Callable[[int], list[int]]
    source_info: Callable[[str, datetime, Sequence[str]], dict]
    client: Callable[[], object]
    cycle_hours: tuple[int, ...] = (0,)
    stale_after_hours: float = 12.0
    keep_runs: int | None = None  # published runs kept; None: FORECAST_KEEP_RUNS
    gfs_static: bool = True  # take GFS's land mask and terrain height when it has none of its own
