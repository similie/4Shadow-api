"""What the ingest needs to know about a model's files on an open-data server."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..grid import Grid


@dataclass(frozen=True)
class Source:
    """A model published as NCEP-style GRIB2 files with ``.idx`` inventories, one file per step and
    product. ``file_url(base_url, run, fhr, product)`` names a step's file; its index is the same
    URL plus ``.idx``. The first product in a catalog is required at every step, others optional."""

    model: str
    grid: Grid
    max_fhr: int
    default_base_url: str
    forecast_hours: Callable[[int], list[int]]
    file_url: Callable[[str, datetime, int, str], str]
    source_info: Callable[[str, datetime, Sequence[str]], dict]
    cycle_hours: tuple[int, ...] = (0, 6, 12, 18)
