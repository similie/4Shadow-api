"""Model sources the ingest can read, by model id (the same ids as catalog.CATALOGS)."""

from __future__ import annotations

from . import aigfs, copernicus_marine, ecmwf, gefs_aerosols, gfs
from .base import ArraySource, Source

SOURCES: dict[str, Source | ArraySource] = {s.model: s for s in (gfs.SOURCE, aigfs.SOURCE, ecmwf.IFS, ecmwf.AIFS,
                                                                  gefs_aerosols.SOURCE, copernicus_marine.SOURCE)}

__all__ = ["SOURCES", "ArraySource", "Source"]
