"""Model sources the ingest can read, by model id (the same ids as catalog.CATALOGS)."""

from __future__ import annotations

from . import aigfs, gfs
from .base import Source

SOURCES: dict[str, Source] = {s.model: s for s in (gfs.SOURCE, aigfs.SOURCE)}

__all__ = ["SOURCES", "Source"]
