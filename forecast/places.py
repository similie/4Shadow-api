"""Place search: place names to coordinates, and coordinates to the nearest town or village.

    python -m forecast.places build                       # cities500: ~200k towns of 500+ people
    python -m forecast.places build --dump allCountries   # every village, plus countries, regions,
                                                          # seas, bays, islands, ports and airports
    python -m forecast.places search "Paris, Texas"
    python -m forecast.places nearest -8.556 125.560

The index is one read-only SQLite file (default data/places/places.sqlite, or FORECAST_PLACES_DB)
built from GeoNames dumps (https://www.geonames.org, CC BY 4.0). A build writes a new file and
renames it into place, and readers reopen when the file changes, so rebuilding never disturbs a
running API. Lookups run on this server only: nobody's place queries go to a third party.

Matching is on normalised names (case, accents and punctuation ignored; other scripts kept), so
"Díli", "dili" and "DILI" are the same. Text after a comma narrows by region or country
("Paris, Texas", "Same, Timor-Leste"); without a comma, trailing words are tried as the country
("paris france"). Results rank exact names before prefixes, then by population and kind of place.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unicodedata
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .service import ForecastError

GEONAMES_URL = "https://download.geonames.org/export/dump"
ATTRIBUTION = "Place names: GeoNames (geonames.org), CC BY 4.0"
FORMAT = "1"
CELL_DEG = 0.1  # spatial cells for nearest-place lookups (~11 km)
LAT_CELLS, LON_CELLS = round(180 / CELL_DEG), round(360 / CELL_DEG)
NEAREST_MAX_KM = 50.0
EXACT, ALTERNATE, PREFIX = "exact", "alternate", "prefix"

# Besides every current populated place (class P), these GeoNames feature codes are kept.
# https://www.geonames.org/export/codes.html
KEEP_CODES = {
    "PCLI", "PCLD", "PCLF", "PCLS", "PCLIX", "PCL", "TERR", "ADM1", "ADM2", "ADM3",  # countries, divisions
    "CONT", "RGN", "PRK",  # continents, regions, parks
    "OCN", "SEA", "GULF", "BAY", "BAYS", "STRT", "SD", "CHN", "LGN", "LK", "LKS", "RSV", "HBR", "FJD", "COVE",
    "PRT", "MAR", "AIRP",  # ports, marinas, airports
    "ISL", "ISLS", "ISLET", "CAPE", "PEN", "PT", "MT", "MTS", "PK", "VLC", "PASS", "PLAT", "DSRT", "VAL",
}
DROP_POPULATED = {"PPLH", "PPLQ", "PPLW", "PPLCH"}  # historical, abandoned, destroyed

KINDS = {
    "PPLC": "capital city", "PPLG": "seat of government", "PPLX": "neighbourhood", "PCLI": "country", "PCLD": "dependent territory",
    "PCLF": "freely associated state", "PCLS": "semi-independent state", "PCLIX": "part of a country",
    "PCL": "country", "TERR": "territory", "ADM1": "region", "ADM2": "district", "ADM3": "subdistrict",
    "CONT": "continent", "RGN": "region", "PRK": "park", "OCN": "ocean", "SEA": "sea", "GULF": "gulf",
    "BAY": "bay", "BAYS": "bays", "STRT": "strait", "SD": "sound", "CHN": "channel", "LGN": "lagoon",
    "LK": "lake", "LKS": "lakes", "RSV": "reservoir", "HBR": "harbour", "FJD": "fjord", "COVE": "cove",
    "PRT": "port", "MAR": "marina", "AIRP": "airport", "ISL": "island", "ISLS": "islands", "ISLET": "islet",
    "CAPE": "cape", "PEN": "peninsula", "PT": "point", "MT": "mountain", "MTS": "mountains", "PK": "peak",
    "VLC": "volcano", "PASS": "pass", "PLAT": "plateau", "DSRT": "desert", "VAL": "valley",
}
# Kinds that cover an area, where a single-point forecast doesn't describe the whole thing.
AREAS = {"PCLI", "PCLD", "PCLF", "PCLS", "PCLIX", "PCL", "TERR", "ADM1", "ADM2", "CONT", "RGN", "PRK",
         "OCN", "SEA", "GULF", "BAYS", "STRT", "SD", "LKS", "ISLS", "PEN", "MTS", "PLAT", "DSRT"}
# Importance = log10(population + 1) + BONUS. Areas (classes A and L) count half their log population,
# since their totals would otherwise outrank the towns in them; water and land features rarely have
# a population, so their bonus alone places them ("Mediterranean" -> the sea, not a region of Turkey).
BONUS = {
    "PPLC": 1.5, "PPLG": 1.0, "PPLA": 1.0, "PPLA2": 0.5,
    "CONT": 6.0, "PCLI": 4.0, "PCL": 3.5, "PCLD": 3.0, "PCLF": 3.0, "PCLS": 3.0, "TERR": 3.0, "PCLIX": 2.0,
    "ADM1": 1.5, "ADM2": 0.5, "RGN": 1.0, "PRK": 0.5,
    "OCN": 6.0, "SEA": 5.0, "GULF": 3.5, "STRT": 3.0, "BAY": 1.5, "SD": 1.5, "LK": 1.5, "LKS": 1.5, "FJD": 1.0,
    "CHN": 1.0, "ISL": 1.0, "ISLS": 1.5, "PEN": 1.5, "MTS": 1.5, "MT": 1.0, "PK": 1.0, "VLC": 1.5, "DSRT": 2.0,
    "CAPE": 0.5, "AIRP": 1.0, "PRT": 1.0, "MAR": 0.5, "HBR": 0.5,
}
COUNTRY_CODES = {"PCLI", "PCL", "PCLD", "PCLF", "PCLS", "PCLIX", "TERR"}
LANDMARK_POPULATION = 1000  # forecasts say "near <town>" for the closest place at least this big
# Score bonus by how the name matched: primary name, ASCII name, alternate name, prefix of a name.
# A place's own name beats another place's alternate name unless that place is ~100x larger ("Dili"
# is also a name for Delhi; "Rio" is Rio de Janeiro's), and a prefix needs ~10^5 times the
# population to beat an exact name ("Paris" -> Paris, Texas before Île-de-France, alias "Paris Region").
MATCH_BONUS = {0: 3.0, 1: 3.0, 2: 1.0, 3: -2.0}

# Official short names where GeoNames' countryInfo.txt differs.
COUNTRY_NAMES = {"TL": "Timor-Leste"}
# Common country names that GeoNames' countryInfo.txt doesn't list. Keys are normalised (see norm).
COUNTRY_ALIASES = {
    "usa": "US", "us": "US", "america": "US", "united states of america": "US", "uk": "GB",
    "great britain": "GB", "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "northern ireland": "GB", "east timor": "TL", "timor leste": "TL", "holland": "NL", "czech republic": "CZ",
    "cote d ivoire": "CI", "ivory coast": "CI", "korea": "KR", "russian federation": "RU", "uae": "AE",
    "drc": "CD", "dr congo": "CD", "burma": "MM", "viet nam": "VN", "lao pdr": "LA", "macedonia": "MK",
    "swaziland": "SZ", "cape verde": "CV", "vatican": "VA", "the netherlands": "NL", "the gambia": "GM",
    "the bahamas": "BS", "png": "PG", "nz": "NZ",
}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE place (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
    fclass TEXT, fcode TEXT, cc TEXT, admin1 TEXT, region TEXT, country TEXT,
    population INTEGER, elevation INTEGER, timezone TEXT, cell INTEGER, importance REAL
);
CREATE TABLE name (key TEXT, place INTEGER, kind INTEGER, importance REAL, PRIMARY KEY (key, place)) WITHOUT ROWID;
CREATE TABLE alias (key TEXT, cc TEXT, admin1 TEXT);
"""
COLUMNS = ("id", "name", "lat", "lon", "fclass", "fcode", "cc", "admin1", "region", "country",
           "population", "elevation", "timezone")


# --------------------------------------------------------------------------- names and coordinates

_ASCII_SEP = re.compile(r"[^a-z0-9]+")


def norm(text: str) -> str:
    """Normalise a name for matching: casefold, drop Latin/Greek/Cyrillic accents (other scripts
    keep their marks, which carry vowels), and treat punctuation and symbols as spaces."""
    if text.isascii():
        return " ".join(_ASCII_SEP.sub(" ", text.lower()).split())
    out, base = [], 0
    for ch in unicodedata.normalize("NFKD", text):
        cat = unicodedata.category(ch)[0]
        if cat == "M":
            if base < 0x0530:
                continue
        elif cat in "PZSC":
            ch, base = " ", 0x20
        else:
            base = ord(ch)
        out.append(ch)
    return " ".join(unicodedata.normalize("NFC", "".join(out)).casefold().split())


_COORDS = re.compile(
    r"^\s*([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,;\s]\s*([-+]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\s*$")


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """'-8.556, 125.56', '48.85 2.35' or '48.85N 2.35E' -> (lat, lon); None if text isn't coordinates."""
    m = _COORDS.match(text)
    if not m:
        return None
    lat, ns, lon, ew = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if ns and ns.upper() == "S":
        lat = -abs(lat)
    if ew and ew.upper() == "W":
        lon = -abs(lon)
    return lat, lon


def cell_of(lat: float, lon: float) -> int:
    i = min(max(int((lat + 90.0) / CELL_DEG), 0), LAT_CELLS - 1)
    j = int(((lon + 180.0) % 360.0) / CELL_DEG) % LON_CELLS
    return i * LON_CELLS + j


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def importance(fclass: str, fcode: str, population: int) -> float:
    p = math.log10(max(population, 0) + 1)
    return (0.5 * p if fclass in ("A", "L") else p) + BONUS.get(fcode, 0.0)


# --------------------------------------------------------------------------- results

@dataclass
class Place:
    id: int
    name: str
    lat: float
    lon: float
    fclass: str
    fcode: str
    cc: str
    admin1: str
    region: str
    country: str
    population: int
    elevation: int | None
    timezone: str
    match: str = EXACT
    distance_km: float | None = None

    @property
    def kind(self) -> str:
        if self.fcode in KINDS:
            return KINDS[self.fcode]
        if self.fclass == "P":
            p = self.population
            size = "city" if p >= 100_000 else "town" if p >= 10_000 else "village" if p > 0 else "settlement"
            return f"{size}, regional capital" if self.fcode == "PPLA" else size
        return "place"

    @property
    def is_area(self) -> bool:
        return self.fcode in AREAS

    @property
    def display(self) -> str:
        """'Paris, Île-de-France, France': the name, then region and country where they add something."""
        parts, name, region = [self.name], norm(self.name), norm(self.region)
        # skip a region that only repeats the name: "Dili, Dili Municipality", "Washington, Washington, D.C."
        if self.fcode not in ("PCLI", "PCL", "ADM1") and region and region != name and not region.startswith(name + " "):
            parts.append(self.region)
        if self.country and not self.fcode.startswith("PCL") and norm(self.country) != norm(self.name):
            parts.append(self.country)
        return ", ".join(parts)

    def to_dict(self) -> dict:
        d = {
            "display_name": self.display, "name": self.name, "region": self.region or None,
            "country": self.country or None, "country_code": self.cc or None, "lat": self.lat, "lon": self.lon,
            "kind": self.kind, "area": self.is_area, "population": self.population, "elevation_m": self.elevation,
            "timezone": self.timezone or None, "match": self.match,
            "geonames_id": self.id, "feature_code": self.fcode,
        }
        if self.distance_km is not None:
            d["distance_km"] = self.distance_km
        return d


@dataclass
class Resolved:
    """Where a forecast request points: coordinates, plus the place they came from or are near."""
    lat: float
    lon: float
    query: str | None = None
    place: Place | None = None  # the place a name matched
    nearby: Place | None = None  # for coordinates: the nearest town or village
    alternatives: list[Place] = field(default_factory=list)  # other places with the same name

    def location_fields(self) -> dict:
        out: dict = {}
        if self.place is not None:
            out["query"] = self.query
            out["place"] = self.place.to_dict()
            out["alternatives"] = [p.to_dict() for p in self.alternatives]
        elif self.nearby is not None:
            out["nearest_place"] = self.nearby.to_dict()
        return out


# --------------------------------------------------------------------------- reading the index

class PlaceIndex:
    """Read-only access to the places database. Safe to share between threads: each thread opens its
    own connection, and reopens it when a rebuild has replaced the file."""

    def __init__(self, path: Path | None):
        self.path = Path(path).resolve() if path else None
        self._local = threading.local()

    @property
    def available(self) -> bool:
        return self.path is not None and self.path.is_file()

    def _con(self) -> sqlite3.Connection:
        if self.path is None:
            raise ForecastError(503, "Place search is not set up on this server")
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            raise ForecastError(503, "Place search is not set up on this server yet") from None
        sig = (st.st_ino, st.st_mtime_ns)
        cached = getattr(self._local, "con", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        if cached is not None:
            cached[1].close()
        con = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        fmt = con.execute("SELECT value FROM meta WHERE key = 'format'").fetchone()
        if not fmt or fmt[0] != FORMAT:
            con.close()
            raise ForecastError(503, "The place index needs rebuilding (python -m forecast.places build)")
        self._local.con = (sig, con)
        return con

    def meta(self) -> dict[str, str]:
        return dict(self._con().execute("SELECT key, value FROM meta"))

    def _places(self, con: sqlite3.Connection, ids: Iterable[int]) -> dict[int, Place]:
        ids = list(ids)
        out: dict[int, Place] = {}
        for n in range(0, len(ids), 500):
            chunk = ids[n:n + 500]
            sql = f"SELECT {', '.join(COLUMNS)} FROM place WHERE id IN ({','.join('?' * len(chunk))})"
            for row in con.execute(sql, chunk):
                out[row[0]] = Place(*row)
        return out

    def search(self, query: str, limit: int = 5) -> list[Place]:
        """Places matching a name, best first. Coordinates return the nearest populated places."""
        coords = parse_coordinates(query)
        if coords is not None:
            return self.nearest(*coords, limit=limit)
        parts = [p for p in (norm(s) for s in query.split(",")) if p]
        if not parts:
            raise ForecastError(400, "Give a place name, such as 'Paris, France', or coordinates 'lat, lon'")
        attempts = [(parts[0], parts[1:])]
        words = parts[0].split()
        if len(parts) == 1:  # "paris france", "san jose costa rica": try trailing words as the country/region
            attempts += [(" ".join(words[:-k]), [" ".join(words[-k:])]) for k in range(1, min(3, len(words) - 1) + 1)]
        con = self._con()
        for name, quals in attempts:
            found = self._lookup(con, name, quals, limit)
            if found:
                return found
        return []

    def _lookup(self, con: sqlite3.Connection, name: str, quals: list[str], limit: int) -> list[Place]:
        cap = 2000 if quals else 200
        cand: dict[int, tuple[int, float]] = {}  # place -> (name kind, score); kind 3 = prefix match
        rows = con.execute("SELECT place, kind, importance FROM name WHERE key = ? ORDER BY importance DESC LIMIT ?",
                           (name, cap)).fetchall()
        for pid, kind, imp in rows:
            cand[pid] = (kind, imp + MATCH_BONUS[kind])
        if len(name) >= 3:
            # Prefixes of places' own names only: alternates include old and foreign names that turn
            # "Nice" into Raqqa (ancient Nicephorium).
            rows = con.execute("SELECT place, kind, importance FROM name WHERE key > ? AND key < ? AND kind <= 1 "
                               "LIMIT 20000", (name, name + "\U0010ffff")).fetchall()
            rows.sort(key=lambda r: -r[2])
            for pid, _kind, imp in rows[:cap]:
                cand.setdefault(pid, (3, imp + MATCH_BONUS[3]))
        if not cand:
            return []
        places = self._places(con, cand)
        # A division named after its town ("Díli" municipality, New York state, the Paris arrondissement)
        # duplicates the town when both match: drop divisions sharing a matched town's country and region.
        towns = {(p.cc, p.admin1) for p in places.values() if p.fclass == "P" and cand[p.id][0] <= 2}
        places = {pid: p for pid, p in places.items()
                  if not (p.fcode.startswith("ADM") and cand[pid][0] <= 2 and (p.cc, p.admin1) in towns)}
        if quals:
            allowed = [self._qualifier(con, q) for q in quals]
            places = {pid: p for pid, p in places.items()
                      if all((p.cc, "") in a or (p.cc, p.admin1) in a for a in allowed)}
        ranked = sorted(places.values(), key=lambda p: (-cand[p.id][1], p.id))
        for p in ranked:
            kind = cand[p.id][0]
            p.match = EXACT if kind <= 1 else ALTERNATE if kind == 2 else PREFIX
        return ranked[:limit]

    def _qualifier(self, con: sqlite3.Connection, key: str) -> set[tuple[str, str]]:
        keys = (key, key.replace(" ", ""))  # "D.C." -> "d c" -> also "dc"
        rows = con.execute("SELECT cc, admin1 FROM alias WHERE key IN (?, ?)", keys)
        return {(cc, a1 or "") for cc, a1 in rows}

    def nearest(self, lat: float, lon: float, limit: int = 5, max_km: float = NEAREST_MAX_KM,
                min_population: int = 0) -> list[Place]:
        """Populated places nearest to a point, closest first, within max_km. With min_population, only
        places at least that big, or seats of a country, region or district (whose size GeoNames may lack)."""
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 360):
            raise ForecastError(400, "lat must be within -90..90 and lon within -180..360")
        lon = (lon + 180.0) % 360.0 - 180.0
        con = self._con()
        i0, j0 = divmod(cell_of(lat, lon), LON_CELLS)
        km_per_cell = CELL_DEG * 111.2 * max(math.cos(math.radians(min(abs(lat) + CELL_DEG, 90.0))), 0.0)
        found: list[tuple[float, int]] = []
        for ring in (1, 3, int(max_km / (CELL_DEG * 111.2)) + 1):
            cells = [i * LON_CELLS + (j % LON_CELLS) for i in range(i0 - ring, i0 + ring + 1) if 0 <= i < LAT_CELLS
                     for j in range(j0 - ring, j0 + ring + 1)]
            cells = sorted(set(cells))
            found = []
            for n in range(0, len(cells), 500):
                chunk = cells[n:n + 500]
                sql = (f"SELECT id, lat, lon FROM place WHERE fclass = 'P' AND cell IN ({','.join('?' * len(chunk))})"
                       " AND (population >= ? OR fcode IN ('PPLC', 'PPLG', 'PPLA', 'PPLA2'))")
                for pid, plat, plon in con.execute(sql, [*chunk, min_population]):
                    d = distance_km(lat, lon, plat, plon)
                    if d <= max_km:
                        found.append((d, pid))
            found.sort()
            covered = ring * km_per_cell  # every place this close is inside the searched cells
            if len(found) >= limit and found[limit - 1][0] <= covered:
                break
        places = self._places(con, [pid for _, pid in found[:limit]])
        out = []
        for d, pid in found[:limit]:
            p = places[pid]
            p.match, p.distance_km = EXACT, round(d, 1)
            out.append(p)
        return out

    def resolve(self, text: str, alternatives: int = 3) -> Resolved:
        """A place name or 'lat, lon' -> Resolved; ForecastError 404 when no place has that name."""
        coords = parse_coordinates(text)
        if coords is not None:
            return self.at(*coords)
        matches = self.search(text, limit=alternatives + 1)
        if not matches:
            raise ForecastError(404, f"No place called {text.strip()!r} was found. Check the spelling, add the "
                                     "country (for example 'Same, Timor-Leste'), or give coordinates as 'lat, lon'.")
        top, rest = matches[0], matches[1:]
        # Places whose own name this is; alternate names (old, foreign) would list York, Nebraska for "New York".
        same_name = [p for p in rest if p.match == EXACT] if top.match != PREFIX else []
        return Resolved(top.lat, top.lon, query=text.strip(), place=top, alternatives=same_name)

    def at(self, lat: float, lon: float) -> Resolved:
        """Coordinates, with a nearby place to name them by when the index is available (never an error):
        the closest town of LANDMARK_POPULATION or more, else the closest settlement of any size."""
        nearby = None
        if self.available:
            try:
                near = self.nearest(lat, lon, limit=1, min_population=LANDMARK_POPULATION) or self.nearest(lat, lon, limit=1)
                nearby = near[0] if near else None
            except (ForecastError, sqlite3.Error):
                pass
        return Resolved(lat, lon, nearby=nearby)


# --------------------------------------------------------------------------- building the index

def _download(name: str, dest_dir: Path) -> Path:
    import requests

    dest = dest_dir / name
    url = f"{GEONAMES_URL}/{name}"
    print(f"downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "4Shadow-places-builder"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def _lines(path: Path) -> Iterator[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.endswith(".txt") and not n.startswith("readme"))
            with z.open(inner) as raw:
                yield from io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    else:
        with open(path, encoding="utf-8", newline="\n") as f:
            yield from f


def _countries(path: Path) -> dict[str, tuple[str, str, str]]:
    """countryInfo.txt -> {ISO2: (ISO3, display name, GeoNames' name)}."""
    out = {}
    for line in _lines(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out[f[0]] = (f[1], COUNTRY_NAMES.get(f[0], f[4]), f[4])
    return out


def _admin1(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """admin1CodesASCII.txt -> {(cc, code): (name, ascii name)}."""
    out = {}
    for line in _lines(path):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and "." in f[0]:
            cc, code = f[0].split(".", 1)
            out[(cc, code)] = (f[1], f[2])
    return out


def _int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def build(dumps: list[Path], country_info: Path, admin1_codes: Path, out: Path, source: str = "") -> dict:
    """Build the index from GeoNames files into out (replaced atomically). Returns counts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp, stage = out.with_name(out.name + ".building"), out.with_name(out.name + ".stage")
    for p in (tmp, stage):
        p.unlink(missing_ok=True)
    os.environ.setdefault("SQLITE_TMPDIR", str(out.parent))  # sorting temp files next to the output, not /tmp
    countries, admin1 = _countries(country_info), _admin1(admin1_codes)
    con = sqlite3.connect(tmp)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-500000;" + SCHEMA)
    con.execute("ATTACH ? AS stage", (str(stage),))
    con.executescript("PRAGMA stage.journal_mode=OFF; PRAGMA stage.synchronous=OFF;")
    con.execute("CREATE TABLE stage.name (key TEXT, place INTEGER, kind INTEGER, importance REAL)")

    aliases: set[tuple[str, str, str]] = set()
    for cc, (iso3, cname, gname) in countries.items():
        aliases.update({(norm(cname), cc, ""), (norm(gname), cc, ""), (cc.lower(), cc, ""), (iso3.lower(), cc, "")})
    for key, cc in COUNTRY_ALIASES.items():
        aliases.add((key, cc, ""))
    for (cc, code), (aname, ascii_name) in admin1.items():
        aliases.update({(norm(aname), cc, code), (norm(ascii_name), cc, code), (code.lower(), cc, code)})

    seen: set[int] = set()
    n_rows = n_places = n_names = 0
    started = time.time()
    places, names = [], []

    def flush():
        con.executemany(f"INSERT OR IGNORE INTO place VALUES ({','.join('?' * 15)})", places)
        con.executemany("INSERT INTO stage.name VALUES (?, ?, ?, ?)", names)
        places.clear()
        names.clear()

    for dump in dumps:
        for line in _lines(dump):
            n_rows += 1
            f = line.rstrip("\n").split("\t")
            if len(f) < 19:
                continue
            fclass, fcode = f[6], f[7]
            if not ((fclass == "P" and fcode not in DROP_POPULATED) or fcode in KEEP_CODES):
                continue
            gid = int(f[0])
            if len(dumps) > 1:
                if gid in seen:
                    continue
                seen.add(gid)
            lat, lon, cc, a1 = float(f[4]), float(f[5]), f[8], f[10]
            pop = int(f[14] or 0)
            elev = _int(f[15]) if f[15] else _int(f[16])  # surveyed elevation, else the DEM value
            elev = None if elev == -9999 else elev  # the DEM's no-data value (sea)
            imp = importance(fclass, fcode, pop)
            region = admin1.get((cc, a1), ("", ""))[0] if a1 and a1 != "00" else ""
            country = countries.get(cc, ("", "", ""))[1]
            # Countries go by their short name ("France", not GeoNames' "Republic of France").
            name = country if fcode in COUNTRY_CODES and country else f[1]
            places.append((gid, name, lat, lon, fclass, fcode, cc, a1, region, country, pop, elev, f[17],
                           cell_of(lat, lon), imp))
            keys: dict[str, int] = {}
            for kind, text in ((0, name), (0, f[1]), (1, f[2])):
                k = norm(text)
                if k:
                    keys.setdefault(k, kind)
            for alt in f[3].split(","):
                k = norm(alt) if alt and len(alt) <= 120 else ""
                if k and not k.startswith("http"):
                    keys.setdefault(k, 2)
            names.extend((k, gid, kind, imp) for k, kind in keys.items())
            if fcode in COUNTRY_CODES and cc:
                aliases.update((k, cc, "") for k in keys)
            elif fcode == "ADM1" and cc and a1:
                aliases.update((k, cc, a1) for k in keys)
            n_places += 1
            n_names += len(keys)
            if len(places) >= 50_000:
                flush()
            if n_rows % 1_000_000 == 0:
                print(f"  {n_rows:,} rows read, {n_places:,} places kept ({time.time() - started:.0f} s)", flush=True)
    flush()
    print(f"sorting {n_names:,} names", flush=True)
    con.execute("INSERT OR IGNORE INTO name SELECT key, place, kind, importance FROM stage.name ORDER BY key, place")
    con.commit()
    con.execute("DETACH stage")
    stage.unlink(missing_ok=True)
    con.executemany("INSERT INTO alias VALUES (?, ?, ?)", sorted(aliases))
    con.execute("CREATE INDEX place_cell ON place (cell)")
    con.execute("CREATE INDEX alias_key ON alias (key)")
    built = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = {"places": n_places, "names": n_names, "rows_read": n_rows}
    meta = {"format": FORMAT, "source": source or ",".join(d.name for d in dumps), "built_at": built,
            "attribution": ATTRIBUTION, **{k: str(v) for k, v in counts.items()}}
    con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    con.commit()
    con.execute("ANALYZE")
    con.close()
    os.replace(tmp, out)
    print(f"built {out} in {time.time() - started:.0f} s: {n_places:,} places, {n_names:,} names", flush=True)
    return counts


def _cli(argv: list[str] | None = None) -> int:
    from .config import get_settings

    ap = argparse.ArgumentParser(prog="python -m forecast.places", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download GeoNames data and build the index")
    b.add_argument("--dump", action="append", help="GeoNames dump: cities500 (default), cities1000, allCountries, "
                                                   "or a country code such as TL. Repeat to combine.")
    b.add_argument("--from", dest="src", type=Path, help="use files already in this directory instead of downloading")
    b.add_argument("--out", type=Path, help="index file (default: FORECAST_PLACES_DB or data/places/places.sqlite)")
    s = sub.add_parser("search", help="look up a place name")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    n = sub.add_parser("nearest", help="nearest towns and villages to a point")
    n.add_argument("lat", type=float)
    n.add_argument("lon", type=float)
    n.add_argument("--limit", type=int, default=5)
    args = ap.parse_args(argv)

    out = args.out if getattr(args, "out", None) else get_settings().places_db
    if args.cmd == "build":
        dumps = args.dump or ["cities500"]
        out.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="geonames-", dir=out.parent))
        try:
            def fetch(name: str) -> Path:
                if args.src:
                    for cand in (args.src / name, args.src / name.replace(".zip", ".txt")):
                        if cand.exists():
                            return cand
                    raise SystemExit(f"{name} not found in {args.src}")
                return _download(name, work)

            files = [fetch(f"{d}.zip") for d in dumps]
            build(files, fetch("countryInfo.txt"), fetch("admin1CodesASCII.txt"), out, source=",".join(dumps))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return 0
    index = PlaceIndex(out)
    results = index.search(args.query, args.limit) if args.cmd == "search" else index.nearest(args.lat, args.lon, args.limit)
    for p in results:
        extra = f"{p.distance_km} km" if p.distance_km is not None else p.match
        print(f"{p.display:<50} {p.kind:<16} pop {p.population:>10,}  {p.lat:9.4f} {p.lon:9.4f}  {extra}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(_cli())
