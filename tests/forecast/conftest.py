from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from forecast.catalog import GFS
from forecast.grid import Grid
from forecast.store import RunWriter

COARSE = Grid(lat0=90.0, dlat=2.5, nlat=73, lon0=0.0, dlon=2.5, nlon=144)
HOURS = list(range(0, 25)) + [27, 30, 33, 36]


def synthetic_value(var_id: str, fhr: int, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray | None:
    shape = lat2d.shape
    if var_id == "t2m":
        return (0.5 * lat2d + 0.1 * fhr).astype(np.float32)
    if var_id == "u10":
        return np.full(shape, 3.0, np.float32)
    if var_id == "v10":
        return np.full(shape, 4.0, np.float32)
    if var_id == "precip":
        return None if fhr == 0 else np.full(shape, 1.0, np.float32)
    if var_id == "dswrf":
        return None if fhr == 0 else np.full(shape, 200.0, np.float32)
    if var_id == "mslp":
        return np.full(shape, 1013.2, np.float32)
    if var_id.startswith(("wave_", "wind_wave_", "swell_")):
        # Sea state: "land" east of 180° is missing. Directions alternate 350°/10° by column,
        # so a point between two columns tests that directions are blended as angles.
        if var_id.endswith("_dir"):
            col = np.rint(lon2d / 2.5).astype(int) if lon2d.size else lon2d
            vals = np.where(col % 2 == 0, 350.0, 10.0)
        elif var_id.endswith("_period"):
            vals = np.full(shape, 9.0)
        else:
            vals = np.full(shape, 2.0 + 0.01 * fhr)
        return np.where(lon2d >= 180.0, np.nan, vals).astype(np.float32)
    return np.full(shape, 50.0, np.float32)


def write_run(root, init: datetime, hours=HOURS, grid=COARSE, keep_runs=None):
    lat2d, lon2d = np.meshgrid(grid.lat0 - np.arange(grid.nlat) * grid.dlat,
                               grid.lon0 + np.arange(grid.nlon) * grid.dlon, indexing="ij")
    writer = RunWriter(root / "gfs", "gfs", init, grid, hours, GFS.stored_variables(), source={"provider": "test"})
    for k, h in enumerate(hours):
        for v in GFS.stored_variables():
            writer.write(v.id, k, synthetic_value(v.id, h, lat2d, lon2d))
    writer.write_static("orog", (lon2d * 2.0).astype(np.float32), {"name": "orog", "units": "m"})
    writer.write_static("land", np.ones(grid.shape, np.float32), {"name": "land", "units": "1"})
    writer.publish(keep_runs=keep_runs)
    return writer.id


@pytest.fixture
def run_init() -> datetime:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now - timedelta(hours=2)


@pytest.fixture
def store_root(tmp_path, run_init):
    write_run(tmp_path, run_init)
    return tmp_path


# --------------------------------------------------------------------------- places (GeoNames sample)

def _geo(gid, name, ascii_, alts, lat, lon, fclass, fcode, cc, a1, pop, tz):
    return "\t".join([str(gid), name, ascii_, alts, str(lat), str(lon), fclass, fcode, cc, "", a1, "", "", "",
                      str(pop), "", "10", tz, "2026-01-01"])


GEONAMES_SAMPLE = [
    _geo(2988507, "Paris", "Paris", "Lutece,Lutetia,Paname,Parigi,París,Париж", 48.85341, 2.3488, "P", "PPLC", "FR", "11", 2138551, "Europe/Paris"),
    _geo(4717560, "Paris", "Paris", "", 33.66094, -95.55551, "P", "PPLA2", "US", "TX", 24171, "America/Chicago"),
    _geo(4647963, "Paris", "Paris", "", 36.302, -88.32671, "P", "PPLA2", "US", "TN", 10156, "America/Chicago"),
    _geo(2988000, "Parisot", "Parisot", "", 44.26, 1.85, "P", "PPL", "FR", "76", 520, "Europe/Paris"),
    _geo(9000001, "Paris Creek", "Paris Creek", "", 40.0, -90.0, "H", "STM", "US", "IL", 0, "America/Chicago"),
    _geo(3012874, "Île-de-France", "Ile-de-France", "Paris Region", 48.5, 2.5, "A", "ADM1", "FR", "11", 12000000, "Europe/Paris"),
    _geo(3017382, "Republic of France", "Republic of France", "France,Frankreich,Francia", 46.0, 2.0, "A", "PCLI", "FR", "00", 66987244, "Europe/Paris"),
    _geo(2968815, "Paris", "Paris", "", 48.8592, 2.3417, "A", "ADM2", "FR", "11", 2138551, "Europe/Paris"),
    _geo(1645456, "Díli", "Dili", "Dili Municipality", -8.58, 125.6, "A", "ADM1", "TL", "DI", 277279, "Asia/Dili"),
    _geo(1273293, "National Capital Territory of Delhi", "National Capital Territory of Delhi", "Delhi,Dili", 28.67, 77.22, "A", "ADM1", "IN", "07", 16787941, "Asia/Kolkata"),
    _geo(8505031, "Mediterranean Region", "Mediterranean Region", "", 37.0, 33.0, "L", "RGN", "TR", "", 10000000, "Europe/Istanbul"),
    _geo(2267057, "Lisbon", "Lisbon", "Lisboa,Lisbonne,Lissabon", 38.71667, -9.13333, "P", "PPLC", "PT", "14", 517802, "Europe/Lisbon"),
    _geo(5160783, "Lisbon", "Lisbon", "", 40.77201, -80.76815, "P", "PPL", "US", "OH", 2717, "America/New_York"),
    _geo(1645457, "Dili", "Dili", "Díli,Dílli", -8.55861, 125.57361, "P", "PPLC", "TL", "DI", 150000, "Asia/Dili"),
    _geo(9000002, "Becora", "Becora", "", -8.5617, 125.6011, "P", "PPL", "TL", "DI", 0, "Asia/Dili"),
    _geo(1636308, "Same", "Same", "", -9.00361, 125.64944, "P", "PPLA", "TL", "MF", 7000, "Asia/Dili"),
    _geo(149996, "Same", "Same", "", -4.0667, 37.7333, "P", "PPL", "TZ", "26", 20000, "Africa/Dar_es_Salaam"),
    _geo(3448439, "São Paulo", "Sao Paulo", "Sampa", -23.5475, -46.63611, "P", "PPLA", "BR", "27", 10021295, "America/Sao_Paulo"),
    _geo(363196, "Mediterranean Sea", "Mediterranean Sea", "Mar Mediterraneo,Mittelmeer", 35.0, 18.0, "H", "SEA", "", "", 0, ""),
    _geo(1273294, "Delhi", "Delhi", "Dili,Dehli", 28.65195, 77.23149, "P", "PPLA", "IN", "07", 11034555, "Asia/Kolkata"),
    _geo(4140963, "Washington", "Washington", "Washington D.C.", 38.89511, -77.03637, "P", "PPLC", "US", "DC", 689545, "America/New_York"),
    _geo(170794, "Ar Raqqah", "Ar Raqqah", "Nicephorium,Raqqa", 35.95, 39.01, "P", "PPLA", "SY", "04", 531952, "Asia/Damascus"),
    _geo(2990440, "Nice", "Nice", "Nizza", 43.70313, 7.26608, "P", "PPLA2", "FR", "93", 342669, "Europe/Paris"),
    _geo(9000003, "Old Dili", "Old Dili", "", -8.55, 125.57, "P", "PPLH", "TL", "DI", 0, "Asia/Dili"),
]
COUNTRY_INFO = ["#ISO\tISO3\tISO-Numeric\tfips\tCountry"] + [
    "\t".join([iso, iso3, "0", iso, name] + [""] * 14) for iso, iso3, name in (
        ("FR", "FRA", "France"), ("US", "USA", "United States"), ("PT", "PRT", "Portugal"),
        ("TL", "TLS", "Timor Leste"), ("TZ", "TZA", "Tanzania"), ("BR", "BRA", "Brazil"), ("IN", "IND", "India"),
        ("SY", "SYR", "Syria"))]
ADMIN1 = ["FR.11\tÎle-de-France\tIle-de-France\t3012874", "FR.76\tOccitanie\tOccitanie\t11071623",
          "US.TX\tTexas\tTexas\t4736286", "US.TN\tTennessee\tTennessee\t4662168", "US.OH\tOhio\tOhio\t5165418",
          "US.IL\tIllinois\tIllinois\t4896861", "PT.14\tLisbon\tLisbon\t2267056", "TL.DI\tDili\tDili\t1645456",
          "TL.MF\tManufahi\tManufahi\t1636309", "TZ.26\tKilimanjaro\tKilimanjaro\t153209",
          "BR.27\tSão Paulo\tSao Paulo\t3448433", "IN.07\tDelhi\tDelhi\t1273293",
          "US.DC\tWashington, D.C.\tWashington, D.C.\t4138106", "SY.04\tRaqqa\tRaqqa\t163842",
          "FR.93\tProvence-Alpes-Côte d'Azur\tProvence-Alpes-Cote d'Azur\t2985244"]


@pytest.fixture(scope="session")
def places_db(tmp_path_factory):
    from forecast.places import build

    src = tmp_path_factory.mktemp("geonames")
    for name, lines in (("sample.txt", GEONAMES_SAMPLE), ("countryInfo.txt", COUNTRY_INFO), ("admin1.txt", ADMIN1)):
        (src / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = src / "places.sqlite"
    build([src / "sample.txt"], src / "countryInfo.txt", src / "admin1.txt", out, source="test sample")
    return out
