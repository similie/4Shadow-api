"""CAMS: air quality from the Copernicus Atmosphere Monitoring Service's global forecast.

The product is "CAMS global atmospheric composition forecasts" (ECMWF's IFS model with atmospheric
chemistry, about 40 km; https://doi.org/10.24381/04a0b097): two runs a day, from 00 and 12 UTC, hourly to
120 hours, on a 0.4° grid, which is stored as it is. We read nitrogen dioxide, carbon monoxide, ozone and
sulphur dioxide at the model's lowest level (about 10 m above the ground), surface PM2.5 and PM10, and the
temperature and pressure there, which turn the gases' mass mixing ratios into µg/m³. From them the ingest
works out the US EPA's Air Quality Index hour by hour (``forecast.aqi``).

It comes from the Atmosphere Data Store (ADS), which needs a free account: CDSAPI_KEY (the ADS's API
token) and, for another server than the ADS, CDSAPI_URL; the dataset's licence must be accepted once on its
page. The ADS prepares each request in a queue, which takes from under a minute to an hour or more. So the
ingest asks for a run on one poll (``prepare`` raises ``RunPending``) and the other models go on; a later
poll finds it ready and downloads it (about 800 MB, every hour of every field), and the file is deleted
once its last hour is read. The request's id is kept in the model's ``.ads`` folder, so a restart picks it
up. One request at a time, checked against the dataset's constraints first, and a failed one is asked
again only after a wait (``RETRY_HOURS``): the ADS's rules for automated clients.

The data are under CC BY 4.0, with ECMWF's notice "Contains modified Copernicus Atmosphere Monitoring
Service information" and its disclaimer.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from ..aqi import HourlyAqi
from ..grid import Grid
from .base import ArraySource, RunPending, StepUnavailable
from .gfs import decode_message

log = logging.getLogger(__name__)

MODEL = "air"
DATASET = "cams-global-atmospheric-composition-forecasts"
DOI_URL = "https://doi.org/10.24381/04a0b097"
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
MAX_FHR = 120
LEVEL = "137"  # the lowest of the model's 137 levels
# The ADS's names for what we ask for, and their short names in the GRIB files it sends.
FIELDS = {"nitrogen_dioxide": "no2", "carbon_monoxide": "co", "ozone": "go3", "sulphur_dioxide": "so2",
          "temperature": "t", "particulate_matter_2.5um": "pm2p5", "particulate_matter_10um": "pm10",
          "surface_pressure": "sp"}
GASES = ("no2", "co", "go3", "so2")  # mass mixing ratios (kg/kg) at the lowest level
AQI = "aqi"  # what the client calls the index it works out
R_DRY = 287.05  # J/(kg K), the gas constant of dry air
# A run is out about ten hours after it starts: younger ones aren't looked for, sparing the ADS a call.
FIRST_LOOK_HOURS = 6
# After a failed request, the next waits an hour, then two; after three, the run is left for the next one.
# The ADS disables accounts that keep sending requests it can't serve.
RETRY_HOURS = (1, 2)

GRID = Grid(lat0=90.0, dlat=0.4, nlat=451, lon0=0.0, dlon=0.4, nlon=900)


def forecast_hours(max_fhr: int = MAX_FHR) -> list[int]:
    """Every hour to 120 h."""
    if not 0 <= max_fhr <= MAX_FHR:
        raise ValueError(f"max_fhr must be within 0..{MAX_FHR}")
    return list(range(max_fhr + 1))


def request(run: datetime, hours: list[int]) -> dict:
    """The ADS request for a run's hours: every field, the whole globe, as GRIB."""
    day = f"{run:%Y-%m-%d}"
    return {"variable": list(FIELDS), "model_level": [LEVEL], "date": [f"{day}/{day}"], "time": [f"{run:%H}:00"],
            "leadtime_hour": [str(h) for h in hours], "type": ["forecast"], "data_format": "grib"}


def grib_messages(path: Path) -> list[tuple[int, int]]:
    """The (offset, length) of each GRIB message in a file, from the lengths in their headers. CAMS's files
    mix editions: the gases are GRIB2, the surface fields GRIB1."""
    found = []
    with open(path, "rb") as fh:
        offset = 0
        while head := fh.read(16):
            if len(head) < 16 or head[:4] != b"GRIB" or head[7] not in (1, 2):
                raise ValueError(f"{path.name}: no GRIB message at byte {offset}")
            if head[7] == 2:
                length = int.from_bytes(head[8:16], "big")
            else:
                length = int.from_bytes(head[4:7], "big")
                if length & 0x800000:  # GRIB1's coding for messages over 8 MB: not in these files
                    raise ValueError(f"{path.name}: a GRIB1 message over 8 MB at byte {offset}")
            found.append((offset, length))
            offset += length
            fh.seek(offset)
    return found


def _utc(when: datetime) -> datetime:
    return when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)


class AtmosphereClient:
    """Asks the ADS for a run, downloads it when it is ready, and reads it an hour at a time.

    ``root`` is the model's folder in the store (its ``.ads`` subfolder holds requests and downloads);
    ``ads`` an ``ecmwf.datastores.Client`` (made from CDSAPI_KEY when not given); ``now`` the clock."""

    def __init__(self, root: Path | None = None, ads=None, now=None) -> None:
        if root is None:
            from ..config import get_settings

            root = get_settings().data_root / MODEL
        self.work = Path(root) / ".ads"
        self._ads = ads
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._run: datetime | None = None
        self._hours: list[int] = []
        self._index: dict[tuple[str, int], tuple[int, int]] = {}
        self._pressure = 1.0  # the lowest level's pressure as a share of the surface's
        self._aqi: HourlyAqi | None = None
        self._next: int | None = None

    def _client(self):
        if self._ads is None:
            key = os.getenv("CDSAPI_KEY")
            if not key:
                raise RuntimeError("No Atmosphere Data Store login: set CDSAPI_KEY (a free account at "
                                   "https://ads.atmosphere.copernicus.eu, then the API token on its profile page)")
            from ecmwf.datastores import Client

            # Its INFO lines repeat the service's notices on every call; ours say what happens.
            logging.getLogger("ecmwf.datastores").setLevel(logging.WARNING)
            url = (os.getenv("CDSAPI_URL") or ADS_URL).rstrip("/").removesuffix("/api") + "/api"
            # Few tries: a server that's down is tried again on the next poll, not held on to for hours.
            self._ads = Client(url=url, key=key, progress=False, maximum_tries=3, retry_after=20)
        return self._ads

    # ------------------------------------------------------------------ which run, and getting it

    def published(self, run: datetime, max_fhr: int = MAX_FHR) -> bool:
        """Whether the ADS has the run's hours to ``max_fhr``, by its dataset's constraints."""
        when = f"{run:%H}:00"
        found = self._client().apply_constraints(DATASET, {
            "variable": [next(iter(FIELDS))], "model_level": [LEVEL], "type": ["forecast"],
            "date": [f"{run:%Y-%m-%d}/{run:%Y-%m-%d}"], "time": [when]})
        return when in found.get("time", []) and str(max_fhr) in found.get("leadtime_hour", [])

    def latest_run(self, max_fhr: int = MAX_FHR) -> datetime | None:
        """The newest run of the last two days the ADS has."""
        now = _utc(self._now())
        cycle = now.replace(hour=12 if now.hour >= 12 else 0, minute=0, second=0, microsecond=0)
        for k in range(4):
            run = cycle - timedelta(hours=12 * k)
            if now - run >= timedelta(hours=FIRST_LOOK_HOURS) and self.published(run, max_fhr):
                return run
        return None

    def _paths(self, run: datetime) -> tuple[Path, Path]:
        stem = f"{run:%Y%m%d%H}"
        return self.work / f"{stem}.grib", self.work / f"{stem}.json"

    def prepare(self, run: datetime, hours: list[int]) -> None:
        """Make sure the run's file is here: download it if the ADS has it ready, else ask for it (once) and
        raise ``RunPending`` while the ADS works on it."""
        self.work.mkdir(parents=True, exist_ok=True)
        grib, state_path = self._paths(run)
        for old in self.work.iterdir():  # other runs' requests and downloads (an older run, a dev slice)
            if old not in (grib, state_path):
                old.unlink(missing_ok=True)
        if grib.exists():
            self._open(run, hours, grib, json.loads(state_path.read_text()) if state_path.exists() else {})
            return
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("request_id") and not set(hours) <= set(state["hours"]):
            state = {}  # asked for fewer hours than now wanted
        rid, now = f"{run:%Y%m%d%H}", _utc(self._now())
        if not state.get("request_id"):
            failures = state.get("failures", 0)
            if failures > len(RETRY_HOURS):
                raise RunPending(f"the ADS failed CAMS run {rid} {failures} times: waiting for the next run")
            if failures:
                again = datetime.fromisoformat(state["failed"]) + timedelta(hours=RETRY_HOURS[failures - 1])
                if now < again:
                    raise RunPending(f"the ADS failed CAMS run {rid}; asking again after {again:%H:%M} UTC")
            remote = self._client().submit(DATASET, request(run, hours))
            state = {"run": rid, "hours": hours, "request_id": remote.request_id, "asked": now.isoformat(),
                     "failures": failures}
            state_path.write_text(json.dumps(state))
            raise RunPending(f"asked the ADS for CAMS run {rid} (request {remote.request_id})")
        from ecmwf.datastores.processing import ProcessingFailedError

        remote = self._client().get_remote(state["request_id"])
        try:
            ready = remote.results_ready
        except ProcessingFailedError as exc:
            failures = self._failed(run, state)
            log.warning("The ADS failed CAMS run %s (request %s, failure %d): %s", rid, state["request_id"],
                        failures, str(exc).strip() or type(exc).__name__)
            raise RunPending(f"the ADS failed CAMS run {rid} (request {state['request_id']})") from None
        if not ready:
            asked = datetime.fromisoformat(state["asked"])
            raise RunPending(f"the ADS is preparing CAMS run {rid} (request {state['request_id']}, asked "
                             f"{asked:%Y-%m-%d %H:%M} UTC)")
        part = grib.with_suffix(".part")
        try:
            remote.download(str(part))
        except requests.RequestException as exc:
            log.warning("CAMS run %s: the download broke off (%s); trying again on the next poll", rid, exc)
            raise RunPending(f"downloading CAMS run {rid} again on the next poll") from None
        os.replace(part, grib)
        try:
            remote.delete()  # done with: off the account's list of requests
        except Exception as exc:  # noqa: BLE001 - only tidying
            log.info("CAMS request %s not deleted from the ADS: %s", state["request_id"], exc)
        self._open(run, hours, grib, state)

    def _failed(self, run: datetime, state: dict) -> int:
        """Note a failed request (the next waits); returns how many the run has had."""
        failures = state.get("failures", 0) + 1
        self._paths(run)[1].write_text(json.dumps({"run": f"{run:%Y%m%d%H}", "failures": failures,
                                                   "failed": _utc(self._now()).isoformat()}))
        return failures

    def _open(self, run: datetime, hours: list[int], grib: Path, state: dict | None = None) -> None:
        """Index the run's file: where each field's message is, by short name and hour."""
        import pygrib  # deferred: only the ingest needs ecCodes

        index: dict[tuple[str, int], tuple[int, int]] = {}
        pressure = None
        with open(grib, "rb") as fh:
            for offset, length in grib_messages(grib):
                fh.seek(offset)
                msg = pygrib.fromstring(fh.read(length))
                index[(msg.shortName, int(msg.step))] = (offset, length)
                if pressure is None and msg.typeOfLevel == "hybrid":
                    # The level's pressure is the mean of its top and bottom (half levels), each a + b x the
                    # surface pressure; the lowest level's a are 0.
                    pv = np.asarray(msg["pv"], np.float64)
                    a, b = pv[: len(pv) // 2], pv[len(pv) // 2 :]
                    if abs(a[-1]) > 1e-6 or abs(a[-2]) > 1e-6:
                        raise ValueError(f"{grib.name}: the lowest level isn't a fixed share of the surface pressure")
                    pressure = float((b[-1] + b[-2]) / 2)
        missing = [(name, h) for h in hours for name in FIELDS.values() if (name, h) not in index]
        if missing:
            grib.unlink(missing_ok=True)
            self._failed(run, state or {})
            raise RuntimeError(f"CAMS run {run:%Y%m%d%H}: the download lacks {len(missing)} fields "
                               f"(first {missing[0]}); asking again later")
        self._run, self._hours, self._index, self._pressure = run, list(hours), index, pressure or 1.0
        self._aqi, self._next = None, None

    # ------------------------------------------------------------------ reading an hour

    def _field(self, grib: Path, name: str, fhr: int, grid: Grid) -> np.ndarray:
        offset, length = self._index[(name, fhr)]
        with open(grib, "rb") as fh:
            fh.seek(offset)
            return decode_message(fh.read(length), grid).astype(np.float64)

    def step(self, run: datetime, fhr: int, names: list[str], grid: Grid) -> dict[str, np.ndarray]:
        """An hour's concentrations in µg/m³ and its AQI. The hours must come in order from the first, for
        the AQI's averages."""
        if run != self._run:
            raise StepUnavailable(f"CAMS run {run:%Y%m%d%H} isn't prepared")
        if fhr == self._hours[0]:
            self._aqi, self._next = HourlyAqi(), fhr
        if fhr != self._next:
            raise RuntimeError(f"CAMS hours out of order: {fhr} after {self._next}")
        grib = self._paths(run)[0]
        f = {name: self._field(grib, name, fhr, grid) for name in FIELDS.values()}
        density = f["sp"] * self._pressure / (R_DRY * f["t"])  # kg/m³ at the lowest level
        out = {gas: f[gas] * density * 1e9 for gas in GASES}
        out["pm2p5"], out["pm10"] = f["pm2p5"] * 1e9, f["pm10"] * 1e9
        out[AQI] = self._aqi.push(pm25=out["pm2p5"], pm10=out["pm10"], o3=f["go3"], co=f["co"], so2=f["so2"],
                                  no2=f["no2"])
        later = [h for h in self._hours if h > fhr]
        self._next = later[0] if later else None
        if not later:  # the last hour: the file has done its job
            grib.unlink(missing_ok=True)
            self._paths(run)[1].unlink(missing_ok=True)
        return {name: out[name].astype(np.float32) for name in names if name in out}


def attribution(run: datetime) -> str:
    return (f"Air quality: Contains modified Copernicus Atmosphere Monitoring Service information {run:%Y} (CAMS "
            f"global atmospheric composition forecasts, {DOI_URL}, CC BY 4.0); neither the European Commission nor "
            "ECMWF is responsible for any use made of it. 4Shadow turns the gases into µg/m³ and adds the US AQI.")


def source_info(base_url: str, run: datetime, products=()) -> dict:
    return {
        "provider": ("Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition forecast "
                     "(ECMWF IFS with chemistry, 0.4°)"),
        "distribution": "Copernicus Atmosphere Data Store (ADS)",
        "url": base_url,
        "dataset": DATASET,
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "attribution": attribution(run),
    }


SOURCE = ArraySource(MODEL, GRID, MAX_FHR, DOI_URL, forecast_hours, source_info, AtmosphereClient,
                     cycle_hours=(0, 12), gfs_static=False,
                     # Each run is out about ten hours after it starts; the next 12 hours after that.
                     stale_after_hours=28)
