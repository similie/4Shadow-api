"""Ingest GFS runs into the array store.

    python -m forecast.ingest                    # ingest the newest complete run if it's new
    python -m forecast.ingest --loop             # ...and keep doing so (systemd service)
    python -m forecast.ingest --run 2026111200 --max-fhr 48   # a specific run / dev slice

One process, one run at a time (guarded by a lock file). Downloads are prefetched in a
small thread pool; decoding happens sequentially in step order because ecCodes is not
thread-safe and because de-accumulating precipitation needs the previous step anyway.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .catalog import GFS, ModelCatalog, Variable
from .config import Settings, get_settings
from .sources import gfs
from .sources.gfs import FcstSpec, GfsClient, IdxRecord
from .store import RunWriter, parse_run_id, prune_runs, read_latest, run_id

log = logging.getLogger("forecast.ingest")


class StepUnavailable(RuntimeError):
    pass


@dataclass
class StepPayload:
    fhr: int
    messages: dict[str, tuple[FcstSpec, bytes]] = field(default_factory=dict)
    static: dict[str, bytes] = field(default_factory=dict)


# --------------------------------------------------------------------------- interval maths


class IntervalState:
    """Turns GRIB running totals / bucket averages into per-interval mean rates."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.prev_fhr = 0
        self.prev_total = np.zeros(shape, dtype=np.float32)  # accumulation at f000 is zero
        self.prev_bucket: tuple[int, int, np.ndarray] | None = None

    def accum(self, fhr: int, total: np.ndarray) -> np.ndarray:
        """Mean rate per hour over (prev_fhr, fhr] from a total-since-init accumulation."""
        rate = (total - self.prev_total) / np.float32(fhr - self.prev_fhr)
        self.prev_fhr, self.prev_total = fhr, total
        return rate

    def average(self, fhr: int, spec: FcstSpec, avg: np.ndarray) -> np.ndarray:
        """Mean over (prev_fhr, fhr] from a bucket average over (spec.start, fhr]."""
        start = spec.start or 0
        prev = self.prev_bucket
        if prev is not None and prev[0] == start and prev[1] < fhr:
            p_start, p_end, p_avg = prev
            out = ((fhr - start) * avg - (p_end - p_start) * p_avg) / np.float32(fhr - p_end)
        else:
            if start != self.prev_fhr:
                log.warning("Average bucket %s-%s does not follow step %s; using bucket mean", start, fhr, self.prev_fhr)
            out = avg
        self.prev_fhr, self.prev_bucket = fhr, (start, fhr, avg)
        return out


# --------------------------------------------------------------------------- download


def download_step(client: GfsClient, catalog: ModelCatalog, run: datetime, fhr: int,
                  include_static: bool, wait_seconds: int = 0) -> StepPayload:
    deadline = time.monotonic() + wait_seconds
    while True:
        records = client.idx(run, fhr)
        if records is not None:
            break
        if time.monotonic() >= deadline:
            raise StepUnavailable(f"{run_id(run)} f{fhr:03d}: index not published")
        time.sleep(30)

    wanted: dict[str, tuple[IdxRecord, FcstSpec]] = {}
    for var in catalog.stored_variables():
        hit = gfs.select_record(records, var.grib_param, var.grib_level, var.kind, fhr)
        if hit is None:
            if not (fhr == 0 and var.kind != "instant"):
                log.warning("%s f%03d: no %s:%s message", run_id(run), fhr, var.grib_param, var.grib_level)
            continue
        wanted[var.id] = hit
    static_wanted: dict[str, IdxRecord] = {}
    if include_static:
        for sf in catalog.static.values():
            hit = gfs.select_record(records, sf.grib_param, sf.grib_level, "instant", fhr)
            if hit is not None:
                static_wanted[sf.id] = hit[0]

    blobs = client.fetch_records(run, fhr, [r for r, _ in wanted.values()] + list(static_wanted.values()))
    return StepPayload(
        fhr=fhr,
        messages={vid: (spec, blobs[rec.num]) for vid, (rec, spec) in wanted.items()},
        static={sid: blobs[rec.num] for sid, rec in static_wanted.items()},
    )


# --------------------------------------------------------------------------- processing


def _convert(var: Variable, values: np.ndarray) -> np.ndarray:
    if var.mul != 1.0:
        values = values * np.float32(var.mul)
    if var.add != 0.0:
        values = values + np.float32(var.add)
    return values


def _clip(var: Variable, values: np.ndarray) -> np.ndarray:
    lo, hi = var.clip
    if lo is not None or hi is not None:
        values = np.clip(values, lo, hi)  # NaN-preserving
    return values


def process_step(writer: RunWriter, catalog: ModelCatalog, step: int, payload: StepPayload,
                 states: dict[str, IntervalState]) -> None:
    for var in catalog.stored_variables():
        item = payload.messages.get(var.id)
        if item is None:
            writer.write(var.id, step, None)
            continue
        spec, blob = item
        values = _convert(var, gfs.decode_message(blob, writer.grid))
        if var.kind == "accum":
            values = states[var.id].accum(payload.fhr, values)
        elif var.kind == "average":
            values = states[var.id].average(payload.fhr, spec, values)
        writer.write(var.id, step, _clip(var, values))
    for sid, blob in payload.static.items():
        sf = catalog.static[sid]
        writer.write_static(sid, gfs.decode_message(blob, writer.grid), {"name": sf.name, "units": sf.units})


def ingest_run(run: datetime, settings: Settings, client: GfsClient | None = None,
               catalog: ModelCatalog = GFS, max_fhr: int | None = None) -> str:
    client = client or GfsClient(settings.gfs_base_url)
    hours = gfs.forecast_hours(settings.max_fhr if max_fhr is None else max_fhr)
    model_root = settings.data_root / catalog.model
    stored = catalog.stored_variables()
    writer = RunWriter(model_root, catalog.model, run, gfs.GRID, hours, stored,
                       source=gfs.source_info(client.base_url, run))
    states = {v.id: IntervalState(gfs.GRID.shape) for v in stored if v.interval}
    log.info("Ingesting %s run %s: %d steps x %d variables", catalog.model, writer.id, len(hours), len(stored))

    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=max(1, settings.download_workers), thread_name_prefix="gfs-dl")
    try:
        futures: dict[int, Future] = {}
        depth = max(2, settings.download_workers * 2)

        def submit(k: int) -> None:
            futures[k] = pool.submit(download_step, client, catalog, run, hours[k], k == 0, settings.step_wait_seconds)

        for k in range(min(depth, len(hours))):
            submit(k)
        next_k = len(futures)
        decode_s = 0.0
        for k, fhr in enumerate(hours):
            payload = futures.pop(k).result()
            if next_k < len(hours):
                submit(next_k)
                next_k += 1
            td = time.monotonic()
            process_step(writer, catalog, k, payload, states)
            decode_s += time.monotonic() - td
            if k % 24 == 0 or k == len(hours) - 1:
                log.info("  f%03d done (%d/%d, %.0fs elapsed, %.0fs decoding)", fhr, k + 1, len(hours),
                         time.monotonic() - t0, decode_s)
        pool.shutdown(wait=True)
        writer.publish(keep_runs=settings.keep_runs, extra={"ingest_seconds": round(time.monotonic() - t0, 1)})
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        writer.abort()
        raise
    log.info("Run %s ingested in %.0fs", writer.id, time.monotonic() - t0)
    return writer.id


def ingest_latest(settings: Settings, client: GfsClient | None = None, catalog: ModelCatalog = GFS,
                  max_fhr: int | None = None, force: bool = False) -> str | None:
    client = client or GfsClient(settings.gfs_base_url)
    max_fhr = settings.max_fhr if max_fhr is None else max_fhr
    candidate = client.latest_complete_run(max_fhr=max_fhr)
    if candidate is None:
        log.warning("No complete %s run found on %s", catalog.model, client.base_url)
        return None
    rid = run_id(candidate)
    current = read_latest(settings.data_root / catalog.model)
    if current is not None and rid <= current and not force:
        log.info("Up to date (latest published run %s)", current)
        return None
    return ingest_run(candidate, settings, client, catalog, max_fhr)


@contextmanager
def ingest_lock(model_root: Path):
    model_root.mkdir(parents=True, exist_ok=True)
    with open(model_root / ".ingest.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest GFS runs into the forecast store.")
    parser.add_argument("--run", help="ingest this run (YYYYMMDDHH) instead of the newest complete one")
    parser.add_argument("--max-fhr", type=int, help="last forecast hour to ingest (default: FORECAST_MAX_FHR or 384)")
    parser.add_argument("--loop", action="store_true", help="keep polling for new runs")
    parser.add_argument("--poll", type=int, default=600, help="seconds between polls with --loop")
    parser.add_argument("--force", action="store_true", help="re-ingest even if already published")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    model_root = settings.data_root / GFS.model
    with ingest_lock(model_root) as acquired:
        if not acquired:
            log.warning("Another ingest is running for %s; exiting", model_root)
            return 0
        if args.run:
            ingest_run(parse_run_id(args.run), settings, max_fhr=args.max_fhr)
            return 0
        while True:
            try:
                ingest_latest(settings, max_fhr=args.max_fhr, force=args.force)
                prune_runs(model_root, settings.keep_runs)
            except Exception:
                log.exception("Ingest cycle failed")
                if not args.loop:
                    return 1
            if not args.loop:
                return 0
            time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
