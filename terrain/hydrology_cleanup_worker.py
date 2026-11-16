# hydrology_cleanup_worker.py
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import redis

from terrain.utils.redis_notifications import ensure_expired_events_enabled

last_sweep: float = 0.0
SWEEP_EVERY = 60
MAX_AGE = 3 * 3600  # match your TTL

PREFIX = "hydro"
REDIS_URL = "redis://localhost:6379/0"
JOBS_DIR = Path("data/hydrology_jobs")

r = redis.from_url(REDIS_URL, decode_responses=True)

# Subscribe to expired events
p = r.pubsub()


print("Cleanup worker listening for Redis expiry events...")

ok = ensure_expired_events_enabled(r)
if not ok:
    print(
        "[cleanup] WARNING: Redis expiry events not enabled; consider fallback cleanup."
    )


def sweep_old_job_dirs(jobs_dir: Path, max_age_seconds: int) -> None:
    now = time.time()
    for job_path in jobs_dir.iterdir():
        if not job_path.is_dir():
            continue
        meta_path = job_path / "job.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        updated_at = float(meta.get("updated_at") or meta.get("created_at") or 0)
        age = now - updated_at
        if age > max_age_seconds:
            print(f"[cleanup-sweep] Removing stale job dir {job_path} age={age:.0f}s")
            shutil.rmtree(job_path, ignore_errors=True)


def redis_db_index_from_url(redis_url: str) -> int:
    pool = redis.ConnectionPool.from_url(redis_url)
    # redis-py stores db in connection_kwargs; default is 0 if missing
    return int(pool.connection_kwargs.get("db", 0))


db = redis_db_index_from_url(REDIS_URL)
pattern = f"__keyevent@{db}__:expired"
p.psubscribe(pattern)


# def check_sweep():
#     now = time.time()
#     if now - last_sweep < SWEEP_EVERY:
#         return
#     sweep_old_job_dirs(JOBS_DIR, MAX_AGE)
#     last_sweep = now


for message in p.listen():
    # print("[cleanup] pubsub:", message)
    if message["type"] != "pmessage":
        continue

    expired_key = message["data"]
    # we only care about hydro:job:<id>
    if not expired_key.startswith(f"{PREFIX}:job:"):
        continue
    print(f"FOUND AND EXPIRED KEY {expired_key}")

    job_id = expired_key.split(":")[-1]
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):  # only ever a uuid4 hex id; never a path
        continue
    job_path = JOBS_DIR / job_id

    if job_path.exists():
        print(f"[cleanup] Removing expired job directory: {job_path}")
        shutil.rmtree(job_path, ignore_errors=True)
