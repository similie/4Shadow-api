# terrain/job_engine.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import redis

from terrain.utils.redis_notifications import ensure_expired_events_enabled

from .utils.station_normalization import StationDict, normalize_station_dict

_JOB_ID = re.compile(r"[0-9a-f]{32}")


@dataclass
class HydrologyJob:
    job_id: str
    status: str
    error: Optional[str]
    params: Dict[str, Any]


class HydrologyJobEngine:
    """
    Central hydrology job manager.

    Responsibilities:
      - Manage job directories on disk
      - Persist job.json for each job (ground truth metadata)
      - Mirror job status into Redis (hash) with TTL
      - Provide queue key + job key helpers

    Redis layout (db 0):

      hydro:queue             (LIST)  - job IDs waiting to be processed
      hydro:job:<job_id>      (HASH)  - status, error, params, timestamps

    Your existing cleanup module can listen to expired events on
    hydro:job:<job_id> and delete the corresponding job directory.
    """

    def __init__(
        self,
        base_hydrology_dir: str | Path = "data/hydrology_copernicus",
        jobs_dir: str | Path = "data/hydrology_jobs",
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "hydro",  # ensures hydro:queue, hydro:job:<id>
        job_ttl_seconds: int = 3 * 3600,  # 3h TTL by default
    ) -> None:
        self.base_hydrology_dir = Path(base_hydrology_dir)
        self.jobs_root = Path(jobs_dir)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.redis = redis.from_url(redis_url)
        self.prefix = prefix
        self.job_ttl_seconds = int(job_ttl_seconds)
        ensure_expired_events_enabled(self.redis)

    # ------------------------------
    # Redis key helpers
    # ------------------------------
    def _queue_key(self) -> str:
        # LIST of pending jobs
        return f"{self.prefix}:queue"  # → "hydro:queue"

    def _job_key(self, job_id: str) -> str:
        # HASH per-job
        return f"{self.prefix}:job:{job_id}"  # → "hydro:job:<id>"

    # ------------------------------
    # Disk paths
    # ------------------------------
    @staticmethod
    def valid_job_id(job_id: str) -> bool:
        """Job ids are uuid4 hex (see _new_job_id). Anything else could name a path outside jobs_root."""
        return bool(_JOB_ID.fullmatch(job_id))

    def _job_path(self, job_id: str) -> Path:
        if not self.valid_job_id(job_id):
            raise FileNotFoundError(f"No job {job_id!r}")
        root = os.path.normpath(self.jobs_root)
        path = os.path.normpath(os.path.join(root, job_id))
        if not path.startswith(root + os.sep):
            raise FileNotFoundError(f"No job {job_id!r}")
        return Path(path)

    # ------------------------------
    # Public API
    # ------------------------------
    def create_job(
        self,
        stations: list[StationDict],
        event_duration_hours: float,
        runoff_coeff: float,
        rain_radius_km: Optional[float],
    ) -> HydrologyJob:
        """
        Create a new hydrology job:

          - allocate job_id
          - write job.json with params (stations, etc.)
          - push job_id into Redis LIST hydro:queue
          - create Redis HASH hydro:job:<id> with status=pending
          - set TTL on the HASH for later cleanup
        """
        job_id = self._new_job_id()
        job_dir = self._job_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        params: Dict[str, Any] = {
            "stations": [normalize_station_dict(s) for s in stations],
            "event_duration_hours": float(event_duration_hours),
            "runoff_coeff": float(runoff_coeff),
            "rain_radius_km": float(rain_radius_km or 10.0),
        }

        meta = {
            "job_id": job_id,
            "status": "pending",
            "error": None,
            "params": params,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Write job.json to disk (source of truth)
        job_json_path = job_dir / "job.json"
        job_json_path.write_text(json.dumps(meta, indent=2))

        # Mirror metadata to Redis hash
        job_key = self._job_key(job_id)
        self.redis.hset(
            job_key,
            mapping={
                "status": "pending",
                "error": "",
                "params": json.dumps(params),
                "created_at": str(meta["created_at"]),
                "updated_at": str(meta["updated_at"]),
            },
        )
        # Set TTL so that your existing cleanup module
        # can catch the expired key event and delete job_dir.
        if self.job_ttl_seconds > 0:
            self.redis.expire(job_key, self.job_ttl_seconds)

        # Enqueue into hydro:queue
        self.redis.rpush(self._queue_key(), job_id)

        return HydrologyJob(
            job_id=job_id,
            status="pending",
            error=None,
            params=params,
        )

    def get_job(self, job_id: str) -> HydrologyJob:
        """
        Load job.json and return a HydrologyJob.
        (This keeps behaviour consistent with your existing code.)
        """
        job_dir = self._job_path(job_id)
        job_json_path = job_dir / "job.json"
        if not job_json_path.exists():
            raise FileNotFoundError(f"Job metadata not found for {job_id}")

        meta = json.loads(job_json_path.read_text())
        return HydrologyJob(
            job_id=meta["job_id"],
            status=meta.get("status", "unknown"),
            error=meta.get("error"),
            params=meta.get("params") or {},
        )

    def _job_json_path(self, job_id: str) -> Path:
        return self._job_path(job_id) / "job.json"

    # ------------------------------------------------------------------
    # Public: update_job
    # ------------------------------------------------------------------
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Update job metadata on disk (job.json) and in Redis.

        - meta is always treated as dict[str, Any]
        - params is stored as a dict in job.json, and JSON-encoded in Redis.
        """
        job_dir = self._job_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        job_json_path = self._job_json_path(job_id)

        # ---- load existing meta from disk, or create fresh ----
        if job_json_path.exists():
            try:
                meta: dict[str, Any] = json.loads(job_json_path.read_text())
            except Exception:
                meta = {"job_id": job_id}
        else:
            meta = {"job_id": job_id}

        # ---- apply updates ----
        if status is not None:
            meta["status"] = status
        if error is not None:
            meta["error"] = error
        if params is not None:
            # store as real dict in job.json (not a string)
            meta["params"] = params

        meta["updated_at"] = time.time()

        # ---- write back to disk ----
        job_json_path.write_text(json.dumps(meta, indent=2))

        # ---- update Redis hash ----
        job_key = self._job_key(job_id)
        redis_mapping: dict[str, str] = {
            "job_id": job_id,
            "updated_at": str(meta["updated_at"]),
        }

        if "status" in meta:
            redis_mapping["status"] = str(meta["status"])
        if "error" in meta and meta["error"] is not None:
            redis_mapping["error"] = str(meta["error"])
        if "params" in meta and meta["params"] is not None:
            # params goes into Redis as JSON string
            redis_mapping["params"] = json.dumps(meta["params"])

        self.redis.hset(job_key, mapping=redis_mapping)

        # Re-apply TTL if desired (so job lives for full job_ttl_seconds
        # after last update)
        if self.job_ttl_seconds > 0:
            self.redis.expire(job_key, self.job_ttl_seconds)

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _new_job_id(self) -> str:
        """
        Very simple ID generator. Replace with UUID if you prefer.
        """
        # You can swap this for uuid4().hex if that's what you had.
        import uuid

        return uuid.uuid4().hex
