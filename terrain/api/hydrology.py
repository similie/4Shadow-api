# terrain/api/hydrology.py
from __future__ import annotations

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Response, Request, status
from pydantic import BaseModel, Field
from terrain.job_engine import HydrologyJobEngine
from terrain.api.tile_read import parse_bbox, parse_srs
from terrain.api.tile_render import (
    RenderContext,
    render_layer_rgba,
    render_data_tile,
    render_base_layer_rgba,
)

# from terrain.rain_interpolation import RainStation
from terrain.utils.station_normalization import StationDict
from terrain.metadata.spectrums import spectrums_payload

router = APIRouter(prefix="/hydrology", tags=["hydrology"])
job_engine = HydrologyJobEngine()
DEFAULT_JOB_DELETE_TTL_S = 10


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class StationIn(BaseModel):
    lat: float
    lon: float
    rain_mm: float = Field(..., description="Rainfall for the event in mm")
    wind_speed_ms: float | None = Field(
        default=None,
        description="Instantaneous wind speed at station (m/s)",
    )
    wind_dir_deg: float | None = Field(
        default=None,
        description="Wind direction (degrees FROM which wind blows, met convention)",
    )


class CreateJobRequest(BaseModel):
    stations: List[StationIn]
    event_duration_hours: float = Field(..., gt=0)
    runoff_coeff: float = Field(0.6, ge=0, le=1)
    rain_radius_km: Optional[float] = Field(..., gt=0)


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    params: dict | None = None


# ---------------------------------------------------------------------
# Tiles endpoint
# ---------------------------------------------------------------------
@router.get("/jobs/{job_id}/tiles/{layer}/{z}/{x}/{y}.png")
def get_tile(job_id: str, layer: str, z: int, x: int, y: int, request: Request):
    if not job_engine.valid_job_id(job_id):
        raise HTTPException(404, "Job not found")
    # We ignore z/x/y for warping, because WMS passes explicit bbox/size/srs.
    # Still useful for caching keys/logging.

    # Parse WMS-ish params
    qp = request.query_params
    srs = qp.get("srs") or qp.get("crs")  # support either
    bbox_str = qp.get("bbox")
    width = int(qp.get("width", "256"))
    height = int(qp.get("height", "256"))

    if not srs:
        raise HTTPException(400, "Missing query param: srs=EPSG:3857")
    if not bbox_str:
        raise HTTPException(400, "Missing query param: bbox=minx,miny,maxx,maxy")

    try:
        dst_crs = parse_srs(srs)
        bbox = parse_bbox(bbox_str)
    except Exception as e:
        raise HTTPException(400, f"Invalid srs/bbox: {e}")

    # Load job params (for rain radius, stations, fsi vmin/vmax, etc.)
    try:
        job = job_engine.get_job(job_id)
        params = job.params or {}
    except FileNotFoundError:
        params = {}

    ctx = RenderContext(
        base_dir=job_engine.base_hydrology_dir,
        jobs_dir=job_engine.jobs_root,
        job_id=job_id,
        params=params,
        dst_crs=dst_crs,
        bbox=bbox,
        width=width,
        height=height,
    )

    # Render RGBA
    rgba = render_layer_rgba(ctx, layer)
    return Response(render_data_tile(rgba, width, height), media_type="image/png")


# ---------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------
@router.post("/jobs", response_model=CreateJobResponse)
def create_hydrology_job(body: CreateJobRequest):
    stations = [
        StationDict(
            lat=s.lat,
            lon=s.lon,
            rain_mm=s.rain_mm,
            wind_speed_ms=s.wind_speed_ms,
            wind_dir_deg=s.wind_dir_deg,
        )
        for s in body.stations
    ]

    job = job_engine.create_job(
        stations=stations,
        event_duration_hours=body.event_duration_hours,
        runoff_coeff=body.runoff_coeff,
        rain_radius_km=body.rain_radius_km,
    )
    return CreateJobResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    try:
        job = job_engine.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        params=job.params,
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
def delete_job(job_id: str, ttl_seconds: int = DEFAULT_JOB_DELETE_TTL_S):
    """
    Mark a job for deletion by expiring its Redis key shortly.
    The cleanup worker listens for the key expiry event and deletes the job dir.
    """
    # Ensure job exists (and gives you a nice 404)
    try:
        _ = job_engine.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found")

    # The key your worker watches: "hydro:job:<id>"
    job_key = f"{job_engine.prefix}:job:{job_id}"
    print("SETTING EXPIRED", job_key)
    # If the key doesn't exist, nothing will expire => cleanup won't fire.
    # Create a small tombstone value if needed.
    r = job_engine.redis
    if not r.exists(job_key):
        r.set(job_key, "delete_requested")

    # Set TTL so "__keyevent@X__:expired" will fire
    ttl_seconds = max(int(ttl_seconds), 1)
    r.expire(job_key, ttl_seconds)

    # Optional: mark status in job.json so UI shows "deleting"
    try:
        job_dir = job_engine._job_path(job_id)
        meta_path = job_dir / "job.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "deleting"
            meta.setdefault("params", {})
            meta["params"]["delete_requested"] = True
            meta["params"]["delete_ttl_seconds"] = ttl_seconds
            meta_path.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    return {"job_id": job_id, "status": "deleting", "ttl_seconds": ttl_seconds}


@router.get("/tiles/{layer}/{z}/{x}/{y}.png")
def get_base_tile(layer: str, z: int, x: int, y: int, request: Request):
    qp = request.query_params
    srs = qp.get("srs") or qp.get("crs")
    bbox_str = qp.get("bbox")
    width = int(qp.get("width", "256"))
    height = int(qp.get("height", "256"))

    if not srs:
        raise HTTPException(400, "Missing query param: srs=EPSG:3857")
    if not bbox_str:
        raise HTTPException(400, "Missing query param: bbox=minx,miny,maxx,maxy")

    try:
        dst_crs = parse_srs(srs)
        bbox = parse_bbox(bbox_str)
    except Exception as e:
        raise HTTPException(400, f"Invalid srs/bbox: {e}")

    ctx = RenderContext(
        base_dir=job_engine.base_hydrology_dir,
        jobs_dir=job_engine.jobs_root,  # or jobs_dir if that's your real attr
        job_id=None,  # ✅ no job context
        params={},  # ✅ base tiles don’t need job params
        dst_crs=dst_crs,
        bbox=bbox,
        width=width,
        height=height,
    )

    rgba = render_base_layer_rgba(ctx, layer)
    return Response(render_data_tile(rgba, width, height), media_type="image/png")


@router.get("/layers/spectrums")
def get_layer_spectrums(scope: str | None = None):
    """
    Returns mapFields keyed by parameter_key:
      { [parameter_key]: { colors: [...], legend: {...} } }
    Compatible with the client's `spectrums` getter.
    """
    include_base = scope in (None, "all", "base")
    include_job = scope in (None, "all", "job")

    if scope not in (None, "all", "base", "job"):
        raise HTTPException(400, "scope must be one of: all, base, job")

    return spectrums_payload(include_base=include_base, include_job=include_job)
