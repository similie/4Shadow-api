# terrain/check_rasters.py

from __future__ import annotations

from pathlib import Path
import rasterio


def describe(path: Path):
    print(f"\n=== {path} ===")
    if not path.exists():
        print("  ❌ MISSING")
        return
    with rasterio.open(path) as src:
        print(f"  CRS:    {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Size:   {src.width} x {src.height}")
        print(f"  Nodata: {src.nodata}")
        print(f"  Transform: {src.transform}")


def main():
    base = Path("data/hydrology")
    jobs_root = Path("data/hydrology_jobs")

    print("==== BASE HYDROLOGY RASTERS ====")
    describe(base / "dem.tif")
    describe(base / "d8_accum.tif")
    describe(base / "dem_3857.tif")
    describe(base / "d8_accum_3857.tif")

    if jobs_root.exists():
        print("\n==== JOB RASTERS ====")
        for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            print(f"\n--- Job {job_dir.name} ---")
            describe(job_dir / "rain_mm.tif")
            describe(job_dir / "discharge_m3s.tif")
            describe(job_dir / "rain_mm_3857.tif")
            describe(job_dir / "discharge_m3s_3857.tif")
    else:
        print("\n(no jobs yet at data/hydrology_jobs)")


if __name__ == "__main__":
    main()
