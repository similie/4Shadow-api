# terrain/preprocess_global.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from terrain.dem_sources.copernicus_s3 import (
    CopernicusDEMConfig,
    CopernicusDEMS3Downloader,
)

from terrain.process.copernicus_vrt import build_dem_vrt  # reuse your existing builder

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PreprocessGlobalConfig:
    dataset: str = "30m"
    download_dir: Path = Path("data/elevation_copernicus")
    output_dir: Path = Path("data/hydrology_copernicus_global")

    # If True, walks the whole bucket. Slow, but resumable because we skip existing files.
    download_all: bool = False

    # Optional bbox: if set, only downloads tiles intersecting bbox (faster)
    bbox: Optional[BBox] = None

    # What to include
    download_wbm: bool = True
    download_qa: bool = False


def _build_vrt_for_pattern(*, tiles_root: Path, vrt_path: Path, pattern: str) -> None:
    """
    Uses your existing VRT builder if it accepts scanning root.
    If build_dem_vrt is DEM-specific, you can replicate it for other patterns.
    """
    # If your build_dem_vrt already finds "*_DEM.tif", keep it for DEM.
    # For masks, create separate builders (see note below).
    raise NotImplementedError


def run_preprocess_global(cfg: PreprocessGlobalConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("🌍 Global Copernicus preprocessing (VRT-first)")
    print(f"  dataset:      {cfg.dataset}")
    print(f"  download_dir: {cfg.download_dir}")
    print(f"  output_dir:   {cfg.output_dir}")
    print(f"  download_all: {cfg.download_all}")
    print(f"  bbox:         {cfg.bbox}")

    # 1) Download (filtered)
    dl_cfg = CopernicusDEMConfig(
        dataset=cfg.dataset,
        out_dir=cfg.download_dir,
        download_dem=True,
        download_wbm=cfg.download_wbm,
        download_qa=cfg.download_qa,
        skip_preview=True,
    )
    downloader = CopernicusDEMS3Downloader(dl_cfg)

    downloaded = downloader.download(
        bbox=cfg.bbox,
        include_all_if_null=cfg.download_all,
    )
    print(f"\n⬇️  Downloaded/available {len(downloaded)} file(s)")

    # 2) Build global DEM VRT
    dem_vrt = cfg.output_dir / "dem.vrt"
    print("\n🧩 Building global DEM VRT ...")
    build_dem_vrt(tiles_root=cfg.download_dir, vrt_path=dem_vrt)
    print(f"  ✔ Saved → {dem_vrt}")

    # 3) Optional: build WBM/QA VRTs
    # If your build_dem_vrt only targets DEM, add sibling builders
    # build_wbm_vrt(...) / build_qa_vrt(...)
    # For now, just document the intended paths:
    if cfg.download_wbm:
        print("\n🧩 (Next) Build WBM VRT →", cfg.output_dir / "wbm.vrt")
    if cfg.download_qa:
        print(
            "\n🧩 (Next) Build QA VRTs →",
            cfg.output_dir / "edm.vrt",
            cfg.output_dir / "flm.vrt",
            cfg.output_dir / "hem.vrt",
        )

    print("\n✅ Global VRT preprocessing complete.")
    print(
        "Next: serve tiles from dem.vrt (and wbm.vrt) using your warp-per-tile reader."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Global Copernicus preprocessing: download + build global VRTs"
    )
    parser.add_argument("--dataset", default="30m", choices=["30m", "90m"])
    parser.add_argument("--download-dir", default="data/elevation_copernicus")
    parser.add_argument("--output-dir", default="data/hydrology_copernicus_global")
    parser.add_argument(
        "--all", action="store_true", help="Download all tiles (slow). Skips existing."
    )
    parser.add_argument(
        "--download-wbm", action="store_true", help="Download WBM water mask AUXFILES"
    )
    parser.add_argument(
        "--download-qa", action="store_true", help="Download EDM/FLM/HEM AUXFILES"
    )
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)

    args = parser.parse_args()
    bbox = None
    if None not in (args.min_lon, args.min_lat, args.max_lon, args.max_lat):
        bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)

    cfg = PreprocessGlobalConfig(
        dataset=args.dataset,
        download_dir=Path(args.download_dir),
        output_dir=Path(args.output_dir),
        download_all=args.all,
        bbox=bbox,
        download_wbm=args.download_wbm,
        download_qa=args.download_qa,
    )
    run_preprocess_global(cfg)
