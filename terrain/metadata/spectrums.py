from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


LegendType = Literal["linear", "percentile", "text", "stacked", "combined"]


# ----------------------------
# Legend schemas (output)
# ----------------------------
class UnitRange(TypedDict, total=False):
    min: float
    max: float
    label: str
    steps: List[Dict[str, float]]  # stacked


class PercentileInfo(TypedDict):
    lo: float
    hi: float


class LegendOut(TypedDict, total=False):
    type: LegendType
    spread: int

    # linear / percentile
    metric: UnitRange
    imperial: UnitRange

    # percentile-only metadata
    percentiles: PercentileInfo
    notes: str

    # text-only
    stepValues: List[str]

    # combined-only
    # metric/imperial will hold nested dicts in combined mode,
    # but TypedDict can’t express that cleanly; we keep it flexible.


@dataclass(frozen=True)
class Spectrum:
    key: str
    name: str
    colors: List[str]
    description: str
    legend: LegendOut


def _linear(
    *,
    spread: int,
    metric: UnitRange,
    imperial: Optional[UnitRange] = None,
) -> LegendOut:
    return {
        "type": "linear",
        "spread": int(spread),
        "metric": metric,
        "imperial": imperial or metric,
    }


def _percentile(
    *,
    spread: int,
    metric: UnitRange,
    imperial: Optional[UnitRange] = None,
    lo: float = 5,
    hi: float = 95,
    notes: str = "Scaled by percentiles for visibility (tile-dependent).",
) -> LegendOut:
    return {
        "type": "percentile",
        "spread": int(spread),
        "metric": metric,
        "imperial": imperial or metric,
        "percentiles": {"lo": float(lo), "hi": float(hi)},
        "notes": notes,
    }


def _text(*, spread: int, step_values: List[str]) -> LegendOut:
    return {"type": "text", "spread": int(spread), "stepValues": step_values}


def _ramp(*hex_colors: str) -> List[str]:
    """
    Provide either:
      - full ramp already (many colors), or
      - 3-5 anchor colors (client can interpolate if it does, otherwise we keep as-is).
    """
    return list(hex_colors)


def _stacked(
    *,
    spread: int,
    metric: UnitRange,
    imperial: Optional[UnitRange] = None,
) -> LegendOut:
    # expects `steps` inside metric/imperial
    out: LegendOut = {"type": "stacked", "spread": int(spread), "metric": metric}
    out["imperial"] = imperial or metric
    return out


def _combined(
    *,
    spread: int,
    metric: Dict[str, UnitRange],
    imperial: Optional[Dict[str, UnitRange]] = None,
) -> LegendOut:
    # matches your air_quality pattern (nested keys)
    return {
        "type": "combined",
        "spread": int(spread),
        "metric": metric,  # type: ignore[typeddict-item]
        "imperial": imperial or metric,  # type: ignore[typeddict-item]
    }


# ----------------------------
# Base spectra (static layers)
# IMPORTANT: Legends should match how you render.
# If a colorizer uses percentile stretching, use _percentile().
# ----------------------------
BASE_SPECTRA: List[Spectrum] = [
    Spectrum(
        name="Digital Elevation Model",
        key="dem",
        description=(
            "Ground elevation above sea level from Copernicus DEM. "
            "Used as the terrain foundation for hydrology and landslide analysis."
        ),
        # NOTE: your current DEM renderer is bucketed + ocean transparent.
        # If you later change DEM to a continuous ramp, update this list.
        colors=[
            "#0A2F51",  # deep ocean/low (if shown)
            "#2E6F40",  # lowlands
            "#B58900",  # mid elevations
            "#F2F2F2",  # high elevations
        ],
        legend=_linear(
            spread=4,
            metric={"min": 0, "max": 3000, "label": "m"},
        ),
    ),
    Spectrum(
        name="Accumulation",
        key="accum",
        description=(
            "Flow accumulation (upstream contributing area proxy). "
            "High values trace drainage paths and channel networks. "
            "Often rendered with log scaling + percentile contrast for visibility."
        ),
        # Matches your current green→yellow→red look.
        colors=["#00FF00", "#FFFF00", "#FF0000"],
        legend=_percentile(
            spread=3,
            metric={"min": 0, "max": 1, "label": "relative"},
            lo=5,
            hi=99.5,
            notes="Log-scaled + percentile-stretched per tile; legend is relative (not absolute cell counts).",
        ),
    ),
    Spectrum(
        name="Slope",
        key="slope",
        description=(
            "Terrain slope (degrees) derived from the DEM. "
            "Steeper slopes generally increase runoff velocity and landslide susceptibility."
        ),
        # colors=[
        #     "#F5F5F5",
        #     "#A6BDDB",
        #     "#2B8CBE",
        #     "#045A8D",
        # ],
        colors=_ramp("#fff5f0", "#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"),
        legend=_linear(
            spread=4,
            metric={"min": 0, "max": 60, "label": "deg"},
        ),
    ),
    Spectrum(
        name="Topographic Wetness Index",
        key="twi",
        description=(
            "Topographic Wetness Index (TWI): a terrain-based indicator of where water tends to "
            "accumulate and persist. Higher values suggest saturation-prone areas."
        ),
        colors=[
            "#F7FCFD",
            "#BFDCE7",
            "#86B8C9",
            "#3A6F8E",
            "#08306B",
        ],
        # Your renderer normalizes by percentiles, so the legend should say that.
        legend=_percentile(
            spread=5,
            metric={"min": 0, "max": 1, "label": "relative"},
            lo=5,
            hi=95,
            notes="Percentile-stretched per tile; values shown are relative contrast, not a fixed global TWI scale.",
        ),
    ),
    Spectrum(
        name="Distance to Channel",
        key="dist_to_channel",
        description=(
            "Distance to the nearest drainage channel derived from flow accumulation. "
            "Smaller values mean closer to valleys/streams (higher flood/saturation potential)."
        ),
        # If your current map is “green blobs”, this matches the present look.
        colors=["#00783C", "#6CD9A7", "#B4FFDC"],
        # If you render as percentile/alpha-fade, make this percentile; otherwise keep linear meters.
        legend=_percentile(
            spread=3,
            metric={"min": 0, "max": 1, "label": "relative"},
            lo=5,
            hi=95,
            notes="Often displayed with contrast/alpha scaling for readability; treat as relative distance intensity.",
        ),
    ),
    Spectrum(
        name="Landslide Susceptibility Index",
        key="lsi_base",
        description=(
            "Static landslide susceptibility derived from terrain properties (e.g., slope, wetness potential, "
            "and proximity to channels). Represents inherent susceptibility independent of rain events."
        ),
        colors=["#143C5A", "#FFD25A", "#FF3C8C"],
        legend=_linear(
            spread=3,
            metric={"min": 0, "max": 1, "label": "index"},
        ),
    ),
]

# ----------------------------
# Job spectra (event layers)
# ----------------------------
JOB_SPECTRA: List[Spectrum] = [
    Spectrum(
        name="Rainfall",
        key="rain",
        description=(
            "Interpolated rainfall depth for the event, based on station observations. "
            "Shows where precipitation occurred and how strong it was."
        ),
        colors=["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
        legend=_linear(spread=5, metric={"min": 0, "max": 200, "label": "mm"}),
    ),
    Spectrum(
        name="Discharge",
        key="discharge",
        description=(
            "Estimated discharge (flow rate) during the event. Higher values indicate stronger channel flows."
        ),
        colors=["#F7FCF0", "#CCEBC5", "#7BCCC4", "#2B8CBE", "#084081"],
        legend=_percentile(
            spread=5,
            metric={"min": 0, "max": 1, "label": "relative"},
            lo=5,
            hi=95,
            notes="Often scaled by percentiles for visibility; absolute discharge ranges vary widely by basin/event.",
        ),
    ),
    Spectrum(
        name="Flood Severity Index",
        key="fsi",
        description=(
            "Flood Severity Index: relative flood hazard combining terrain, drainage, discharge, and rainfall."
        ),
        colors=["#1A9850", "#FEE08B", "#D73027"],
        legend=_linear(spread=3, metric={"min": 0, "max": 1, "label": "index"}),
    ),
    Spectrum(
        name="Landslide Tigger",
        key="lsi_trigger",
        description=(
            "Rainfall trigger field representing event-specific forcing for landslides. "
            "Higher values indicate rainfall conditions more likely to activate landslide processes."
        ),
        colors=["#143C5A", "#FFD25A", "#FF3C8C"],
        legend=_linear(spread=3, metric={"min": 0, "max": 1, "label": "index"}),
    ),
    Spectrum(
        name="Landslipe Hazards",
        key="lsi_hazard",
        description=(
            "Event landslide hazard combining base susceptibility and rainfall trigger. "
            "Higher values indicate jointly elevated landslide risk."
        ),
        colors=["#143C5A", "#FFD25A", "#FF3C8C"],
        legend=_linear(spread=3, metric={"min": 0, "max": 1, "label": "index"}),
    ),
]


def spectrums_payload(
    *, include_base: bool = True, include_job: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "<layer_key>": {
          "colors": [...],
          "description": "...",
          "legend": {...}
        },
        ...
      }
    This matches your client expectation shape.
    """
    out: Dict[str, Dict[str, Any]] = {}
    defs: List[Spectrum] = []
    if include_base:
        defs += BASE_SPECTRA
    if include_job:
        defs += JOB_SPECTRA

    for s in defs:
        out[s.key] = {
            "colors": s.colors,
            "description": s.description,
            "legend": s.legend,
            "parameter_key": s.key,
            "parameter_name": s.name,
        }

    return out


# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional


# @dataclass(frozen=True)
# class SpectrumDef:
#     parameter_key: str
#     colors: List[str]  # hex strings min->max
#     units: str  # label shown in legend
#     min: float
#     max: float
#     spread: int = 3  # matches your client default
#     description: Optional[str] = None


# # ---------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------
# def _ramp(*hex_colors: str) -> List[str]:
#     """
#     Provide either:
#       - full ramp already (many colors), or
#       - 3-5 anchor colors (client can interpolate if it does, otherwise we keep as-is).
#     """
#     return list(hex_colors)


# # ---------------------------------------------------------------------
# # Base (static) layers
# # ---------------------------------------------------------------------
# BASE_SPECTRUM_DEFS = [
#     SpectrumDef(
#         parameter_key="dem",
#         units="m",
#         min=0,
#         max=3000,
#         description="Ground elevation above sea level from Copernicus DEM. Ocean is rendered transparent.",
#         colors=_ramp("#0A2F51", "#2E6F40", "#B58900", "#F2F2F2"),  # exact buckets used
#     ),
#     SpectrumDef(
#         parameter_key="accum",
#         units="log10(cells)",
#         min=0,
#         max=6,  # representative; since you log10() and auto-scale
#         description="Upstream contributing area proxy (flow accumulation). Rendered as log-scaled and contrast-stretched per tile.",
#         colors=_ramp("#00FF00", "#FFFF00", "#FF0000"),
#     ),
#     SpectrumDef(
#         parameter_key="slope",
#         units="deg",
#         min=0,
#         max=60,
#         description="Terrain steepness in degrees derived from the DEM.",
#         colors=_ramp("#F5F5F5", "#A6BDDB", "#2B8CBE", "#045A8D"),
#     ),
#     SpectrumDef(
#         parameter_key="twi",
#         units="index",
#         min=0,
#         max=20,
#         description="Topographic Wetness Index: where water tends to accumulate and saturate. Current renderer normalizes by percentiles per tile.",
#         colors=_ramp("#F7FCFD", "#BFDCE7", "#86B8C9", "#3A6F8E", "#08306B"),
#     ),
#     SpectrumDef(
#         parameter_key="dist_to_channel",
#         units="m",
#         min=0,
#         max=2000,
#         description="Distance to nearest drainage channel derived from flow accumulation. Current renderer fades alpha with distance.",
#         colors=_ramp("#00783C", "#6CD9A7", "#B4FFDC"),
#     ),
#     SpectrumDef(
#         parameter_key="lsi_base",
#         units="index",
#         min=0,
#         max=1,
#         description="Static landslide susceptibility (terrain-driven).",
#         colors=_ramp("#143C5A", "#FFD25A", "#FF3C8C"),
#     ),
# ]
# # BASE_SPECTRUM_DEFS: List[SpectrumDef] = [
# #     SpectrumDef(
# #         parameter_key="dem",
# #         units="m",
# #         description="Ground elevation above sea level derived from Copernicus GLO-30 digital elevation data. This layer represents the physical terrain and underpins all hydrology and landslide calculations.",
# #         min=0,
# #         max=3000,
# #         colors=_ramp(
# #             "#081d58",
# #             "#253494",
# #             "#225ea8",
# #             "#1d91c0",
# #             "#41b6c4",
# #             "#7fcdbb",
# #             "#c7e9b4",
# #             "#edf8b1",
# #             "#ffffd9",
# #         ),
# #     ),
# #     SpectrumDef(
# #         parameter_key="accum",
# #         units="cells",
# #         description="The number of upstream grid cells contributing flow to each location. High values indicate valleys, streams, and drainage paths where water naturally concentrates.",
# #         min=0,
# #         max=100000,  # display range; accumulation is huge — treat as visual/log in UI
# #         colors=_ramp("#00000000", "#2c7fb8", "#41b6c4", "#a1dab4", "#ffffcc"),
# #     ),
# #     SpectrumDef(
# #         parameter_key="slope",
# #         units="deg",
# #         description="Steepness of the terrain derived from the elevation model. Steeper slopes generally increase runoff speed and landslide susceptibility.",
# #         min=0,
# #         max=60,
# #         colors=_ramp("#fff5f0", "#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"),
# #     ),
# #     SpectrumDef(
# #         parameter_key="twi",
# #         units="TWI",
# #         description="A terrain-based indicator of where water tends to accumulate and persist. High TWI values identify areas prone to saturation, poor drainage, and elevated landslide risk.",
# #         min=0,
# #         max=20,  # common-ish display range; your computed values may differ
# #         colors=_ramp("#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"),
# #     ),
# #     SpectrumDef(
# #         parameter_key="dist_to_channel",
# #         units="m",
# #         description="Horizontal distance to the nearest drainage channel or stream derived from flow accumulation. Smaller values indicate proximity to rivers or valley bottoms where flooding and saturation are more likely.",
# #         min=0,
# #         max=2000,
# #         colors=_ramp("#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"),
# #     ),
# #     SpectrumDef(
# #         parameter_key="lsi_base",
# #         units="index",
# #         description="Intrinsic landslide susceptibility derived from terrain properties such as slope, wetness potential, curvature, and proximity to channels. This layer is static and represents where landslides are more likely in general.",
# #         min=0,
# #         max=1,
# #         colors=_ramp("#143c5a", "#ffd25a", "#ff3c8c"),
# #     ),
# #     SpectrumDef(
# #         parameter_key="fsi",
# #         units="index",
# #         description="Relative flood hazard indicator combining terrain, drainage, discharge, and rainfall. Higher values indicate locations more likely to experience damaging flood conditions during the event.",
# #         min=0,
# #         max=1,
# #         colors=_ramp("#1a9850", "#fee08b", "#d73027"),
# #     ),
# # ]

# # ---------------------------------------------------------------------
# # Job (event) layers
# # ---------------------------------------------------------------------
# JOB_SPECTRUM_DEFS: List[SpectrumDef] = [
#     SpectrumDef(
#         parameter_key="rain",
#         units="mm",
#         description="Interpolated rainfall depth across the region for the selected event, based on station observations and spatial weighting. Shows where precipitation occurred and how intense it was.",
#         min=0,
#         max=200,
#         colors=_ramp("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
#     ),
#     SpectrumDef(
#         parameter_key="discharge",
#         units="m³/s",
#         description="Estimated water flow rate moving through channels during the event. Higher values indicate stronger and potentially hazardous flows.",
#         min=0,
#         max=500,
#         colors=_ramp("#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"),
#     ),
#     SpectrumDef(
#         parameter_key="lsi_trigger",
#         units="index",
#         description="Event-specific rainfall forcing that increases landslide likelihood. Highlights where rainfall intensity and duration are sufficient to activate landslide processes on susceptible terrain.",
#         min=0,
#         max=1,
#         colors=_ramp("#143c5a", "#ffd25a", "#ff3c8c"),
#     ),
#     SpectrumDef(
#         parameter_key="lsi_hazard",
#         units="index",
#         description="Combined landslide hazard for the event, calculated by multiplying base susceptibility with rainfall triggering. Areas with high values represent locations where terrain conditions and rainfall jointly indicate elevated landslide risk.",
#         min=0,
#         max=1,
#         colors=_ramp("#143c5a", "#ffd25a", "#ff3c8c"),
#     ),
# ]


# def build_spectrums_payload(
#     *, include_base: bool = True, include_job: bool = True
# ) -> Dict[str, dict]:
#     out: Dict[str, dict] = {}
#     defs: List[SpectrumDef] = []
#     if include_base:
#         defs += BASE_SPECTRUM_DEFS
#     if include_job:
#         defs += JOB_SPECTRUM_DEFS

#     for d in defs:
#         out[d.parameter_key] = {
#             "colors": d.colors,
#             "parameter_key": d.parameter_key,
#             "description": d.description,
#             "legend": {
#                 "spread": d.spread,
#                 "imperial": {
#                     "min": round(d.min),
#                     "max": round(d.max),
#                     "label": d.units,
#                 },
#                 "metric": {"min": round(d.min), "max": round(d.max), "label": d.units},
#             },
#         }
#     return out
