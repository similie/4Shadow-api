"""MCP (Model Context Protocol) server: forecast tools for AI assistants.

api.py serves it at /v2/mcp over streamable HTTP. It is stateless (each request stands alone, so
any uvicorn worker can answer and nothing is kept between calls) and answers with plain JSON rather
than an SSE stream, which passes cleanly through the API-key router. The tools call the same
service.point_forecast as GET /v2/forecast, so the two interfaces can't disagree.

Design rules, learned from the hosts that call it:
  * Tools return compact Markdown text, not structured output. Hosts hand the text to the model
    as is, and some cap it (Experts truncates at 8000 characters).
  * Parameters are plain types with "empty" defaults, not `X | None`: Microsoft 365 Copilot drops
    tools whose schemas use anyOf/null.
  * Experts hashes each tool's name, description and schema when an admin approves a connection,
    and asks for re-approval when they change. Change them deliberately.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import numpy as np
from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .catalog import CATALOGS
from .service import ForecastError, get_run, point_forecast
from .store import RunStore
from .timeutil import resolve_timezone

PATH = "/v2/mcp"
MODEL = "gfs"
MAX_DAYS = 16
MAX_HOURS = 384
MAX_HOURLY_VARS = 8
DEFAULT_HOURLY_VARS = ("t2m", "rh2m", "precip", "wind_speed", "wind_dir", "gust", "tcc")
LIST_TTL_MS = 60 * 60 * 1000  # how long hosts may cache the tool list; the list is the same for everyone

INSTRUCTIONS = (
    "4Shadow gives weather forecasts for any point on Earth, from NOAA's GFS model (0.25° grid, "
    "about 25 km) updated four times a day, up to 16 days ahead. Use get_forecast for current "
    "conditions and a daily outlook, and get_hourly_forecast for hour-by-hour detail. Both take "
    "decimal latitude and longitude; if you only have a place name, use its coordinates. Times are "
    "local to the location. Values are model estimates for a grid cell, not observations: local "
    "effects (mountains, coasts, thunderstorms) can differ, and reliability falls after about 5 days. "
    "Data: NOAA GFS (public domain), processed by 4Shadow (Similie)."
)

READ_ONLY = ToolAnnotations.model_validate({
    "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False,
})

SHORT_LABELS = {
    "t2m": "Temp", "d2m": "Dew pt", "rh2m": "RH", "q2m": "Spec hum", "u10": "U wind", "v10": "V wind",
    "wind_speed": "Wind", "wind_dir": "Dir", "gust": "Gust", "mslp": "MSLP", "sp": "Sfc pres",
    "precip": "Rain", "tcc": "Cloud", "cape": "CAPE", "pwat": "Precip water", "vis": "Visibility",
    "dswrf": "Solar", "snod": "Snow depth",
}
_COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
SOURCE_LINE = "Source: NOAA GFS (public domain), processed by 4Shadow (Similie), https://4shadow.io"

Lat = Annotated[float, Field(description="Latitude in decimal degrees, -90 to 90 (south is negative).")]
Lon = Annotated[float, Field(description="Longitude in decimal degrees, -180 to 180 (west is negative).")]
Tz = Annotated[str, Field(description="IANA time zone for times and days, e.g. 'Asia/Dili' or 'UTC'. "
                                      "Empty uses the location's own time zone.")]


def compass(deg: float) -> str:
    return _COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def _fmt(v: float | None, decimals: int = 1) -> str:
    return "–" if v is None else f"{v:.{decimals}f}"


def _offset(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    h, m = divmod(abs(seconds) // 60, 60)
    return f"UTC{sign}{h:02d}:{m:02d}"


def _parse_iso(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def _header(title: str, r, body: dict, now: datetime) -> list[str]:
    loc, tz = body["location"], body["timezone"]
    grid = loc["nearest_grid_point"]
    age_h = (now - r.init_time).total_seconds() / 3600
    terrain = []
    if loc.get("model_elevation_m") is not None:
        terrain.append(f"model terrain height {loc['model_elevation_m']:.0f} m")
    if loc.get("land_fraction") is not None:
        terrain.append(f"land {loc['land_fraction'] * 100:.0f}%")
    return [
        f"## {title} for {loc['lat']:.4f}, {loc['lon']:.4f}",
        f"Time zone {tz['name']} ({_offset(tz['utc_offset_seconds'])}). "
        f"GFS run {r.init_time:%Y-%m-%d %H:%M} UTC ({age_h:.0f} h old), forecast to {r.valid_times[-1]:%Y-%m-%d %H:%M} UTC. "
        f"Grid cell {grid['lat']:.2f}, {grid['lon']:.2f}" + (f"; {', '.join(terrain)}." if terrain else "."),
    ]


def _tool_error(exc: ForecastError) -> ToolError:
    if exc.status == 503:
        return ToolError("No forecast data is available yet. Try again in about 10 minutes.")
    return ToolError(exc.message)


def _interval_total(values: list[float | None], hours: list[float], lo: int, hi: int, per_hour: bool) -> float | None:
    """Combine interval values (rates over the interval ending at each step) for steps lo..hi inclusive:
    the total when the unit is per hour (mm/h -> mm), else the time-weighted mean."""
    x = np.array([np.nan if v is None else v for v in values[lo:hi + 1]], dtype=np.float64)
    w = np.asarray(hours[lo:hi + 1], dtype=np.float64)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return None
    total = float((x[ok] * w[ok]).sum())
    return total if per_hour else total / float(w[ok].sum())


def build_server(stores: dict[str, RunStore]) -> MCPServer:
    cat = CATALOGS[MODEL]
    server = MCPServer(
        "4Shadow Forecast",
        version=__version__,
        website_url="https://4shadow.io",
        instructions=INSTRUCTIONS,
        cache_hints={m: CacheHint(ttl_ms=LIST_TTL_MS, scope="public") for m in ("server/discover", "tools/list")},
    )

    @server.tool(
        title="Weather forecast",
        description=(
            "Current weather and a day-by-day forecast (up to 16 days) for any point on Earth: "
            "minimum and maximum temperature, rain total, wind, gusts, humidity, cloud cover and solar "
            "energy per local day."
        ),
        annotations=READ_ONLY,
        structured_output=False,
    )
    def get_forecast(
        latitude: Lat,
        longitude: Lon,
        days: Annotated[int, Field(description="Number of days, 1 to 16, starting today.")] = 7,
        timezone: Tz = "",
    ) -> str:
        days = min(max(int(days), 1), MAX_DAYS)
        now = datetime.now(UTC)
        try:
            r, body = point_forecast(stores, latitude, longitude, model=MODEL, hours=1, start="now",
                                     tz=timezone or None, daily=True, now=now)
        except ForecastError as exc:
            raise _tool_error(exc) from exc
        zone, _ = resolve_timezone(latitude, longitude, timezone or None)
        lines = _header("Forecast", r, body, now)

        cur, units = body.get("current"), body["units"]

        def now_value(vid: str, label: str) -> str:
            return f"{label} {_fmt(cur[vid], 0 if units[vid] == '%' else cat.variables[vid].decimals)} {units[vid]}"

        if cur:
            parts = []
            for vid, label in (("t2m", "temperature"), ("d2m", "dew point"), ("rh2m", "humidity")):
                if cur.get(vid) is not None:
                    parts.append(now_value(vid, label))
            if cur.get("wind_speed") is not None:
                wind = f"wind {cur['wind_speed']} m/s"
                if cur.get("wind_dir") is not None:
                    wind += f" from {compass(cur['wind_dir'])} ({cur['wind_dir']:.0f}°)"
                parts.append(wind)
            for vid, label in (("gust", "gusts"), ("precip", "precipitation"), ("tcc", "cloud cover"),
                               ("mslp", "sea-level pressure")):
                if cur.get(vid) is not None:
                    parts.append(now_value(vid, label))
            local = _parse_iso(cur["time"]).astimezone(zone)
            lines += ["", f"### Now ({local:%a %d %b %H:%M} local)", "; ".join(parts) + "."]

        daily = body.get("daily")
        if daily and daily.get("date"):
            du = body.get("daily_units", {})
            cols = [(k, h) for k, h in (
                ("t2m_min", f"Min {du.get('t2m_min', '')}"), ("t2m_max", f"Max {du.get('t2m_max', '')}"),
                ("precip_sum", f"Rain {du.get('precip_sum', '')}"), ("wind_speed_max", f"Wind max {du.get('wind_speed_max', '')}"),
                ("gust_max", f"Gust max {du.get('gust_max', '')}"), ("rh2m_mean", f"RH {du.get('rh2m_mean', '')}"),
                ("tcc_mean", f"Cloud {du.get('tcc_mean', '')}"), ("dswrf_sum", f"Solar {du.get('dswrf_sum', '')}"),
            ) if k in daily]
            lines += ["", "### Daily (local days)", "| Date | " + " | ".join(h.strip() for _, h in cols) + " |",
                      "|---" * (len(cols) + 1) + "|"]
            partial, decimals = False, {"rh2m_mean": 0, "tcc_mean": 0}
            for n, date in enumerate(daily["date"][:days]):
                d = datetime.fromisoformat(date)
                label = f"{d:%a %d %b}"
                covered = daily["hours_covered"][n]
                if covered < 23:  # 23 and 25 are whole days with a DST change
                    label += f" ({covered:.0f} h)"
                    partial = True
                lines.append(f"| {label} | " + " | ".join(_fmt(daily[k][n], decimals.get(k, 1)) for k, _ in cols) + " |")
            if partial:
                lines.append("A day marked (N h) is only partly covered by this run, so its values cover those hours only.")
        lines += ["", "Rain is the day's total. Model estimates for the grid cell, not observations.", SOURCE_LINE]
        return "\n".join(lines)

    @server.tool(
        title="Hourly forecast",
        description=(
            "Hour-by-hour forecast table for any point on Earth, starting now, for up to 384 hours "
            "(16 days). Rows are hourly up to 48 hours, then every 3 hours up to 7 days, then every "
            "6 hours. Choose the variables to show; call list_variables for the full list."
        ),
        annotations=READ_ONLY,
        structured_output=False,
    )
    def get_hourly_forecast(
        latitude: Lat,
        longitude: Lon,
        hours: Annotated[int, Field(description="How many hours ahead to cover, 1 to 384.")] = 24,
        variables: Annotated[str, Field(description=(
            "Comma-separated variable ids, at most 8, e.g. 't2m,precip,wind_speed'. Empty shows "
            "temperature, humidity, rain, wind speed and direction, gusts and cloud cover."))] = "",
        timezone: Tz = "",
    ) -> str:
        hours = min(max(int(hours), 1), MAX_HOURS)
        now = datetime.now(UTC)
        try:
            run = get_run(stores, MODEL, "latest")
            available = {v.id for v in cat.available(run.variables)}
            ids = [s.strip() for s in variables.split(",") if s.strip()] or [v for v in DEFAULT_HOURLY_VARS if v in available]
            if len(ids) > MAX_HOURLY_VARS:
                raise ToolError(f"At most {MAX_HOURLY_VARS} variables per call; you asked for {len(ids)}.")
            r, body = point_forecast(stores, latitude, longitude, model=MODEL, run=run.id, variables=",".join(ids),
                                     hours=hours, start="now", tz=timezone or None, daily=False, now=now)
        except ForecastError as exc:
            raise _tool_error(exc) from exc
        zone, _ = resolve_timezone(latitude, longitude, timezone or None)
        ts, units = body["timeseries"], body["units"]
        times = [_parse_iso(t) for t in ts["time"]]
        dt = [float(h) for h in ts["interval_hours"]]
        fhr = [round((t - r.init_time).total_seconds() / 3600) for t in times]
        every = 1 if hours <= 48 else 3 if hours <= 168 else 6
        rows = [0] + [n for n in range(1, len(times)) if fhr[n] % every == 0]

        headers, cells = ["Local time"], [[f"{times[n].astimezone(zone):%a %d %H:%M}"] for n in rows]
        for vid in ids:
            var = cat.variables[vid]
            per_hour = var.interval and var.units.endswith("/h")
            unit = var.units[:-2] if per_hour else var.units
            headers.append(SHORT_LABELS.get(vid, vid) + ("" if vid == "wind_dir" else f" {unit}"))
            values = ts[vid]
            for k, n in enumerate(rows):
                if var.interval:
                    lo = n if k == 0 else rows[k - 1] + 1
                    v = _interval_total(values, dt, lo, n, per_hour)
                else:
                    v = values[n]
                if vid == "wind_dir":
                    cells[k].append("–" if v is None else f"{compass(v)} {v:.0f}°")
                else:
                    cells[k].append(_fmt(v, var.decimals))

        lines = _header("Hourly forecast", r, body, now)
        lines += ["", "| " + " | ".join(headers) + " |", "|---" * len(headers) + "|"]
        lines += ["| " + " | ".join(c) + " |" for c in cells]
        notes = []
        if any(cat.variables[v].interval for v in ids):
            notes.append("Rain and solar values cover the period since the previous row (rain as a total, solar as a mean).")
        if every > 1:
            notes.append(f"Rows every {every} hours after the first.")
        lines += [""] + notes + ["Model estimates for the grid cell, not observations.", SOURCE_LINE]
        return "\n".join(lines)

    @server.tool(
        title="List forecast variables",
        description="Lists the weather variables get_hourly_forecast can show, with their ids, units and meaning.",
        annotations=READ_ONLY,
        structured_output=False,
    )
    def list_variables() -> str:
        try:
            available = {v.id for v in cat.available(get_run(stores, MODEL, "latest").variables)}
        except ForecastError:
            available = set(cat.variables)
        lines = ["| id | name | units | meaning |", "|---|---|---|---|"]
        for v in cat.variables.values():
            if v.id in available:
                meaning = v.description + (" Mean over the period ending at each time." if v.interval else "")
                lines.append(f"| {v.id} | {v.name} | {v.units} | {meaning} |")
        return "\n".join(lines)

    return server


class _PostOnly:
    """Answers anything but POST with 405, as the spec allows for a server without an SSE stream.

    Left to the SDK, a GET opens a stream for server-initiated messages and holds it open
    forever (this server never sends any), and a DELETE ends a session that stateless mode
    never created. Through the router either would tie up a connection or count as a call.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] != "POST":
            response = JSONResponse({"detail": "This MCP endpoint takes POST only (stateless, no SSE stream)."},
                                    status_code=405, headers={"Allow": "POST"})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def http_routes(server: MCPServer) -> list[BaseRoute]:
    """Routes serving `server` at PATH; api.py adds them and runs `server.session_manager`.

    Transport security is passed explicitly: with none, the SDK accepts only Host: localhost, so
    every request arriving through the router would get 421. DNS-rebinding protection guards
    servers that trust whoever can reach them (on a user's own machine); this one is public and
    the router authenticates each request with an API key, so a rebinding page has nothing to borrow.
    """
    app = server.streamable_http_app(
        streamable_http_path=PATH,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return [Route(r.path, _PostOnly(r.app)) if isinstance(r, Route) and r.path == PATH else r for r in app.routes]
