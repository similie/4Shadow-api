import json
import re

import pytest
from fastapi.testclient import TestClient

from forecast.api import create_app
from forecast.config import Settings
from forecast.mcp_server import compass

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
VERSION = "2025-11-25"


@pytest.fixture
def mcp(store_root):
    with TestClient(create_app(Settings(data_root=store_root))) as client:  # runs the lifespan
        yield client


def rpc(client, method, params=None, *, id_=1, version=VERSION):
    headers = dict(HEADERS)
    if method != "initialize":
        headers["MCP-Protocol-Version"] = version
    msg = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
    if id_ is not None:
        msg["id"] = id_
    return client.post("/v2/mcp", headers=headers, content=json.dumps(msg))


def call(client, name, **arguments):
    r = rpc(client, "tools/call", {"name": name, "arguments": arguments})
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    return result["content"][0]["text"], result.get("isError", False)


def table(text):
    """Rows of the first Markdown table in text, as lists of cells (header row first)."""
    rows = [line for line in text.splitlines() if line.startswith("|") and not line.startswith("|---")]
    return [[c.strip() for c in row.strip("|").split("|")] for row in rows]


def test_handshake_and_tool_list(mcp):
    r = rpc(mcp, "initialize", {"protocolVersion": VERSION, "capabilities": {},
                                "clientInfo": {"name": "test", "version": "0"}})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    init = r.json()["result"]
    assert init["serverInfo"]["name"] == "4Shadow Forecast" and "tools" in init["capabilities"]
    assert "mcp-session-id" not in r.headers  # stateless: any worker can answer any request

    assert rpc(mcp, "notifications/initialized", id_=None).status_code == 202

    tools = {t["name"]: t for t in rpc(mcp, "tools/list").json()["result"]["tools"]}
    assert set(tools) == {"get_forecast", "get_hourly_forecast", "list_variables"}
    for t in tools.values():
        assert t["annotations"]["readOnlyHint"] is True and t["annotations"]["destructiveHint"] is False
        assert "anyOf" not in json.dumps(t["inputSchema"])  # Copilot drops tools with nullable params
        assert "outputSchema" not in t
    assert tools["get_forecast"]["inputSchema"]["required"] == ["latitude", "longitude"]
    assert "timezone" in tools["get_hourly_forecast"]["inputSchema"]["properties"]


def test_get_forecast(mcp):
    text, err = call(mcp, "get_forecast", latitude=10.3, longitude=20.7, timezone="UTC")
    assert not err, text
    assert text.startswith("## Forecast for 10.3000, 20.7000")
    assert "Time zone UTC (UTC+00:00)" in text and "### Now" in text and "humidity 50 %" in text
    assert "wind 5.0 m/s from SW (217°)" in text
    rows = table(text)
    assert rows[0][:4] == ["Date", "Min °C", "Max °C", "Rain mm"]
    assert 2 <= len(rows) - 1 <= 3  # the 36 h synthetic run spans two or three UTC days
    assert "Source: NOAA GFS" in text and len(text) < 8000


def test_2026_stateless_protocol(mcp):
    """MCP 2026-07-28: no handshake; each request carries its version and capabilities."""
    meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}
    headers = {**HEADERS, "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "get_forecast"}
    msg = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
           "params": {"name": "get_forecast", "arguments": {"latitude": 0, "longitude": 0, "days": 1}, "_meta": meta}}
    r = mcp.post("/v2/mcp", headers=headers, content=json.dumps(msg))
    assert r.status_code == 200, r.text
    assert r.json()["result"]["content"][0]["text"].startswith("## Forecast for 0.0000, 0.0000")


def test_get_forecast_limits_days(mcp):
    text, _ = call(mcp, "get_forecast", latitude=0, longitude=0, days=1, timezone="UTC")
    assert len(table(text)) == 2


def test_hourly_rows_and_rain_totals(mcp):
    text, err = call(mcp, "get_hourly_forecast", latitude=0, longitude=0, hours=6,
                     variables="t2m,precip,wind_dir", timezone="UTC")
    assert not err, text
    rows = table(text)
    assert rows[0] == ["Local time", "Temp °C", "Rain mm", "Dir"]
    assert len(rows) - 1 == 7  # now .. now+6 h, hourly
    assert {r[2] for r in rows[1:]} == {"1.00"}  # 1 mm/h for one hour
    assert {r[3] for r in rows[1:]} == {"SW 217°"}

    # Over 48 h the rows thin to 3-hourly and rain becomes the total since the previous row.
    text, _ = call(mcp, "get_hourly_forecast", latitude=0, longitude=0, hours=72, variables="precip", timezone="UTC")
    rows = table(text)[1:]
    hours = [int(re.search(r"(\d\d):00$", r[0]).group(1)) for r in rows]
    assert all(h % 3 == 0 for h in hours[1:])
    assert rows[0][1] == "1.00" and {r[1] for r in rows[2:]} == {"3.00"}
    assert "Rows every 3 hours" in text


def test_tool_errors_are_reported_to_the_model(mcp):
    text, err = call(mcp, "get_forecast", latitude=95, longitude=0)
    assert err and "lat must be within" in text
    text, err = call(mcp, "get_hourly_forecast", latitude=0, longitude=0, variables="t2m,nope")
    assert err and "Unknown variable 'nope'" in text and "precip" in text
    text, err = call(mcp, "get_hourly_forecast", latitude=0, longitude=0, variables=",".join(["t2m"] * 9))
    assert err and "At most 8" in text
    text, err = call(mcp, "get_forecast", latitude=0, longitude=0, timezone="Mars/Base")
    assert err and "Unknown timezone" in text


def test_no_data_yet(tmp_path):
    with TestClient(create_app(Settings(data_root=tmp_path))) as client:
        text, err = call(client, "get_forecast", latitude=0, longitude=0)
    assert err and "No forecast data is available yet" in text


def test_list_variables(mcp):
    text, err = call(mcp, "list_variables")
    rows = table(text)
    assert not err and rows[0] == ["id", "name", "units", "meaning"]
    ids = {r[0] for r in rows[1:]}
    assert {"t2m", "precip", "wind_speed", "wind_dir"} <= ids


def test_only_post_and_not_in_openapi(mcp):
    for method in ("GET", "DELETE"):
        r = mcp.request(method, "/v2/mcp", headers={"Accept": "text/event-stream"})
        assert r.status_code == 405 and r.headers["allow"] == "POST"
    assert "/v2/mcp" not in mcp.get("/v2/openapi.json").json()["paths"]


def test_compass():
    assert [compass(d) for d in (0, 11, 12, 90, 217, 348.75, 359)] == ["N", "N", "NNE", "E", "SW", "N", "N"]
