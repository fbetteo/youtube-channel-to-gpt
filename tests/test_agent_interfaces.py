import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("yt_dlp", types.SimpleNamespace(YoutubeDL=object))

from src.transcript_api import app


client = TestClient(app)


def test_public_developer_openapi_is_limited_to_api_v1():
    response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert paths
    assert all(path.startswith("/api/v1") for path in paths)
    assert "/internal/openapi.json" not in paths
    assert "/payments/create-checkout-session" not in paths


def test_agent_discovery_docs_are_public():
    llms = client.get("/llms.txt")
    agent_docs = client.get("/docs/agent.md")

    assert llms.status_code == 200
    assert "youtube-transcript-agent" in llms.text
    assert agent_docs.status_code == 200
    assert "MCP" in agent_docs.text


def test_developer_api_still_requires_api_key():
    response = client.get("/api/v1/account/credits")

    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


def test_hosted_mcp_lists_tools_without_secret():
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )

    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "get_transcript" in names
    assert "start_channel_job" in names
    assert "get_download_url" in names


def test_hosted_mcp_tool_call_requires_bearer_auth():
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_transcript",
                "arguments": {"video_url": "FOp280ZAxhg"},
            },
        },
    )

    assert response.status_code == 401
