"""
Agent-facing discovery docs.

These endpoints are intentionally public and only point agents at the
developer API surface, CLI, skill, and MCP entrypoints.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from fastapi.openapi.utils import get_openapi


router = APIRouter(tags=["Agent Discovery"])
ROOT_DIR = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (ROOT_DIR / relative_path).read_text(encoding="utf-8")


@router.get("/llms.txt", include_in_schema=False)
async def llms_txt():
    return PlainTextResponse(_read_text("llms.txt"), media_type="text/plain")


@router.get("/docs/agent.md", include_in_schema=False)
async def agent_markdown():
    return PlainTextResponse(_read_text("docs/agent.md"), media_type="text/markdown")


@router.get("/api/v1/openapi.json", include_in_schema=False)
async def developer_openapi(request: Request):
    """Return an OpenAPI schema limited to the public developer API."""
    routes = [
        route
        for route in request.app.routes
        if getattr(route, "path", "").startswith("/api/v1")
        and getattr(route, "path", "") != "/api/v1/openapi.json"
    ]

    return get_openapi(
        title="YouTube Transcript Developer API",
        version="1.0.0",
        description=(
            "Developer API for turning YouTube videos, channels, and playlists "
            "into agent-ready transcript context."
        ),
        routes=routes,
    )
