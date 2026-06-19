"""
Minimal hosted MCP endpoint for the YouTube Transcript Developer API.

Implements the JSON-RPC methods agents need for tool discovery and tool calls.
The local CLI exposes the same tools over stdio for local coding agents.
"""

import json
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from api_key_auth import validate_api_key_value
from routers import developer_api


router = APIRouter(tags=["MCP"])

MCP_PROTOCOL_VERSION = "2025-06-18"


class McpToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


def _json_rpc_result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _json_rpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _content_text(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _content_json(payload: Any) -> Dict[str, Any]:
    return _content_text(json.dumps(payload, indent=2, default=str))


def _tool_schema(properties: Dict[str, Any], required: Optional[list[str]] = None):
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "get_transcript",
        "description": "Return the transcript for a single YouTube video URL or ID.",
        "inputSchema": _tool_schema(
            {
                "video_url": {
                    "type": "string",
                    "description": "YouTube video URL or video ID.",
                },
                "include_timestamps": {
                    "type": "boolean",
                    "description": "Include transcript timestamps.",
                    "default": False,
                },
            },
            ["video_url"],
        ),
    },
    {
        "name": "get_channel_info",
        "description": "Return metadata for a YouTube channel before starting a job.",
        "inputSchema": _tool_schema(
            {"channel": {"type": "string", "description": "Handle, name, or ID."}},
            ["channel"],
        ),
    },
    {
        "name": "list_channel_videos",
        "description": "List videos and duration metadata for a YouTube channel.",
        "inputSchema": _tool_schema(
            {"channel": {"type": "string", "description": "Handle, name, or ID."}},
            ["channel"],
        ),
    },
    {
        "name": "start_channel_job",
        "description": "Start an async job that downloads channel transcripts.",
        "inputSchema": _tool_schema(
            {
                "channel": {"type": "string", "description": "Handle, name, or ID."},
                "max_videos": {
                    "type": "integer",
                    "description": "Optional maximum videos to process.",
                    "minimum": 1,
                },
                "include_timestamps": {"type": "boolean", "default": False},
                "concatenate_all": {"type": "boolean", "default": False},
            },
            ["channel"],
        ),
    },
    {
        "name": "start_playlist_job",
        "description": "Start an async job that downloads playlist transcripts.",
        "inputSchema": _tool_schema(
            {
                "playlist": {
                    "type": "string",
                    "description": "YouTube playlist URL or ID.",
                },
                "max_videos": {
                    "type": "integer",
                    "description": "Optional maximum videos to process.",
                    "minimum": 1,
                },
                "include_timestamps": {"type": "boolean", "default": False},
                "concatenate_all": {"type": "boolean", "default": False},
            },
            ["playlist"],
        ),
    },
    {
        "name": "get_job_status",
        "description": "Poll an async transcript job and get a download URL when ready.",
        "inputSchema": _tool_schema(
            {"job_id": {"type": "string", "description": "Transcript job ID."}},
            ["job_id"],
        ),
    },
    {
        "name": "get_download_url",
        "description": "Return the ZIP download URL for a completed job.",
        "inputSchema": _tool_schema(
            {"job_id": {"type": "string", "description": "Transcript job ID."}},
            ["job_id"],
        ),
    },
]


async def _api_key_data_from_request(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="MCP tool calls require Authorization: Bearer <api_key>.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = auth_header.split(" ", 1)[1].strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is empty.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await validate_api_key_value(api_key)


def _transcript_options(args: Dict[str, Any]) -> developer_api.TranscriptOptions:
    return developer_api.TranscriptOptions(
        include_timestamps=bool(args.get("include_timestamps", False)),
        include_video_title=bool(args.get("include_video_title", True)),
        include_video_id=bool(args.get("include_video_id", True)),
        include_video_url=bool(args.get("include_video_url", True)),
        include_view_count=bool(args.get("include_view_count", False)),
        concatenate_all=bool(args.get("concatenate_all", False)),
    )


async def _call_tool(
    name: str,
    args: Dict[str, Any],
    api_key_data: Dict[str, Any],
    request: Request,
) -> Dict[str, Any]:
    if name == "get_transcript":
        response = await developer_api.get_single_transcript(
            developer_api.SingleTranscriptRequest(
                video_url=args["video_url"],
                include_timestamps=bool(args.get("include_timestamps", False)),
            ),
            api_key_data,
        )
        return _content_json(response.model_dump())

    if name == "get_channel_info":
        response = await developer_api.get_channel_info(args["channel"], api_key_data)
        return _content_json(response.model_dump())

    if name == "list_channel_videos":
        response = await developer_api.list_channel_videos(args["channel"], api_key_data)
        return _content_json(response.model_dump())

    if name == "start_channel_job":
        response = await developer_api.download_channel_transcripts(
            developer_api.ChannelTranscriptRequest(
                channel=args["channel"],
                max_videos=args.get("max_videos"),
                options=_transcript_options(args),
            ),
            api_key_data,
        )
        return _content_json(response.model_dump())

    if name == "start_playlist_job":
        response = await developer_api.download_playlist_transcripts(
            developer_api.PlaylistTranscriptRequest(
                playlist=args["playlist"],
                max_videos=args.get("max_videos"),
                options=_transcript_options(args),
            ),
            api_key_data,
        )
        return _content_json(response.model_dump())

    if name in {"get_job_status", "get_download_url"}:
        response = await developer_api.get_job_status(args["job_id"], api_key_data)
        payload = response.model_dump()
        if payload.get("download_url"):
            payload["download_url"] = (
                str(request.base_url).rstrip("/") + payload["download_url"]
            )
        if name == "get_download_url" and not payload.get("download_ready"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is not ready for download. Status: {payload['status']}",
            )
        return _content_json(payload)

    raise HTTPException(status_code=404, detail=f"Unknown MCP tool: {name}")


async def _handle_json_rpc(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return _json_rpc_result(
            request_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "youtube-transcript-agent",
                    "version": "1.0.0",
                },
            },
        )

    if method == "tools/list":
        return _json_rpc_result(request_id, {"tools": TOOLS})

    if method == "tools/call":
        call = McpToolCall.model_validate(params)
        api_key_data = await _api_key_data_from_request(request)
        result = await _call_tool(call.name, call.arguments, api_key_data, request)
        return _json_rpc_result(request_id, result)

    if method == "notifications/initialized":
        return _json_rpc_result(request_id, {})

    return _json_rpc_error(request_id, -32601, f"Method not found: {method}")


@router.get("/mcp")
async def mcp_get(request: Request):
    """Offer a lightweight SSE capability check for hosted MCP clients."""
    if "text/event-stream" not in request.headers.get("accept", ""):
        return {
            "service": "youtube-transcript-agent-mcp",
            "transport": "streamable-http",
            "endpoint": "/mcp",
            "auth": "Authorization: Bearer <api_key>",
        }

    async def events():
        yield (
            "event: message\n"
            'data: {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n\n'
        )

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/mcp")
async def mcp_post(request: Request):
    try:
        payload = await request.json()
        if isinstance(payload, list):
            results = [await _handle_json_rpc(item, request) for item in payload]
            return JSONResponse(results)
        return JSONResponse(await _handle_json_rpc(payload, request))
    except HTTPException as e:
        if e.status_code == status.HTTP_401_UNAUTHORIZED:
            raise
        return JSONResponse(
            _json_rpc_error(None, -32000, str(e.detail)),
            status_code=e.status_code,
        )
    except Exception as e:
        return JSONResponse(
            _json_rpc_error(None, -32603, f"Internal error: {e}"),
            status_code=500,
        )


@router.delete("/mcp")
async def mcp_delete():
    return Response(status_code=204)
