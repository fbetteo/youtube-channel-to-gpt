# YouTube Transcript Agent API

Turn any YouTube video, playlist, or channel into agent-ready context.

This project exposes a developer API, npm CLI, installable agent skill, and MCP endpoint for agents that need YouTube transcripts as working memory.

## Agent Interfaces

- Developer API: `GET /api/v1/openapi.json`
- Agent docs: `GET /docs/agent.md`
- LLM discovery: `GET /llms.txt`
- Hosted MCP: `POST /mcp` with `Authorization: Bearer <api_key>`
- Local MCP: `npx youtube-transcript-agent mcp`
- CLI binary: `ytx`

## Developer API

All `/api/v1/*` transcript endpoints require `X-API-Key`.

Public API base URL:

```bash
API_BASE_URL=https://api.youtubetranscripts.fbetteo.com
```

```bash
curl -X POST "$API_BASE_URL/api/v1/transcripts/single" \
  -H "X-API-Key: $YOUTUBE_TRANSCRIPT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=FOp280ZAxhg","include_timestamps":true}'
```

Start a channel transcript job:

```bash
curl -X POST "$API_BASE_URL/api/v1/transcripts/channel" \
  -H "X-API-Key: $YOUTUBE_TRANSCRIPT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"channel":"@mkbhd","max_videos":25,"options":{"include_timestamps":false,"concatenate_all":true}}'
```

Poll and download:

```bash
curl -H "X-API-Key: $YOUTUBE_TRANSCRIPT_API_KEY" \
  "$API_BASE_URL/api/v1/jobs/$JOB_ID"

curl -L -H "X-API-Key: $YOUTUBE_TRANSCRIPT_API_KEY" \
  "$API_BASE_URL/api/v1/jobs/$JOB_ID/download" \
  -o transcripts.zip
```

## CLI

The CLI reads auth from `YOUTUBE_TRANSCRIPT_API_KEY` first, then local config created by `ytx auth set-key`.

```bash
npx youtube-transcript-agent auth set-key yt_live_xxx
npx youtube-transcript-agent transcript FOp280ZAxhg --timestamps
npx youtube-transcript-agent channel download @mkbhd --max 25 --wait --output mkbhd.zip
npx youtube-transcript-agent playlist download PLxxxx --wait
npx youtube-transcript-agent mcp
```

Most users do not need to set `YOUTUBE_TRANSCRIPT_API_BASE_URL`; the CLI defaults to the hosted production API. Use it only for local development, staging, self-hosted deployments, or debugging.

## Hosted MCP

The hosted MCP endpoint accepts JSON-RPC over Streamable HTTP at `/mcp`.

Supported tools:

- `get_transcript`
- `get_channel_info`
- `list_channel_videos`
- `start_channel_job`
- `start_playlist_job`
- `get_job_status`
- `get_download_url`

Use bearer auth:

```bash
curl -X POST "$API_BASE_URL/mcp" \
  -H "Authorization: Bearer $YOUTUBE_TRANSCRIPT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

## Local Setup

### Prerequisites

- Python 3.12+
- YouTube Data API key
- Optional WebShare proxy credentials

### Installation

```bash
git clone <repository-url>
cd youtube-channel-to-gpt
pip install -r requirements.txt
```

Create `.env`:

```bash
YOUTUBE_API_KEY=your_youtube_api_key
TRANSCRIPT_API_KEY=your_internal_secret_key
API_BASE_URL=http://localhost:8000
WEBSHARE_PROXY_USERNAME=optional_proxy_username
WEBSHARE_PROXY_PASSWORD=optional_proxy_password
```

Start the API:

```bash
python src/transcript_api.py
```

The API will be available at `http://localhost:8000`.
