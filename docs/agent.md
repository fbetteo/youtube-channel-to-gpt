# YouTube Transcript Agent Guide

Turn YouTube videos, playlists, and channels into agent-ready transcript context.

## Quick Start

Use the CLI when you are running locally:

```bash
npx youtube-transcript-agent auth set-key <api_key>
npx youtube-transcript-agent transcript FOp280ZAxhg --timestamps
```

Use the hosted MCP endpoint when your agent supports remote MCP:

```text
Endpoint: https://your-api.example.com/mcp
Auth: Authorization: Bearer <api_key>
```

Use the Developer API directly when you need explicit HTTP calls:

```bash
curl -X POST "$API_BASE_URL/api/v1/transcripts/single" \
  -H "X-API-Key: $YOUTUBE_TRANSCRIPT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"video_url":"FOp280ZAxhg","include_timestamps":false}'
```

## Common Agent Workflows

### Summarize One Video

```bash
npx youtube-transcript-agent transcript <video_url_or_id> --timestamps
```

Then summarize the transcript, extract claims, or transform it into notes.

### Build A Knowledge Base From A Channel

```bash
npx youtube-transcript-agent channel download @channel --max 100 --wait --output channel-transcripts.zip
```

Use `--max` for first-pass research. Increase it after confirming the channel is relevant.

### Download Playlist Transcripts

```bash
npx youtube-transcript-agent playlist download <playlist_url_or_id> --wait --output playlist-transcripts.zip
```

This is best for course material, conference talks, podcasts, and launch playlists.

### Poll Until A Job Is Ready

```bash
npx youtube-transcript-agent jobs status <job_id>
npx youtube-transcript-agent jobs download <job_id> --output transcripts.zip
```

If `download_ready` is false, wait and poll again.

## MCP Tool Use

The MCP server exposes these tools:

- `get_transcript`: returns a single transcript immediately.
- `get_channel_info`: validates a channel and returns metadata.
- `list_channel_videos`: previews channel videos.
- `start_channel_job`: starts an async channel transcript job.
- `start_playlist_job`: starts an async playlist transcript job.
- `get_job_status`: polls an async job.
- `get_download_url`: returns a ZIP download URL for a completed job.

For large jobs, prefer `start_channel_job` or `start_playlist_job`, then `get_job_status`.
Do not ask the MCP server to inline ZIP files into chat context.

## Authentication

Never paste API keys into prompts unless the tool explicitly requires it. Prefer:

```bash
export YOUTUBE_TRANSCRIPT_API_KEY=<api_key>
```

For hosted MCP, configure the client with:

```text
Authorization: Bearer <api_key>
```

For local CLI use:

```bash
npx youtube-transcript-agent auth set-key <api_key>
```

## API Base URL

Set a custom API base URL for local development or staging:

```bash
export YOUTUBE_TRANSCRIPT_API_BASE_URL=http://localhost:8000
```

All CLI commands and local MCP calls will use that base URL.
