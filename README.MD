# YouTube Transcript Agent

Turn any YouTube video, playlist, or channel into agent-ready transcript context.

This package provides:

- `ytx`, a CLI for transcripts, channel jobs, playlist jobs, polling, and ZIP downloads
- a local stdio MCP server for coding agents and desktop assistants
- docs for using the hosted `/mcp` endpoint and Developer API

## Install

Run without installing:

```bash
npx youtube-transcript-agent --help
```

Or install globally:

```bash
npm install -g youtube-transcript-agent
ytx --help
```

## Authentication

The CLI reads auth in this order:

1. `YOUTUBE_TRANSCRIPT_API_KEY`
2. local config created by `ytx auth set-key`

Set an API key:

```bash
npx youtube-transcript-agent auth set-key <api_key>
```

Or use environment variables:

```bash
export YOUTUBE_TRANSCRIPT_API_KEY=<api_key>
```

PowerShell:

```powershell
$env:YOUTUBE_TRANSCRIPT_API_KEY="<api_key>"
```

## CLI Usage

Get account credits:

```bash
npx youtube-transcript-agent credits
```

Fetch one transcript:

```bash
npx youtube-transcript-agent transcript FOp280ZAxhg --timestamps
```

Download channel transcripts:

```bash
npx youtube-transcript-agent channel download "@mkbhd" --max 25 --wait --output channel.zip
```

Download playlist transcripts:

```bash
npx youtube-transcript-agent playlist download <playlist_url_or_id> --max 25 --wait --output playlist.zip
```

Poll a job:

```bash
npx youtube-transcript-agent jobs status <job_id>
```

Download a completed job:

```bash
npx youtube-transcript-agent jobs download <job_id> --output transcripts.zip
```

## Local MCP

Start the local stdio MCP server:

```bash
npx youtube-transcript-agent mcp
```

This command waits for JSON-RPC messages on stdin. Agent clients should launch it with:

```text
Command: npx
Args: youtube-transcript-agent mcp
Env:
  YOUTUBE_TRANSCRIPT_API_KEY=<api_key>
```

Available MCP tools:

- `get_transcript`
- `get_channel_info`
- `list_channel_videos`
- `start_channel_job`
- `start_playlist_job`
- `get_job_status`
- `get_download_url`

Manual smoke test:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | npx youtube-transcript-agent mcp
```

## Hosted MCP

If your API server is deployed with hosted MCP enabled:

```text
Endpoint: https://api.youtubetranscripts.fbetteo.com/mcp
Auth: Authorization: Bearer <api_key>
```

List tools:

```bash
curl -X POST "https://api.youtubetranscripts.fbetteo.com/mcp" \
  -H "Content-Type: application/json" \
  --data-raw '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Call a tool:

```bash
curl -X POST "https://api.youtubetranscripts.fbetteo.com/mcp" \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  --data-raw '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_transcript","arguments":{"video_url":"FOp280ZAxhg"}}}'
```

## Agent Skill

The repo includes an installable skill template at:

```text
agent-skill/skill.md
```

For distribution, publish that file as the root `skill.md` in a separate public skill repo.

## Troubleshooting

## Advanced API Host Override

Most users do not need to set an API URL. The package defaults to the hosted production API.

Only set a custom base URL for local development, staging, self-hosted deployments, or debugging:

```bash
export YOUTUBE_TRANSCRIPT_API_BASE_URL=http://127.0.0.1:8000
```

Or save it in local CLI config:

```bash
npx youtube-transcript-agent auth set-key <api_key> --base-url http://127.0.0.1:8000
```

## Troubleshooting

If `fetch failed` appears and the API logs show nothing, the CLI is not reaching the API host. Check:

```bash
echo "$YOUTUBE_TRANSCRIPT_API_BASE_URL"
echo "$API_BASE_URL"
cat ~/.youtube-transcript-agent/config.json 2>/dev/null
```

The CLI base URL priority is:

1. `YOUTUBE_TRANSCRIPT_API_BASE_URL`
2. `API_BASE_URL`
3. saved local config
4. package default

On Windows PowerShell, quote channel handles:

```powershell
node cli/dist/cli.js channel download "@mkbhd" --max 2 --wait
```

PowerShell aliases `curl` to `Invoke-WebRequest`; use `curl.exe` or `Invoke-RestMethod`.
