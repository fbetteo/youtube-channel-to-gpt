# Turn YouTube Research Into Agent Context With MCP

Agents need tools, not tabs. YouTube Transcript Agent exposes transcript workflows through MCP so an agent can fetch video context, start larger channel or playlist jobs, and poll until results are ready.

Hosted MCP:

```text
Endpoint: https://your-api.example.com/mcp
Auth: Authorization: Bearer <api_key>
```

Local MCP:

```bash
npx youtube-transcript-agent mcp
```

## Tools

- `get_transcript`: fetch one video transcript.
- `get_channel_info`: validate a channel.
- `list_channel_videos`: preview videos before spending credits.
- `start_channel_job`: create an async channel transcript job.
- `start_playlist_job`: create an async playlist transcript job.
- `get_job_status`: poll progress.
- `get_download_url`: get the ZIP URL after completion.

## Why MCP

Direct HTTP works, but it wastes prompt space and can expose secrets inside chat context. MCP gives agents a smaller, stable tool surface.

For large outputs, the MCP server returns metadata and download links. The CLI handles ZIP downloads:

```bash
npx youtube-transcript-agent jobs download <job_id> --output transcripts.zip
```

That keeps the agent focused on planning and analysis while the transcript system handles retrieval.
