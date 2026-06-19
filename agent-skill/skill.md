# YouTube Transcript Agent Skill

Use this skill when the user wants to turn YouTube videos, playlists, or channels into transcript context for research, summaries, knowledge bases, custom GPTs, or long-context analysis.

## Install The CLI

If `ytx` is not available, run commands with `npx`:

```bash
npx youtube-transcript-agent --help
```

If the user has an API key, prefer an environment variable:

```bash
export YOUTUBE_TRANSCRIPT_API_KEY=<api_key>
```

On Windows PowerShell:

```powershell
$env:YOUTUBE_TRANSCRIPT_API_KEY="<api_key>"
```

If the API is local or staging:

```bash
export YOUTUBE_TRANSCRIPT_API_BASE_URL=http://localhost:8000
```

## Recipes

### Summarize A Video

```bash
npx youtube-transcript-agent transcript <video_url_or_id> --timestamps
```

Read the returned transcript and produce the requested summary, outline, claims, quotes, or action plan.

### Build A Knowledge Base From A Channel

```bash
npx youtube-transcript-agent channel download <channel> --max 100 --wait --output transcripts.zip
```

Use a smaller `--max` first if the channel is large. After the ZIP is ready, inspect or extract the files before answering.

### Download Playlist Transcripts

```bash
npx youtube-transcript-agent playlist download <playlist_url_or_id> --wait --output playlist-transcripts.zip
```

Use this for courses, podcast series, conference tracks, and curated research playlists.

### Poll Until A Job Is Ready

```bash
npx youtube-transcript-agent jobs status <job_id>
```

When `download_ready` is true:

```bash
npx youtube-transcript-agent jobs download <job_id> --output transcripts.zip
```

## Guidance

- Do not expose API keys in chat output.
- Use the CLI for large ZIP downloads.
- Use the hosted MCP or local MCP for tool-calling agents.
- Use direct API calls only when the user explicitly wants HTTP examples or the CLI is unavailable.
