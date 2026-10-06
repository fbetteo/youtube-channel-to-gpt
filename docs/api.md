# API conventions

## Choose the surface

- Website endpoints: `src/transcript_api.py`, including `/download/transcript/raw`, `/channel/*`, `/playlist/*`, `/user/*`, `/payments/*`.
- Developer endpoints: `src/routers/developer_api.py`, under `/api/v1`, with their own Pydantic models and API-key dependency. Public schema: `/api/v1/openapi.json`.
- Hosted MCP: `src/routers/mcp.py`; tool calls reuse developer handlers. Keep tool schemas and response mapping aligned.
- Full schema/docs: `/internal/docs`, `/internal/redoc`, `/internal/openapi.json`, through their access check. Default FastAPI docs routes are disabled.

## Follow existing contracts

- Use Pydantic request validation and existing shared formatting/language models and helpers.
- Use `HTTPException` for HTTP failures; preserve deliberate status codes in broader exception handling. Backend errors generally use `detail`; frontend proxies often map it to `error`.
- Responses vary: text for raw transcripts, JSON for metadata/progress, ZIP for bulk downloads, JSON-RPC for MCP. Preserve the surface's format.
- Bulk work returns a job ID and uses background dispatch/polling. Persist accepted work before scheduling it; see [jobs-and-credits.md](jobs-and-credits.md).
- Channels/playlists share service logic but keep separate paths and request fields. Website playlist selection uses `playlist_name`; developer requests have their own models. Inspect consumers before renaming fields.
- Keep blocking SDK/network calls off async request paths using existing executor or `asyncio.to_thread` patterns where applicable.
- Reuse uploads-playlist / `playlistItems.list` discovery and batched metadata helpers; avoid introducing expensive per-video searches.

## Single-video summary

Website `POST /summaries/single` (`src/transcript_api.py`, logic in
`src/summary_service.py`) returns JSON: transcript plus an OpenAI summary. Not
exposed by the developer API, CLI, or MCP yet. Body:

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "preferred_language": null,
  "summary_language": null,
  "length": "concise"
}
```

Languages are nullable codes validated like `preferred_language`; the summary
defaults to the selected caption language. `length` is `concise` or `detailed`.
The response carries `transcript`, `transcript_language`, `transcript_type`, and
`summary` = `{tldr, takeaways[{text, start_seconds, timestamp, url}],
sections[{title, summary, start_seconds, timestamp, url}]}` (sections only for
videos of 10+ minutes), plus `summary_language`, `length`, `model`,
`prompt_version`, `cached`, and token `usage`.

Timestamps are grounded: captions are grouped into ~30 s numbered blocks, the
model cites block numbers under a strict JSON schema, and the server maps them to
real start times. An invalid citation keeps its text with null time fields.

Access and cost: optional JWT like the raw route. One credit covers a video's
transcript and summary: a signed-in user charged for that video (raw download or
summary) in the last 24 h (`video_access`) is not charged again. Otherwise 1 credit,
refunded on our failures (transcript timeout 504, provider error 502, video over
`SUMMARY_MAX_INPUT_CHARS` 422, unexpected 500). A video without captions returns
400 and stays charged, matching raw downloads. Anonymous users consume the same
in-memory limiter as `/download/transcript/raw`. 503 before charging when
`SUMMARY_OPENAI_API_KEY` is unset. Transcripts and summaries are cached in
`video_transcripts` / `video_summaries` when those tables exist; cache errors
fall back to live work. Raw downloads share the same transcript cache
(`summary_service.load_transcript`). Plan and next phases: `docs/plans/single-video-summary.md`.

## Contract changes

Check [frontend consumers](frontend.md), `cli/src/cli.ts`, and MCP tools. Update public usage docs (`README_TRANSCRIPT_API.md`, `docs/agent.md`, `llms.txt`) when usage changes. Playlist batch example types live in `docs/frontend/playlistBatchApi.contract.ts`.

Read [auth.md](auth.md) for endpoint-specific access. Commented routes and historical Copilot examples are not the current API specification.
