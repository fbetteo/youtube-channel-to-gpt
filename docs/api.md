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

## Single-video summary beta

Website `POST /summaries/single` requires a Supabase bearer JWT and returns
`text/event-stream`. It is separate from the plain-text transcript route and is
not currently exposed by the developer API, CLI, or MCP. Body:

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "preferred_language": null,
  "summary_language": "es",
  "summary_length": "concise"
}
```

Languages are nullable BCP-47 codes. Summary language defaults to the actual
selected caption language; length is `concise` or `detailed`. Events, in order:
`status`, `transcript_ready`, `status`, `summary_ready`, `done`. Heartbeat comments
keep the stream active during work. `transcript_ready` includes text with source
timestamps, video ID, language/type, and normalized transcript SHA-256. A summary
contains overview, takeaways, optional sections, limitations, server-resolved
timestamp sources, output language, preset, model/version, hash, and token usage.

Preflight failures use HTTP status codes: 400 URL, 401/403 authentication, 422
request fields, 429 quota/concurrency (with Retry-After), 503 missing provider key.
After streaming begins, extraction failure emits `error` and `done: failed`;
generation failure emits `summary_failed` and `done: partial`, retaining the
transcript. Success terminates with `done: completed`. Clients must not treat a
200 or an interrupted stream without `done` as success. Errors contain safe
`code`/`message` fields; provider request bodies are not exposed.

This beta does not persist results or deduct download credits. Allowances and
concurrency limits are process-local; each admitted attempt counts, including
failure/cancellation. Retries are new explicit requests, with no automatic SDK
retry. Input is estimated with the model's `tiktoken` encoding, including captions,
source IDs, prompt, schema, and 256 tokens for provider framing. Successful results
include `estimated_input_tokens`; actual billed prompt tokens remain in `usage`.
The local estimate is not an exact provider count. Oversize failures state the
estimate and limit in their message, and oversized transcripts are never truncated.
Extraction has a 30-second timeout; LLM generation defaults to 20 seconds; total
stream processing is bounded to 52 seconds. Browser disconnect cancels async
work; an already-running synchronous caption-fetch thread may finish afterward.

## Contract changes

Check [frontend consumers](frontend.md), `cli/src/cli.ts`, and MCP tools. Update public usage docs (`README_TRANSCRIPT_API.md`, `docs/agent.md`, `llms.txt`) when usage changes. Playlist batch example types live in `docs/frontend/playlistBatchApi.contract.ts`.

Read [auth.md](auth.md) for endpoint-specific access. Commented routes and historical Copilot examples are not the current API specification.
