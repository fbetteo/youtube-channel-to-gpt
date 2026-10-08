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
- Discovery uses flat yt-dlp listings (`youtube_service._fetch_all_channel_videos`: the channel's `videos`, `shorts`, and `streams` tabs in parallel, typed `video`/`short`/`live`). Avoid adding per-video metadata requests to discovery paths.

## Bulk download formats

`GET /channel/download/results/{job_id}` and `GET /api/v1/jobs/{job_id}/download` take `?format=txt|srt|vtt|json` (default `txt`); `POST /download-all-content` takes the same as a body field `format`. The CLI passes `--format`. Every format, `txt` included, is rendered (`src/transcript_formats.py`) from the `.json` segments file the worker stores next to the `.txt`; `txt` applies the job's `formatting_options` headers/timestamps with `job_videos.title`/`view_count`. `json` follows the same options: header flags pick the `title`/`video_id`/`url`/`view_count` fields (`language` and `is_generated` are always present), and `include_timestamps` gives timed `segments` instead of one `text` string. `srt`/`vtt` are always timed and ignore the options. Non-txt formats are one file per video and ignore `concatenate_all`. The website download picker previews these shapes (`lib/download-options.ts` in the frontend); keep both in sync. Videos from jobs created before segment storage existed have no `.json` and are served from the stored `.txt` (also the fallback if a `txt` render cannot read the segments).

Download-time overrides: the same endpoints accept optional `include_timestamps`, `include_video_title`, `include_video_id`, `include_video_url`, `include_view_count`, and `concatenate_all` (query parameters on the GETs, body fields on `/download-all-content`, which already had `concatenate_all`). Omitted/null keeps the job's saved `formatting_options` (`transcript_formats.merge_formatting_options`; ZIP builders apply it via `youtube_service.apply_option_overrides`). They change rendered videos only: stored-`.txt` videos keep their original text, though concatenation still applies to them. No credits are charged for re-downloads. `/user/download-history` items include `formattingOptions` (the job's saved options) so the website can prefill its download picker. While the worker still writes both files, `render_text` must match its `.txt` byte for byte: `tests/test_transcript_segments_cache_and_formats.py` checks this, and `scripts/compare_rendered_txt.py` (read-only, needs DB/S3 config) compares real stored jobs.

## Channel video listing

`GET /api/v1/channels/{channel}/videos?limit=N` (default 100, max 2000; CLI `--limit`, MCP `limit`) returns up to `limit` videos plus `has_more`. Order is the channel job order: long-form, then Shorts, then streams, each newest first, so `limit=N` previews what `max_videos=N` downloads. Each tab is capped in yt-dlp (`playlistend`, `get_all_channel_videos(max_per_tab=...)`) so large channels answer well inside the 60 s nginx timeout; an uncapped full listing of a big channel does not. Developer channel jobs with `max_videos` use the same cap during discovery. `video_count` from `/channels/{channel}/info` is usually null (yt-dlp gets no count for channel tabs) and `title` is the channel name, not the tab title.

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
`SUMMARY_OPENAI_API_KEY` is unset. Transcripts use the shared transcript cache
(below); summaries are cached in `video_summaries` when that table exists. Cache
errors fall back to live work. Plan and next phases: `docs/plans/single-video-summary.md`.

## Shared transcript cache

`src/transcript_cache.py`. One cache for every path: website raw downloads and
summaries, developer `/api/v1/transcripts/single` (so MCP `get_transcript` and
`ytx transcript`), and the Lambda worker. Segments live in S3 at
`transcript-cache/{video_id}/{requested_language|auto}.json`, in the worker's
format (`tests/test_transcript_cache.py` checks they match). Lookups read S3
directly, so worker-written entries hit immediately. Postgres `transcript_cache`
indexes entries (language, size, `source`, `hit_count`, `last_hit_at`, no
segments) for batch lookups and stats; a hit or write upserts it. Single paths go
through `summary_service.load_transcript` / `load_transcript_within`: on timeout
the fetch keeps running and still fills the cache, so a retry is usually a hit.
`TRANSCRIPT_CACHE_MAX_AGE_DAYS` means the same on the API as on the worker.

Keys use the requested language: website single requests send none (`auto`);
developer single requests and developer jobs default to `en`, so those two share
entries but differ from website `auto` entries.

Developer single: 1 credit, refunded on timeout (504 after 45 s, below nginx's
60 s), missing captions (400) or errors (500); an invalid URL charges nothing. It
records `video_access`, so a website summary within 24 h is free. `title` is
null until titles are cached.

Setup: `db_youtube_transcripts/migration_add_transcript_cache.py`, then
`scripts/backfill_transcript_cache.py` (dry run; `--apply` copies old
`video_transcripts` rows to S3 and indexes existing S3 entries). The old
`video_transcripts` table is no longer read or written.

## Contract changes

Check [frontend consumers](frontend.md), `cli/src/cli.ts`, and MCP tools. Update public usage docs (`README_TRANSCRIPT_API.md`, `docs/agent.md`, `llms.txt`) when usage changes. Playlist batch example types live in `docs/frontend/playlistBatchApi.contract.ts`.

Read [auth.md](auth.md) for endpoint-specific access. Commented routes and historical Copilot examples are not the current API specification.
