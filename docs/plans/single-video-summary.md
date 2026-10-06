# Plan: single-video AI summary

Status: phases 1 (backend) and 2 (frontend) implemented: OpenAI, 1 credit total for signed-in
users, anonymous users on the shared raw-transcript limit, JSON (no streaming). Second goal (pgvector over all transcripts) is
out of scope here, but step 3 below is designed to be its foundation.

## Goal

A signed-in user pastes one YouTube URL and gets, in the same flow, the transcript
plus a structured summary with clickable timestamps, in seconds.

## What similar products do

| Product | Output shape | Worth copying |
| --- | --- | --- |
| Eightify | ~8 key insights, each timestamped; focus (insightful/actionable), list or Q&A, length | Timestamped takeaways; small set of presets |
| summarize.tech | Topic segments, each with a time range and a paragraph | Sections for long videos (lectures, podcasts) |
| NoteGPT | Summary + key points + timestamped chapters + transcript + mind map | Transcript and summary side by side |
| NoteLM / BibiGPT | Summary with clickable timestamps and chapters | Clickable `&t=` links back to YouTube |

Common pattern: **TL;DR → key takeaways (timestamped) → sections/chapters
(timestamped) → full transcript below**. Users value jumping to the moment in the
video more than prose quality. Our edge: we already have reliable transcript
extraction, languages, credits, and an API/MCP surface for agents.

## Output format (v1)

```json
{
  "tldr": "2–3 sentences",
  "takeaways": [{ "text": "...", "source": 12 }],
  "sections": [{ "title": "...", "summary": "...", "source": 4 }],
  "language": "es",
  "length": "concise | detailed"
}
```

- `sections` only for videos longer than ~10 minutes. `concise` caps takeaways
  at 5; `detailed` at 10 plus longer section summaries.
- **Grounded timestamps**: the server splits captions into ~30 s blocks numbered
  `[0]…[n]` and sends those to the model. The model returns block numbers
  (`source`), never times. The server maps each number to the block's real start
  time and drops invalid numbers. This removes hallucinated timestamps, the most
  visible failure in this category.
- Output language defaults to the transcript language; user can pick another.

## Design

1. **Endpoint** `POST /summaries/single` (website, Supabase JWT) in a new
   `src/routers/summaries.py`, logic in `src/summary_service.py`.
   Body: `youtube_url`, `preferred_language`, `summary_language`, `length`.
   Returns JSON `{ transcript, transcript_language, summary, usage }`.
2. **Transcript**: reuse `youtube_service` caption fetching (same retries,
   proxy, language selection). Keep the raw segments with start times; do not
   re-implement extraction.
3. **Cache tables** (one migration, additive):
   - `video_transcripts(video_id, language, is_generated, segments jsonb, text,
     fetched_at)`, unique `(video_id, language)`.
   - `video_summaries(video_id, language, summary_language, length,
     prompt_version, model, summary jsonb, input_tokens, output_tokens,
     created_at)`, unique on everything that changes the output.
   A repeat request for a popular video costs zero LLM and proxy usage.
   `video_transcripts` is exactly the corpus pgvector will embed later.
4. **LLM call**: one request with a JSON schema (structured output), a fixed
   max output, and a timeout (~25 s). Thin provider wrapper so the model can be
   swapped by setting. Key in a dedicated env var, never in the frontend.
5. **Size limit**: a 1 h video is roughly 10–15k tokens; 3 h roughly 40k. v1
   sends the whole transcript in one call up to a cap (e.g. 60k input tokens) and
   rejects longer videos with a clear message. Map-reduce over chunks is a later
   step only if users ask for very long videos.
6. **Credits**: reserve before the LLM call, charge on success, refund on LLM
   failure/timeout (same rule we adopted for jobs: our failures are free).
   Cache hits are still charged (the user gets the value) but cost us nothing.
7. **Safety**: the transcript is untrusted text. System prompt says to treat it
   only as content to summarize and ignore instructions inside it; output is
   schema-constrained; no tool use. Log tokens per request, never transcript text.
8. **Frontend** (separate repo): a "Summarize" action on the single-video tool.
   Show stages ("Fetching transcript…", "Summarizing…"), then TL;DR, takeaways
   and sections with `youtube.com/watch?v=ID&t=123s` links, transcript below.
   Copy as Markdown and download `.md`. Next.js proxy route must allow ~60 s.

## Delivery: JSON first, streaming later

Real-time does not require streaming. Transcript fetch (2–8 s) plus a small
model on a 1 h video (~5–15 s) fits in one request with a staged spinner. A
plain JSON response avoids SSE through Vercel and nginx buffering, and keeps
schema validation simple. If latency feels slow in testing, phase 3 can return
the transcript first and stream the summary.

## Phases

1. **Backend v1**: migration, service, endpoint, credit reservation/refund,
   cache, tests with a mocked LLM (schema validity, timestamp mapping, invalid
   source ids dropped, refund on failure, cache hit skips the LLM).
2. **Frontend v1**: action, rendering, Markdown copy/download, proxy route.
3. **Then, by demand**: streaming; `POST /api/v1/summaries` + MCP tool for
   agents; focus presets (actionable, Q&A) like Eightify; chat with the video
   (needs pgvector — the second goal).

## Evaluation before launch

A fixed set of ~20 videos: short/long, lecture/podcast/tutorial, English and
non-English, auto vs manual captions, one music video. Automatic checks: valid
JSON, every timestamp exists, output language correct, latency and tokens per
video. Manual check: does each takeaway match the moment it links to.

## Decisions needed from you

1. **Model/provider**: OpenAI (SDK already in `pyproject.toml`) or Anthropic
   (e.g. Claude Haiku 4.5). Pick a fast, cheap model with structured output;
   verify current pricing when implementing.
2. **Price**: summary = +1 credit on top of the transcript, or a bundled price.
3. **Anonymous users**: none (recommended, LLM cost) or a small free allowance.
4. **Streaming in v1**: recommended no.

## Cleanup

`docs/api.md` ("Single-video summary beta") and the "Summary beta" section of
`docs/development.md` describe an earlier implementation that is no longer in
the repository. Remove or replace them when this plan is implemented.
