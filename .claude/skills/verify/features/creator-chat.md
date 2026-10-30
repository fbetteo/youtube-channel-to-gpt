# Creator chat

Visitors pick a creator and chat with an LLM that answers only from that creator's video transcripts, citing numbered video moments with timestamp links. It is free: anonymous visitors get a few messages a day per IP, then must sign in; signed-in users get a higher daily cap.

## Sub-features

- `GET /creators` lists enabled creators with `video_count`; `GET /creators/{slug}` returns one, 404 when unknown or not enabled.
- `POST /creators/{slug}/chat` streams Server-Sent Events: `sources` (search query and numbered sources with YouTube URLs), many `delta` (answer text), then `done` (`usage`, `remaining`) or `error`.
- Follow-ups: with `history`, the message is rewritten into a standalone question before search; `sources.query` shows it.
- Limits: `CREATOR_CHAT_ANONYMOUS_DAILY` (default 3) per client IP, then 401 `Sign in to keep chatting.`; `CREATOR_CHAT_USER_DAILY` (default 50) per user, then 429. Validation errors (empty message) are 422.

## How to get to it (user POV)

The website's creators page lists the cards; picking one opens its chat. The Next.js proxy forwards the visitor's IP (`X-Client-IP` with `X-Proxy-Secret`) and, when signed in, the Supabase bearer token.

## Driving it with verify.py

```bash
$V up --openai                 # real SUMMARY_OPENAI_API_KEY: each message costs ~$0.002
$V seed-creator dankulkov      # smallest creator; copies its production chunks (read-only) and enables it
MSYS_NO_PATHCONV=1 $V call GET /creators --run creator-chat
curl -sN -H "Content-Type: application/json" -H "X-Proxy-Secret: ytx-verify-proxy" -H "X-Client-IP: 203.0.113.7" \
  http://127.0.0.1:8765/creators/dankulkov/chat -d '{"message":"How should I price my mobile app?"}'
```

Repeat the curl from the same `X-Client-IP` until the 4th message returns 401; a different IP still gets 200. For signed-in, send `Authorization: Bearer $($V jwt <uuid>)` and check `remaining` counts down from 49. For a follow-up, send `history` with the previous user and assistant turns and check `sources.query` is a standalone question.

## Gotchas

- `curl -N` (or a streaming client) is needed to see events arrive over time; `$V call` waits for the whole body.
- Without `--openai` the chat returns 503; `GET /creators` still works.
- Limits are in memory: they reset when the API restarts and are counted per API process.
- `seed-creator` needs the creator added and indexed in production first (`scripts/creators.py add` and `index`).
