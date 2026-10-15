# Website single video

The website's main tool turns one YouTube URL into a plain-text transcript. Signed-in users pay 1 credit; anonymous visitors get a small rate-limited allowance. The summary variant needs an OpenAI key.

## Sub-features

- `raw-signed-in` returns transcript text and deducts 1 credit.
- `raw-anonymous` returns transcript text without credits, within the anonymous limit.
- `raw-anonymous-limit` refuses anonymous requests past the limit.
- `raw-timestamps` honours `include_timestamps` (default true).
- `summary-unavailable` returns 503 from `/summaries/single` when no OpenAI key is configured.

## How to get to it (user POV)

- Website home page or tools pages: paste a URL and download. The frontend proxy calls `POST /download/transcript/raw {"youtube_url": ..., "include_timestamps": ...}` with the user's JWT when signed in.
- Website summary tool: `POST /summaries/single`.

## Driving it with verify.py

Preconditions:

- `$V doctor` exits 0.
- A fresh UUID that has read its credits once (100 credits).

- **Signed-in download.** Run `MSYS_NO_PATHCONV=1 $V call POST /download/transcript/raw --user <uuid> --json '{"youtube_url":"https://www.youtube.com/watch?v=jNQXAC9IVRw","include_timestamps":false}' --run website`. 200 with transcript text mentioning the elephants.
- **Charged.** Run `MSYS_NO_PATHCONV=1 $V call GET /user/credits --user <uuid> --run website`. `credits: 99`.
- **Timestamps.** Repeat the download with `"include_timestamps":true`. The text has time markers. Credits drop again.
- **Anonymous.** Run the download without `--user`. 200 with the transcript; no user row is created (`$V sql "SELECT count(*) FROM user_credits"` is unchanged).
- **Anonymous limit.** Repeat the anonymous download until it is refused. Record the request count and the refusal status and `detail`.
- **Summary unavailable.** Run `MSYS_NO_PATHCONV=1 $V call POST /summaries/single --user <uuid> --json '{"youtube_url":"jNQXAC9IVRw"}' --run website`. Status 503, and credits unchanged.

## Gotchas

- Anonymous limits are in memory and per process: restarting the instance resets them. The window and count are split between `src/rate_limiter.py` and `check_anonymous_rate_limit`; read both before asserting a number.
- Without `X-Proxy-Secret`, the client IP is the last `X-Forwarded-For` hop or `127.0.0.1`, so all anonymous calls from this machine share one allowance.
- Real YouTube is fetched through the `.env` proxies; a proxy failure refunds the signed-in credit. Check the balance rather than assuming a charge.
