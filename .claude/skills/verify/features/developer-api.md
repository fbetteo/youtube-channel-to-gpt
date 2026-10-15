# Developer API and CLI

Developers create API keys on the website and use them with the `/api/v1` REST API or the `ytx` CLI. Each single-video transcript costs 1 credit from the same balance as the website; failures on our side are refunded.

## Sub-features

- `key-create` returns a full `yt_live_` key once, plus its prefix.
- `key-list` lists the user's keys without the secret; `key-revoke` deactivates one.
- `dev-credits` returns the balance and per-key usage counters.
- `dev-single` returns a transcript and charges 1 credit.
- `dev-auth` rejects a missing or revoked key.
- `cli-credits` and `cli-transcript` do the same through `ytx`.

## How to get to it (user POV)

- Website API keys page: `POST /user/api-keys`, `GET /user/api-keys`, `DELETE /user/api-keys/{key_id}` (JWT).
- REST: `GET /api/v1/account/credits`, `POST /api/v1/transcripts/single` with `X-API-Key`.
- CLI: `ytx credits`, `ytx transcript <video>` (built to `node cli/dist/cli.js`).

## Driving it with verify.py

Preconditions:

- `$V doctor` exits 0.
- A fresh UUID that has read its credits once: `MSYS_NO_PATHCONV=1 $V call GET /user/credits --user <uuid> --run devapi` (100 credits).

- **Create key.** Run `MSYS_NO_PATHCONV=1 $V call POST /user/api-keys --user <uuid> --json '{"name":"verify"}' --run devapi`. Status 200 with `api_key` starting `yt_live_` and the message `Save this API key now`.
- **List keys.** Run `MSYS_NO_PATHCONV=1 $V call GET /user/api-keys --user <uuid> --run devapi`. The key appears with its `key_prefix` and `is_active: true`.
- **Credits by key.** Run `MSYS_NO_PATHCONV=1 $V call GET /api/v1/account/credits --api-key <key> --run devapi`. 200 with `credits: 100` and `api_key_name: verify`.
- **Single transcript.** Run `MSYS_NO_PATHCONV=1 $V call POST /api/v1/transcripts/single --api-key <key> --json '{"video_url":"jNQXAC9IVRw"}' --run devapi`. 200 with `video_id: jNQXAC9IVRw`, non-empty `transcript`, `credits_used: 1`.
- **Charged.** Run the credits-by-key command again. `credits: 99`, `total_credits_via_api: 1`.
- **CLI.** After `npm ci && npm run build`, run `YOUTUBE_TRANSCRIPT_API_BASE_URL=http://127.0.0.1:8765 YOUTUBE_TRANSCRIPT_API_KEY=<key> node cli/dist/cli.js credits --json`. Exit 0 and the same `credits` value. Save stdout to the run folder as evidence.
- **Revoke.** Run `MSYS_NO_PATHCONV=1 $V call DELETE /user/api-keys/<key_id> --user <uuid> --run devapi`, then the credits-by-key command. The second call is rejected (401 or 403).

## Gotchas

- `POST /user/api-keys` returns 500 for a user who never read their credits: the `user_credits` row does not exist yet (foreign key). Always do the first-visit step.
- Transcripts come from real YouTube through the `.env` proxies. A proxy block or a video without captions returns an error and refunds or charges per [jobs-and-credits](../../../../docs/jobs-and-credits.md); check credits after every call instead of assuming.
- The developer single path defaults to language `en`, the website path to `auto`. The shared S3 cache is disabled in this instance, so every call is a live fetch and the server log shows `Transcript cache miss ... ClientError` and `cache write failed ... InvalidAccessKeyId`.
- Without `YOUTUBE_TRANSCRIPT_API_BASE_URL` the CLI calls the hosted production API.
