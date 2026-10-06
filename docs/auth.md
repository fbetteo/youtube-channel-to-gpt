# Authentication and authorization

| Surface | Mechanism / source |
| --- | --- |
| Authenticated website routes | Supabase JWT bearer; `validate_jwt` in `src/transcript_api.py` |
| Single-video website download | Optional JWT via `get_user_or_anonymous`, otherwise anonymous limiting |
| Single-video website summary | Required JWT via `validate_jwt`; requires a nonempty `sub` |
| Developer API | `X-API-Key`; `src/api_key_auth.py` |
| Hosted MCP tool calls | `Authorization: Bearer <api_key>`; `src/routers/mcp.py` |
| API-key management | Supabase JWT routes under `/user/api-keys` |
| Stripe webhook | Stripe signature verification on raw request body |

Website JWT validation uses HS256 and `SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS`, with audience verification disabled. User identity comes from `sub`. Missing JWT returns 401; invalid JWT returns 403. Optional auth instead treats invalid JWT as anonymous. Preserve those distinctions unless deliberately changing the contract.

Developer keys use a `yt_live_` prefix, are stored as SHA-256 hashes, and are returned in full only at creation. Validation checks existence, active status, and expiry and records usage. Revocation is scoped to the owner. Tier settings exist; check handler enforcement before claiming a limit applies.

## Working rules

- Reuse the dependency for the target surface; Supabase JWTs and developer keys are different credentials.
- Authentication identifies a user; user-owned operations also need ownership checks. Follow download/cancel handlers rather than trusting a supplied user ID.
- Keep full tokens, keys, and secrets out of new logs/docs.
- Coordinate sessions with frontend Supabase browser/server/middleware clients; see [frontend.md](frontend.md).

## Current inconsistencies

Some website status/discovery endpoints have no auth dependency. Legacy `/internal/job/...` HTTP callbacks also have no auth dependency in their current handlers; an `internal` path does not enforce access control. Completion results ignore the reported `s3_key` and derive `{user_id}/{job_id}/{video_id}.txt` from the job owner, so a caller cannot point a job at another user's object. Set `ENABLE_HTTP_RESULT_CALLBACKS=false` once SQS delivery is configured to make these routes return 404; the worker then reports results only through SQS. Do not describe them as protected or copy that pattern for new private operations. SQS access uses AWS credentials/IAM.

Anonymous limiting is in-memory and process-local. Window/request limits are split between `src/rate_limiter.py` and `check_anonymous_rate_limit` in the API; inspect both rather than assuming the older “3 per hour” guidance.
