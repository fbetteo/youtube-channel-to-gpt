---
name: verify
description: Launch and drive an isolated local instance of the YouTube transcript API (FastAPI backend in this repo) over HTTP to prove behavior with evidence. Covers website routes (JWT), the /api/v1 developer API (API keys), hosted MCP, and Stripe test-mode subscription billing. Use after changing backend behavior, before claiming a feature works, or when asked to run, verify, or demo the API.
---

# Verify the transcript API

The user-facing surface of this repo is HTTP: website routes in `src/transcript_api.py` (Supabase JWT), the developer API under `/api/v1` (`X-API-Key`), and hosted MCP at `/mcp` (JSON-RPC, `Authorization: Bearer <api_key>`). The `ytx` CLI (`cli/`) is a thin client over `/api/v1`. The Next.js frontend lives in another repo and is not driven here.

`.env` points at **production** Postgres, S3, Lambda and live Stripe. Never verify with the plain dev command. Every step below goes through the helper, which starts an isolated instance:

| Dependency | What the instance uses |
| --- | --- |
| Postgres | throwaway Docker container `ytx-verify-pg` on port 55432, empty at launch |
| Website auth | local JWT secret; the helper mints tokens for any user UUID |
| Stripe | `STRIPE_SECRET_KEY_TEST_TRANSCRIPTS` from `.env` (test mode only); local webhook secret, so the helper signs webhooks itself |
| AWS (S3 cache, Lambda, SQS) | deliberately invalid credentials: cache reads/writes fail and fall back to live fetches, Lambda dispatch fails |
| OpenAI (summaries) | unset, so `/summaries/single` returns 503 |
| YouTube | **real** network, through the proxies in `.env` |

Helper: `.claude/skills/verify/verify.py`, run with `poetry run python` from the repo root. In Git Bash, prefix commands that pass URL paths with `MSYS_NO_PATHCONV=1` (Git Bash rewrites `/user/credits` into a Windows path; the helper refuses such paths). In cmd or PowerShell no prefix is needed. Examples below set `V="poetry run python .claude/skills/verify/verify.py"` in Git Bash.

## Launch

Prerequisites: Docker Desktop running (`docker version` shows a Server), `poetry install` done.

```bash
poetry run python .claude/skills/verify/verify.py up
```

It refuses to start when an instance is recorded and alive, when port 8765 or 55432 is taken, or when the container already exists. Only one instance per machine: ports and container name are fixed. Never point it at, or kill, a server it did not start.

Steps it runs: start `pgvector/pgvector:pg15` (production is Supabase Postgres 15 with pgvector) and wait for a real query (not just `pg_isready`); create `user_credits` (production gets it from Supabase, no repo script does); run `db_youtube_transcripts/schema`, `schema_api_keys` and every `migration_*` as modules; start uvicorn on `127.0.0.1:8765` with the isolated environment. Ready when it prints `Ready: http://127.0.0.1:8765 (probe user has 100 welcome credits)`. That probe proves auth, the database and the schema together, not just that the port answers. It takes about 30 s.

## Doctor

Run first whenever anything looks off. It only reads (the probe user already exists after `up`):

```bash
poetry run python .claude/skills/verify/verify.py doctor
```

Each line is `OK` or `FAIL`: state file present, API process alive, port 8765 owned by that PID, container running, `user_subscriptions` and `api_keys` tables present, `GET /user/credits` with a minted JWT returns 200, Stripe key is test mode. Exit code 1 if any check fails. If the API died, read the `server_log` path in `scripts/e2e_results/verify/instance.json`, then `down` and `up`.

## Drive

All commands take `--run NAME` to group evidence under `scripts/e2e_results/verify/NAME/`.

```bash
# HTTP as a signed-in website user (any UUID; first credits call creates the user with 100 credits)
MSYS_NO_PATHCONV=1 $V call GET /user/credits --user 11111111-1111-4111-8111-111111111111 --run myrun
MSYS_NO_PATHCONV=1 $V call POST /user/api-keys --user <uuid> --json '{"name":"verify"}' --run myrun
# HTTP as a developer (key from the create call above)
MSYS_NO_PATHCONV=1 $V call POST /api/v1/transcripts/single --api-key yt_live_... --json '{"video_url":"jNQXAC9IVRw"}' --run myrun
# Raw JWT for curl or other clients
$V jwt <uuid>
# Observe database state (SELECT/WITH only; change state through the API)
$V sql "SELECT user_id, credits, permanent_credits FROM user_credits" --run myrun
# Stripe test-mode subscription lifecycle: real Stripe objects, events delivered signed to /payments/webhook
$V stripe-subscribe --user <uuid> [--lookup-key transcript_monthly_1000] --run myrun
$V stripe-advance --days 32 --run myrun
$V stripe-cancel --run myrun
$V stripe-forward --run myrun   # deliver events Stripe produced late
```

`call` prints the status and body. Website errors use `{"detail": ...}`. Per-feature recipes, stable paths and expected end states are in [features/README.md](features/README.md). Drive every entry point a feature file lists before calling the feature proven.

For the CLI, build once (`npm ci && npm run build`), then point it at the instance:

```bash
YOUTUBE_TRANSCRIPT_API_BASE_URL=http://127.0.0.1:8765 YOUTUBE_TRANSCRIPT_API_KEY=yt_live_... node cli/dist/cli.js credits
```

## Evidence

Every `call`, `sql` and Stripe step appends a JSON line (request, status, response, rows, webhook results) to `scripts/e2e_results/verify/<run>/evidence.jsonl`. The server log is `scripts/e2e_results/verify/server-<timestamp>.log`. Both are gitignored and survive `down`. API keys and tokens in evidence belong to the throwaway database only.

Proof standards:

- Exercise the real user path: the HTTP routes the website, CLI or MCP call. Do not call `CreditManager` or SQL setters to create state; `sql` is for observing.
- Capture the action and the resulting state: the request, its response, and a follow-up read (`GET /user/credits`, `sql`) that shows the change.
- Verify side effects alongside responses: credit balances, `user_subscriptions` rows, `stripe_checkout_credits` idempotency rows, API-key usage counters, server-log lines.
- Mocks only at existing production boundaries. Stripe runs in real test mode. `stripe-subscribe` creates the subscription Checkout would create (same metadata) because hosted Checkout needs a browser; credits still arrive only through the signed webhook.
- Isolation was observed, not assumed: the server log shows `Transcript cache miss ...: ClientError` and `cache write failed ... InvalidAccessKeyId`, and no production S3 write happens. Re-check those lines if you change `instance_env()` in the helper.

## Cleanup

```bash
poetry run python .claude/skills/verify/verify.py down
```

Stops the API process tree by recorded PID (never by name), deletes the Stripe test clocks this instance created (which removes their test customers and subscriptions), removes the `ytx-verify-pg` container only if it carries the `ytx-verify=1` label, and deletes `instance.json`. Evidence and server logs stay in `scripts/e2e_results/verify/`. Run `down` after every failed `up` too.

## Helper reference

`verify.py up | doctor | down | jwt USER_ID | call METHOD PATH [--user UUID | --api-key KEY | --bearer KEY] [--json BODY] [--run NAME] | sql QUERY [--run NAME] | stripe-subscribe --user UUID [--lookup-key KEY] [--run NAME] | stripe-advance [--days N] [--run NAME] | stripe-cancel [--run NAME] | stripe-forward [--run NAME]`. The constants at the top of the file (ports, container name, secrets) and `instance_env()` define the isolation.
