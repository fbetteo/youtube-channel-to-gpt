# Development and runtime

## Commands

Run from the repo root. Python uses Poetry (`pyproject.toml`, `poetry.lock`), requiring `>=3.12,<3.14`. Keep the existing manager and lockfile.

```bat
poetry install
poetry run uvicorn src.transcript_api:app --reload --host 127.0.0.1 --port 8000
poetry run pytest tests
```

CLI/local MCP uses npm and Node >=18:

```bat
npm ci
npm run build
node cli/dist/cli.js --help
```

For local CLI requests, use `set YOUTUBE_TRANSCRIPT_API_BASE_URL=http://127.0.0.1:8000` in cmd; otherwise the CLI defaults to the hosted API.

## Validation

Choose affected files in `tests/`: agent interfaces, playlist lifecycle, discovery persistence, video metadata reliability, language selection, and formatting/localized titles have dedicated coverage.

Avoid bare `pytest` discovery for routine checks: root and `src/` contain manual scripts that can contact AWS/YouTube or require configured services. Do not treat live download/payment scripts as unit tests. Build the CLI for TypeScript changes; check frontend changes in its repo.

Live Developer API walkthrough (spends real credits: 1 + `--max-videos`): `scripts/e2e_api_workflow.py --channel @handle --max-videos 5 [-i] [--label tag]` uses `YOUTUBE_TRANSCRIPT_API_KEY` and `YOUTUBE_TRANSCRIPT_API_BASE_URL` like the CLI. It covers credits, channel info and listing, a single video, a channel job with polling, the ZIP download, and a credit check. Results go to the gitignored `scripts/e2e_results/` (a per-run folder with `report.json`, `run.log` and the ZIP, plus a shared `runs.csv` for comparing runs).

## Configuration and operations

### Single-video summaries

Set `SUMMARY_OPENAI_API_KEY` on the Python server to enable `POST
/summaries/single` (a separate variable from any legacy assistant key; never put
it in the frontend). Optional: run
`poetry run python db_youtube_transcripts/migration_add_video_summary_cache.py`
to create the additive tables (`video_transcripts`, `video_summaries`, `video_access`); without them every request is live and a summary after a paid transcript costs a second credit.

Set the same random value as `PROXY_SHARED_SECRET` here and `TRANSCRIPT_API_PROXY_SECRET` in the frontend so anonymous limits use the visitor's IP.

| Setting | Default | Purpose |
| --- | --- | --- |
| `SUMMARY_MODEL` | `gpt-4.1-mini` | Must support strict JSON-schema output; part of the cache key |
| `SUMMARY_TIMEOUT_SECONDS` | `25` | Provider deadline (no SDK retries) |
| `SUMMARY_MAX_INPUT_CHARS` | `240000` | Caption characters accepted (~60k tokens); longer videos get 422 |
| `SUMMARY_MAX_OUTPUT_TOKENS` | `3000` | Output ceiling; truncated output is rejected and refunded |

Worst case is ~25 s transcript + ~25 s generation, so the frontend proxy needs a
60 s route duration. Live quality, latency, and token cost still need checking
with a real key on the evaluation set in the plan.

### Existing services

- `src/config_v2.py` loads `.env` and defines API, YouTube, AWS/S3, SQS, proxy, CORS, and timeout settings. Other values are read directly in the API and database modules.
- Auth: `SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS`. Database: `DB_HOST_YOUTUBE_TRANSCRIPTS`, `DB_NAME_YOUTUBE_TRANSCRIPTS`, `DB_USERNAME_YOUTUBE_TRANSCRIPTS`, `DB_PASSWORD_YOUTUBE_TRANSCRIPTS`, `DB_PORT_YOUTUBE_TRANSCRIPTS`.
- Payments: `STRIPE_SECRET_KEY_LIVE`, `STRIPE_WEBHOOK_SECRET_TRANSCRIPTS`. Redirects: `FRONTEND_URL_YOUTUBE_TRANSCRIPTS`.
- `DOCS_SECRET_KEY` gates `/internal/docs`, `/internal/redoc`, `/internal/openapi.json`, and `/debug/*` through a `?secret=` query parameter. When unset, all of them return 404.
- AWS: `AWS_DEFAULT_REGION`, `LAMBDA_FUNCTION_NAME`, `S3_BUCKET_NAME`, `LAMBDA_RESULTS_QUEUE_URL`. Inspect worker environment reads too; backend settings do not automatically configure Lambda.
- `API_BASE_URL` is used for worker callback fallback; frontend `TRANSCRIPT_API_URL` points proxies at this API. `CORS_ORIGINS` controls browser origins. Local backend defaults use port 8000; old Copilot instructions used 8001.
- `env.example` is incomplete. Take variable names from implementation and never copy secret values into docs.
- For SQS delivery, run `poetry run python src/sqs_result_consumer.py` separately with queue/database settings. API startup does not run the consumer. With SQS working, set `ENABLE_HTTP_RESULT_CALLBACKS=false` on the API to disable the unauthenticated HTTP fallback (default `true`). If the worker cannot reach SQS while it is disabled, that video's result is lost and the job timeout marks it failed.
- Database startup failures are logged without preventing API startup; health success does not prove database/AWS readiness.
- Background dispatch/timeout tasks run in the API process and do not survive a restart. A sweeper started with the API (`youtube_service.run_stale_job_sweeper`, at startup and every 5 minutes) closes active jobs whose `updated_at` is older than `JOB_TIMEOUT_MINUTES`: unsent videos are skipped, in-flight ones failed, neither charged.

Operational references: `SQS_CALLBACK_SETUP.md`, `S3_ZIP_SETUP.md`, `CONCURRENT_LAMBDA_GUIDE.md`, worker-local guides. Read when needed and verify against current code before deployment.
