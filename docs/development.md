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

## Configuration and operations

### Summary beta

Set `SUMMARY_OPENAI_API_KEY` on the Python server to enable authenticated
single-video summaries. Do not put this key in the frontend or a `NEXT_PUBLIC_*`
variable. The separate variable avoids accidentally reusing a legacy assistant
key. No database migration is required. Install the locked `tiktoken` dependency
with `poetry install`, then pre-cache its encoding using
`poetry run python scripts/prepare_summary_tokenizer.py`. The first initialization
downloads public tokenizer data; use the same `TIKTOKEN_CACHE_DIR` and service
user for preparation and runtime if deployment needs an explicit persistent cache.

| Setting | Default | Purpose |
| --- | --- | --- |
| `SUMMARY_MODEL` | `gpt-4.1-mini` | Structured-output text model |
| `SUMMARY_MAX_INPUT_TOKENS` | `50000` | Model-tokenizer estimate for captions, source IDs, prompt, and schema, plus 256 tokens for provider framing |
| `SUMMARY_MAX_OUTPUT_TOKENS` | `2000` | Generation ceiling; incomplete output is rejected |
| `SUMMARY_TIMEOUT_SECONDS` | `20` | Provider deadline; must be positive and no more than 20 seconds |
| `SUMMARY_REQUESTS_PER_HOUR` | `10` | Admitted attempts per signed-in user per API process |
| `SUMMARY_MAX_CONCURRENT` | `4` | Concurrent summary flows per API process; at most one per user per process |

The beta does not deduct existing download credits, save summaries, or automatically
retry requests. Limits reset on restart and multiply across API processes;
they are not a durable account-wide billing mechanism. Keep rollout bounded until
shared quotas/persistence are added. Input above the ceiling is rejected without
calling the model, while the retrieved transcript remains available. With the
token-based bound, supported duration varies with caption density and language.
The tokenizer must recognize `SUMMARY_MODEL`; unsupported models produce an
explicit error rather than falling back to a potentially wrong encoding. The
local estimate is returned as `estimated_input_tokens`; actual billed tokens are
reported separately in `usage` after generation. Provider-specific schema framing
means these counts may differ slightly.

The frontend summary proxy uses a 60-second route duration and 55-second deadline;
the backend stream has a 52-second total budget (30 for extraction, up to 20 for
generation). Confirm the deployed hosting plan permits this duration. Nginx or
other reverse proxies must pass SSE promptly; the backend supplies
`X-Accel-Buffering: no`, but infrastructure can override that header. No production
proxy settings were changed as part of implementation.

Backend verification: `poetry run pytest tests/test_single_video_summary.py
tests/test_transcript_language_selection.py tests/test_transcript_formatting_and_localized_titles.py
tests/test_agent_interfaces.py`. In the frontend repo, run
`node --test tests/single-video-summary.test.cjs` and `npm run build`.
Provider calls are mocked in tests, including an HTTP MockTransport exercising
the installed OpenAI SDK. Live model quality, token cost, and deployed latency
still need evaluation with a configured project key.

### Existing services

- `src/config_v2.py` loads `.env` and defines API, YouTube, AWS/S3, SQS, proxy, CORS, and timeout settings. Other values are read directly in the API and database modules.
- Auth: `SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS`. Database: `DB_HOST_YOUTUBE_TRANSCRIPTS`, `DB_NAME_YOUTUBE_TRANSCRIPTS`, `DB_USERNAME_YOUTUBE_TRANSCRIPTS`, `DB_PASSWORD_YOUTUBE_TRANSCRIPTS`, `DB_PORT_YOUTUBE_TRANSCRIPTS`.
- Payments: `STRIPE_SECRET_KEY_LIVE`, `STRIPE_WEBHOOK_SECRET_TRANSCRIPTS`. Redirects: `FRONTEND_URL_YOUTUBE_TRANSCRIPTS`.
- AWS: `AWS_DEFAULT_REGION`, `LAMBDA_FUNCTION_NAME`, `S3_BUCKET_NAME`, `LAMBDA_RESULTS_QUEUE_URL`. Inspect worker environment reads too; backend settings do not automatically configure Lambda.
- `API_BASE_URL` is used for worker callback fallback; frontend `TRANSCRIPT_API_URL` points proxies at this API. `CORS_ORIGINS` controls browser origins. Local backend defaults use port 8000; old Copilot instructions used 8001.
- `env.example` is incomplete. Take variable names from implementation and never copy secret values into docs.
- For SQS delivery, run `poetry run python src/sqs_result_consumer.py` separately with queue/database settings. API startup does not run the consumer.
- Database startup failures are logged without preventing API startup; health success does not prove database/AWS readiness.
- Background dispatch/timeout tasks run in the API process. Persisted job state does not make those tasks survive a restart.

Operational references: `SQS_CALLBACK_SETUP.md`, `S3_ZIP_SETUP.md`, `CONCURRENT_LAMBDA_GUIDE.md`, worker-local guides. Read when needed and verify against current code before deployment.
