# Jobs, credits, and payments

## Job lifecycle

- Normalize/deduplicate available videos with `youtube_service.normalize_videos_for_job` before reserving credits.
- Create durable transcript jobs through `hybrid_job_manager` / `JobManager`. Reservation and job/video insertion can share a database transaction. Accepted paid work must not silently become file-only after a database error.
- Persist discovery through `db_youtube_transcripts/discovery_job_manager.py` before scheduling it. Distinguish discovery, transcript, and batch IDs.
- `youtube_service.prefetch_and_dispatch_task` coordinates metadata/dispatch. Lambda uses `InvocationType="Event"` and alias `prod`: invocation acceptance does not mean transcript completion. `max_concurrent` currently does not impose a dispatch semaphore.
- Workers upload text under `{user_id}/{job_id}/{video_id}.txt` in S3, plus the caption segments as `{video_id}.json` beside it (used for non-txt download formats; see [api.md](api.md)), try SQS result delivery first, and fall back to HTTP callbacks when available.
- Worker transcript cache: before calling YouTube, the worker reads `transcript-cache/{video_id}/{requested_language|auto}.json` from the same bucket and reuses it. `TRANSCRIPT_CACHE_MAX_AGE_DAYS` (Lambda env): unset (default) reuses entries forever; `N` re-fetches and overwrites entries older than N days; `0` never reuses. Fetches always write it back, and no code deletes entries; clean the prefix in S3 directly if needed. Cache read/write errors fall back to a live fetch. A cache hit completes and is charged like any other completed video. This cache is separate from the Postgres `video_transcripts` cache used by single-video downloads and summaries.
- SQS requires a separate `src/sqs_result_consumer.py` process. Both delivery paths use `src/job_result_processor.py`; preserve shared processing and duplicate-result handling.
- Timeout, cancellation, dispatch failure, and late results have distinct paths. The only timeout is a periodic sweep (`run_stale_job_sweeper`) that reconciles active jobs with no progress for `JOB_TIMEOUT_MINUTES`, including jobs orphaned by an API restart; large jobs that keep receiving results are not cut off. The worker bounds its own fetch to the Lambda's remaining time minus `DEADLINE_SAFETY_SECONDS` and reports a retriable `DeadlineExceeded` failure instead of being killed silently. Lambda timeout should therefore stay well above that margin. Cancelled jobs ignore late results and preserve completed downloads. Inspect terminal accounting in `JobManager` before changing it.

## Credits and database

- Single authenticated raw downloads deduct one credit before the attempt.
- Bulk jobs reserve credits for accepted videos. Completed transcripts and non-retriable worker failures (no captions, unavailable/private video) consume credits. Retriable failures (proxy blocks, network errors, worker deadline) are our side and are not charged; a missing `retriable` flag is treated as chargeable for older workers; unprocessed work can be refunded during finalization/cancellation. A failure does not automatically mean a refund. Videos with no worker result before the job timeout (throttling, crash, lost message) are marked failed but not charged, so finalization refunds them.
- `JobManager` owns transactional progress, terminal transitions, cancellation, and refunds. Keep balance/job accounting consistent and preserve protection against repeated results or finalization.
- The asyncpg pool and transaction helper live in `db_youtube_transcripts/database.py`. Transactional locks/updates prevent concurrent workers applying the same progress/balance change independently. Legacy synchronous connections also exist.
- Schema/migration entry points: `db_youtube_transcripts/schema.py`, `schema_api_keys.py`, `migration_*.py`. Review execution effects before running against a configured database; code changes do not migrate deployed tables.

## Payments

Checkout/webhooks live in `src/transcript_api.py`. `PRICE_CREDITS_MAP` is server-side. Checkout links the Supabase user through session metadata, including `project="transcript-api"`; the signature-verified webhook grants credits on `checkout.session.completed` for this project.

Keep frontend checkout payloads, referral metadata, and redirect URLs aligned. Browser success pages do not grant credits. The webhook grants credits only when `payment_status` is `paid` or `no_payment_required` (100%-off promotion codes), and records each Checkout session in `stripe_checkout_credits` in the same transaction as the credit grant, so redelivered events return `duplicate` without adding credits. That table comes from `db_youtube_transcripts/migration_add_stripe_checkout_credits.py`; until it exists, the webhook returns 500 and Stripe retries.

Relevant tests: `tests/test_playlist_job_lifecycle.py`, `tests/test_discovery_job_manager.py`.
