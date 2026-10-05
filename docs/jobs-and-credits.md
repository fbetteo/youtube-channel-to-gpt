# Jobs, credits, and payments

## Job lifecycle

- Normalize/deduplicate available videos with `youtube_service.normalize_videos_for_job` before reserving credits.
- Create durable transcript jobs through `hybrid_job_manager` / `JobManager`. Reservation and job/video insertion can share a database transaction. Accepted paid work must not silently become file-only after a database error.
- Persist discovery through `db_youtube_transcripts/discovery_job_manager.py` before scheduling it. Distinguish discovery, transcript, and batch IDs.
- `youtube_service.prefetch_and_dispatch_task` coordinates metadata/dispatch. Lambda uses `InvocationType="Event"` and alias `prod`: invocation acceptance does not mean transcript completion. `max_concurrent` currently does not impose a dispatch semaphore.
- Workers upload text under `{user_id}/{job_id}/{video_id}.txt` in S3, try SQS result delivery first, and fall back to HTTP callbacks when available.
- SQS requires a separate `src/sqs_result_consumer.py` process. Both delivery paths use `src/job_result_processor.py`; preserve shared processing and duplicate-result handling.
- Timeout, cancellation, dispatch failure, and late results have distinct paths. Cancelled jobs ignore late results and preserve completed downloads. Inspect terminal accounting in `JobManager` before changing it.

## Credits and database

- Single authenticated raw downloads deduct one credit before the attempt.
- Bulk jobs reserve credits for accepted videos. Completed transcripts and processing failures consume credits; unprocessed work can be refunded during finalization/cancellation. A failure does not automatically mean a refund.
- `JobManager` owns transactional progress, terminal transitions, cancellation, and refunds. Keep balance/job accounting consistent and preserve protection against repeated results or finalization.
- The asyncpg pool and transaction helper live in `db_youtube_transcripts/database.py`. Transactional locks/updates prevent concurrent workers applying the same progress/balance change independently. Legacy synchronous connections also exist.
- Schema/migration entry points: `db_youtube_transcripts/schema.py`, `schema_api_keys.py`, `migration_*.py`. Review execution effects before running against a configured database; code changes do not migrate deployed tables.

## Payments

Checkout/webhooks live in `src/transcript_api.py`. `PRICE_CREDITS_MAP` is server-side. Checkout links the Supabase user through session metadata, including `project="transcript-api"`; the signature-verified webhook grants credits on `checkout.session.completed` for this project.

Keep frontend checkout payloads, referral metadata, and redirect URLs aligned. Browser success pages do not grant credits. The current handler has no explicit persisted event/session deduplication; do not assume payment events can be replayed safely when testing.

Relevant tests: `tests/test_playlist_job_lifecycle.py`, `tests/test_discovery_job_manager.py`.
