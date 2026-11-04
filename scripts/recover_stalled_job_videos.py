#!/usr/bin/env python3
"""
Mark videos failed by the stale-job sweeper as completed when the worker's
transcript is already in S3.

The sweeper fails in-flight videos when no result arrives for JOB_TIMEOUT_MINUTES,
even if the worker finished and only the result delivery (SQS consumer) stalled.
Those videos were not charged and their credits were refunded at finalization;
recovering them does not charge again.

Dry run by default:
    poetry run python scripts/recover_stalled_job_videos.py <job_id> --bucket <bucket>
    poetry run python scripts/recover_stalled_job_videos.py <job_id> --bucket <bucket> --apply
"""

import argparse
import asyncio
import os
import sys

import asyncpg
import boto3
from dotenv import load_dotenv

load_dotenv()

SWEEPER_ERROR_PREFIX = "No progress for "


def list_transcript_keys(bucket: str, prefix: str) -> set:
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    keys = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.update(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".txt"))
    return keys


async def main(job_id: str, bucket: str, apply: bool) -> None:
    conn = await asyncpg.connect(
        database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
        host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
        user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
        password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
        port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
        statement_cache_size=0,
    )
    try:
        job = await conn.fetchrow(
            "SELECT user_id, status, completed, failed_count, skipped_count FROM jobs WHERE job_id = $1",
            job_id,
        )
        if not job:
            sys.exit(f"Job {job_id} not found")
        if job["status"] not in {"completed", "completed_with_errors"}:
            sys.exit(f"Job {job_id} is {job['status']}; only finished jobs can be recovered")

        stalled = await conn.fetch(
            """
            SELECT video_id FROM job_videos
            WHERE job_id = $1 AND status = 'failed' AND error_message LIKE $2
            """,
            job_id,
            SWEEPER_ERROR_PREFIX + "%",
        )
        prefix = f"{job['user_id']}/{job_id}/"
        in_s3 = list_transcript_keys(bucket, prefix)
        recoverable = [r["video_id"] for r in stalled if f"{prefix}{r['video_id']}.txt" in in_s3]

        print(
            f"Job {job_id} ({job['status']}): completed={job['completed']} "
            f"failed={job['failed_count']} skipped={job['skipped_count']}"
        )
        print(f"Failed by the sweeper: {len(stalled)}; transcript in S3: {len(recoverable)}")
        if not recoverable:
            return
        if not apply:
            print("Dry run. Re-run with --apply to mark them completed.")
            return

        async with conn.transaction():
            result = await conn.execute(
                """
                UPDATE job_videos
                SET status = 'completed', s3_key = $3 || video_id || '.txt',
                    error_message = NULL, updated_at = NOW()
                WHERE job_id = $1 AND video_id = ANY($2::text[])
                  AND status = 'failed' AND error_message LIKE $4
                """,
                job_id,
                recoverable,
                prefix,
                SWEEPER_ERROR_PREFIX + "%",
            )
            updated = int(result.rsplit(" ", 1)[-1])
            await conn.execute(
                """
                UPDATE jobs
                SET completed = completed + $2, failed_count = failed_count - $2,
                    status = CASE WHEN failed_count - $2 = 0 AND skipped_count = 0
                                  THEN 'completed' ELSE status END,
                    updated_at = NOW(), version = version + 1
                WHERE job_id = $1
                """,
                job_id,
                updated,
            )
        print(f"Marked {updated} videos completed.")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("job_id")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.bucket:
        sys.exit("Pass --bucket or set S3_BUCKET_NAME")
    asyncio.run(main(args.job_id, args.bucket, args.apply))
