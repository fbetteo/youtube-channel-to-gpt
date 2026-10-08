"""
Migration: Add the transcript_cache index table (src/transcript_cache.py).

The shared transcript cache keeps segments in S3 under
transcript-cache/{video_id}/{requested_language|auto}.json (the same objects
the Lambda worker uses). This table indexes them: one small row per cached
transcript, for fast batch lookups ("which of these videos are cached?") and
cache stats. It never stores segments.

Additive only (CREATE TABLE IF NOT EXISTS); no existing data is changed.
Optional at deploy time: without the table the cache still works from S3,
only indexing and hit counts are skipped.
    python db_youtube_transcripts/migration_add_transcript_cache.py

Then fill it (and copy the old video_transcripts rows to S3) with
scripts/backfill_transcript_cache.py.
"""

from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

connection = psycopg2.connect(
    database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
    host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
    user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
    password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
    port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
)
connection.autocommit = True

with connection.cursor() as c:
    print("Creating table transcript_cache...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_cache (
            video_id TEXT NOT NULL,
            requested_language TEXT NOT NULL,  -- 'auto' or the requested code
            language TEXT NOT NULL,            -- caption track actually used
            is_generated BOOLEAN NOT NULL,
            segment_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL,   -- when YouTube was called
            source TEXT NOT NULL,              -- first writer: single, worker, backfill
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_hit_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (video_id, requested_language)
        );
        """
    )
    print("✓ transcript_cache created successfully")

connection.close()
