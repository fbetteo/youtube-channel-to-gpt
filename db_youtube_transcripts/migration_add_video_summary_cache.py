"""
Migration: Add video_transcripts and video_summaries cache tables and the
video_access table for single-video AI summaries (src/summary_service.py).

Additive only (CREATE TABLE IF NOT EXISTS); no existing data is changed.
Optional at deploy time: without these tables everything still works, but
uncached, and a summary after a paid transcript costs a second credit.
    python db_youtube_transcripts/migration_add_video_summary_cache.py

video_transcripts is also the intended corpus for future pgvector embeddings.
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
    print("Creating table video_transcripts...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS video_transcripts (
            video_id TEXT NOT NULL,
            requested_language TEXT NOT NULL,  -- 'auto' or the requested code
            language TEXT NOT NULL,            -- caption track actually used
            is_generated BOOLEAN NOT NULL,
            segments JSONB NOT NULL,           -- [{text, start, duration}]
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (video_id, requested_language)
        );
        """
    )
    print("✓ video_transcripts created successfully")

    print("Creating table video_summaries...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS video_summaries (
            video_id TEXT NOT NULL,
            transcript_language TEXT NOT NULL,
            summary_language TEXT NOT NULL,
            length TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            summary JSONB NOT NULL,            -- model output with block numbers
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (video_id, transcript_language, summary_language,
                         length, prompt_version, model)
        );
        """
    )
    print("✓ video_summaries created successfully")

    # One credit covers a video's transcript and its summary for 24 hours.
    print("Creating table video_access...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS video_access (
            user_id UUID NOT NULL,
            video_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, video_id)
        );
        """
    )
    print("✓ video_access created successfully")

connection.close()
