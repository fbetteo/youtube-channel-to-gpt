"""
Migration: Add composite index on jobs(user_id, source_id) for efficient
"new videos" detection queries.

Run this once against your database:
    python db_youtube_transcripts/migration_add_source_index.py
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
    try:
        # Composite index for looking up jobs by user + source (used by /new-videos endpoints)
        print("Creating index idx_jobs_user_source on jobs(user_id, source_id)...")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_user_source ON jobs(user_id, source_id);"
        )
        print("✓ idx_jobs_user_source created successfully")

        # Composite index for looking up completed videos per user efficiently
        # This powers the get_completed_video_ids_for_user query
        print(
            "Creating index idx_job_videos_status_video_id on job_videos(status, video_id)..."
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_videos_status_video_id "
            "ON job_videos(status, video_id) WHERE status = 'completed';"
        )
        print("✓ idx_job_videos_status_video_id created successfully")

        print("\n🎉 Migration completed successfully!")

    except Exception as e:
        print(f"❌ Error running migration: {e}")
        raise

connection.close()
