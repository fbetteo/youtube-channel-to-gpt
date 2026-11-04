"""
Migration: Drop jobs.videos_metadata.

The column held a JSONB copy of every video's metadata. Nothing read it: the
same metadata lives in job_videos columns. It made every SELECT * FROM jobs
pull a large blob. Deploy the code that stopped writing it first, then run:
    python db_youtube_transcripts/migration_drop_jobs_videos_metadata.py
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
    print("Dropping jobs.videos_metadata...")
    c.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS videos_metadata;")
    print("✓ jobs.videos_metadata dropped")

connection.close()
