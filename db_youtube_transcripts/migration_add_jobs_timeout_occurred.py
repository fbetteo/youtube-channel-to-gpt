"""
Migration: Add jobs.timeout_occurred.

Production already has this column; no earlier schema file created it, so fresh
databases (local, verify) failed every job status query. Safe to run anywhere:
    python db_youtube_transcripts/migration_add_jobs_timeout_occurred.py
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
    print("Adding jobs.timeout_occurred...")
    c.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS timeout_occurred BOOLEAN NOT NULL DEFAULT false;"
    )
    print("✓ jobs.timeout_occurred present")

connection.close()
