"""
Migration: Enable pgvector and add the tables behind creator chat (RAG over a
creator's transcripts).

creators lists the channels shown on the creators page; we curate it by hand.
transcript_chunks holds ~1-2 minute pieces of each indexed video's captions,
with their embedding, so a question can be matched to the closest pieces of
one creator's videos (WHERE channel_id = ...).

Embeddings are OpenAI text-embedding-3-small shortened to 512 dimensions:
about a third of the storage of the full 1536 with a small quality loss.
Changing model or size means a new column type and re-indexing every creator.

There is no vector index yet: an exact scan of one creator's chunks is fast at
this size and never misses a match. Add HNSW when a creator gets slow.

Additive only; no existing data is changed. Idempotent.
    python db_youtube_transcripts/migration_add_creator_chat.py

RLS is enabled with no policies (see migration_enable_rls.py): the backend
connects as the table owner and keeps access; the public Supabase REST API
gets none.
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
    print("Enabling extension vector...")
    c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✓ vector enabled")

    print("Creating table creators...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS creators (
            channel_id TEXT PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,         -- URL: /creators/{slug}
            name TEXT NOT NULL,
            handle TEXT NULL,                  -- e.g. @marc-lou
            category TEXT NOT NULL,            -- e.g. indie-hacking, business
            avatar_url TEXT NULL,
            tagline TEXT NULL,
            enabled BOOLEAN NOT NULL DEFAULT FALSE,  -- shown on the page
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    c.execute("ALTER TABLE creators ENABLE ROW LEVEL SECURITY;")
    print("✓ creators created successfully")

    print("Creating table transcript_chunks...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_chunks (
            video_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,      -- 0, 1, 2... within the video
            channel_id TEXT NOT NULL REFERENCES creators(channel_id) ON DELETE CASCADE,
            video_title TEXT NULL,
            published_at TIMESTAMPTZ NULL,
            start_seconds REAL NOT NULL,       -- for youtube.com/watch?v=..&t=..
            end_seconds REAL NOT NULL,
            text TEXT NOT NULL,
            embedding vector(512) NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (video_id, chunk_index)
        );
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_transcript_chunks_channel ON transcript_chunks (channel_id);"
    )
    c.execute("ALTER TABLE transcript_chunks ENABLE ROW LEVEL SECURITY;")
    print("✓ transcript_chunks created successfully")

connection.close()
