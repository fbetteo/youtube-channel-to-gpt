"""
Migration: Let every user have a default API key the dashboard can show again.

is_default marks the key created for the user by GET /user/api-keys/default.
key_ciphertext holds that key encrypted with API_KEY_ENCRYPTION_KEY (Fernet),
so it can be shown again; other keys keep it NULL and are shown once.
Validation still uses key_hash.

The partial unique index allows one active default per user. Revoking the
default frees the slot, so the next request creates a new one.

Additive only; no existing data is changed. Idempotent.
    python db_youtube_transcripts/migration_add_default_api_key.py
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
    print("Adding api_keys.is_default and api_keys.key_ciphertext...")
    c.execute(
        """
        ALTER TABLE api_keys
            ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS key_ciphertext TEXT NULL;
        """
    )
    print("✓ columns added")

    print("Creating one-active-default-per-user index...")
    c.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_one_default_per_user
            ON api_keys(user_id) WHERE is_default AND is_active;
        """
    )
    print("✓ index created")

connection.close()
