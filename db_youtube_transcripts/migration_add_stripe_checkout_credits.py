"""
Migration: Add stripe_checkout_credits so the Stripe webhook grants credits
once per Checkout session, even when Stripe redelivers an event.

Additive only (CREATE TABLE IF NOT EXISTS); no existing data is changed.
Run this once against your database BEFORE deploying the webhook change:
    python db_youtube_transcripts/migration_add_stripe_checkout_credits.py
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
    print("Creating table stripe_checkout_credits...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stripe_checkout_credits (
            session_id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            credits INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    print("✓ stripe_checkout_credits created successfully")

connection.close()
