"""
Migration: subscriptions with expiring monthly and top-up credits.

- user_credits.permanent_credits: the part of `credits` that never expires
  (welcome credits, legacy packs, balance held when first subscribing). The
  first subscription payment sets it from the whole balance, so existing rows
  need no backfill. See src/credit_ledger.py and docs/jobs-and-credits.md.
- credit_lots: expiring credits, spent soonest-expiry first: one `monthly`
  row per plan period (expires at the period end) and one `topup` row per
  top-up (30 days after purchase or when the subscription ends).
- user_subscriptions: one row per Stripe subscription, synced by the webhook.

Additive only (ADD COLUMN / CREATE TABLE IF NOT EXISTS); no existing data is
changed and spending code keeps working without it. Run this once against
your database BEFORE deploying the subscription webhook change:
    python db_youtube_transcripts/migration_add_subscriptions.py
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
    print("Adding user_credits.permanent_credits...")
    c.execute(
        """
        ALTER TABLE user_credits
        ADD COLUMN IF NOT EXISTS permanent_credits INTEGER NOT NULL DEFAULT 0;
        """
    )
    print("Creating table credit_lots...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS credit_lots (
            id BIGSERIAL PRIMARY KEY,
            user_id UUID NOT NULL,
            amount INTEGER NOT NULL CHECK (amount > 0),
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            kind TEXT NOT NULL DEFAULT 'topup' CHECK (kind IN ('monthly', 'topup')),
            source_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_credit_lots_user_id ON credit_lots (user_id);
        CREATE INDEX IF NOT EXISTS idx_credit_lots_expires_at ON credit_lots (expires_at);
        """
    )
    print("Creating table user_subscriptions...")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_subscriptions (
            stripe_subscription_id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            stripe_customer_id TEXT NOT NULL,
            status TEXT NOT NULL,
            lookup_key TEXT,
            monthly_credits INTEGER NOT NULL DEFAULT 0,
            cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
            current_period_end TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_user_subscriptions_user_id
            ON user_subscriptions (user_id);
        """
    )
    print("Done: subscription schema ready")

connection.close()
