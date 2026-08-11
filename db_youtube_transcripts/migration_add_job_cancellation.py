"""Add durable cancellation/refund counters to transcript jobs.

Run once before deploying application code:
    python -m db_youtube_transcripts.migration_add_job_cancellation
"""

import asyncio

from .database import close_db_pool, get_db_connection, init_db_pool


async def migrate() -> None:
    await init_db_pool()
    try:
        async with get_db_connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    ALTER TABLE jobs
                    ADD COLUMN IF NOT EXISTS skipped_count INTEGER NOT NULL DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS refunded_credits INTEGER NOT NULL DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS api_key_id UUID NULL
                    """
                )
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(migrate())
