"""Create shared storage for video and playlist discovery jobs.

Run before deploying the application changes:
    python -m db_youtube_transcripts.migration_add_video_discovery_jobs
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
                    CREATE TABLE IF NOT EXISTS video_discovery_jobs (
                        job_id UUID PRIMARY KEY,
                        job_type VARCHAR(50) NOT NULL,
                        source_type VARCHAR(20) NULL,
                        source_id TEXT NULL,
                        status VARCHAR(30) NOT NULL DEFAULT 'processing',
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        error_message TEXT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        completed_at TIMESTAMP NULL,
                        expires_at TIMESTAMP NOT NULL DEFAULT NOW() + INTERVAL '48 hours'
                    );

                    CREATE INDEX IF NOT EXISTS idx_video_discovery_jobs_status
                        ON video_discovery_jobs(status);
                    CREATE INDEX IF NOT EXISTS idx_video_discovery_jobs_expires_at
                        ON video_discovery_jobs(expires_at);
                    """
                )
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(migrate())
