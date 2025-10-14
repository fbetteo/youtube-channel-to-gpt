import os
import asyncpg
import psycopg2
from dotenv import load_dotenv
from typing import Optional
from contextlib import asynccontextmanager

load_dotenv()

### YOUTUBE TRANSCRIPTS ###


# Sync connection for backwards compatibility
def get_connection_youtube_transcripts():
    conn = psycopg2.connect(
        database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
        host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
        user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
        password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
        port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
    )
    conn.autocommit = True
    return conn


def get_db_youtube_transcripts():
    conn = get_connection_youtube_transcripts()
    try:
        yield conn
    finally:
        conn.close()


# Async database connection pool for job management
_connection_pool: Optional[asyncpg.Pool] = None


async def init_db_pool():
    """Initialize the async database connection pool"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = await asyncpg.create_pool(
            database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
            host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
            user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
            password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
            port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
            min_size=10,  # Increased minimum connections
            max_size=50,  # Increased maximum connections for concurrent Lambda callbacks
            statement_cache_size=0,  # Disable prepared statements for pgbouncer compatibility
            command_timeout=30,  # 30 second command timeout
            server_settings={
                "application_name": "transcript_api",
                "tcp_keepalives_idle": "600",  # Send keepalive every 10 minutes
                "tcp_keepalives_interval": "30",  # Retry keepalive every 30 seconds
                "tcp_keepalives_count": "3",  # 3 failed keepalives = dead connection
            },
        )
    return _connection_pool


async def close_db_pool():
    """Close the async database connection pool"""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None


async def get_db_pool() -> asyncpg.Pool:
    """Get the async database connection pool"""
    if _connection_pool is None:
        await init_db_pool()
    return _connection_pool


@asynccontextmanager
async def get_db_connection():
    """Get an async database connection from the pool"""
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        yield connection


@asynccontextmanager
async def get_db_transaction():
    """Get an async database transaction"""
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            yield connection
