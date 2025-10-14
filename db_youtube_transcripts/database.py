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
            min_size=5,  # Minimum connections in pool
            max_size=20,  # Maximum connections in pool
            statement_cache_size=0,  # Disable prepared statements for pgbouncer compatibility
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
