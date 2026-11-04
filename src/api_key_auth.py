"""
API Key Authentication Module

Handles validation and management of developer API keys.
API keys are an alternative to JWT authentication for programmatic access.
"""

import hashlib
import os
import secrets
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from fastapi import HTTPException, Header, status

logger = logging.getLogger(__name__)


def get_key_hint() -> str:
    """Where to get a key; appended to auth errors so agents can tell their user."""
    frontend_url = os.getenv(
        "FRONTEND_URL_YOUTUBE_TRANSCRIPTS", "http://localhost:3000"
    )
    return f"Get your API key at {frontend_url}/dashboard/api-keys"


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix)
        - full_key: The complete API key to show to user ONCE
        - key_hash: SHA-256 hash to store in database
        - key_prefix: First 8 chars for identification
    """
    # Create key with identifiable prefix using secure random token
    key_suffix = secrets.token_urlsafe(32)  # ~43 chars, 256 bits of entropy
    full_key = f"yt_live_{key_suffix}"

    # Hash the key for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    # Store prefix for identification (visible in UI) - first 16 chars
    key_prefix = full_key[:16]  # "yt_live_" + first 8 chars of random suffix

    return full_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """Hash an API key for database lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def validate_api_key_value(api_key: str) -> Dict[str, Any]:
    """
    Validate a raw API key value.

    Returns:
        Dict with user_id, key_id, rate_limit_tier, and other key metadata

    Raises:
        HTTPException 403 if key is invalid, inactive, or expired
    """
    # Hash the provided key
    key_hash = hash_api_key(api_key)

    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            # Look up the key
            row = await conn.fetchrow(
                """
                SELECT 
                    key_id, user_id, name, is_active, rate_limit_tier,
                    expires_at, total_requests, total_credits_used, created_at
                FROM api_keys
                WHERE key_hash = $1
                """,
                key_hash,
            )

            if not row:
                logger.warning(f"Invalid API key attempted: {api_key[:12]}...")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Invalid API key. {get_key_hint()}",
                )

            # Check if key is active
            if not row["is_active"]:
                logger.warning(f"Inactive API key used: {row['key_id']}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key has been deactivated. {get_key_hint()}",
                )

            # Check if key has expired
            if row["expires_at"] and row["expires_at"] < datetime.utcnow():
                logger.warning(f"Expired API key used: {row['key_id']}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key has expired. {get_key_hint()}",
                )

            # Update last_used_at and increment total_requests
            await conn.execute(
                """
                UPDATE api_keys 
                SET last_used_at = NOW(), total_requests = total_requests + 1
                WHERE key_id = $1
                """,
                row["key_id"],
            )

            logger.info(
                f"API key validated for user {row['user_id']} (key: {row['name']})"
            )

            return {
                "key_id": str(row["key_id"]),
                "user_id": str(row["user_id"]),
                "key_name": row["name"],
                "rate_limit_tier": row["rate_limit_tier"],
                "total_requests": row["total_requests"] + 1,  # Include this request
                "total_credits_used": row["total_credits_used"],
                "created_at": (
                    row["created_at"].isoformat() if row["created_at"] else None
                ),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate API key",
        )


async def validate_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Dict[str, Any]:
    """
    Validate an API key from the X-API-Key header.

    Raises:
        HTTPException 401 if key is missing
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API key is required. Provide it via the X-API-Key header. {get_key_hint()}",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return await validate_api_key_value(x_api_key)


async def create_api_key(
    user_id: str,
    name: str,
    rate_limit_tier: str = "standard",
    expires_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create a new API key for a user.

    Args:
        user_id: The user's UUID
        name: A friendly name for the key
        rate_limit_tier: 'free', 'standard', or 'premium'
        expires_at: Optional expiration datetime

    Returns:
        Dict with key_id, api_key (ONLY TIME IT'S RETURNED), and metadata
    """
    full_key, key_hash, key_prefix = generate_api_key()

    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO api_keys (user_id, key_hash, key_prefix, name, rate_limit_tier, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING key_id, created_at
                """,
                user_id,
                key_hash,
                key_prefix,
                name,
                rate_limit_tier,
                expires_at,
            )

            logger.info(f"Created API key {row['key_id']} for user {user_id}")

            return {
                "key_id": str(row["key_id"]),
                "api_key": full_key,  # ONLY returned here, never stored
                "key_prefix": key_prefix,
                "name": name,
                "rate_limit_tier": rate_limit_tier,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": row["created_at"].isoformat(),
                "message": "Save this API key now. It won't be shown again.",
            }

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


DEFAULT_KEY_QUERY = """
    SELECT key_id, key_prefix, name, key_ciphertext, created_at
    FROM api_keys
    WHERE user_id = $1 AND is_default AND is_active
"""


async def get_or_create_default_api_key(user_id: str) -> Dict[str, Any]:
    """
    Return the user's default API key in full, creating it on first call.

    Only the default key keeps an encrypted copy (key_ciphertext) so the
    dashboard can show it again; other keys are shown once. Validation still
    goes through key_hash. Revoking the default makes the next call create a
    new one. The user_credits row must already exist (foreign key).
    """
    encryption_key = os.getenv("API_KEY_ENCRYPTION_KEY")
    if not encryption_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Default API keys are not configured on this server.",
        )
    fernet = Fernet(encryption_key)

    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        row = await conn.fetchrow(DEFAULT_KEY_QUERY, user_id)
        if not row:
            full_key, key_hash, key_prefix = generate_api_key()
            await conn.execute(
                """
                INSERT INTO api_keys
                    (user_id, key_hash, key_prefix, name, is_default, key_ciphertext)
                VALUES ($1, $2, $3, 'Default', TRUE, $4)
                ON CONFLICT DO NOTHING
                """,
                user_id,
                key_hash,
                key_prefix,
                fernet.encrypt(full_key.encode()).decode(),
            )
            # Re-read: a concurrent request may have created the default first.
            row = await conn.fetchrow(DEFAULT_KEY_QUERY, user_id)
            logger.info(f"Created default API key {row['key_id']} for user {user_id}")

    return {
        "key_id": str(row["key_id"]),
        "api_key": fernet.decrypt(row["key_ciphertext"].encode()).decode(),
        "key_prefix": row["key_prefix"],
        "name": row["name"],
        "created_at": row["created_at"].isoformat(),
    }


async def list_user_api_keys(user_id: str) -> list[Dict[str, Any]]:
    """
    List all API keys for a user (without the actual key values).

    Args:
        user_id: The user's UUID

    Returns:
        List of API key metadata dicts
    """
    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    key_id, key_prefix, name, is_active, is_default, rate_limit_tier,
                    last_used_at, total_requests, total_credits_used,
                    created_at, expires_at
                FROM api_keys
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                user_id,
            )

            return [
                {
                    "key_id": str(row["key_id"]),
                    "key_prefix": row["key_prefix"],
                    "name": row["name"],
                    "is_active": row["is_active"],
                    "is_default": row["is_default"],
                    "rate_limit_tier": row["rate_limit_tier"],
                    "last_used_at": (
                        row["last_used_at"].isoformat() if row["last_used_at"] else None
                    ),
                    "total_requests": row["total_requests"],
                    "total_credits_used": row["total_credits_used"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                    "expires_at": (
                        row["expires_at"].isoformat() if row["expires_at"] else None
                    ),
                }
                for row in rows
            ]

    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys",
        )


async def revoke_api_key(user_id: str, key_id: str) -> bool:
    """
    Revoke (deactivate) an API key.

    Args:
        user_id: The user's UUID (for authorization check)
        key_id: The key's UUID to revoke

    Returns:
        True if revoked, False if not found
    """
    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            result = await conn.execute(
                """
                UPDATE api_keys 
                SET is_active = FALSE, updated_at = NOW()
                WHERE key_id = $1 AND user_id = $2
                """,
                key_id,
                user_id,
            )

            # Check if any row was updated
            rows_affected = int(result.split()[-1])

            if rows_affected > 0:
                logger.info(f"Revoked API key {key_id} for user {user_id}")
                return True
            else:
                logger.warning(
                    f"API key {key_id} not found or not owned by user {user_id}"
                )
                return False

    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )


async def increment_api_key_credits_used(key_id: str, credits: int) -> None:
    """
    Increment the total credits used counter for an API key.
    Called when credits are consumed via API key authentication.

    Args:
        key_id: The API key's UUID
        credits: Number of credits to add to the counter
    """
    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE api_keys 
                SET total_credits_used = total_credits_used + $1
                WHERE key_id = $2
                """,
                credits,
                key_id,
            )

    except Exception as e:
        logger.error(f"Error incrementing API key credits: {e}")
        # Don't raise - this is a non-critical tracking operation


# Rate limit tiers configuration
RATE_LIMIT_TIERS = {
    "free": {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "max_videos_per_job": 10000,
        "concurrent_jobs": 1,
    },
    "standard": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "max_videos_per_job": 10000,
        "concurrent_jobs": 5,
    },
    "premium": {
        "requests_per_minute": 120,
        "requests_per_hour": 5000,
        "max_videos_per_job": 10000,
        "concurrent_jobs": 20,
    },
}


def get_rate_limits(tier: str) -> Dict[str, int]:
    """Get rate limit configuration for a tier."""
    return RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["standard"])
