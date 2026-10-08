"""
Shared transcript cache for every path: single downloads, summaries, the
developer API/MCP, and the Lambda worker.

S3 holds the segments, in the same objects and format the worker reads and
writes (lambda-transcript-processor/lambda_function.py):
    transcript-cache/{video_id}/{requested_language|auto}.json
Postgres `transcript_cache` indexes them (language, size, hits) for fast
batch lookups and stats; see db_youtube_transcripts/migration_add_transcript_cache.py.

S3 is the source of truth: lookups read S3 directly, so worker-written entries
hit even before they are indexed, and a hit indexes them. Every failure is
logged and treated as a miss, so the cache can never fail a request.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import youtube_service

logger = logging.getLogger(__name__)

CACHE_PREFIX = "transcript-cache"
SEGMENTS_SCHEMA_VERSION = 1

# Same env var and meaning as the worker: unset = reuse forever,
# N = entries older than N days are misses, 0 = never reuse.
_max_age = os.getenv("TRANSCRIPT_CACHE_MAX_AGE_DAYS", "").strip()
MAX_AGE_DAYS: Optional[float] = float(_max_age) if _max_age else None


def requested_language_key(preferred_language: Optional[str]) -> str:
    """'auto' when no language was requested, else the normalized code."""
    return youtube_service.normalize_preferred_language(preferred_language) or "auto"


def _object_key(video_id: str, language_key: str) -> str:
    return f"{CACHE_PREFIX}/{video_id}/{language_key}.json"


def cache_key(video_id: str, preferred_language: Optional[str]) -> str:
    return _object_key(video_id, requested_language_key(preferred_language))


def build_document(
    video_id: str,
    preferred_language: Optional[str],
    language: str,
    is_generated: bool,
    segments: List[Dict[str, Any]],
    fetched_at: Optional[int] = None,
) -> Dict[str, Any]:
    """Canonical record, identical to the worker's build_segments_document."""
    return {
        "schema_version": SEGMENTS_SCHEMA_VERSION,
        "video_id": video_id,
        "requested_language": requested_language_key(preferred_language),
        "language": language,
        "is_generated": is_generated,
        "fetched_at": fetched_at or int(time.time()),
        "segments": segments,
    }


def _is_usable(document: Dict[str, Any]) -> bool:
    if document.get("schema_version") != SEGMENTS_SCHEMA_VERSION or not isinstance(
        document.get("segments"), list
    ):
        return False
    if MAX_AGE_DAYS is None:
        return True
    age_seconds = time.time() - float(document.get("fetched_at") or 0)
    return age_seconds <= MAX_AGE_DAYS * 86400


async def load(video_id: str, preferred_language: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the cached document, or None on miss/expiry/error."""
    if MAX_AGE_DAYS is not None and MAX_AGE_DAYS <= 0:
        return None
    key = cache_key(video_id, preferred_language)
    try:
        s3_client, bucket = youtube_service.get_s3_client()
        response = await asyncio.to_thread(s3_client.get_object, Bucket=bucket, Key=key)
        document = json.loads(response["Body"].read())
    except Exception as e:
        # Missing keys surface as NoSuchKey, or AccessDenied without ListBucket.
        logger.info(f"Transcript cache miss for {key}: {type(e).__name__}")
        return None
    if not _is_usable(document):
        logger.info(f"Transcript cache entry {key} unusable or expired")
        return None
    logger.info(f"Transcript cache hit for {key}")
    await _index(document, hit=True)
    return document


async def save(document: Dict[str, Any], source: str) -> None:
    """Write the S3 object and index it. Best effort: never raises."""
    # requested_language is already the key form ('auto' or a normalized code).
    key = _object_key(document["video_id"], document["requested_language"])
    try:
        s3_client, bucket = youtube_service.get_s3_client()
        body = json.dumps(document, ensure_ascii=False).encode("utf-8")
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning(f"Transcript cache write failed for {key}: {e}")
        return
    await _index(document, source=source)


async def _index(document: Dict[str, Any], source: str = "unknown", hit: bool = False) -> None:
    """Upsert the index row; a hit also bumps hit_count. Never raises."""
    try:
        from db_youtube_transcripts.database import get_db_connection

        segments = document["segments"]
        async with get_db_connection() as conn:
            await conn.execute(
                """
                INSERT INTO transcript_cache
                    (video_id, requested_language, language, is_generated,
                     segment_count, char_count, fetched_at, source,
                     hit_count, last_hit_at)
                VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7), $8,
                        CASE WHEN $9 THEN 1 ELSE 0 END,
                        CASE WHEN $9 THEN NOW() END)
                ON CONFLICT (video_id, requested_language) DO UPDATE SET
                    language = EXCLUDED.language,
                    is_generated = EXCLUDED.is_generated,
                    segment_count = EXCLUDED.segment_count,
                    char_count = EXCLUDED.char_count,
                    fetched_at = EXCLUDED.fetched_at,
                    hit_count = transcript_cache.hit_count
                        + CASE WHEN $9 THEN 1 ELSE 0 END,
                    last_hit_at = CASE WHEN $9 THEN NOW()
                        ELSE transcript_cache.last_hit_at END
                """,
                document["video_id"],
                document["requested_language"],
                document["language"],
                bool(document["is_generated"]),
                len(segments),
                sum(len(str(s.get("text", ""))) for s in segments),
                float(document.get("fetched_at") or time.time()),
                source,
                hit,
            )
    except Exception as e:
        logger.warning(f"Transcript cache index failed for {document.get('video_id')}: {e}")


def to_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """Metadata in the shape single downloads and summaries already return."""
    return {
        "video_id": document["video_id"],
        "transcript_language": document["language"],
        "transcript_type": "auto-generated" if document["is_generated"] else "manual",
    }
