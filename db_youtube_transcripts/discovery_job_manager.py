"""PostgreSQL-backed storage for short-lived discovery and orchestration jobs."""

import json
import logging
import time
from typing import Any, Dict, Optional

from .database import get_db_connection, get_db_transaction


logger = logging.getLogger(__name__)


def _json_payload(value: Dict[str, Any]) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


class DiscoveryJobManager:
    """Shared job state for work that must survive process and replica changes."""

    @staticmethod
    async def create_job(
        job_id: str,
        job_type: str,
        source_type: Optional[str],
        source_id: Optional[str],
        payload: Dict[str, Any],
        ttl_hours: int = 48,
    ) -> str:
        job_payload = dict(payload)
        job_payload["job_id"] = job_id
        status = str(job_payload.get("status", "processing"))

        async with get_db_transaction() as tx:
            # Opportunistic cleanup keeps this intentionally small table bounded
            # without requiring another scheduler.
            await tx.execute(
                "DELETE FROM video_discovery_jobs WHERE expires_at <= NOW()"
            )
            await tx.execute(
                """
                INSERT INTO video_discovery_jobs (
                    job_id, job_type, source_type, source_id, status,
                    payload, expires_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::jsonb,
                    NOW() + make_interval(hours => $7::int)
                )
                """,
                job_id,
                job_type,
                source_type,
                source_id,
                status,
                _json_payload(job_payload),
                ttl_hours,
            )
        return job_id

    @staticmethod
    async def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        if not updates:
            return await DiscoveryJobManager.get_job(job_id)

        updates = dict(updates)
        status = updates.get("status")
        error_message = updates.get("error") or updates.get("error_message")
        is_terminal = status in {"completed", "completed_with_errors", "failed"}

        async with get_db_connection() as conn:
            record = await conn.fetchrow(
                """
                UPDATE video_discovery_jobs
                SET payload = payload || $2::jsonb,
                    status = COALESCE($3, status),
                    error_message = COALESCE($4, error_message),
                    completed_at = CASE
                        WHEN $5 THEN COALESCE(completed_at, NOW())
                        ELSE completed_at
                    END,
                    updated_at = NOW()
                WHERE job_id = $1
                  AND expires_at > NOW()
                  AND status NOT IN ('completed', 'completed_with_errors', 'failed')
                RETURNING payload, status, error_message
                """,
                job_id,
                _json_payload(updates),
                status,
                error_message,
                is_terminal,
            )
        return DiscoveryJobManager._record_to_job(record)

    @staticmethod
    async def get_job(
        job_id: str, stale_after_minutes: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        async with get_db_transaction() as tx:
            if stale_after_minutes:
                timeout_message = (
                    "Discovery worker stopped reporting progress. Please try again."
                )
                timeout_payload = {
                    "status": "failed",
                    "error": timeout_message,
                    "end_time": time.time(),
                }
                await tx.execute(
                    """
                    UPDATE video_discovery_jobs
                    SET status = 'failed',
                        error_message = $3,
                        payload = payload || $4::jsonb,
                        completed_at = COALESCE(completed_at, NOW()),
                        updated_at = NOW()
                    WHERE job_id = $1
                      AND status IN ('queued', 'processing')
                      AND updated_at < NOW() - make_interval(mins => $2::int)
                      AND expires_at > NOW()
                    """,
                    job_id,
                    stale_after_minutes,
                    timeout_message,
                    _json_payload(timeout_payload),
                )

            record = await tx.fetchrow(
                """
                SELECT payload, status, error_message
                FROM video_discovery_jobs
                WHERE job_id = $1 AND expires_at > NOW()
                """,
                job_id,
            )
        return DiscoveryJobManager._record_to_job(record)

    @staticmethod
    def _record_to_job(record: Any) -> Optional[Dict[str, Any]]:
        if not record:
            return None
        payload = record["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        result = dict(payload or {})
        result["status"] = record["status"]
        if record["error_message"] and not result.get("error"):
            result["error"] = record["error_message"]
        return result
