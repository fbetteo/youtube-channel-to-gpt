#!/usr/bin/env python3
"""
Job Manager - Database-backed job management with async operations
Replaces the file-based job system with PostgreSQL for better concurrency control
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from .database import get_db_connection, get_db_transaction

# Configure logging
logger = logging.getLogger(__name__)


class JobManager:
    """Database-backed job management with atomic operations and race condition prevention"""

    @staticmethod
    async def create_job_with_videos(
        job_id: str,
        user_id: str,
        source_type: str,  # 'channel' or 'playlist'
        source_id: str,
        source_name: str,
        videos: List[Dict[str, Any]],
        formatting_options: Dict[str, Any] = None,
        **job_metadata,
    ) -> str:
        """
        Create a new job with associated videos in a single transaction.
        This ensures atomicity - either both job and videos are created, or neither.

        Args:
            job_id: Unique job identifier
            user_id: User UUID
            source_type: 'channel' or 'playlist'
            source_id: Channel ID or Playlist ID
            source_name: Human readable name
            videos: List of video dictionaries with id, title, etc.
            formatting_options: Dict of formatting preferences
            **job_metadata: Additional job fields (credits_reserved, etc.)

        Returns:
            job_id on success

        Raises:
            Exception on database error
        """
        try:
            async with get_db_transaction() as tx:
                # Prepare job data with defaults
                job_data = {
                    "job_id": job_id,
                    "user_id": user_id,
                    "status": "initializing",
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_name": source_name,
                    "total_videos": len(videos),
                    "processed_count": 0,
                    "completed": 0,
                    "failed_count": 0,
                    "credits_reserved": job_metadata.get(
                        "credits_reserved", len(videos)
                    ),
                    "credits_used": 0,
                    "reservation_id": job_metadata.get("reservation_id"),
                    # Formatting options
                    "include_timestamps": (
                        formatting_options.get("include_timestamps", False)
                        if formatting_options
                        else False
                    ),
                    "include_video_title": (
                        formatting_options.get("include_video_title", True)
                        if formatting_options
                        else True
                    ),
                    "include_video_id": (
                        formatting_options.get("include_video_id", True)
                        if formatting_options
                        else True
                    ),
                    "include_video_url": (
                        formatting_options.get("include_video_url", True)
                        if formatting_options
                        else True
                    ),
                    "include_view_count": (
                        formatting_options.get("include_view_count", False)
                        if formatting_options
                        else False
                    ),
                    "concatenate_all": (
                        formatting_options.get("concatenate_all", False)
                        if formatting_options
                        else False
                    ),
                    # Lambda processing
                    "lambda_dispatched_count": 0,
                    "prefetch_completed": False,
                    # Metadata storage
                    "videos_metadata": job_metadata.get("videos_metadata"),
                    "formatting_options": formatting_options,
                    # Source-specific fields
                    "channel_name": source_id if source_type == "channel" else None,
                    "playlist_id": source_id if source_type == "playlist" else None,
                }

                # Insert job record
                job_insert_query = """
                    INSERT INTO jobs (
                        job_id, user_id, status, source_type, source_id, source_name,
                        total_videos, processed_count, completed, failed_count,
                        credits_reserved, credits_used, reservation_id,
                        include_timestamps, include_video_title, include_video_id, 
                        include_video_url, include_view_count, concatenate_all,
                        lambda_dispatched_count, prefetch_completed,
                        videos_metadata, formatting_options, channel_name, playlist_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                    )
                """

                await tx.execute(
                    job_insert_query,
                    job_data["job_id"],
                    job_data["user_id"],
                    job_data["status"],
                    job_data["source_type"],
                    job_data["source_id"],
                    job_data["source_name"],
                    job_data["total_videos"],
                    job_data["processed_count"],
                    job_data["completed"],
                    job_data["failed_count"],
                    job_data["credits_reserved"],
                    job_data["credits_used"],
                    job_data["reservation_id"],
                    job_data["include_timestamps"],
                    job_data["include_video_title"],
                    job_data["include_video_id"],
                    job_data["include_video_url"],
                    job_data["include_view_count"],
                    job_data["concatenate_all"],
                    job_data["lambda_dispatched_count"],
                    job_data["prefetch_completed"],
                    (
                        json.dumps(job_data["videos_metadata"])
                        if job_data["videos_metadata"]
                        else None
                    ),
                    (
                        json.dumps(job_data["formatting_options"])
                        if job_data["formatting_options"]
                        else None
                    ),
                    job_data["channel_name"],
                    job_data["playlist_id"],
                )

                # Insert videos for the job
                if videos:
                    video_insert_query = """
                        INSERT INTO job_videos (
                            video_id, job_id, title, url, description, published_at,
                            channel_id, channel_title, duration_iso, duration_seconds,
                            duration_category, view_count, language, status
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                        )
                    """

                    for video in videos:
                        # Parse published date if available
                        published_at = None
                        if video.get("publishedAt"):
                            try:
                                published_at = datetime.fromisoformat(
                                    video["publishedAt"].replace("Z", "+00:00")
                                )
                            except Exception:
                                pass

                        await tx.execute(
                            video_insert_query,
                            video["id"],  # video_id
                            job_id,  # job_id
                            video.get("title", ""),  # title
                            video.get(
                                "url", f"https://www.youtube.com/watch?v={video['id']}"
                            ),  # url
                            video.get("description"),  # description
                            published_at,  # published_at
                            video.get("channel_id"),  # channel_id
                            video.get("channel_title"),  # channel_title
                            video.get("duration_iso"),  # duration_iso (PT4M13S format)
                            video.get("duration_seconds"),  # duration_seconds
                            video.get(
                                "duration", "unknown"
                            ),  # duration_category (short/medium/long)
                            video.get("view_count"),  # view_count
                            video.get("language"),  # language
                            "pending",  # status (default to pending)
                        )

                logger.info(
                    f"Created job {job_id} with {len(videos)} videos in database"
                )
                return job_id

        except Exception as e:
            logger.error(f"Failed to create job {job_id} in database: {str(e)}")
            raise

    @staticmethod
    async def get_job_from_db(
        job_id: str, include_videos: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get job data from database with optional video details.
        Optimized for fast job status queries.

        Args:
            job_id: Job identifier
            include_videos: Whether to include video details (slower query)

        Returns:
            Job dictionary or None if not found
        """
        try:
            async with get_db_connection() as conn:
                # Get job data (always fast)
                job_query = "SELECT * FROM jobs WHERE job_id = $1"
                job_record = await conn.fetchrow(job_query, job_id)

                if not job_record:
                    return None

                # Convert to dict and handle JSON fields
                job = dict(job_record)

                # Parse JSON fields
                if job["videos_metadata"]:
                    job["videos_metadata"] = json.loads(job["videos_metadata"])
                if job["formatting_options"]:
                    job["formatting_options"] = json.loads(job["formatting_options"])

                # Convert timestamps to Unix timestamps for backwards compatibility
                if job["start_time"]:
                    job["start_time"] = job["start_time"].timestamp()
                if job["end_time"]:
                    job["end_time"] = job["end_time"].timestamp()
                if job["lambda_dispatch_time"]:
                    job["lambda_dispatch_time"] = job[
                        "lambda_dispatch_time"
                    ].timestamp()

                if include_videos:
                    # Get video data (potentially slower for large jobs)
                    videos_query = """
                        SELECT video_id as id, title, url, description, published_at, 
                               channel_id, channel_title, duration_iso, duration_seconds,
                               duration_category as duration, view_count, language, 
                               status, file_path, s3_key, file_size, processed_at,
                               error_message, retry_count
                        FROM job_videos 
                        WHERE job_id = $1 
                        ORDER BY created_at
                    """
                    video_records = await conn.fetch(videos_query, job_id)
                    job["videos"] = [dict(v) for v in video_records]

                    # Separate completed files for backwards compatibility
                    files = []
                    for video in job["videos"]:
                        if video["status"] == "completed" and (
                            video["file_path"] or video["s3_key"]
                        ):
                            file_info = {
                                "video_id": video["id"],
                                "title": video["title"],
                                "file_path": video["file_path"],
                                "s3_key": video["s3_key"],
                                "file_size": video["file_size"],
                            }
                            files.append(file_info)

                    job["files"] = files

                return job

        except Exception as e:
            logger.error(f"Failed to get job {job_id} from database: {str(e)}")
            return None

    @staticmethod
    async def update_job_progress_db(
        job_id: str, **updates
    ) -> Optional[Dict[str, Any]]:
        """
        Update job progress with atomic operations. Prevents race conditions
        by using SQL atomic increments and proper WHERE clauses.

        Supports both individual field updates and atomic increments:
        - completed_increment=1: Atomically increment completed count
        - status='processing': Update status
        - files_append=file_info: Add completed file (updates video record)

        Args:
            job_id: Job identifier
            **updates: Fields to update (supports _increment suffix for atomic ops)

        Returns:
            Updated job record or None on error
        """
        try:
            async with get_db_transaction() as tx:
                # Build dynamic UPDATE query for jobs table
                set_clauses = []
                params = [job_id]
                param_count = 2

                # Handle special operations
                files_to_update = []

                for key, value in updates.items():
                    if key.endswith("_increment"):
                        # Atomic increment operations
                        base_key = key.replace("_increment", "")
                        if base_key in [
                            "completed",
                            "failed_count",
                            "processed_count",
                            "credits_used",
                        ]:
                            set_clauses.append(
                                f"{base_key} = {base_key} + ${param_count}"
                            )
                            params.append(value)
                            param_count += 1

                    elif key == "files_append":
                        # This represents a completed video - update job_videos table
                        files_to_update.append(value)

                    elif key in [
                        "status",
                        "reservation_id",
                        "lambda_dispatched_count",
                        "lambda_dispatch_time",
                        "prefetch_completed",
                        "error_message",
                        "end_time",
                        "duration",
                        "videos_metadata",
                    ]:
                        # Direct field updates
                        if key == "lambda_dispatch_time":
                            # Convert Unix timestamp to PostgreSQL timestamp
                            value = (
                                datetime.fromtimestamp(value)
                                if isinstance(value, (int, float))
                                else value
                            )
                        elif key == "videos_metadata":
                            value = json.dumps(value) if value else None

                        set_clauses.append(f"{key} = ${param_count}")
                        params.append(value)
                        param_count += 1

                # Update jobs table if there are changes
                if set_clauses:
                    # Add updated_at timestamp
                    set_clauses.append("updated_at = NOW()")

                    update_query = f"""
                        UPDATE jobs 
                        SET {', '.join(set_clauses)}
                        WHERE job_id = $1
                        RETURNING *
                    """

                    job_record = await tx.fetchrow(update_query, *params)

                    if not job_record:
                        logger.warning(
                            f"Job {job_id} not found for update - checking if it exists"
                        )
                        # Check if job exists at all
                        job_exists = await tx.fetchval(
                            "SELECT 1 FROM jobs WHERE job_id = $1", job_id
                        )
                        if job_exists:
                            logger.warning(
                                f"Job {job_id} exists but update returned no rows - possible race condition"
                            )
                        else:
                            logger.error(f"Job {job_id} does not exist in database")
                        return None

                # Update video records for completed files
                for file_info in files_to_update:
                    video_id = file_info.get("video_id")
                    if video_id:
                        await tx.execute(
                            """
                            UPDATE job_videos 
                            SET status = 'completed',
                                file_path = $3,
                                s3_key = $4,
                                file_size = $5,
                                processed_at = NOW(),
                                updated_at = NOW()
                            WHERE job_id = $1 AND video_id = $2
                        """,
                            job_id,
                            video_id,
                            file_info.get("file_path"),
                            file_info.get("s3_key"),
                            file_info.get("file_size", 0),
                        )

                # Return updated job data
                return await JobManager.get_job_from_db(job_id, include_videos=False)

        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {str(e)}")
            return None

    @staticmethod
    async def mark_video_completed(
        job_id: str, video_id: str, file_info: Dict[str, Any]
    ) -> bool:
        """
        Mark a specific video as completed and update job counters atomically.
        This is called by Lambda functions when they finish processing.

        Args:
            job_id: Job identifier
            video_id: Video identifier
            file_info: Dict with file_path, s3_key, file_size, etc.

        Returns:
            True on success, False on error
        """
        try:
            async with get_db_transaction() as tx:
                # Update video status
                video_result = await tx.execute(
                    """
                    UPDATE job_videos 
                    SET status = 'completed',
                        file_path = $3,
                        s3_key = $4,
                        file_size = $5,
                        processed_at = NOW(),
                        updated_at = NOW()
                    WHERE job_id = $1 AND video_id = $2 AND status != 'completed'
                """,
                    job_id,
                    video_id,
                    file_info.get("file_path"),
                    file_info.get("s3_key"),
                    file_info.get("file_size", 0),
                )

                # Only increment job counters if video was actually updated
                if video_result == "UPDATE 1":
                    await tx.execute(
                        """
                        UPDATE jobs 
                        SET completed = completed + 1,
                            processed_count = processed_count + 1,
                            credits_used = credits_used + 1,
                            updated_at = NOW()
                        WHERE job_id = $1
                    """,
                        job_id,
                    )

                    logger.debug(
                        f"Marked video {video_id} as completed for job {job_id}"
                    )
                    return True
                else:
                    logger.debug(
                        f"Video {video_id} was already completed for job {job_id}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to mark video {video_id} as completed: {str(e)}")
            return False

    @staticmethod
    async def mark_video_failed(job_id: str, video_id: str, error_message: str) -> bool:
        """
        Mark a specific video as failed and update job counters atomically.

        Args:
            job_id: Job identifier
            video_id: Video identifier
            error_message: Error description

        Returns:
            True on success, False on error
        """
        try:
            async with get_db_transaction() as tx:
                # Update video status
                video_result = await tx.execute(
                    """
                    UPDATE job_videos 
                    SET status = 'failed',
                        error_message = $3,
                        processed_at = NOW(),
                        retry_count = retry_count + 1,
                        updated_at = NOW()
                    WHERE job_id = $1 AND video_id = $2 AND status NOT IN ('completed', 'failed')
                """,
                    job_id,
                    video_id,
                    error_message,
                )

                # Only increment job counters if video was actually updated
                if video_result == "UPDATE 1":
                    await tx.execute(
                        """
                        UPDATE jobs 
                        SET failed_count = failed_count + 1,
                            processed_count = processed_count + 1,
                            credits_used = credits_used + 1,
                            updated_at = NOW()
                        WHERE job_id = $1
                    """,
                        job_id,
                    )

                    logger.debug(
                        f"Marked video {video_id} as failed for job {job_id}: {error_message}"
                    )
                    return True
                else:
                    logger.debug(
                        f"Video {video_id} was already processed for job {job_id}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to mark video {video_id} as failed: {str(e)}")
            return False

    @staticmethod
    async def get_job_videos_status(job_id: str) -> List[Dict[str, Any]]:
        """
        Get status of all videos in a job. Useful for progress monitoring.

        Args:
            job_id: Job identifier

        Returns:
            List of video status dictionaries
        """
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT video_id, title, status, file_path, s3_key, 
                           error_message, retry_count, processed_at
                    FROM job_videos 
                    WHERE job_id = $1 
                    ORDER BY created_at
                """
                records = await conn.fetch(query, job_id)
                return [dict(r) for r in records]

        except Exception as e:
            logger.error(f"Failed to get video status for job {job_id}: {str(e)}")
            return []

    @staticmethod
    async def get_jobs_by_user(
        user_id: str, limit: int = 50, status_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get jobs for a specific user. Useful for user dashboard.

        Args:
            user_id: User identifier
            limit: Maximum number of jobs to return
            status_filter: Optional status filter ('completed', 'processing', etc.)

        Returns:
            List of job dictionaries
        """
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT job_id, source_type, source_name, status, 
                           total_videos, completed, failed_count, 
                           created_at, updated_at, duration
                    FROM jobs 
                    WHERE user_id = $1
                """
                params = [user_id]

                if status_filter:
                    query += " AND status = $2"
                    params.append(status_filter)

                query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)

                records = await conn.fetch(query, *params)
                return [dict(r) for r in records]

        except Exception as e:
            logger.error(f"Failed to get jobs for user {user_id}: {str(e)}")
            return []

    @staticmethod
    async def update_job_status_safe(
        job_id: str, new_status: str, expected_current_status: str = None
    ) -> bool:
        """
        Update job status with optional conflict detection.
        Prevents race condition overwrites by checking current status.

        Args:
            job_id: Job identifier
            new_status: Status to set
            expected_current_status: Only update if current status matches this

        Returns:
            True if update succeeded, False if conflict or error
        """
        try:
            async with get_db_connection() as conn:
                if expected_current_status:
                    # Conditional update to prevent race conditions
                    result = await conn.execute(
                        """
                        UPDATE jobs 
                        SET status = $2, updated_at = NOW(), version = version + 1
                        WHERE job_id = $1 AND status = $3
                    """,
                        job_id,
                        new_status,
                        expected_current_status,
                    )

                    success = result == "UPDATE 1"
                    if not success:
                        logger.warning(
                            f"Status update conflict for job {job_id}: "
                            f"expected '{expected_current_status}', trying to set '{new_status}'"
                        )
                    else:
                        logger.info(
                            f"Job {job_id}: Status updated {expected_current_status} -> {new_status}"
                        )

                    return success
                else:
                    # Force update (for system operations)
                    await conn.execute(
                        """
                        UPDATE jobs 
                        SET status = $2, updated_at = NOW(), version = version + 1
                        WHERE job_id = $1
                    """,
                        job_id,
                        new_status,
                    )

                    logger.info(f"Job {job_id}: Status force updated to {new_status}")
                    return True

        except Exception as e:
            logger.error(f"Failed to update status for job {job_id}: {str(e)}")
            return False


# Convenience functions for backwards compatibility with existing code
async def save_job_to_db(job_id: str, job_data: Dict[str, Any]) -> None:
    """Backwards compatibility wrapper for job creation"""
    videos = job_data.pop("videos", [])

    await JobManager.create_job_with_videos(
        job_id=job_id,
        user_id=job_data["user_id"],
        source_type=job_data.get("source_type", "channel"),
        source_id=job_data.get(
            "source_id", job_data.get("channel_id", job_data.get("playlist_id"))
        ),
        source_name=job_data.get(
            "source_name", job_data.get("channel_name", "Unknown")
        ),
        videos=videos,
        formatting_options=job_data.get("formatting_options", {}),
        **job_data,
    )


async def load_job_from_db(job_id: str) -> Optional[Dict[str, Any]]:
    """Backwards compatibility wrapper for job retrieval"""
    return await JobManager.get_job_from_db(job_id, include_videos=True)


async def update_job_progress_db(job_id: str, **updates) -> Optional[Dict[str, Any]]:
    """Backwards compatibility wrapper for job updates"""
    return await JobManager.update_job_progress_db(job_id, **updates)
