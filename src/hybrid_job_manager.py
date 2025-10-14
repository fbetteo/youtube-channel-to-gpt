#!/usr/bin/env python3
"""
Job Migration Wrapper - Provides seamless transition from file-based to database-based job management
Allows both systems to coexist during migration period
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Any

# Add the db_youtube_transcripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db_youtube_transcripts"))

from db_youtube_transcripts.job_manager import JobManager
from db_youtube_transcripts.database import init_db_pool, close_db_pool

logger = logging.getLogger(__name__)

# Migration control flags
USE_DATABASE = True  # Set to True to enable database operations
FALLBACK_TO_FILES = True  # Set to True to fallback to file operations on DB errors


class HybridJobManager:
    """
    Hybrid job manager that can use both database and file systems.
    Provides backwards compatibility during migration.
    """

    def __init__(self):
        self._db_initialized = False

    async def _ensure_db_initialized(self):
        """Ensure database pool is initialized"""
        if not self._db_initialized and USE_DATABASE:
            try:
                await init_db_pool()
                self._db_initialized = True
                logger.info("Database pool initialized for job management")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                if not FALLBACK_TO_FILES:
                    raise

    async def create_job(
        self, job_id: str, job_data: Dict[str, Any], videos: List[Dict[str, Any]] = None
    ) -> str:
        """
        Create a job using the appropriate backend (database preferred, file fallback)
        """
        if videos is None:
            videos = job_data.get("videos", [])

        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                logger.info(
                    f"Creating job {job_id} in database with {len(videos)} videos"
                )

                # Extract required fields for database
                source_type = "playlist" if job_data.get("playlist_id") else "channel"
                source_id = (
                    job_data.get("playlist_id")
                    or job_data.get("channel_id")
                    or job_data.get("source_id")
                )
                source_name = job_data.get("source_name") or job_data.get(
                    "channel_name", "Unknown"
                )

                logger.debug(
                    f"Job data: user_id={job_data.get('user_id')}, source_type={source_type}, source_id={source_id}"
                )

                # Ensure user exists in user_credits table (required for foreign key)
                try:
                    from transcript_api import CreditManager

                    CreditManager.create_user_credits(job_data["user_id"], 0)
                    logger.debug(
                        f"Ensured user {job_data['user_id']} exists in user_credits table"
                    )
                except Exception as user_error:
                    logger.warning(f"Could not ensure user exists: {user_error}")

                # Create formatting options dict
                formatting_options = {
                    "include_timestamps": job_data.get("include_timestamps", False),
                    "include_video_title": job_data.get("include_video_title", True),
                    "include_video_id": job_data.get("include_video_id", True),
                    "include_video_url": job_data.get("include_video_url", True),
                    "include_view_count": job_data.get("include_view_count", False),
                    "concatenate_all": job_data.get("concatenate_all", False),
                }

                await JobManager.create_job_with_videos(
                    job_id=job_id,
                    user_id=job_data["user_id"],
                    source_type=source_type,
                    source_id=source_id,
                    source_name=source_name,
                    videos=videos,
                    formatting_options=formatting_options,
                    credits_reserved=job_data.get("credits_reserved", len(videos)),
                    reservation_id=job_data.get("reservation_id"),
                    videos_metadata=job_data.get("videos_metadata"),
                )

                logger.info(f"Successfully created job {job_id} in database")

                # Also save to file system for backwards compatibility (during transition)
                if FALLBACK_TO_FILES:
                    try:
                        from youtube_service import save_job_to_file

                        save_job_to_file(job_id, job_data)
                        logger.debug(f"Also saved job {job_id} to file system")
                    except Exception as file_error:
                        logger.warning(
                            f"Failed to save job {job_id} to file system: {file_error}"
                        )

                return job_id

            except Exception as db_error:
                logger.error(
                    f"Failed to create job {job_id} in database: {db_error}",
                    exc_info=True,
                )
                if not FALLBACK_TO_FILES:
                    raise

        # Fallback to file system
        if FALLBACK_TO_FILES:
            try:
                from youtube_service import save_job_to_file

                save_job_to_file(job_id, job_data)
                logger.info(f"Created job {job_id} in file system (fallback)")
                return job_id
            except Exception as file_error:
                logger.error(
                    f"Failed to create job {job_id} in file system: {file_error}"
                )
                raise

        raise Exception(f"Failed to create job {job_id} in any backend")

    async def get_job(
        self, job_id: str, include_videos: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a job using the appropriate backend (database preferred, file fallback)
        """
        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                job = await JobManager.get_job_from_db(job_id, include_videos)
                if job:
                    logger.debug(f"Retrieved job {job_id} from database")
                    return job
            except Exception as db_error:
                logger.error(f"Failed to get job {job_id} from database: {db_error}")
                if not FALLBACK_TO_FILES:
                    raise

        # Fallback to file system
        if FALLBACK_TO_FILES:
            try:
                from youtube_service import load_job_from_file

                job = load_job_from_file(job_id)
                if job:
                    logger.debug(f"Retrieved job {job_id} from file system (fallback)")
                    return job
            except Exception as file_error:
                logger.error(
                    f"Failed to get job {job_id} from file system: {file_error}"
                )

        return None

    async def update_job(self, job_id: str, **updates) -> Optional[Dict[str, Any]]:
        """
        Update a job using the appropriate backend (database preferred, file fallback)
        """
        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                result = await JobManager.update_job_progress_db(job_id, **updates)
                if result:
                    logger.debug(f"Updated job {job_id} in database")

                    # Also update file system during transition
                    if FALLBACK_TO_FILES:
                        try:
                            from youtube_service import update_job_progress

                            update_job_progress(job_id, **updates)
                            logger.debug(f"Also updated job {job_id} in file system")
                        except Exception as file_error:
                            logger.warning(
                                f"Failed to update job {job_id} in file system: {file_error}"
                            )

                    return result
            except Exception as db_error:
                logger.error(f"Failed to update job {job_id} in database: {db_error}")
                if not FALLBACK_TO_FILES:
                    raise

        # Fallback to file system
        if FALLBACK_TO_FILES:
            try:
                from youtube_service import update_job_progress

                update_job_progress(job_id, **updates)
                logger.debug(f"Updated job {job_id} in file system (fallback)")
                # Return updated job data
                return await self.get_job(job_id, include_videos=False)
            except Exception as file_error:
                logger.error(
                    f"Failed to update job {job_id} in file system: {file_error}"
                )
                raise

        return None

    async def mark_video_completed(
        self, job_id: str, video_id: str, file_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Mark a video as completed using the appropriate backend and return updated job data
        """
        success = False

        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                success = await JobManager.mark_video_completed(
                    job_id, video_id, file_info
                )
                if success:
                    logger.debug(f"Marked video {video_id} as completed in database")
                    # Return updated job data
                    return await self.get_job(job_id, include_videos=False)
            except Exception as db_error:
                logger.error(
                    f"Failed to mark video {video_id} completed in database: {db_error}"
                )
                if not FALLBACK_TO_FILES:
                    raise

        # Also update file system (during transition) or as fallback
        if FALLBACK_TO_FILES and not success:
            try:
                from youtube_service import update_job_progress, load_job_from_file

                updated_job = update_job_progress(
                    job_id,
                    files_append=file_info,
                    completed_increment=1,
                    credits_used_increment=1,
                )
                logger.debug(f"Marked video {video_id} as completed in file system")
                return updated_job if updated_job else load_job_from_file(job_id)
            except Exception as file_error:
                logger.error(
                    f"Failed to mark video {video_id} completed in file system: {file_error}"
                )
                raise

        return None

    async def mark_video_failed(
        self, job_id: str, video_id: str, error_message: str
    ) -> Optional[Dict[str, Any]]:
        """
        Mark a video as failed using the appropriate backend and return updated job data
        """
        success = False

        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                success = await JobManager.mark_video_failed(
                    job_id, video_id, error_message
                )
                if success:
                    logger.debug(f"Marked video {video_id} as failed in database")
                    # Return updated job data
                    return await self.get_job(job_id, include_videos=False)
            except Exception as db_error:
                logger.error(
                    f"Failed to mark video {video_id} failed in database: {db_error}"
                )
                if not FALLBACK_TO_FILES:
                    raise

        # Also update file system (during transition) or as fallback
        if FALLBACK_TO_FILES and not success:
            try:
                from youtube_service import update_job_progress, load_job_from_file

                updated_job = update_job_progress(
                    job_id, failed_count_increment=1, credits_used_increment=1
                )
                logger.debug(f"Marked video {video_id} as failed in file system")
                return updated_job if updated_job else load_job_from_file(job_id)
            except Exception as file_error:
                logger.error(
                    f"Failed to mark video {video_id} failed in file system: {file_error}"
                )
                raise

        return None

    async def update_job_status_safe(
        self, job_id: str, new_status: str, expected_current_status: str = None
    ) -> bool:
        """
        Update job status safely with conflict detection
        """
        success = False

        # Try database first
        if USE_DATABASE:
            try:
                await self._ensure_db_initialized()
                success = await JobManager.update_job_status_safe(
                    job_id, new_status, expected_current_status
                )
                if success:
                    logger.debug(
                        f"Updated job {job_id} status to {new_status} in database"
                    )
            except Exception as db_error:
                logger.error(
                    f"Failed to update job {job_id} status in database: {db_error}"
                )
                if not FALLBACK_TO_FILES:
                    raise

        # Also update file system (during transition) or as fallback
        if FALLBACK_TO_FILES:
            try:
                from youtube_service import update_job_progress

                update_job_progress(job_id, status=new_status)
                logger.debug(
                    f"Updated job {job_id} status to {new_status} in file system"
                )
                success = True
            except Exception as file_error:
                logger.error(
                    f"Failed to update job {job_id} status in file system: {file_error}"
                )
                if not success:  # Only raise if database also failed
                    raise

        return success

    async def cleanup(self):
        """Clean up resources"""
        if self._db_initialized:
            try:
                await close_db_pool()
                logger.info("Database pool closed")
            except Exception as e:
                logger.error(f"Failed to close database pool: {e}")


# Global instance for easy importing
hybrid_job_manager = HybridJobManager()


# Convenience functions that match the existing API
async def save_job_hybrid(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save a job using hybrid approach (database + file fallback)"""
    await hybrid_job_manager.create_job(job_id, job_data)


async def load_job_hybrid(job_id: str) -> Optional[Dict[str, Any]]:
    """Load a job using hybrid approach (database + file fallback)"""
    return await hybrid_job_manager.get_job(job_id)


async def update_job_progress_hybrid(
    job_id: str, **updates
) -> Optional[Dict[str, Any]]:
    """Update job progress using hybrid approach (database + file fallback)"""
    return await hybrid_job_manager.update_job(job_id, **updates)


async def mark_video_completed_hybrid(
    job_id: str, video_id: str, file_info: Dict[str, Any]
) -> bool:
    """Mark video as completed using hybrid approach"""
    return await hybrid_job_manager.mark_video_completed(job_id, video_id, file_info)


async def mark_video_failed_hybrid(
    job_id: str, video_id: str, error_message: str
) -> bool:
    """Mark video as failed using hybrid approach"""
    return await hybrid_job_manager.mark_video_failed(job_id, video_id, error_message)


async def update_job_status_safe_hybrid(
    job_id: str, new_status: str, expected_current_status: str = None
) -> bool:
    """Update job status safely using hybrid approach"""
    return await hybrid_job_manager.update_job_status_safe(
        job_id, new_status, expected_current_status
    )
