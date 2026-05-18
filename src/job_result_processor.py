#!/usr/bin/env python3
"""
Shared processing logic for Lambda transcript results.

This is used by both the legacy HTTP callback endpoints and the SQS consumer so
job progress, finalization, and logging all follow the same path.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "db_youtube_transcripts"))

from db_youtube_transcripts.job_manager import JobManager


logger = logging.getLogger(__name__)


async def _get_job_status(job_id: str) -> Dict[str, Any]:
    job = await JobManager.get_job_status_from_db(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    return job


def _log_execution_timing(job: Dict[str, Any], video_id: str) -> None:
    dispatch_time = job.get("lambda_dispatch_time")
    if not dispatch_time:
        return

    if isinstance(dispatch_time, datetime):
        dispatch_time = dispatch_time.timestamp()

    execution_time = time.time() - dispatch_time
    if execution_time > 300:
        logger.warning(
            f"Video {video_id} took {execution_time / 60:.1f} minutes to complete "
            f"(potential Lambda delay/timeout recovery)"
        )
    elif execution_time > 120:
        logger.info(
            f"Video {video_id} took {execution_time / 60:.1f} minutes to complete"
        )


def _log_timeout_state(job: Dict[str, Any], video_id: str, event_label: str) -> None:
    if job.get("timeout_occurred"):
        logger.info(
            f"Video {video_id} {event_label} after timeout was triggered - "
            f"this is a late-arriving Lambda response"
        )


async def _finalize_job_if_needed(job_id: str) -> Dict[str, Any]:
    finalized_job = await JobManager.finalize_job_if_complete(job_id)
    if not finalized_job:
        raise ValueError(f"Job {job_id} not found during finalization")

    if finalized_job["processed_count"] >= finalized_job[
        "total_videos"
    ] and finalized_job.get("finalized"):
        logger.info(
            f"Job {job_id} finalized with status={finalized_job['status']} "
            f"(refund_applied={finalized_job.get('refund_applied', False)})"
        )

    return finalized_job


async def process_video_completion(
    job_id: str, completion_data: Dict[str, Any]
) -> Dict[str, Any]:
    video_id = completion_data["video_id"]
    job = await _get_job_status(job_id)

    _log_execution_timing(job, video_id)
    _log_timeout_state(job, video_id, "completed")

    was_updated = await JobManager.mark_video_completed(
        job_id=job_id,
        video_id=video_id,
        file_info={
            "s3_key": completion_data["s3_key"],
            "transcript_length": completion_data.get("transcript_length", 0),
            "status": "completed",
        },
    )

    if was_updated:
        logger.info(f"Video {video_id} completed for job {job_id}")
    else:
        logger.info(f"Video {video_id} completion for job {job_id} was already applied")

    updated_job = await _finalize_job_if_needed(job_id)
    return {
        "status": "updated",
        "job_id": job_id,
        "video_id": video_id,
        "job_status": updated_job["status"],
        "processed_count": updated_job["processed_count"],
        "total_videos": updated_job["total_videos"],
    }


async def process_video_failure(
    job_id: str, failure_data: Dict[str, Any]
) -> Dict[str, Any]:
    video_id = failure_data["video_id"]
    job = await _get_job_status(job_id)

    _log_timeout_state(job, video_id, "failed")

    was_updated = await JobManager.mark_video_failed(
        job_id=job_id,
        video_id=video_id,
        error_message=failure_data.get("error", "Unknown error"),
    )

    if was_updated:
        logger.warning(
            f"Video {video_id} failed for job {job_id}: {failure_data.get('error', 'Unknown error')} "
            f"(error_type={failure_data.get('error_type', 'unknown')}, "
            f"stage={failure_data.get('stage', 'unknown')}, "
            f"retriable={failure_data.get('retriable', False)}, "
            f"attempts={failure_data.get('attempts', 1)})"
        )
    else:
        logger.info(f"Video {video_id} failure for job {job_id} was already applied")

    updated_job = await _finalize_job_if_needed(job_id)
    return {
        "status": "updated",
        "job_id": job_id,
        "video_id": video_id,
        "job_status": updated_job["status"],
        "processed_count": updated_job["processed_count"],
        "total_videos": updated_job["total_videos"],
    }


async def process_lambda_result_message(message: Dict[str, Any]) -> Dict[str, Any]:
    event_type = message.get("event_type")
    job_id = message.get("job_id")

    if not event_type:
        raise ValueError("Lambda result message is missing event_type")
    if not job_id:
        raise ValueError("Lambda result message is missing job_id")

    if event_type == "video_completed":
        return await process_video_completion(job_id, message)
    if event_type == "video_failed":
        return await process_video_failure(job_id, message)

    raise ValueError(f"Unsupported event_type: {event_type}")
