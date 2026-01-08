"""
Developer API Router - v1

Programmatic access to YouTube transcript services via API keys.
These endpoints mirror the frontend functionality but are designed for automation.

All endpoints require authentication via X-API-Key header.
Credits are deducted from the API key owner's account.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator

import youtube_service
from api_key_auth import (
    validate_api_key,
    get_rate_limits,
    increment_api_key_credits_used,
)
from hybrid_job_manager import hybrid_job_manager

logger = logging.getLogger(__name__)

# Create router with /api/v1 prefix
router = APIRouter(prefix="/api/v1", tags=["Developer API"])


# =============================================
# REQUEST/RESPONSE MODELS
# =============================================


class TranscriptOptions(BaseModel):
    """Common formatting options for transcripts."""

    include_timestamps: bool = Field(
        default=False, description="Include [MM:SS] timestamps in transcript text"
    )
    include_video_title: bool = Field(
        default=True, description="Include video title in transcript header"
    )
    include_video_id: bool = Field(
        default=True, description="Include video ID in transcript header"
    )
    include_video_url: bool = Field(
        default=True, description="Include video URL in transcript header"
    )
    include_view_count: bool = Field(
        default=False, description="Include view count in transcript header"
    )
    concatenate_all: bool = Field(
        default=False,
        description="Return all transcripts in a single file instead of individual files",
    )


class SingleTranscriptRequest(BaseModel):
    """Request to download a single video transcript."""

    video_url: str = Field(
        ..., description="YouTube video URL or video ID (e.g., 'dQw4w9WgXcQ')"
    )
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in transcript"
    )


class ChannelTranscriptRequest(BaseModel):
    """Request to download all transcripts from a channel."""

    channel: str = Field(
        ...,
        description="YouTube channel name, handle (@channel), or channel ID",
        examples=["@mkbhd", "MKBHD", "UCBcRF18a7Qf58cCRy5xuWwQ"],
    )
    max_videos: Optional[int] = Field(
        default=None,
        description="Maximum number of videos to process. None = all videos.",
        ge=1,
        le=2000,
    )
    options: TranscriptOptions = Field(
        default_factory=TranscriptOptions, description="Transcript formatting options"
    )


class PlaylistTranscriptRequest(BaseModel):
    """Request to download all transcripts from a playlist."""

    playlist: str = Field(
        ...,
        description="YouTube playlist ID or URL",
        examples=["PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"],
    )
    max_videos: Optional[int] = Field(
        default=None,
        description="Maximum number of videos to process. None = all videos.",
        ge=1,
        le=2000,
    )
    options: TranscriptOptions = Field(
        default_factory=TranscriptOptions, description="Transcript formatting options"
    )


class JobResponse(BaseModel):
    """Response when a job is created."""

    job_id: str
    status: str
    total_videos: int
    source_type: str
    source_name: str
    credits_reserved: int
    message: str


class JobStatusResponse(BaseModel):
    """Response with job status and progress."""

    job_id: str
    status: str
    total_videos: int
    processed_count: int
    completed: int
    failed_count: int
    credits_used: int
    source_type: str
    source_name: str
    download_ready: bool
    download_url: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    error_message: Optional[str] = None


class CreditsResponse(BaseModel):
    """Response with user's credit balance."""

    credits: int
    api_key_name: str
    total_api_requests: int
    total_credits_via_api: int


class SingleTranscriptResponse(BaseModel):
    """Response with single video transcript."""

    video_id: str
    title: str
    transcript: str
    language: str
    character_count: int
    credits_used: int = 1


class ChannelInfoResponse(BaseModel):
    """Response with channel information."""

    channel_id: str
    title: str
    description: Optional[str]
    subscriber_count: Optional[int]
    video_count: Optional[int]
    thumbnail_url: Optional[str]


class VideoInfo(BaseModel):
    """Video metadata."""

    id: str
    title: str
    url: str
    duration_seconds: Optional[int]
    duration_category: Optional[str]
    view_count: Optional[int]
    published_at: Optional[str]


class ChannelVideosResponse(BaseModel):
    """Response with channel videos list."""

    channel_id: str
    channel_name: str
    total_videos: int
    videos: List[VideoInfo]
    duration_breakdown: Dict[str, int]


# =============================================
# HELPER FUNCTIONS
# =============================================


async def get_user_credits(user_id: str) -> int:
    """Get credit balance for a user."""
    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            "SELECT credits FROM user_credits WHERE user_id = $1", user_id
        )
        return row["credits"] if row else 0


async def reserve_credits(user_id: str, amount: int) -> str:
    """Reserve credits for a job. Returns reservation_id."""
    from db_youtube_transcripts.database import get_db_connection

    reservation_id = str(uuid.uuid4())

    async with get_db_connection() as conn:
        # Atomic deduction
        result = await conn.execute(
            """
            UPDATE user_credits 
            SET credits = credits - $1 
            WHERE user_id = $2 AND credits >= $1
            """,
            amount,
            user_id,
        )

        rows_affected = int(result.split()[-1])
        if rows_affected == 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {amount} credits.",
            )

    return reservation_id


async def finalize_credits(
    user_id: str, reservation_id: str, credits_used: int, credits_reserved: int
) -> None:
    """Finalize credit usage, refunding unused credits."""
    from db_youtube_transcripts.database import get_db_connection

    credits_to_refund = credits_reserved - credits_used

    if credits_to_refund > 0:
        async with get_db_connection() as conn:
            await conn.execute(
                "UPDATE user_credits SET credits = credits + $1 WHERE user_id = $2",
                credits_to_refund,
                user_id,
            )
            logger.info(f"Refunded {credits_to_refund} credits to user {user_id}")


# =============================================
# ENDPOINTS
# =============================================


@router.get("/", summary="API Health Check")
async def api_root():
    """
    Health check for the Developer API.
    """
    return {
        "service": "YouTube Transcript Developer API",
        "version": "v1",
        "status": "online",
        "documentation": "/api/v1/docs",
    }


@router.get("/account/credits", response_model=CreditsResponse, summary="Get Credits")
async def get_credits(api_key_data: Dict = Depends(validate_api_key)):
    """
    Get your current credit balance.

    Credits are shared between API access and web dashboard.
    """
    credits = await get_user_credits(api_key_data["user_id"])

    return CreditsResponse(
        credits=credits,
        api_key_name=api_key_data["key_name"],
        total_api_requests=api_key_data["total_requests"],
        total_credits_via_api=api_key_data["total_credits_used"],
    )


@router.post(
    "/transcripts/single",
    response_model=SingleTranscriptResponse,
    summary="Single Video Transcript",
)
async def get_single_transcript(
    request: SingleTranscriptRequest, api_key_data: Dict = Depends(validate_api_key)
):
    """
    Get transcript for a single YouTube video.

    **Cost:** 1 credit per video

    Returns the full transcript text immediately (synchronous).
    """
    user_id = api_key_data["user_id"]

    # Check credits
    credits = await get_user_credits(user_id)
    if credits < 1:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits. Need 1 credit for single video transcript.",
        )

    try:
        # Extract video ID
        video_id = youtube_service.extract_youtube_id(request.video_url)

        # Reserve credit
        await reserve_credits(user_id, 1)

        # Get transcript using existing service
        transcript_text, _, metadata = await youtube_service.get_single_transcript(
            video_id=video_id,
            output_dir=None,
            include_timestamps=request.include_timestamps,
        )

        # Track credit usage on API key
        await increment_api_key_credits_used(api_key_data["key_id"], 1)

        return SingleTranscriptResponse(
            video_id=video_id,
            title=metadata.get("title", "Unknown"),
            transcript=transcript_text,
            language=metadata.get("transcript_language", "unknown"),
            character_count=len(transcript_text),
            credits_used=1,
        )

    except ValueError as e:
        # Refund credit if extraction failed
        await finalize_credits(user_id, "", 0, 1)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting single transcript: {e}", exc_info=True)
        await finalize_credits(user_id, "", 0, 1)
        raise HTTPException(status_code=500, detail=f"Failed to get transcript: {e}")


@router.get(
    "/channels/{channel}/info",
    response_model=ChannelInfoResponse,
    summary="Get Channel Info",
)
async def get_channel_info(
    channel: str, api_key_data: Dict = Depends(validate_api_key)
):
    """
    Get information about a YouTube channel.

    **Cost:** Free (no credits)

    Use this to validate a channel before starting a batch download.
    """
    try:
        info = await youtube_service.get_channel_info(channel)

        return ChannelInfoResponse(
            channel_id=info.get("id", ""),
            title=info.get("title", ""),
            description=info.get("description"),
            subscriber_count=info.get("subscriberCount"),
            video_count=info.get("videoCount"),
            thumbnail_url=info.get("thumbnailUrl"),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting channel info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get channel info: {e}")


@router.get(
    "/channels/{channel}/videos",
    response_model=ChannelVideosResponse,
    summary="List Channel Videos",
)
async def list_channel_videos(
    channel: str, api_key_data: Dict = Depends(validate_api_key)
):
    """
    List all videos from a YouTube channel.

    **Cost:** Free (no credits)

    Returns video metadata including duration categories.
    Use this to preview what will be downloaded.
    """
    try:
        # Get channel info first
        channel_info = await youtube_service.get_channel_info(channel)
        channel_id = channel_info.get("channelId", channel)
        logger.info(f"Listing videos for channel ID: {channel_id}")

        # Get all videos
        videos = await youtube_service.get_all_channel_videos(channel_id)

        # Calculate duration breakdown
        duration_breakdown = {"short": 0, "medium": 0, "long": 0}
        video_list = []

        for v in videos:
            category = v.get("duration_category", "medium")
            duration_breakdown[category] = duration_breakdown.get(category, 0) + 1

            video_list.append(
                VideoInfo(
                    id=v.get("id", ""),
                    title=v.get("title", ""),
                    url=v.get(
                        "url", f"https://www.youtube.com/watch?v={v.get('id', '')}"
                    ),
                    duration_seconds=v.get("duration_seconds"),
                    duration_category=category,
                    view_count=v.get("view_count"),
                    published_at=v.get("published_at"),
                )
            )

        return ChannelVideosResponse(
            channel_id=channel_id,
            channel_name=channel_info.get("title", channel),
            total_videos=len(video_list),
            videos=video_list,
            duration_breakdown=duration_breakdown,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing channel videos: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list channel videos: {e}"
        )


@router.post(
    "/transcripts/channel",
    response_model=JobResponse,
    summary="Download Channel Transcripts",
)
async def download_channel_transcripts(
    request: ChannelTranscriptRequest, api_key_data: Dict = Depends(validate_api_key)
):
    """
    Start downloading all transcripts from a YouTube channel.

    **Cost:** 1 credit per video

    This is an asynchronous operation. Use the returned `job_id` to:
    - Check progress: `GET /api/v1/jobs/{job_id}`
    - Download results: `GET /api/v1/jobs/{job_id}/download`

    Credits are reserved upfront and unused credits are refunded when the job completes.
    """
    user_id = api_key_data["user_id"]

    try:
        # 1. Get channel info
        channel_info = await youtube_service.get_channel_info(request.channel)
        channel_id = channel_info.get("channelId", request.channel)
        channel_name = channel_info.get("title", request.channel)

        # 2. Get all videos
        all_videos = await youtube_service.get_all_channel_videos(channel_id)

        if not all_videos:
            raise HTTPException(status_code=404, detail="No videos found in channel")

        # 3. Apply max_videos limit if specified
        if request.max_videos and request.max_videos < len(all_videos):
            all_videos = all_videos[: request.max_videos]

        num_videos = len(all_videos)

        # 4. Check rate limits
        rate_limits = get_rate_limits(api_key_data["rate_limit_tier"])
        if num_videos > rate_limits["max_videos_per_job"]:
            raise HTTPException(
                status_code=400,
                detail=f"Exceeds maximum videos per job ({rate_limits['max_videos_per_job']}). "
                f"Use max_videos parameter to limit, or upgrade your API tier.",
            )

        # 5. Check credits
        credits = await get_user_credits(user_id)
        if credits < num_videos:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {num_videos} credits, have {credits}.",
            )

        # 6. Reserve credits
        reservation_id = await reserve_credits(user_id, num_videos)

        # 7. Create job
        job_id = str(uuid.uuid4())

        # Convert videos to the format expected by create_job
        videos_for_job = [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "url": v.get("url", f"https://www.youtube.com/watch?v={v.get('id')}"),
                "duration_seconds": v.get("duration_seconds"),
                "duration_category": v.get("duration_category"),
                "view_count": v.get("view_count"),
            }
            for v in all_videos
        ]

        job_data = {
            "status": "initializing",
            "channel_name": channel_name,
            "channel_info": channel_info,
            "source_id": channel_id,
            "source_name": channel_name,
            "source_type": "channel",
            "total_videos": num_videos,
            "completed": 0,
            "failed_count": 0,
            "processed_count": 0,
            "files": [],
            "videos": videos_for_job,
            "start_time": time.time(),
            "user_id": user_id,
            "credits_reserved": num_videos,
            "credits_used": 0,
            "reservation_id": reservation_id,
            "videos_metadata": {},
            "prefetch_completed": False,
            "lambda_dispatched_count": 0,
            "formatting_options": {
                "include_timestamps": request.options.include_timestamps,
                "include_video_title": request.options.include_video_title,
                "include_video_id": request.options.include_video_id,
                "include_video_url": request.options.include_video_url,
                "include_view_count": request.options.include_view_count,
                "concatenate_all": request.options.concatenate_all,
            },
            "api_key_id": api_key_data["key_id"],  # Track which API key initiated
        }

        # Save job to database
        await hybrid_job_manager.create_job(job_id, job_data, videos_for_job)
        logger.info(
            f"API: Created job {job_id} for channel {channel_name} ({num_videos} videos)"
        )

        # 8. Start background processing
        asyncio.create_task(youtube_service.prefetch_and_dispatch_task(job_id))

        # 9. Track credits on API key (will be updated as videos complete)
        await increment_api_key_credits_used(api_key_data["key_id"], num_videos)

        return JobResponse(
            job_id=job_id,
            status="initializing",
            total_videos=num_videos,
            source_type="channel",
            source_name=channel_name,
            credits_reserved=num_videos,
            message=f"Job created. Processing {num_videos} videos from '{channel_name}'. "
            f"Check status at GET /api/v1/jobs/{job_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting channel download: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start channel download: {e}"
        )


@router.post(
    "/transcripts/playlist",
    response_model=JobResponse,
    summary="Download Playlist Transcripts",
)
async def download_playlist_transcripts(
    request: PlaylistTranscriptRequest, api_key_data: Dict = Depends(validate_api_key)
):
    """
    Start downloading all transcripts from a YouTube playlist.

    **Cost:** 1 credit per video

    This is an asynchronous operation. Use the returned `job_id` to check progress
    and download results when complete.
    """
    user_id = api_key_data["user_id"]

    try:
        # 1. Extract playlist ID
        playlist_id = youtube_service.extract_playlist_id(request.playlist)

        # 2. Get playlist info
        playlist_info = await youtube_service.get_playlist_info(playlist_id)
        playlist_name = playlist_info.get("title", playlist_id)

        # 3. Get all videos
        all_videos = await youtube_service.get_all_playlist_videos(playlist_id)

        if not all_videos:
            raise HTTPException(status_code=404, detail="No videos found in playlist")

        # 4. Apply max_videos limit if specified
        if request.max_videos and request.max_videos < len(all_videos):
            all_videos = all_videos[: request.max_videos]

        num_videos = len(all_videos)

        # 5. Check rate limits
        rate_limits = get_rate_limits(api_key_data["rate_limit_tier"])
        if num_videos > rate_limits["max_videos_per_job"]:
            raise HTTPException(
                status_code=400,
                detail=f"Exceeds maximum videos per job ({rate_limits['max_videos_per_job']}). "
                f"Use max_videos parameter to limit.",
            )

        # 6. Check credits
        credits = await get_user_credits(user_id)
        if credits < num_videos:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {num_videos} credits, have {credits}.",
            )

        # 7. Reserve credits
        reservation_id = await reserve_credits(user_id, num_videos)

        # 8. Create job
        job_id = str(uuid.uuid4())

        videos_for_job = [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "url": v.get("url", f"https://www.youtube.com/watch?v={v.get('id')}"),
                "duration_seconds": v.get("duration_seconds"),
                "duration_category": v.get("duration_category"),
                "view_count": v.get("view_count"),
            }
            for v in all_videos
        ]

        job_data = {
            "status": "initializing",
            "playlist_id": playlist_id,
            "playlist_info": playlist_info,
            "source_id": playlist_id,
            "source_name": playlist_name,
            "source_type": "playlist",
            "total_videos": num_videos,
            "completed": 0,
            "failed_count": 0,
            "processed_count": 0,
            "files": [],
            "videos": videos_for_job,
            "start_time": time.time(),
            "user_id": user_id,
            "credits_reserved": num_videos,
            "credits_used": 0,
            "reservation_id": reservation_id,
            "videos_metadata": {},
            "prefetch_completed": False,
            "lambda_dispatched_count": 0,
            "formatting_options": {
                "include_timestamps": request.options.include_timestamps,
                "include_video_title": request.options.include_video_title,
                "include_video_id": request.options.include_video_id,
                "include_video_url": request.options.include_video_url,
                "include_view_count": request.options.include_view_count,
                "concatenate_all": request.options.concatenate_all,
            },
            "api_key_id": api_key_data["key_id"],
        }

        await hybrid_job_manager.create_job(job_id, job_data, videos_for_job)
        logger.info(
            f"API: Created job {job_id} for playlist {playlist_name} ({num_videos} videos)"
        )

        # 9. Start background processing
        asyncio.create_task(youtube_service.prefetch_and_dispatch_task(job_id))

        # 10. Track credits
        await increment_api_key_credits_used(api_key_data["key_id"], num_videos)

        return JobResponse(
            job_id=job_id,
            status="initializing",
            total_videos=num_videos,
            source_type="playlist",
            source_name=playlist_name,
            credits_reserved=num_videos,
            message=f"Job created. Processing {num_videos} videos from playlist '{playlist_name}'. "
            f"Check status at GET /api/v1/jobs/{job_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting playlist download: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start playlist download: {e}"
        )


@router.get(
    "/jobs/{job_id}", response_model=JobStatusResponse, summary="Get Job Status"
)
async def get_job_status(job_id: str, api_key_data: Dict = Depends(validate_api_key)):
    """
    Get the status of a transcript download job.

    Poll this endpoint to track progress. When `download_ready` is true,
    use the `download_url` to retrieve results.
    """
    user_id = api_key_data["user_id"]

    try:
        job = await hybrid_job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Verify ownership
        if str(job.get("user_id")) != str(user_id):
            raise HTTPException(status_code=403, detail="Access denied")

        job_status = job.get("status", "unknown")
        download_ready = job_status in ["completed", "completed_with_errors"]

        # Calculate elapsed time
        start_time = job.get("start_time")
        elapsed = None
        if start_time:
            if isinstance(start_time, (int, float)):
                elapsed = time.time() - start_time
            else:
                # It's a datetime
                elapsed = time.time() - start_time.timestamp()

        return JobStatusResponse(
            job_id=job_id,
            status=job_status,
            total_videos=job.get("total_videos", 0),
            processed_count=job.get("processed_count", 0),
            completed=job.get("completed", 0),
            failed_count=job.get("failed_count", 0),
            credits_used=job.get("credits_used", 0),
            source_type=job.get("source_type", "unknown"),
            source_name=job.get("source_name", "unknown"),
            download_ready=download_ready,
            download_url=f"/api/v1/jobs/{job_id}/download" if download_ready else None,
            elapsed_seconds=elapsed,
            error_message=job.get("error_message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")


@router.get("/jobs/{job_id}/download", summary="Download Job Results")
async def download_job_results(
    job_id: str, api_key_data: Dict = Depends(validate_api_key)
):
    """
    Download the transcript results for a completed job.

    Returns a ZIP file containing all transcripts.
    Only available when job status is 'completed' or 'completed_with_errors'.
    """
    from fastapi.responses import Response

    user_id = api_key_data["user_id"]

    try:
        job = await hybrid_job_manager.get_job(job_id, include_videos=True)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Verify ownership
        if str(job.get("user_id")) != str(user_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Verify job is ready
        job_status = job.get("status")
        if job_status not in ["completed", "completed_with_errors"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job not ready for download. Status: {job_status}",
            )

        # Create ZIP from S3
        logger.info(f"API: Creating ZIP for job {job_id}")
        zip_start = time.time()

        try:
            zip_buffer = await youtube_service.create_transcript_zip_from_s3_concurrent(
                job_id
            )
        except Exception as e:
            logger.warning(f"Concurrent download failed, trying sequential: {e}")
            zip_buffer = await youtube_service.create_transcript_zip_from_s3_sequential(
                job_id
            )

        zip_time = time.time() - zip_start
        zip_size = len(zip_buffer.getvalue())

        logger.info(
            f"API: Created ZIP for job {job_id} in {zip_time:.2f}s ({zip_size:,} bytes)"
        )

        # Create filename
        source_name = job.get("source_name", "transcripts")
        safe_name = youtube_service.sanitize_filename(source_name)
        filename = f"{safe_name}_transcripts.zip"

        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Job-ID": job_id,
                "X-Generation-Time": f"{zip_time:.2f}s",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download results: {e}")


@router.get("/jobs", summary="List Jobs")
async def list_jobs(
    api_key_data: Dict = Depends(validate_api_key),
    limit: int = 20,
    status_filter: Optional[str] = None,
):
    """
    List your recent transcript download jobs.

    Optionally filter by status: 'processing', 'completed', 'failed', etc.
    """
    user_id = api_key_data["user_id"]

    try:
        from db_youtube_transcripts.database import get_db_connection

        async with get_db_connection() as conn:
            if status_filter:
                rows = await conn.fetch(
                    """
                    SELECT job_id, status, source_type, source_name, total_videos,
                           completed, failed_count, credits_used, created_at
                    FROM jobs
                    WHERE user_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    status_filter,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT job_id, status, source_type, source_name, total_videos,
                           completed, failed_count, credits_used, created_at
                    FROM jobs
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    limit,
                )

            return {
                "jobs": [
                    {
                        "job_id": str(row["job_id"]),
                        "status": row["status"],
                        "source_type": row["source_type"],
                        "source_name": row["source_name"],
                        "total_videos": row["total_videos"],
                        "completed": row["completed"],
                        "failed_count": row["failed_count"],
                        "credits_used": row["credits_used"],
                        "created_at": (
                            row["created_at"].isoformat() if row["created_at"] else None
                        ),
                    }
                    for row in rows
                ],
                "count": len(rows),
            }

    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}")
