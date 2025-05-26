#!/usr/bin/env python3
"""
YouTube Service - Service layer for YouTube transcript downloading and processing
"""
import asyncio
import io
import logging
import os
import re
import tempfile
import time
import uuid
import zipfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

# Using the Pydantic v2 compatible settings
from config_v2 import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize YouTube API client with retry for transient failures
try:
    youtube = build("youtube", "v3", developerKey=settings.youtube_api_key)
    logger.info("YouTube API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize YouTube API client: {str(e)}")
    # Fallback to None, will re-attempt connection when needed
    youtube = None

# Initialize YouTube Transcript API with proxy configuration if credentials are provided
try:
    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        proxy_config = WebshareProxyConfig(
            proxy_username=settings.webshare_proxy_username,
            proxy_password=settings.webshare_proxy_password,
        )
        ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
    else:
        ytt_api = YouTubeTranscriptApi()
    logger.info("YouTube Transcript API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize YouTube Transcript API: {str(e)}")
    # We'll handle this in the function calls

# Dictionary to track channel download jobs
channel_download_jobs: Dict[str, Dict[str, Any]] = {}


def extract_youtube_id(url: str) -> str:
    """
    Extract YouTube video ID from various URL formats.

    Args:
        url: YouTube URL or video ID

    Returns:
        The YouTube video ID

    Raises:
        ValueError: If unable to extract a valid YouTube video ID
    """
    # Common YouTube URL patterns
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|\/watch\?v=)([^#\&\?\/\s]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matches, perhaps the URL is already just the ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    raise ValueError(f"Could not extract YouTube video ID from URL: {url}")


def sanitize_filename(filename: str, max_len: int = 30) -> str:
    """
    Sanitizes a filename to be safe for all operating systems.

    Args:
        filename: The filename to sanitize
        max_len: Maximum length for the filename

    Returns:
        A sanitized filename
    """
    # Remove characters invalid in Windows/Linux filenames
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
    # Replace multiple spaces with a single space
    sanitized = re.sub(r"\s+", " ", sanitized)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    # Prevent empty filenames
    if not sanitized:
        sanitized = "_"
    return sanitized[:max_len]


async def get_video_info(video_id: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a YouTube video.

    Args:
        video_id: The YouTube video ID

    Returns:
        Dictionary with video metadata (title, channel, views, etc.)

    Raises:
        ValueError: If video ID is invalid or video doesn't exist
    """
    try:
        # This is a blocking call, so we run it in a thread
        def _fetch_video_info():
            video_request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id,
            )
            return video_request.execute()

        # Run the blocking API call in a thread pool
        video_response = await asyncio.to_thread(_fetch_video_info)

        if not video_response.get("items"):
            raise ValueError(f"No video found with ID: {video_id}")

        video_data = video_response["items"][0]
        snippet = video_data["snippet"]
        statistics = video_data["statistics"]
        content_details = video_data["contentDetails"]

        # Format duration string (PT1H2M3S -> 1:02:03)
        duration = content_details.get("duration", "PT0S")
        duration_str = _format_duration(duration)

        # Extract thumbnail URLs
        thumbnails = snippet.get("thumbnails", {})
        best_thumbnail = (
            thumbnails.get("maxres")
            or thumbnails.get("high")
            or thumbnails.get("medium")
            or thumbnails.get("default")
            or {}
        )

        # Build metadata dictionary
        metadata = {
            "id": video_id,
            "title": snippet.get("title", "Untitled"),
            "description": snippet.get("description", ""),
            "channelId": snippet.get("channelId", ""),
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt": snippet.get("publishedAt", ""),
            "thumbnail": best_thumbnail.get("url", ""),
            "duration": duration_str,
            "viewCount": int(statistics.get("viewCount", 0)),
            "likeCount": int(statistics.get("likeCount", 0)),
            "commentCount": int(statistics.get("commentCount", 0)),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        return metadata

    except Exception as e:
        logger.error(f"Error fetching video info for {video_id}: {str(e)}")
        raise ValueError(f"Failed to get metadata for video {video_id}: {str(e)}")


def _format_duration(duration_str: str) -> str:
    """
    Format ISO 8601 duration string to a human-readable format.

    Args:
        duration_str: ISO 8601 duration string (e.g., 'PT1H2M3S')

    Returns:
        Formatted duration string (e.g., '1:02:03')
    """
    match_hours = re.search(r"(\d+)H", duration_str)
    match_minutes = re.search(r"(\d+)M", duration_str)
    match_seconds = re.search(r"(\d+)S", duration_str)

    hours = int(match_hours.group(1)) if match_hours else 0
    minutes = int(match_minutes.group(1)) if match_minutes else 0
    seconds = int(match_seconds.group(1)) if match_seconds else 0

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


async def get_channel_info(channel_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a YouTube channel.

    Args:
        channel_name: Channel name or ID

    Returns:
        Dictionary with channel information (title, description, stats, etc.)

    Raises:
        ValueError: If channel is not found
    """
    try:
        # First, determine if this is a channel ID or a channel name
        is_channel_id = re.match(r"^UC[\w-]{22}$", channel_name) is not None

        def _fetch_channel_info():
            if is_channel_id:
                # If it's a channel ID, fetch directly
                request = youtube.channels().list(
                    part="snippet,statistics", id=channel_name
                )
            else:
                # Otherwise search for the channel
                search_request = youtube.search().list(
                    part="snippet", q=channel_name, type="channel", maxResults=1
                )
                search_response = search_request.execute()

                if not search_response.get("items"):
                    raise ValueError(f"No channel found with name: {channel_name}")

                channel_id = search_response["items"][0]["id"]["channelId"]
                request = youtube.channels().list(
                    part="snippet,statistics", id=channel_id
                )

            return request.execute()

        # Run the blocking API call in a thread pool
        response = await asyncio.to_thread(_fetch_channel_info)

        if not response.get("items"):
            raise ValueError(f"No channel found with identifier: {channel_name}")

        channel_info = response["items"][0]
        channel_id = channel_info["id"]

        return {
            "title": channel_info["snippet"]["title"],
            "description": channel_info["snippet"].get("description", ""),
            "thumbnail": channel_info["snippet"]["thumbnails"]["high"]["url"],
            "videoCount": int(channel_info["statistics"].get("videoCount", 0)),
            "subscriberCount": int(
                channel_info["statistics"].get("subscriberCount", 0)
            ),
            "viewCount": int(channel_info["statistics"].get("viewCount", 0)),
            "channelId": channel_id,
        }

    except Exception as e:
        logger.error(f"Error fetching channel info for {channel_name}: {str(e)}")
        raise ValueError(f"Failed to get channel information: {str(e)}")


async def get_channel_videos(
    channel_id: str, max_results: int = None
) -> List[Dict[str, str]]:
    """
    Get a list of videos from a YouTube channel.

    Args:
        channel_id: The YouTube channel ID
        max_results: Maximum number of videos to retrieve (default: 30)

    Returns:
        List of dictionaries containing video IDs and titles

    Raises:
        ValueError: If channel ID is invalid or no videos found
    """
    try:

        def _fetch_channel_videos():
            # Get videos with medium duration
            medium_videos = (
                youtube.search()
                .list(
                    part="snippet",
                    channelId=channel_id,
                    type="video",
                    videoDuration="medium",
                    # maxResults=max_results // 2,
                    order="date",
                )
                .execute()
            )

            # Get videos with long duration
            long_videos = (
                youtube.search()
                .list(
                    part="snippet",
                    channelId=channel_id,
                    type="video",
                    videoDuration="long",
                    # maxResults=max_results // 2,
                    order="date",
                )
                .execute()
            )

            # Get videos with long duration
            short_videos = (
                youtube.search()
                .list(
                    part="snippet",
                    channelId=channel_id,
                    type="video",
                    videoDuration="short",
                    # maxResults=max_results // 2,
                    order="date",
                )
                .execute()
            )

            return medium_videos, long_videos, short_videos

        # Run the blocking API call in a thread pool
        medium_videos, long_videos, short_videos = await asyncio.to_thread(
            _fetch_channel_videos
        )

        # Combine and process the results
        video_data = []
        for video in (
            medium_videos.get("items", [])
            + long_videos.get("items", [])
            + short_videos.get("items", [])
        ):
            if video["id"]["kind"] == "youtube#video":
                video_id = video["id"]["videoId"]
                video_title = video["snippet"]["title"]
                video_data.append(
                    {
                        "id": video_id,
                        "title": video_title,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    }
                )

        # Limit to max_results
        video_data = video_data[:max_results]

        if not video_data:
            logger.warning(f"No videos found for channel: {channel_id}")

        return video_data

    except Exception as e:
        logger.error(f"Error fetching videos for channel {channel_id}: {str(e)}")
        raise ValueError(f"Failed to get videos for channel: {str(e)}")


async def get_single_transcript(
    video_id: str, output_dir: Optional[str] = None, include_timestamps: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Get transcript for a single YouTube video.

    Args:
        video_id: YouTube video ID
        output_dir: Optional directory to save the transcript file
        include_timestamps: Whether to include timestamps in the transcript

    Returns:
        Tuple of (transcript text, file path if saved or None)

    Raises:
        ValueError: If transcript cannot be retrieved
    """
    start_time = time.time()
    logger.info(f"Starting transcript fetch for video {video_id}")

    try:
        # This is a blocking call, use asyncio.to_thread with retry
        fetch_start = time.time()
        try:
            # Use retry operation to handle transient network issues
            fetched_transcript = await retry_operation(
                lambda: asyncio.to_thread(ytt_api.fetch, video_id),
                max_retries=2,  # Try up to 3 times total (initial + 2 retries)
                retry_delay=1.0,
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch transcript for {video_id} after retries: {str(e)}"
            )
            raise ValueError(f"YouTube API failed to return transcript: {str(e)}")

        fetch_end = time.time()
        logger.info(
            f"API fetch took {fetch_end - fetch_start:.3f}s for video {video_id}"
        )

        # Get raw data for better performance
        raw_data_start = time.time()
        transcript_data = fetched_transcript.to_raw_data()
        raw_data_end = time.time()
        logger.info(
            f"Raw data conversion took {raw_data_end - raw_data_start:.3f}s for video {video_id}"
        )

        # Format transcript - optimize by using string builder approach
        format_start = time.time()

        if include_timestamps:
            # Format with timestamps [MM:SS] Text
            # Pre-allocate the list to avoid resizing
            transcript_lines = [""] * len(transcript_data)
            for i, segment in enumerate(transcript_data):
                start_time_sec = segment["start"]
                minutes = int(start_time_sec // 60)
                seconds = int(start_time_sec % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}] "
                transcript_lines[i] = f"{timestamp}{segment['text']}"
            transcript_text = "\n".join(transcript_lines)
        else:
            # Simple concatenation without timestamps - join for better performance
            # Use a list comprehension instead of generator expression for potentially better performance
            transcript_text = " ".join([segment["text"] for segment in transcript_data])

        format_end = time.time()
        logger.info(
            f"Transcript formatting took {format_end - format_start:.3f}s for video {video_id}"
        )

        # Save to file if output directory is specified
        file_path = None
        if output_dir:
            file_start = time.time()

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Try to get video metadata, but continue even if it fails
            video_title = None
            try:
                metadata_start = time.time()
                video_info = await asyncio.wait_for(
                    get_video_info(video_id),
                    timeout=5.0,  # Don't wait more than 5 seconds for metadata
                )
                video_title = video_info.get("title", "Untitled")
                metadata_end = time.time()
                logger.info(
                    f"Video metadata fetch took {metadata_end - metadata_start:.3f}s for video {video_id}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Metadata fetch timed out for video {video_id}, using fallback title"
                )
                video_title = "Untitled_Video"
            except Exception as e:
                logger.warning(
                    f"Failed to get metadata for video {video_id}, using fallback title: {str(e)}"
                )
                video_title = "Untitled_Video"

            # Create a sanitized filename from the video title (or ID if title not available)
            safe_title = sanitize_filename(video_title) if video_title else video_id
            file_path = os.path.join(output_dir, f"{safe_title}_{video_id}.txt")

            # Write transcript to file - more efficient by writing once
            write_start = time.time()
            with open(file_path, "w", encoding="utf-8") as f:
                # Write header with available info
                if video_title:
                    f.write(f"Video Title: {video_title}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n\n")
                f.write(transcript_text)
            write_end = time.time()
            logger.info(
                f"File writing took {write_end - write_start:.3f}s for video {video_id}"
            )

            file_end = time.time()
            logger.info(
                f"Total file operations took {file_end - file_start:.3f}s for video {video_id}"
            )

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(
            f"Total transcript processing took {total_time:.3f}s for video {video_id}"
        )

        return transcript_text, file_path

    except Exception as e:
        logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
        raise ValueError(f"Failed to get transcript for video {video_id}: {str(e)}")


async def start_channel_transcript_download(
    channel_name: str, max_results: int, user_id: str
) -> str:
    """
    Start the asynchronous process of downloading transcripts for a channel.

    Args:
        channel_name: Channel name or ID
        max_results: Maximum number of videos to process
        user_id: User identifier for directory organization

    Returns:
        Job ID for tracking the download process

    Raises:
        ValueError: If channel not found or other errors occur
    """
    try:
        # Get channel info to validate channel existence and get channel ID
        channel_info = await get_channel_info(channel_name)
        channel_id = channel_info["channelId"]

        # Get list of videos from the channel
        videos = await get_channel_videos(channel_id, max_results)

        if not videos:
            raise ValueError(f"No videos found for channel: {channel_name}")

        # Create a unique job ID
        job_id = str(uuid.uuid4())  # Initialize job entry in the tracking dictionary
        channel_download_jobs[job_id] = {
            "status": "processing",
            "channel_name": channel_name,
            "channel_id": channel_id,
            "total_videos": len(videos),
            "processed_count": 0,
            "failed_count": 0,
            "completed": 0,
            "files": [],
            "videos": videos,
            "start_time": time.time(),
            "user_id": user_id,
            "credits_deducted": 0,
            "initial_user_credits": None,  # Will be set when first credit is deducted
            "credits_reserved": len(videos),  # Total credits that will be deducted
        }

        # Start the background task to process transcripts
        # Note: This is where the actual download will happen in the background
        asyncio.create_task(download_channel_transcripts_task(job_id))

        return job_id

    except Exception as e:
        logger.error(f"Error starting transcript download for {channel_name}: {str(e)}")
        raise ValueError(f"Failed to start transcript download: {str(e)}")


async def download_channel_transcripts_task(job_id: str) -> None:
    """
    Background task that downloads transcripts for all videos in a job.
    This function updates the job status as it progresses.

    Args:
        job_id: The job identifier
    """
    if job_id not in channel_download_jobs:
        logger.error(f"Job {job_id} not found for processing")
        return

    job = channel_download_jobs[job_id]
    videos = job["videos"]
    user_id = job["user_id"]

    # Create directory for this specific job
    output_dir = os.path.join(settings.temp_dir, user_id, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Process videos concurrently with a limit on parallelism
    # Process in batches to avoid overwhelming the API
    batch_size = 5  # Process 5 videos at a time

    for i in range(0, len(videos), batch_size):
        batch_videos = videos[i : i + batch_size]
        tasks = []

        # Create tasks for each video in the current batch
        for video in batch_videos:
            video_id = video["id"]
            tasks.append(process_single_video(job_id, video_id, output_dir))

        # Run the batch concurrently and wait for all to complete
        await asyncio.gather(*tasks)

    # Mark job as completed when all videos are processed
    job["status"] = "completed"
    job["end_time"] = time.time()
    job["duration"] = job["end_time"] - job["start_time"]

    logger.info(
        f"Job {job_id} completed. Processed {job['completed']}/{job['total_videos']} videos successfully."
    )


async def process_single_video(job_id: str, video_id: str, output_dir: str) -> None:
    """
    Process a single video for transcript download.
    Deducts 1 credit per video attempt and updates job statistics.

    Args:
        job_id: The job identifier
        video_id: YouTube video ID to process
        output_dir: Directory to save transcript files
    """
    if job_id not in channel_download_jobs:
        logger.error(f"Job {job_id} not found for video {video_id}")
        return

    job = channel_download_jobs[job_id]
    user_id = job["user_id"]
    video_dir = os.path.join(
        output_dir, video_id
    )  # Use video ID as subdirectory for isolation

    # Deduct credit before attempting transcript download
    try:
        # Import CreditManager here to avoid circular imports
        from transcript_api import CreditManager

        # Store initial credits on first deduction
        if job["initial_user_credits"] is None:
            job["initial_user_credits"] = CreditManager.get_user_credits(user_id)
            logger.info(
                f"Job {job_id}: User {user_id} started with {job['initial_user_credits']} credits"
            )

        # Deduct 1 credit for this video attempt
        if CreditManager.deduct_credit(user_id):
            job["credits_deducted"] += 1
            logger.info(
                f"Job {job_id}: Deducted credit for video {video_id}. Total deducted: {job['credits_deducted']}"
            )
        else:
            # This shouldn't happen since we checked upfront, but log it
            logger.error(
                f"Job {job_id}: Failed to deduct credit for video {video_id} (insufficient credits)"
            )
            job["failed_count"] += 1
            job["processed_count"] += 1
            return

    except Exception as e:
        logger.error(
            f"Job {job_id}: Error deducting credit for video {video_id}: {str(e)}"
        )
        job["failed_count"] += 1
        job["processed_count"] += 1
        return

    try:
        # Create a separate subdirectory for each video to isolate failures
        os.makedirs(video_dir, exist_ok=True)

        # Add a timeout to prevent any single video from taking too long
        try:
            # Get transcript with timeout to prevent hanging
            transcript_task = asyncio.create_task(
                get_single_transcript(video_id, video_dir, include_timestamps=False)
            )
            _, file_path = await asyncio.wait_for(
                transcript_task, timeout=60.0
            )  # 60 second timeout

            if file_path:
                # Update job statistics
                job["files"].append(file_path)
                job["completed"] += 1
                logger.info(
                    f"Downloaded transcript for video {video_id} ({job['completed']}/{job['total_videos']})"
                )

        except asyncio.TimeoutError:
            logger.error(
                f"Transcript processing for {video_id} timed out after 60 seconds"
            )
            raise ValueError(f"Transcript processing timed out")

    except Exception as e:
        # Log failure but continue with other videos
        job["failed_count"] += 1
        logger.error(f"Failed to process video {video_id} for job {job_id}: {str(e)}")

    finally:
        # Increment processed count
        job["processed_count"] += 1


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the current status of a transcript download job.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with job status information including credit usage

    Raises:
        ValueError: If job ID is not found
    """
    if job_id not in channel_download_jobs:
        raise ValueError(f"Job not found with ID: {job_id}")

    job = channel_download_jobs[job_id]

    # Calculate progress percentage
    total = job["total_videos"]
    processed = job["processed_count"]
    progress = (processed / total * 100) if total > 0 else 0

    # Get current user credits for reference
    current_credits = None
    try:
        from transcript_api import CreditManager

        current_credits = CreditManager.get_user_credits(job["user_id"])
    except Exception as e:
        logger.warning(f"Could not get current credits for job {job_id}: {str(e)}")

    # Return relevant status information including credit tracking
    status_info = {
        "status": job["status"],
        "channel_name": job["channel_name"],
        "total_videos": total,
        "processed_count": processed,
        "completed": job["completed"],
        "failed_count": job["failed_count"],
        "progress": round(progress, 2),
        "start_time": job["start_time"],
        "end_time": job.get("end_time"),
        "duration": job.get("duration"),
        "credits_deducted": job["credits_deducted"],
        "credits_reserved": job["credits_reserved"],
        "initial_user_credits": job["initial_user_credits"],
        "current_user_credits": current_credits,
    }

    # Add credit usage summary
    if job["initial_user_credits"] is not None:
        status_info["credits_used_this_job"] = job["credits_deducted"]
        status_info["credits_remaining_for_job"] = max(
            0, job["credits_reserved"] - job["credits_deducted"]
        )

    return status_info


async def create_transcript_zip(job_id: str) -> Optional[io.BytesIO]:
    """
    Create a ZIP archive of all transcripts for a completed job.

    Args:
        job_id: The job identifier

    Returns:
        BytesIO object containing the ZIP file, or None if job not completed

    Raises:
        ValueError: If job ID not found or job not completed
    """
    if job_id not in channel_download_jobs:
        raise ValueError(f"Job not found with ID: {job_id}")

    job = channel_download_jobs[job_id]

    if job["status"] != "completed":
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job["files"]:
        raise ValueError("No transcript files found for this job")

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = io.BytesIO()

    # Create a ZIP file with all transcripts
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_path in job["files"]:
            # Extract filename from path
            filename = os.path.basename(file_path)
            # Add file to ZIP if it exists
            if os.path.exists(file_path):
                zip_file.write(file_path, filename)

    # Seek to beginning of buffer for response
    zip_buffer.seek(0)

    return zip_buffer


def get_safe_channel_name(job_id: str) -> str:
    """
    Get a filesystem-safe version of the channel name for a job.

    Args:
        job_id: The job identifier

    Returns:
        Safe channel name for use in filenames

    Raises:
        ValueError: If job ID not found
    """
    if job_id not in channel_download_jobs:
        raise ValueError(f"Job not found with ID: {job_id}")

    channel_name = channel_download_jobs[job_id]["channel_name"]
    return sanitize_filename(channel_name)


async def retry_operation(operation, max_retries=3, retry_delay=1.0, *args, **kwargs):
    """
    Retry an async operation with exponential backoff.

    Args:
        operation: Async function to retry
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (will increase exponentially)
        *args, **kwargs: Arguments to pass to the operation

    Returns:
        Result of the operation if successful

    Raises:
        Last exception encountered if all retries fail
    """
    last_exception = None
    current_retry = 0

    while current_retry <= max_retries:
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            current_retry += 1

            if current_retry <= max_retries:
                # Log the exception but continue with retry
                logger.warning(
                    f"Operation failed (attempt {current_retry}/{max_retries}): {str(e)}. "
                    f"Retrying in {retry_delay:.1f} seconds..."
                )
                await asyncio.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
            else:
                # Log the final failure
                logger.error(f"Operation failed after {max_retries} retries: {str(e)}")
                raise last_exception

    # This should never be reached due to the raise in the else clause
    raise last_exception


def cleanup_old_jobs(max_age_hours: int = 24) -> None:
    """
    Clean up old jobs and their associated files.

    Args:
        max_age_hours: Maximum age of jobs to keep in hours
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    jobs_to_remove = []

    # Find old jobs
    for job_id, job in channel_download_jobs.items():
        job_start_time = job.get("start_time", 0)
        job_age = current_time - job_start_time

        if job_age > max_age_seconds:
            jobs_to_remove.append(job_id)

            # Try to remove files
            for file_path in job.get("files", []):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old file {file_path}: {str(e)}")

            # Try to remove job directory
            try:
                user_id = job.get("user_id")
                if user_id:
                    job_dir = os.path.join(settings.temp_dir, user_id, job_id)
                    if os.path.exists(job_dir) and os.path.isdir(job_dir):
                        # Remove all files in the directory
                        for root, dirs, files in os.walk(job_dir):
                            for file in files:
                                try:
                                    os.remove(os.path.join(root, file))
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to remove file in cleanup: {str(e)}"
                                    )

                        # Try to remove the directory
                        try:
                            os.rmdir(job_dir)
                            logger.info(f"Removed old job directory: {job_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to remove job directory: {str(e)}")
            except Exception as e:
                logger.warning(f"Error during job cleanup for {job_id}: {str(e)}")

    # Remove old jobs from the dictionary
    for job_id in jobs_to_remove:
        try:
            del channel_download_jobs[job_id]
            logger.info(f"Removed old job: {job_id}")
        except Exception as e:
            logger.warning(f"Error removing job {job_id} from dictionary: {str(e)}")

    logger.info(f"Cleanup complete. Removed {len(jobs_to_remove)} old jobs.")


# ytt_api = YouTubeTranscriptApi()

# asd = ytt_api.fetch("cESaIUWoCJQ")

# bbb = ytt_api.fetch("nCuaNmeVfQY")

# asd
