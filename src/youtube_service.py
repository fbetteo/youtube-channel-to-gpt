#!/usr/bin/env python3
"""
YouTube Service - Service layer for YouTube transcript downloading and processing
"""
import asyncio
import gc
import io
import json
import logging
import os
import psutil
import re
import tempfile
import threading
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

# Memory tracking logger - separate logger for memory metrics
memory_logger = logging.getLogger("memory_tracker")
memory_logger.setLevel(logging.INFO)

# Memory tracking variables
_memory_monitoring_active = False
_memory_monitor_thread = None
_memory_stats = {
    "peak_memory_mb": 0,
    "peak_memory_percent": 0,
    "last_gc_time": time.time(),
    "gc_count": 0,
    "memory_warnings": 0,
}


class MemoryTracker:
    """Memory tracking utility for monitoring system resources."""

    def __init__(
        self,
        warning_threshold_percent: float = 85.0,
        critical_threshold_percent: float = 95.0,
    ):
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            # Process memory info
            process_memory = self.process.memory_info()
            process_memory_mb = process_memory.rss / 1024 / 1024
            process_memory_percent = self.process.memory_percent()

            # System memory info
            system_memory = psutil.virtual_memory()
            system_memory_mb = system_memory.used / 1024 / 1024
            system_memory_percent = system_memory.percent

            # Update peak tracking
            global _memory_stats
            if process_memory_mb > _memory_stats["peak_memory_mb"]:
                _memory_stats["peak_memory_mb"] = process_memory_mb
            if process_memory_percent > _memory_stats["peak_memory_percent"]:
                _memory_stats["peak_memory_percent"] = process_memory_percent

            return {
                "timestamp": time.time(),
                "process_memory_mb": round(process_memory_mb, 2),
                "process_memory_percent": round(process_memory_percent, 2),
                "system_memory_mb": round(system_memory_mb, 2),
                "system_memory_percent": round(system_memory_percent, 2),
                "system_memory_available_mb": round(
                    system_memory.available / 1024 / 1024, 2
                ),
                "peak_memory_mb": round(_memory_stats["peak_memory_mb"], 2),
                "peak_memory_percent": round(_memory_stats["peak_memory_percent"], 2),
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {str(e)}")
            return {}

    def log_memory_usage(
        self, context: str = "", level: str = "info"
    ) -> Dict[str, Any]:
        """Log current memory usage with context."""
        memory_info = self.get_memory_info()
        if not memory_info:
            return {}

        log_msg = (
            f"Memory usage {context}: "
            f"Process: {memory_info['process_memory_mb']}MB ({memory_info['process_memory_percent']}%), "
            f"System: {memory_info['system_memory_percent']}%, "
            f"Available: {memory_info['system_memory_available_mb']}MB, "
            f"Peak: {memory_info['peak_memory_mb']}MB"
        )

        # Check thresholds and log accordingly
        if memory_info["process_memory_percent"] >= self.critical_threshold:
            memory_logger.critical(f"CRITICAL: {log_msg}")
            _memory_stats["memory_warnings"] += 1
        elif memory_info["process_memory_percent"] >= self.warning_threshold:
            memory_logger.warning(f"WARNING: {log_msg}")
            _memory_stats["memory_warnings"] += 1
        elif level == "debug":
            memory_logger.debug(log_msg)
        else:
            memory_logger.info(log_msg)

        return memory_info

    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and log results."""
        try:
            before_memory = self.get_memory_info()

            # Force garbage collection
            collected = gc.collect()
            _memory_stats["gc_count"] += 1
            _memory_stats["last_gc_time"] = time.time()

            after_memory = self.get_memory_info()

            memory_freed = before_memory.get("process_memory_mb", 0) - after_memory.get(
                "process_memory_mb", 0
            )

            memory_logger.info(
                f"Garbage collection completed: "
                f"Objects collected: {collected}, "
                f"Memory freed: {memory_freed:.2f}MB, "
                f"Memory before: {before_memory.get('process_memory_mb', 0):.2f}MB, "
                f"Memory after: {after_memory.get('process_memory_mb', 0):.2f}MB"
            )

            return {
                "objects_collected": collected,
                "memory_freed_mb": memory_freed,
                "before_memory": before_memory,
                "after_memory": after_memory,
            }
        except Exception as e:
            logger.error(f"Error during garbage collection: {str(e)}")
            return {}


# Global memory tracker instance
memory_tracker = MemoryTracker()


def start_memory_monitoring(interval_seconds: int = 30) -> None:
    """Start background memory monitoring."""
    global _memory_monitoring_active, _memory_monitor_thread

    if _memory_monitoring_active:
        logger.info("Memory monitoring already active")
        return

    def monitor_memory():
        while _memory_monitoring_active:
            try:
                memory_tracker.log_memory_usage("periodic_check", "debug")
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring thread: {str(e)}")
                time.sleep(interval_seconds)

    _memory_monitoring_active = True
    _memory_monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    _memory_monitor_thread.start()

    memory_logger.info(f"Started memory monitoring with {interval_seconds}s interval")
    memory_tracker.log_memory_usage("monitoring_start")


def stop_memory_monitoring() -> None:
    """Stop background memory monitoring."""
    global _memory_monitoring_active
    _memory_monitoring_active = False
    memory_logger.info("Stopped memory monitoring")


def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics."""
    current_memory = memory_tracker.get_memory_info()
    return {
        **current_memory,
        "gc_count": _memory_stats["gc_count"],
        "last_gc_time": _memory_stats["last_gc_time"],
        "memory_warnings": _memory_stats["memory_warnings"],
        "monitoring_active": _memory_monitoring_active,
    }


# Initialize YouTube API client with retry for transient failures
try:
    youtube = build("youtube", "v3", developerKey=settings.youtube_api_key)
    logger.info("YouTube API client initialized successfully")

    # Initialize memory monitoring when the service starts
    memory_tracker.log_memory_usage("service_initialization")

except Exception as e:
    logger.error(f"Failed to initialize YouTube API client: {str(e)}")
    # Fallback to None, will re-attempt connection when needed
    youtube = None


def get_youtube_client():
    """
    Create a thread-safe YouTube API client for each worker.
    This prevents thread safety issues with the global client.
    """
    try:
        return build("youtube", "v3", developerKey=settings.youtube_api_key)
    except Exception as e:
        logger.error(f"Failed to create YouTube API client: {str(e)}")
        return youtube  # Fallback to global client


# Remove global ytt_api initialization, only keep YouTube API client
def get_ytt_api() -> YouTubeTranscriptApi:
    """
    Create a new YouTubeTranscriptApi instance with proxy config if needed.
    Thread-safe implementation that creates fresh instances.
    Returns:
        YouTubeTranscriptApi instance
    """
    try:
        if settings.webshare_proxy_username and settings.webshare_proxy_password:
            proxy_config = WebshareProxyConfig(
                proxy_username=settings.webshare_proxy_username,
                proxy_password=settings.webshare_proxy_password,
                retries_when_blocked=1,
            )
            return YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            return YouTubeTranscriptApi()
    except Exception as e:
        logger.error(f"Error creating YouTubeTranscriptApi: {str(e)}")
        # Fallback to basic instance
        return YouTubeTranscriptApi()


# Dictionary to track channel download jobs
channel_download_jobs: Dict[str, Dict[str, Any]] = {}

# Add persistent job storage
JOBS_STORAGE_DIR = os.path.join(settings.temp_dir, "jobs")
os.makedirs(JOBS_STORAGE_DIR, exist_ok=True)


def save_job_to_file(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save job data to persistent storage"""
    try:
        job_file = os.path.join(JOBS_STORAGE_DIR, f"{job_id}.json")
        # Convert non-serializable objects to serializable format
        serializable_data = {
            k: v
            for k, v in job_data.items()
            if k not in ["videos_metadata"]  # Skip large objects
        }
        # Add basic info about metadata
        if "videos_metadata" in job_data:
            serializable_data["metadata_count"] = len(job_data["videos_metadata"])

        with open(job_file, "w") as f:
            json.dump(serializable_data, f, default=str)
        logger.debug(f"Saved job {job_id} to persistent storage")
    except Exception as e:
        logger.error(f"Failed to save job {job_id}: {e}")


def load_job_from_file(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job data from persistent storage"""
    try:
        job_file = os.path.join(JOBS_STORAGE_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                job_data = json.load(f)
            logger.debug(f"Loaded job {job_id} from persistent storage")
            return job_data
        return None
    except Exception as e:
        logger.error(f"Failed to load job {job_id}: {e}")
        return None


def update_job_progress(job_id: str, **updates):
    """Update job progress and save to persistent storage"""
    if job_id in channel_download_jobs:
        # Update in-memory job
        channel_download_jobs[job_id].update(updates)
        # Save to persistent storage
        save_job_to_file(job_id, channel_download_jobs[job_id])
        logger.debug(f"Updated job {job_id} progress: {updates}")
    else:
        logger.error(f"Cannot update progress for unknown job: {job_id}")


def recover_jobs_from_storage() -> List[str]:
    """Recover jobs from persistent storage on startup"""
    recovered_jobs = []
    try:
        if os.path.exists(JOBS_STORAGE_DIR):
            for filename in os.listdir(JOBS_STORAGE_DIR):
                if filename.endswith(".json"):
                    job_id = filename[:-5]  # Remove .json extension
                    job_data = load_job_from_file(job_id)
                    if job_data:
                        # Only recover jobs that are still processing
                        if job_data.get("status") == "processing":
                            # Mark as failed since we lost the process
                            job_data["status"] = "failed"
                            job_data["error"] = "Process restarted during execution"
                            job_data["end_time"] = time.time()
                            job_data["duration"] = job_data.get(
                                "end_time", time.time()
                            ) - job_data.get("start_time", time.time())

                        channel_download_jobs[job_id] = job_data
                        recovered_jobs.append(job_id)

        logger.info(f"Recovered {len(recovered_jobs)} jobs from storage")
        return recovered_jobs
    except Exception as e:
        logger.error(f"Error recovering jobs: {e}")
        return []


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


async def pre_fetch_videos_metadata(
    video_ids: List[str], batch_size: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-fetch metadata for multiple videos in parallel batches using YouTube API batch requests.

    Args:
        video_ids: List of YouTube video IDs
        batch_size: Number of videos to process in each API batch (max 50 for YouTube API)

    Returns:
        Dictionary mapping video_id to metadata dict, or empty dict if failed
    """
    logger.info(
        f"Pre-fetching metadata for {len(video_ids)} videos in batches of {batch_size}"
    )

    metadata_results = {}

    # Process videos in batches using YouTube API batch requests
    for batch_num, i in enumerate(range(0, len(video_ids), batch_size), 1):
        batch_video_ids = video_ids[i : i + batch_size]

        logger.info(
            f"Processing metadata batch {batch_num}/{(len(video_ids) + batch_size - 1) // batch_size} ({len(batch_video_ids)} videos)"
        )

        try:
            # Use YouTube API batch request for efficiency
            batch_results = await get_videos_metadata_batch(batch_video_ids)

            # Process results
            for video_id in batch_video_ids:
                if video_id in batch_results:
                    metadata_results[video_id] = batch_results[video_id]
                else:
                    logger.warning(f"No metadata returned for video {video_id}")
                    metadata_results[video_id] = {}  # Empty dict for failed metadata

        except Exception as e:
            logger.warning(
                f"Metadata batch {batch_num} failed: {str(e)}, using fallback data"
            )
            for video_id in batch_video_ids:
                metadata_results[video_id] = {}

        # Small delay between batches to be respectful to the API
        if i + batch_size < len(video_ids):
            await asyncio.sleep(0.2)  # Reduced delay since we're using fewer API calls

    successful_fetches = sum(1 for metadata in metadata_results.values() if metadata)
    logger.info(
        f"Successfully pre-fetched metadata for {successful_fetches}/{len(video_ids)} videos"
    )

    return metadata_results


async def get_videos_metadata_batch(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for multiple videos in a single YouTube API batch request.

    Args:
        video_ids: List of YouTube video IDs (max 50)

    Returns:
        Dictionary mapping video_id to metadata dict
    """
    if not video_ids:
        return {}

    if len(video_ids) > 50:
        raise ValueError("YouTube API batch request supports maximum 50 video IDs")

    try:
        # Join video IDs with commas for batch request
        video_ids_str = ",".join(video_ids)

        def _fetch_videos_batch():
            # Use thread-safe client
            client = get_youtube_client()
            request = client.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_ids_str,
                maxResults=50,
            )
            return request.execute()

        # Run the blocking API call in a thread pool with timeout
        video_response = await asyncio.wait_for(
            asyncio.to_thread(_fetch_videos_batch),
            timeout=10.0,  # 10 second timeout for batch request
        )

        if not video_response.get("items"):
            logger.warning(f"No videos found for batch of {len(video_ids)} video IDs")
            return {}

        results = {}

        # Process each video in the response
        for video_data in video_response["items"]:
            video_id = video_data["id"]
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

            results[video_id] = metadata

        logger.info(
            f"Successfully fetched metadata for {len(results)}/{len(video_ids)} videos in batch"
        )
        return results

    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching metadata batch for {len(video_ids)} videos")
        raise
    except Exception as e:
        logger.error(f"Error fetching metadata batch: {str(e)}")
        raise


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
            # Use thread-safe client
            client = get_youtube_client()
            video_request = client.videos().list(
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


def _parse_duration_to_seconds(duration_str: str) -> int:
    """
    Parse ISO 8601 duration string to total seconds.

    Args:
        duration_str: ISO 8601 duration string (e.g., 'PT1H2M3S')

    Returns:
        Total duration in seconds
    """
    match_hours = re.search(r"(\d+)H", duration_str)
    match_minutes = re.search(r"(\d+)M", duration_str)
    match_seconds = re.search(r"(\d+)S", duration_str)

    hours = int(match_hours.group(1)) if match_hours else 0
    minutes = int(match_minutes.group(1)) if match_minutes else 0
    seconds = int(match_seconds.group(1)) if match_seconds else 0

    return hours * 3600 + minutes * 60 + seconds


def _categorize_duration(duration_seconds: int) -> str:
    """
    Categorize video duration based on YouTube's duration categories.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Duration category: 'short' (≤60s), 'medium' (60s-20min), 'long' (>20min)
    """
    if duration_seconds <= 60:
        return "short"
    elif duration_seconds <= 1200:  # 20 minutes
        return "medium"
    else:
        return "long"


async def get_channel_info(channel_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a YouTube channel.

    Args:
        channel_name: Channel name, handle, or channel ID

    Returns:
        Dictionary with channel information (title, description, stats, etc.)

    Raises:
        ValueError: If channel is not found
    """
    try:
        # Check if this is a channel ID (UC format)
        is_channel_id = re.match(r"^UC[\w-]{22}$", channel_name) is not None

        def _fetch_channel_info():
            # Use thread-safe client
            client = get_youtube_client()
            if is_channel_id:
                # If it's a channel ID, fetch directly
                logger.info(f"Using channel ID directly: {channel_name}")
                request = client.channels().list(
                    part="snippet,statistics,contentDetails", id=channel_name
                )
            else:
                # For channel names/handles, try to fetch directly
                # This will work for some formats but may fail for others
                logger.info(f"Attempting direct channel lookup for: {channel_name}")
                request = client.channels().list(
                    part="snippet,statistics,contentDetails", forHandle=channel_name
                )

            return request.execute()

        # Run the blocking API call in a thread pool
        response = await asyncio.to_thread(_fetch_channel_info)

        if not response.get("items"):
            # If forHandle failed, try one more approach for usernames/custom URLs
            if not is_channel_id:
                logger.info(f"Direct lookup failed, trying forUsername: {channel_name}")
                try:

                    def _fetch_by_username():
                        return (
                            youtube.channels()
                            .list(
                                part="snippet,statistics,contentDetails",
                                forUsername=channel_name,
                            )
                            .execute()
                        )

                    response = await asyncio.to_thread(_fetch_by_username)
                except Exception as e:
                    logger.warning(f"forUsername lookup also failed: {str(e)}")

            if not response.get("items"):
                raise ValueError(
                    f"Channel not found: '{channel_name}'. "
                    f"Please provide a valid channel ID (UC...), handle (@username), or exact channel username."
                )

        channel_info = response["items"][0]
        channel_id = channel_info["id"]

        logger.info(
            f"Successfully resolved '{channel_name}' to channel ID '{channel_id}' (title: '{channel_info['snippet']['title']}')"
        )

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
            "uploadsPlaylistId": channel_info.get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads"),
        }

    except Exception as e:
        logger.error(f"Error fetching channel info for {channel_name}: {str(e)}")
        raise ValueError(f"Failed to get channel information: {str(e)}")


# async def get_channel_videos(
#     channel_id: str, max_results: int = None
# ) -> List[Dict[str, str]]:
#     """
#     Get a list of videos from a YouTube channel using uploads playlist (quota efficient).

#     Args:
#         channel_id: The YouTube channel ID
#         max_results: Maximum number of videos to retrieve (default: None for all)

#     Returns:
#         List of dictionaries containing video IDs, titles, and duration categories

#     Raises:
#         ValueError: If channel ID is invalid or no videos found
#     """
#     try:
#         logger.info(f"Fetching videos for channel {channel_id} using uploads playlist")

#         # First get the channel info to get the uploads playlist ID
#         channel_info = await get_channel_info(channel_id)
#         uploads_playlist_id = channel_info.get("uploadsPlaylistId")

#         if not uploads_playlist_id:
#             raise ValueError(
#                 f"Could not find uploads playlist for channel {channel_id}"
#             )

#         def _fetch_playlist_videos():
#             """Fetch videos from uploads playlist using playlistItems.list (1 quota unit)"""
#             playlist_response = (
#                 youtube.playlistItems()
#                 .list(
#                     part="snippet",
#                     playlistId=uploads_playlist_id,
#                     maxResults=max_results or 50,  # Default to 50 if no limit specified
#                 )
#                 .execute()
#             )
#             return playlist_response

#         # Run the blocking API call in a thread pool
#         playlist_response = await asyncio.to_thread(_fetch_playlist_videos)

#         logger.info(
#             f"Found {len(playlist_response.get('items', []))} videos in uploads playlist for channel {channel_id}"
#         )

#         # Extract video IDs and basic info
#         video_data = []
#         video_ids = []

#         for item in playlist_response.get("items", []):
#             snippet = item["snippet"]
#             # Skip private videos (they appear with empty titles)
#             if not snippet.get("title") or snippet.get("title") == "Private video":
#                 continue

#             video_id = snippet["resourceId"]["videoId"]
#             video_title = snippet["title"]
#             video_data.append(
#                 {
#                     "id": video_id,
#                     "title": video_title,
#                     "url": f"https://www.youtube.com/watch?v={video_id}",
#                     "duration": None,  # Will be filled after batch call
#                 }
#             )
#             video_ids.append(video_id)

#         # Get duration information with batch call (efficient: 1 quota unit for up to 50 videos)
#         if video_ids:
#             logger.info(f"Fetching duration information for {len(video_ids)} videos")

#             def _fetch_video_durations():
#                 """Fetch video durations using videos.list batch call (1 quota unit)"""
#                 return (
#                     youtube.videos()
#                     .list(part="contentDetails", id=",".join(video_ids))
#                     .execute()
#                 )

#             durations_response = await asyncio.to_thread(_fetch_video_durations)

#             # Create a mapping of video_id to duration category
#             duration_map = {}
#             for video_item in durations_response.get("items", []):
#                 video_id = video_item["id"]
#                 duration_str = video_item["contentDetails"]["duration"]
#                 duration_seconds = _parse_duration_to_seconds(duration_str)
#                 duration_category = _categorize_duration(duration_seconds)
#                 duration_map[video_id] = duration_category

#             # Update video_data with duration categories
#             for video in video_data:
#                 video["duration"] = duration_map.get(video["id"], "unknown")

#         logger.info(f"Final list of videos to download ({len(video_data)}):")
#         for v in video_data:
#             logger.info(f"  - {v['id']}: {v['title']} ({v['duration']})")

#         if not video_data:
#             logger.warning(f"No videos found for channel: {channel_id}")

#         return video_data

#     except Exception as e:
#         logger.error(f"Error fetching videos for channel {channel_id}: {str(e)}")
#         raise ValueError(f"Failed to get videos for channel: {str(e)}")


def _fetch_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all videos from a channel using uploads playlist (quota efficient).
    Returns a list of video metadata dicts (id, title, publishedAt, duration category, etc.).
    """
    logger.info(f"Fetching all videos from uploads playlist for channel {channel_id}")

    # Get channel info to access uploads playlist
    client = get_youtube_client()
    request = client.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()

    if not response.get("items"):
        raise ValueError(f"Channel {channel_id} not found")

    uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"][
        "uploads"
    ]

    all_videos = []
    next_page_token = None

    # Paginate through all videos in the uploads playlist
    while True:
        request = client.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]
            # Skip private videos (they appear with empty titles)
            if not snippet.get("title") or snippet.get("title") == "Private video":
                continue

            video_id = snippet["resourceId"]["videoId"]
            all_videos.append(
                {
                    "id": video_id,
                    "title": snippet["title"],
                    "publishedAt": snippet.get("publishedAt"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "duration": None,  # Will be filled after batch call
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    # Get duration information in batches (50 videos per API call)
    logger.info(f"Fetching duration information for {len(all_videos)} videos")

    for i in range(0, len(all_videos), 50):
        batch_videos = all_videos[i : i + 50]
        video_ids = [video["id"] for video in batch_videos]

        # Fetch durations for this batch
        request = client.videos().list(part="contentDetails", id=",".join(video_ids))
        durations_response = request.execute()

        # Create a mapping of video_id to duration category
        duration_map = {}
        for video_item in durations_response.get("items", []):
            video_id = video_item["id"]
            duration_str = video_item["contentDetails"]["duration"]
            duration_seconds = _parse_duration_to_seconds(duration_str)
            duration_category = _categorize_duration(duration_seconds)
            duration_map[video_id] = duration_category

        # Update this batch with duration categories
        for video in batch_videos:
            video["duration"] = duration_map.get(video["id"], "unknown")

    logger.info(
        f"Found {len(all_videos)} videos in uploads playlist for channel {channel_id}"
    )

    # Log duration distribution
    duration_counts = {}
    for video in all_videos:
        category = video["duration"]
        duration_counts[category] = duration_counts.get(category, 0) + 1

    logger.info(f"Duration distribution: {duration_counts}")

    return all_videos


async def get_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]:
    """
    Async wrapper for _fetch_all_channel_videos with better error handling and logging.

    Args:
        channel_id: YouTube channel ID

    Returns:
        List of video dictionaries with metadata
    """
    try:
        logger.info(f"Fetching all videos for channel {channel_id}")
        videos = await asyncio.to_thread(_fetch_all_channel_videos, channel_id)
        logger.info(
            f"Successfully fetched {len(videos)} videos for channel {channel_id}"
        )
        return videos
    except Exception as e:
        logger.error(f"Error in get_all_channel_videos for {channel_id}: {str(e)}")
        raise ValueError(f"Failed to fetch all videos: {str(e)}")


async def get_single_transcript(
    video_id: str,
    output_dir: Optional[str] = None,
    include_timestamps: bool = False,
    include_video_title: bool = True,
    include_video_id: bool = True,
    include_video_url: bool = True,
    include_view_count: bool = False,
    pre_fetched_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Get transcript for a single YouTube video, with language fallback.

    Args:
        video_id: YouTube video ID
        output_dir: Optional directory to save the transcript file
        include_timestamps: Whether to include timestamps in the transcript
        include_video_title: Whether to include video title in file header
        include_video_id: Whether to include video ID in file header
        include_video_url: Whether to include video URL in file header
        include_view_count: Whether to include view count in file header
        pre_fetched_metadata: Optional pre-fetched metadata to use instead of fetching

    Returns:
        Tuple of (transcript text, file path if saved, metadata dict)

    Raises:
        ValueError: If transcript cannot be retrieved
    """
    start_time = time.time()

    # Track memory at start of transcript processing
    memory_tracker.log_memory_usage(f"transcript_start[{video_id}]")

    logger.info(f"Starting transcript fetch for video {video_id}")

    try:
        # Create fresh API instance for thread safety
        ytt_api = get_ytt_api()
        fetch_start = time.time()

        # 1. List available transcripts with better error handling
        try:
            transcript_list = await retry_operation(
                lambda ytt_api=ytt_api: asyncio.to_thread(ytt_api.list, video_id),
                max_retries=2,
            )
        except Exception as e:
            logger.error(f"Failed to list transcripts for video {video_id}: {str(e)}")
            raise ValueError(f"No transcripts available for video {video_id}: {str(e)}")

        transcript = None
        # 2. Try to find English, otherwise take the first available
        try:
            transcript = transcript_list.find_transcript(["en"])
            logger.info(f"Found English transcript for video {video_id}")
        except Exception:
            logger.warning(
                f"No English transcript found for {video_id}. Trying first available."
            )
            try:
                # Get the first transcript in the list
                first_transcript_in_list = next(iter(transcript_list))
                transcript = first_transcript_in_list
                logger.info(
                    f"Using first available transcript ({transcript.language_code}) for video {video_id}"
                )
            except StopIteration:
                logger.error(f"No transcripts available at all for video {video_id}")
                raise ValueError(f"No transcripts available for video {video_id}")

        # 3. Fetch the selected transcript with better error handling
        try:
            fetched_transcript = await retry_operation(
                lambda: asyncio.to_thread(transcript.fetch),
                max_retries=2,
            )
        except Exception as e:
            logger.error(f"Failed to fetch transcript for video {video_id}: {str(e)}")
            raise ValueError(
                f"Failed to fetch transcript for video {video_id}: {str(e)}"
            )

        fetch_end = time.time()
        logger.info(
            f"API fetch took {fetch_end - fetch_start:.3f}s for video {video_id}"
        )

        # Clean up API instance immediately after use to prevent memory leaks
        del ytt_api

        # Track memory after API fetch
        memory_tracker.log_memory_usage(f"transcript_fetched[{video_id}]", "debug")

        # Create metadata with language info
        selected_language = transcript.language_code
        metadata = {
            "video_id": video_id,
            "transcript_language": selected_language,
            "transcript_type": (
                "manual" if not transcript.is_generated else "auto-generated"
            ),
        }

        # Get raw data for better performance
        raw_data_start = time.time()
        transcript_data = fetched_transcript.to_raw_data()
        raw_data_end = time.time()
        logger.info(
            f"Raw data conversion took {raw_data_end - raw_data_start:.3f}s for video {video_id}"
        )

        # Track memory after raw data conversion
        memory_tracker.log_memory_usage(f"transcript_raw_data[{video_id}]", "debug")

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

        # Track memory after formatting (this can be memory-intensive for long videos)
        memory_tracker.log_memory_usage(f"transcript_formatted[{video_id}]")

        # Save to file if output directory is specified
        file_path = None
        if output_dir:
            file_start = time.time()

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Use pre-fetched metadata if available, otherwise fetch it
            video_title = None
            video_view_count = None

            if pre_fetched_metadata:
                logger.info("Using pre-fetched metadata")
                # Use pre-fetched metadata (much faster)
                video_title = pre_fetched_metadata.get("title", "Untitled_Video")
                video_view_count = pre_fetched_metadata.get("viewCount", 0)
                logger.debug(
                    f"Using pre-fetched metadata for video {video_id}: {video_title}"
                )
            else:
                logger.info("Fetching video metadata - no prefetch")
                # Fallback to fetching metadata (slower, with timeout)
                try:
                    metadata_start = time.time()
                    video_info = await asyncio.wait_for(
                        get_video_info(video_id),
                        timeout=10.0,  # Increased timeout to 10 seconds when not pre-fetched
                    )
                    video_title = video_info.get("title", "Untitled_Video")
                    video_view_count = video_info.get("viewCount", 0)
                    metadata_end = time.time()
                    logger.info(
                        f"Video metadata fetch took {metadata_end - metadata_start:.3f}s for video {video_id}"
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Metadata fetch timed out for video {video_id}, using fallback title"
                    )
                    video_title = "Untitled_Video"
                    video_view_count = 0
                except Exception as e:
                    logger.warning(
                        f"Failed to get metadata for video {video_id}, using fallback title: {str(e)}"
                    )
                    video_title = "Untitled_Video"
                    video_view_count = 0

            # Create a sanitized filename from the video title (or ID if title not available)
            safe_title = sanitize_filename(video_title) if video_title else video_id
            file_path = os.path.join(output_dir, f"{safe_title}_{video_id}.txt")

            # Write transcript to file - more efficient by writing once
            write_start = time.time()
            with open(file_path, "w", encoding="utf-8") as f:
                # Write header with available info based on formatting options
                if include_video_title and video_title:
                    f.write(f"Video Title: {video_title}\n")
                if include_video_id:
                    f.write(f"Video ID: {video_id}\n")
                if include_video_url:
                    f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
                if include_view_count and video_view_count is not None:
                    f.write(f"View Count: {video_view_count:,}\n")

                # Add separator if any header was written
                if (
                    (include_video_title and video_title)
                    or include_video_id
                    or include_video_url
                    or (include_view_count and video_view_count is not None)
                ):
                    f.write("\n")

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

        # Track memory at completion and log final stats
        final_memory = memory_tracker.log_memory_usage(
            f"transcript_complete[{video_id}]"
        )

        logger.info(
            f"Total transcript processing took {total_time:.3f}s for video {video_id}"
        )

        # Clean up large variables to help with memory management
        del transcript_data
        if "transcript_lines" in locals():
            del transcript_lines

        # Force garbage collection if memory usage is high
        if (
            final_memory.get("process_memory_percent", 0)
            > memory_tracker.warning_threshold
        ):
            memory_tracker.force_garbage_collection()

        return transcript_text, file_path, metadata

    except Exception as e:
        # Track memory on error too
        memory_tracker.log_memory_usage(f"transcript_error[{video_id}]")
        logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
        raise ValueError(f"Failed to get transcript for video {video_id}: {str(e)}")


# async def start_channel_transcript_download(
#     channel_name: str,
#     max_results: int,
#     user_id: str,
#     include_timestamps: bool = False,
#     include_video_title: bool = True,
#     include_video_id: bool = True,
#     include_video_url: bool = True,
#     include_view_count: bool = False,
#     concatenate_all: bool = False,
# ) -> str:
#     """
#     Start the asynchronous process of downloading transcripts for a channel.

#     Args:
#         channel_name: Channel name or ID
#         max_results: Maximum number of videos to process
#         user_id: User identifier for directory organization

#     Returns:
#         Job ID for tracking the download process

#     Raises:
#         ValueError: If channel not found or other errors occur
#     """
#     try:
#         # Get channel info to validate channel existence and get channel ID
#         channel_info = await get_channel_info(channel_name)
#         channel_id = channel_info["channelId"]

#         # Get list of videos from the channel
#         videos = await get_channel_videos(channel_id, max_results)

#         logger.info(
#             f"Preparing to download {len(videos)} videos for channel '{channel_name}' (ID: {channel_id})"
#         )
#         if not videos:
#             logger.warning(f"No videos found for channel: {channel_name}")
#             raise ValueError(f"No videos found for channel: {channel_name}")

#         # Log the full list of videos to be downloaded
#         logger.info("Videos to be downloaded:")
#         for v in videos:
#             logger.info(f"  - {v['id']}: {v['title']}")

#         # Pre-fetch metadata for all videos
#         video_ids = [v["id"] for v in videos]
#         logger.info(f"Pre-fetching metadata for {len(video_ids)} videos...")
#         videos_metadata = await pre_fetch_videos_metadata(video_ids)
#         logger.info(f"Metadata pre-fetching completed")

#         # Create a unique job ID
#         job_id = str(uuid.uuid4())  # Initialize job entry in the tracking dictionary
#         channel_download_jobs[job_id] = {
#             "status": "processing",
#             "channel_name": channel_name,
#             "channel_id": channel_id,
#             "total_videos": len(videos),
#             "processed_count": 0,
#             "failed_count": 0,
#             "completed": 0,
#             "files": [],
#             "videos": videos,
#             "start_time": time.time(),
#             "user_id": user_id,
#             "credits_reserved": len(videos),  # Total credits reserved upfront
#             "credits_used": 0,  # Track actual credits used per video
#             "reservation_id": None,  # Will be set when credits are reserved
#             "videos_metadata": videos_metadata,  # Pre-fetched metadata
#             # Formatting options
#             "include_timestamps": include_timestamps,
#             "include_video_title": include_video_title,
#             "include_video_id": include_video_id,
#             "include_video_url": include_video_url,
#             "include_view_count": include_view_count,
#             "concatenate_all": concatenate_all,
#         }

#         # Save job to persistent storage immediately after creation
#         save_job_to_file(job_id, channel_download_jobs[job_id])
#         logger.info(f"Created and saved job {job_id} for channel {channel_name}")

#         # Start the background task to process transcripts
#         asyncio.create_task(download_channel_transcripts_task(job_id))

#         return job_id

#     except Exception as e:
#         logger.error(f"Error starting transcript download for {channel_name}: {str(e)}")
#         raise ValueError(f"Failed to start transcript download: {str(e)}")


async def start_selected_videos_transcript_download(
    channel_name: str,
    videos: List[Dict[str, Any]],
    user_id: str,
    include_timestamps: bool = False,
    include_video_title: bool = True,
    include_video_id: bool = True,
    include_video_url: bool = True,
    include_view_count: bool = False,
    concatenate_all: bool = False,
) -> str:
    """
    Start transcript download for a user-selected list of videos from a channel.

    Args:
        channel_name: Channel name or ID
        videos: List of video dictionaries containing at least 'id' and 'title'
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

        # Count and log the video selection
        num_videos = len(videos)
        logger.info(
            f"Preparing to download {num_videos} selected videos for channel '{channel_name}' (ID: {channel_id})"
        )

        if not videos:
            logger.warning(f"No videos selected for channel: {channel_name}")
            raise ValueError(f"No videos selected for channel: {channel_name}")

        # Log the list of videos to be downloaded
        logger.info("Selected videos to be downloaded:")
        video_ids = []
        for v in videos:
            try:
                # Handle both dictionary access and object attribute access
                video_id = v.id if hasattr(v, "id") else v["id"]
                video_title = v.title if hasattr(v, "title") else v["title"]
                logger.info(f"  - {video_id}: {video_title}")
                video_ids.append(video_id)
            except Exception as e:
                logger.error(f"Error accessing video properties: {str(e)}")
                logger.error(f"Video object type: {type(v)}")

        # Pre-fetch metadata for all selected videos
        logger.info(f"Pre-fetching metadata for {len(video_ids)} selected videos...")
        videos_metadata = await pre_fetch_videos_metadata(video_ids)
        logger.info("Metadata pre-fetching completed")

        # Create a unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job entry in the tracking dictionary
        channel_download_jobs[job_id] = {
            "status": "processing",
            "channel_name": channel_name,
            "channel_id": channel_id,
            "total_videos": num_videos,
            "processed_count": 0,
            "failed_count": 0,
            "completed": 0,
            "files": [],
            "videos": videos,  # Store the selected videos list
            "start_time": time.time(),
            "user_id": user_id,
            "credits_reserved": num_videos,  # Total credits reserved upfront
            "credits_used": 0,  # Track actual credits used per video
            "reservation_id": None,  # Will be set when credits are reserved
            "videos_metadata": videos_metadata,  # Pre-fetched metadata
            # Formatting options
            "include_timestamps": include_timestamps,
            "include_video_title": include_video_title,
            "include_video_id": include_video_id,
            "include_video_url": include_video_url,
            "include_view_count": include_view_count,
            "concatenate_all": concatenate_all,
        }

        # Save job to persistent storage immediately after creation
        save_job_to_file(job_id, channel_download_jobs[job_id])
        logger.info(
            f"Created and saved selected videos job {job_id} for channel {channel_name}"
        )

        # Start the background task to process transcripts
        # Uses the same task processor as the full channel download
        asyncio.create_task(download_channel_transcripts_task(job_id))

        return job_id

    except Exception as e:
        logger.error(
            f"Error starting selected videos transcript download for {channel_name}: {str(e)}"
        )
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

    # Reserve credits upfront for all videos (Phase 2: Batch Credit Management)
    try:
        from transcript_api import CreditManager

        credits_to_reserve = job["credits_reserved"]
        logger.info(
            f"Job {job_id}: Reserving {credits_to_reserve} credits for user {user_id}"
        )

        reservation_id = CreditManager.reserve_credits(user_id, credits_to_reserve)
        job["reservation_id"] = reservation_id

        logger.info(
            f"Job {job_id}: Successfully reserved {credits_to_reserve} credits, reservation: {reservation_id}"
        )

    except Exception as e:
        logger.error(f"Job {job_id}: Failed to reserve credits: {str(e)}")
        job["status"] = "failed"
        job["end_time"] = time.time()
        job["duration"] = job["end_time"] - job["start_time"]
        job["error"] = f"Credit reservation failed: {str(e)}"

        # Save failed job status to persistent storage
        update_job_progress(
            job_id,
            status=job["status"],
            end_time=job["end_time"],
            duration=job["duration"],
            error=job["error"],
        )

        return

    # Track memory at start of batch processing
    memory_tracker.log_memory_usage(f"batch_start[{job_id}]")

    # Create directory for this specific job
    output_dir = os.path.join(settings.temp_dir, user_id, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Process videos concurrently with a limit on parallelism
    # Process in batches to avoid overwhelming the API
    batch_size = 10  # Process 5 videos at a time

    for batch_num, i in enumerate(range(0, len(videos), batch_size), 1):
        batch_videos = videos[i : i + batch_size]

        # Track memory before each batch
        memory_tracker.log_memory_usage(f"batch_{batch_num}_start[{job_id}]")

        tasks = []

        # Create tasks for each video in the current batch
        for video in batch_videos:
            try:
                # Handle both dictionary access and object attribute access
                video_id = video.id if hasattr(video, "id") else video["id"]
                tasks.append(process_single_video(job_id, video_id, output_dir))
            except Exception as e:
                logger.error(f"Error accessing video ID: {str(e)}")
                logger.error(f"Video object type: {type(video)}")

        # Run the batch concurrently and wait for all to complete
        await asyncio.gather(*tasks)

        # Add small delay between batches to reduce API pressure and prevent memory corruption
        if batch_num < len(range(0, len(videos), batch_size)):
            await asyncio.sleep(1.0)  # 1 second delay between batches

        # Track memory after each batch and force GC if needed
        batch_memory = memory_tracker.log_memory_usage(
            f"batch_{batch_num}_complete[{job_id}]"
        )

        # Force garbage collection after each batch if memory usage is getting high
        if (
            batch_memory.get("process_memory_percent", 0)
            > memory_tracker.warning_threshold
        ):
            logger.info(
                f"High memory usage detected after batch {batch_num}, forcing garbage collection"
            )
            memory_tracker.force_garbage_collection()

    # Finalize credit usage - refund unused credits (Phase 2: Batch Credit Management)
    try:
        if job.get("reservation_id"):
            CreditManager.finalize_credit_usage(
                user_id=user_id,
                reservation_id=job["reservation_id"],
                credits_used=job["credits_used"],
                credits_reserved=job["credits_reserved"],
            )
            logger.info(
                f"Job {job_id}: Finalized credit usage - used {job['credits_used']}/{job['credits_reserved']} credits"
            )
    except Exception as e:
        logger.error(f"Job {job_id}: Error finalizing credit usage: {str(e)}")
        # Don't fail the job for credit finalization errors

    # Mark job as completed when all videos are processed
    job["status"] = "completed"
    job["end_time"] = time.time()
    job["duration"] = job["end_time"] - job["start_time"]

    # Save final job status to persistent storage
    update_job_progress(
        job_id, status=job["status"], end_time=job["end_time"], duration=job["duration"]
    )

    # Final memory tracking
    final_memory = memory_tracker.log_memory_usage(f"batch_final[{job_id}]")

    logger.info(
        f"Job {job_id} completed. Processed {job['completed']}/{job['total_videos']} videos successfully."
    )

    # Log memory summary for the entire job
    memory_logger.info(
        f"Job {job_id} memory summary: "
        f"Peak memory: {_memory_stats['peak_memory_mb']:.2f}MB ({_memory_stats['peak_memory_percent']:.2f}%), "
        f"GC runs: {_memory_stats['gc_count']}, "
        f"Memory warnings: {_memory_stats['memory_warnings']}, "
        f"Final memory: {final_memory.get('process_memory_mb', 0):.2f}MB"
    )


async def process_single_video(job_id: str, video_id: str, output_dir: str) -> None:
    """
    Process a single video for transcript download.
    Note: Credits are handled at the job level via reservation system.

    Args:
        job_id: The job identifier
        video_id: YouTube video ID to process
        output_dir: Directory to save transcript files
    """
    if job_id not in channel_download_jobs:
        logger.error(f"Job {job_id} not found for video {video_id}")
        return

    job = channel_download_jobs[job_id]
    video_dir = os.path.join(
        output_dir, video_id
    )  # Use video ID as subdirectory for isolation

    try:
        # Create a separate subdirectory for each video to isolate failures
        os.makedirs(video_dir, exist_ok=True)

        # Add a timeout to prevent any single video from taking too long
        try:
            # Get pre-fetched metadata for this video
            video_metadata = job.get("videos_metadata", {}).get(video_id, {})

            # Log if we're using pre-fetched metadata
            if video_metadata:
                logger.debug(f"Using pre-fetched metadata for video {video_id}")
            else:
                logger.debug(f"No pre-fetched metadata for video {video_id}")

            # Get transcript with timeout to prevent hanging - use job formatting options and pre-fetched metadata
            transcript_task = asyncio.create_task(
                get_single_transcript(
                    video_id,
                    video_dir,
                    include_timestamps=job["include_timestamps"],
                    include_video_title=job["include_video_title"],
                    include_video_id=job["include_video_id"],
                    include_video_url=job["include_video_url"],
                    include_view_count=job["include_view_count"],
                    pre_fetched_metadata=video_metadata,
                )
            )
            _, file_path, metadata = await asyncio.wait_for(
                transcript_task, timeout=35.0
            )  # 35 second timeout

            if file_path:
                # Update job statistics and track credit usage
                job["files"].append(
                    {
                        "file_path": file_path,
                        "video_id": video_id,
                        "language": metadata.get("transcript_language"),
                        "type": metadata.get("transcript_type"),
                    }
                )
                job["completed"] += 1
                job["credits_used"] += 1  # Track credit usage for finalization

                # Update progress with persistence
                update_job_progress(
                    job_id,
                    files=job["files"],
                    completed=job["completed"],
                    credits_used=job["credits_used"],
                )

                logger.info(
                    f"Downloaded transcript for video {video_id} ({job['completed']}/{job['total_videos']})"
                )

        except asyncio.TimeoutError:
            logger.error(
                f"Transcript processing for {video_id} timed out after 60 seconds"
            )
            job["credits_used"] += 1  # Count failed attempts as credit usage
            raise ValueError("Transcript processing timed out")

    except Exception as e:
        # Log failure but continue with other videos
        job["failed_count"] += 1
        job["credits_used"] += 1  # Count failed attempts as credit usage

        # Update progress with persistence
        update_job_progress(
            job_id, failed_count=job["failed_count"], credits_used=job["credits_used"]
        )

        logger.error(f"Failed to process video {video_id} for job {job_id}: {str(e)}")

    finally:
        # Increment processed count and update progress
        job["processed_count"] += 1
        update_job_progress(job_id, processed_count=job["processed_count"])


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the current status of a transcript download job with persistence fallback.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with job status information including credit usage

    Raises:
        ValueError: If job ID is not found
    """
    # First try in-memory storage
    # if job_id in channel_download_jobs:
    #     job = channel_download_jobs[job_id]
    #     logger.debug(f"Found job {job_id} in memory")
    # else:
    #     # Fallback to persistent storage
    job = load_job_from_file(job_id)
    if job:
        # Restore to in-memory storage
        channel_download_jobs[job_id] = job
        logger.info(f"Restored job {job_id} from persistent storage to memory")
    else:
        logger.error(f"Job not found in memory or persistent storage: {job_id}")
        raise ValueError(f"Job not found with ID: {job_id}")

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
        "credits_reserved": job["credits_reserved"],
        "credits_used": job["credits_used"],
        "current_user_credits": current_credits,
        # Formatting options
        "include_timestamps": job.get("include_timestamps", False),
        "include_video_title": job.get("include_video_title", True),
        "include_video_id": job.get("include_video_id", True),
        "include_video_url": job.get("include_video_url", True),
        "include_view_count": job.get("include_view_count", False),
        "concatenate_all": job.get("concatenate_all", False),
    }

    # Add credit usage summary
    status_info["credits_used_this_job"] = job["credits_used"]
    status_info["credits_remaining_for_job"] = max(
        0, job["credits_reserved"] - job["credits_used"]
    )

    return status_info


async def create_transcript_zip(job_id: str) -> Optional[io.BytesIO]:
    """
    Create a ZIP archive of all transcripts for a completed job.
    If concatenate_all is True, creates a single concatenated file instead of individual files.

    Args:
        job_id: The job identifier

    Returns:
        BytesIO object containing the ZIP file, or None if job not completed

    Raises:
        ValueError: If job ID not found or job not completed
    """
    if job_id not in channel_download_jobs:
        raise ValueError(f"Job not found with ID: {job_id}")

    # job = channel_download_jobs[job_id]
    job = load_job_from_file(job_id)

    if job["status"] != "completed":
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job["files"]:
        raise ValueError("No transcript files found for this job")

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = io.BytesIO()

    # Check if we should concatenate all transcripts into a single file
    if job.get("concatenate_all", False):
        # Create a single concatenated file
        concatenated_content = await create_concatenated_transcript(job_id)

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Use the channel name for the concatenated file
            safe_channel_name = sanitize_filename(job["channel_name"])
            filename = f"{safe_channel_name}_all_transcripts.txt"
            zip_file.writestr(filename, concatenated_content)
    else:
        # Create a ZIP file with individual transcript files
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for file_info in job["files"]:
                file_path = file_info["file_path"]
                filename = os.path.basename(file_path)
                zip_file.write(file_path, filename)

    # Seek to beginning of buffer for response
    zip_buffer.seek(0)

    return zip_buffer


async def create_concatenated_transcript(job_id: str) -> str:
    """
    Create a single concatenated transcript from all individual transcript files.

    Args:
        job_id: The job identifier

    Returns:
        String containing all transcripts concatenated with separators

    Raises:
        ValueError: If job not found
    """
    if job_id not in channel_download_jobs:
        raise ValueError(f"Job not found with ID: {job_id}")

    job = channel_download_jobs[job_id]
    concatenated_parts = []

    # Add header with channel information
    concatenated_parts.append(f"CHANNEL: {job['channel_name']}")
    concatenated_parts.append(f"TOTAL VIDEOS: {job['completed']}")
    concatenated_parts.append(f"GENERATED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    concatenated_parts.append("=" * 80)
    concatenated_parts.append("")

    # Read each transcript file and add it to the concatenated content
    for i, file_info in enumerate(job["files"], 1):
        file_path = file_info["file_path"]
        video_id = file_info["video_id"]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add section separator
            concatenated_parts.append(f"[VIDEO {i}/{job['completed']}]")
            concatenated_parts.append("-" * 60)
            concatenated_parts.append(content)
            concatenated_parts.append("")  # Empty line between videos
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")  # Empty line between sections

        except Exception as e:
            logger.error(f"Error reading transcript file {file_path}: {str(e)}")
            # Add error message to concatenated content
            concatenated_parts.append(
                f"[VIDEO {i}/{job['completed']}] - ERROR READING TRANSCRIPT"
            )
            concatenated_parts.append(f"Video ID: {video_id}")
            concatenated_parts.append(f"Error: {str(e)}")
            concatenated_parts.append("")
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")

    return "\n".join(concatenated_parts)


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
    # Track memory before cleanup
    memory_tracker.log_memory_usage("cleanup_start")

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

    # Force garbage collection after cleanup and track memory
    if jobs_to_remove:
        memory_tracker.force_garbage_collection()
        memory_tracker.log_memory_usage("cleanup_complete")


# Convenience functions for external use
def init_memory_monitoring(interval_seconds: int = 30, auto_start: bool = True) -> None:
    """
    Initialize memory monitoring for the YouTube service.

    Args:
        interval_seconds: How often to log memory stats in background
        auto_start: Whether to start monitoring immediately
    """
    if auto_start:
        start_memory_monitoring(interval_seconds)

    memory_logger.info(
        f"Memory monitoring initialized with {interval_seconds}s interval"
    )


def get_memory_report() -> Dict[str, Any]:
    """
    Get a comprehensive memory report for the service.

    Returns:
        Dictionary with current memory stats and historical data
    """
    current_stats = get_memory_stats()

    return {
        "current": current_stats,
        "thresholds": {
            "warning_percent": memory_tracker.warning_threshold,
            "critical_percent": memory_tracker.critical_threshold,
        },
        "recommendations": _get_memory_recommendations(current_stats),
    }


def _get_memory_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate memory usage recommendations based on current stats."""
    recommendations = []

    memory_percent = stats.get("process_memory_percent", 0)

    if memory_percent > 90:
        recommendations.append(
            "CRITICAL: Memory usage above 90%. Consider restarting the service."
        )
    elif memory_percent > 75:
        recommendations.append(
            "HIGH: Memory usage above 75%. Monitor closely and consider reducing batch sizes."
        )
    elif memory_percent > 50:
        recommendations.append(
            "MODERATE: Memory usage above 50%. Consider running garbage collection."
        )

    if stats.get("memory_warnings", 0) > 5:
        recommendations.append(
            "Multiple memory warnings detected. Consider reducing concurrent operations."
        )

    if stats.get("gc_count", 0) == 0:
        recommendations.append(
            "No garbage collections performed yet. This is normal for new processes."
        )

    return recommendations


def log_memory_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """
    Log a memory checkpoint with a custom name.
    Useful for tracking memory at specific points in your application.

    Args:
        checkpoint_name: Name for this checkpoint

    Returns:
        Current memory information
    """
    return memory_tracker.log_memory_usage(f"checkpoint[{checkpoint_name}]")


# ytt_api = YouTubeTranscriptApi()

# asd = ytt_api.fetch("cESaIUWoCJQ")

# bbb = ytt_api.fetch("nCuaNmeVfQY")

# asd
