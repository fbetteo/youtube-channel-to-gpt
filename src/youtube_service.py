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

ENABLE_MEMORY_TRACKING = False
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
        if not ENABLE_MEMORY_TRACKING:
            return {}
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


# Dictionary to track channel download jobs
# channel_download_jobs: Dict[str, Dict[str, Any]] = {}

# Add persistent job storage
JOBS_STORAGE_DIR = os.path.join(settings.temp_dir, "jobs")
os.makedirs(JOBS_STORAGE_DIR, exist_ok=True)


def save_job_to_file(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save job data to persistent storage"""
    try:
        job_file = os.path.join(JOBS_STORAGE_DIR, f"{job_id}.json")
        # Convert non-serializable objects to serializable format
        # Include videos_metadata since it's needed for prefetch functionality
        serializable_data = {}
        for k, v in job_data.items():
            if k == "videos" and v:
                # Convert Pydantic VideoInfo objects to dictionaries
                serializable_videos = []
                for video in v:
                    if hasattr(video, "dict"):
                        # It's a Pydantic object, convert to dict
                        serializable_videos.append(video.dict())
                    elif hasattr(video, "id"):
                        # It's an object with attributes, convert to dict
                        serializable_videos.append(
                            {
                                "id": video.id,
                                "title": getattr(video, "title", ""),
                                "url": getattr(
                                    video,
                                    "url",
                                    f"https://www.youtube.com/watch?v={video.id}",
                                ),
                                "duration": getattr(video, "duration", None),
                                "publishedAt": getattr(video, "publishedAt", None),
                            }
                        )
                    else:
                        # It's already a dict or other serializable format
                        serializable_videos.append(video)
                serializable_data[k] = serializable_videos
            else:
                serializable_data[k] = v

        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, default=str, ensure_ascii=False, indent=2)
        logger.debug(f"Saved job {job_id} to persistent storage")
    except Exception as e:
        logger.error(f"Failed to save job {job_id}: {e}")


def load_job_from_file(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job data from persistent storage"""
    try:
        job_file = os.path.join(JOBS_STORAGE_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            logger.debug(f"Loaded job {job_id} from persistent storage")
            return job_data
        return None
    except Exception as e:
        logger.error(f"Failed to load job {job_id}: {e}")
        return None


def update_job_progress(job_id: str, **updates):
    """Update job progress and save to persistent storage with atomic operations"""
    import time
    import shutil
    import tempfile
    from threading import Lock

    # Use a global lock for all job updates (simpler and more memory efficient)
    if not hasattr(update_job_progress, "_global_lock"):
        update_job_progress._global_lock = Lock()

    # Use the global lock instead of per-job locks to prevent memory accumulation
    with update_job_progress._global_lock:
        # Try to update with atomic file operations
        max_retries = 3
        for attempt in range(max_retries):
            try:
                job_file = os.path.join(JOBS_STORAGE_DIR, f"{job_id}.json")

                # Create temporary file for atomic write
                temp_filename = None
                try:
                    # Load current data
                    if os.path.exists(job_file):
                        with open(job_file, "r", encoding="utf-8") as f:
                            job = json.load(f)
                    else:
                        job = {}

                    # Apply updates, handling special increment operations
                    for key, value in updates.items():
                        if key.endswith("_increment"):
                            # Handle atomic increments
                            base_key = key.replace("_increment", "")
                            job[base_key] = job.get(base_key, 0) + value
                        elif key == "files_append":
                            # Handle appending to files list
                            if "files" not in job:
                                job["files"] = []
                            job["files"].append(value)
                        else:
                            # Regular update
                            job[key] = value

                    # Write to temporary file first (atomic operation)
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=JOBS_STORAGE_DIR,
                        prefix=f".{job_id}_",
                        suffix=".tmp",
                        delete=False,
                    ) as temp_file:
                        json.dump(
                            job, temp_file, default=str, indent=2, ensure_ascii=False
                        )
                        temp_file.flush()
                        os.fsync(temp_file.fileno())  # Force write to disk
                        temp_filename = temp_file.name

                    # Atomic move: replace the original file
                    if os.name == "nt":  # Windows
                        if os.path.exists(job_file):
                            os.replace(temp_filename, job_file)
                        else:
                            shutil.move(temp_filename, job_file)
                    else:  # Unix/Linux
                        os.rename(temp_filename, job_file)

                    logger.debug(f"Updated job {job_id} progress: {updates}")
                    return job

                except Exception as inner_e:
                    # Clean up temporary file if it exists
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except Exception:
                            pass
                    raise inner_e

            except (
                json.JSONDecodeError,
                FileNotFoundError,
                PermissionError,
                OSError,
            ) as e:
                logger.warning(
                    f"Attempt {attempt + 1} to update job {job_id} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2**attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(
                        f"Failed to update job {job_id} after {max_retries} attempts: {e}"
                    )
                    return None


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

                        # channel_download_jobs[job_id] = job_data
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


def extract_playlist_id(url_or_id: str) -> str:
    """
    Extract YouTube playlist ID from various URL formats or validate existing ID.

    Args:
        url_or_id: YouTube playlist URL or playlist ID

    Returns:
        The YouTube playlist ID

    Raises:
        ValueError: If unable to extract a valid YouTube playlist ID
    """
    # Common YouTube playlist URL patterns
    patterns = [
        r"[?&]list=([a-zA-Z0-9_-]+)",  # ?list=PLxxxxxx or &list=PLxxxxxx
        r"playlist\?list=([a-zA-Z0-9_-]+)",  # playlist?list=PLxxxxxx
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            playlist_id = match.group(1)
            # Validate that it looks like a valid playlist ID
            if re.match(r"^[a-zA-Z0-9_-]{10,}$", playlist_id):
                return playlist_id

    # If no pattern matches, perhaps the input is already just the ID
    if re.match(r"^[a-zA-Z0-9_-]{10,}$", url_or_id):
        return url_or_id

    raise ValueError(f"Could not extract YouTube playlist ID from URL: {url_or_id}")


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

    # Debug: Log sample of what was fetched
    if metadata_results:
        sample_video_id = list(metadata_results.keys())[0]
        sample_metadata = metadata_results[sample_video_id]
        logger.debug(f"Sample prefetch result for {sample_video_id}: {sample_metadata}")

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


async def get_playlist_info(playlist_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a YouTube playlist.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        Dictionary with playlist information (title, description, video count, etc.)

    Raises:
        ValueError: If playlist is not found
    """
    try:

        def _fetch_playlist_info():
            # Use thread-safe client
            client = get_youtube_client()
            request = client.playlists().list(
                part="snippet,contentDetails,status", id=playlist_id
            )
            return request.execute()

        # Run the blocking API call in a thread pool
        response = await asyncio.to_thread(_fetch_playlist_info)

        if not response.get("items"):
            raise ValueError(
                f"Playlist not found: '{playlist_id}'. "
                f"Please provide a valid playlist ID or ensure the playlist is public."
            )

        playlist_info = response["items"][0]
        snippet = playlist_info["snippet"]
        content_details = playlist_info["contentDetails"]

        logger.info(
            f"Successfully fetched playlist '{playlist_id}' (title: '{snippet['title']}')"
        )

        # Get best thumbnail
        thumbnails = snippet.get("thumbnails", {})
        best_thumbnail = (
            thumbnails.get("maxres")
            or thumbnails.get("high")
            or thumbnails.get("medium")
            or thumbnails.get("default")
            or {}
        )

        return {
            "title": snippet["title"],
            "description": snippet.get("description", ""),
            "thumbnail": best_thumbnail.get("url", ""),
            "videoCount": int(content_details.get("itemCount", 0)),
            "channelTitle": snippet.get("channelTitle", ""),
            "channelId": snippet.get("channelId", ""),
            "playlistId": playlist_id,
            "publishedAt": snippet.get("publishedAt", ""),
        }

    except Exception as e:
        logger.error(f"Error fetching playlist info for {playlist_id}: {str(e)}")
        raise ValueError(f"Failed to get playlist information: {str(e)}")


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


def _fetch_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all videos from a playlist (quota efficient).
    Returns a list of video metadata dicts (id, title, publishedAt, duration category, etc.).
    """
    logger.info(f"Fetching all videos from playlist {playlist_id}")

    client = get_youtube_client()
    all_videos = []
    next_page_token = None

    # Paginate through all videos in the playlist
    while True:
        request = client.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
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

    logger.info(f"Found {len(all_videos)} videos in playlist {playlist_id}")

    # Log duration distribution
    duration_counts = {}
    for video in all_videos:
        category = video["duration"]
        duration_counts[category] = duration_counts.get(category, 0) + 1

    logger.info(f"Duration distribution: {duration_counts}")

    return all_videos


async def get_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]:
    """
    Async wrapper for _fetch_all_playlist_videos with better error handling and logging.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        List of video dictionaries with metadata
    """
    try:
        logger.info(f"Fetching all videos for playlist {playlist_id}")
        videos = await asyncio.to_thread(_fetch_all_playlist_videos, playlist_id)
        logger.info(
            f"Successfully fetched {len(videos)} videos for playlist {playlist_id}"
        )
        return videos
    except Exception as e:
        logger.error(f"Error in get_all_playlist_videos for {playlist_id}: {str(e)}")
        raise ValueError(f"Failed to fetch all videos: {str(e)}")


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


async def dispatch_lambdas_concurrently(
    job_id: str,
    videos: List[Any],
    videos_metadata: Dict[str, Dict[str, Any]],
    user_id: str,
    formatting_options: Dict[str, Any],
    max_concurrent: int = 20,
) -> int:
    """
    Dispatch Lambda functions concurrently in true async fire-and-forget mode.

    Args:
        job_id: The job identifier
        videos: List of video objects to process
        videos_metadata: Pre-fetched metadata for videos
        user_id: User identifier
        formatting_options: Formatting options for transcripts
        max_concurrent: Not used - kept for compatibility

    Returns:
        Number of Lambda dispatches attempted
    """
    import boto3
    import json

    # Create Lambda client once (thread-safe)
    lambda_client = boto3.client("lambda")

    logger.info(
        f"Job {job_id}: Async fire-and-forget dispatch of {len(videos)} Lambda functions"
    )

    async def dispatch_single_lambda(video):
        """Dispatch a single Lambda function asynchronously"""
        try:
            video_id = video.id if hasattr(video, "id") else video["id"]
            pre_fetched_metadata = videos_metadata.get(video_id, {})

            lambda_payload = {
                "video_id": video_id,
                "job_id": job_id,
                "user_id": user_id,
                "formatting_options": formatting_options,
                "pre_fetched_metadata": pre_fetched_metadata,
                "api_base_url": settings.api_base_url,
            }

            # Use asyncio.to_thread for truly async Lambda dispatch
            await asyncio.to_thread(
                lambda_client.invoke,
                FunctionName=settings.lambda_function_name,
                InvocationType="Event",  # Asynchronous invocation
                Payload=json.dumps(lambda_payload),
            )

            logger.debug(f"Job {job_id}: Dispatched Lambda for video {video_id}")
            return True

        except Exception as e:
            logger.error(
                f"Job {job_id}: Failed to dispatch Lambda for video {video_id}: {str(e)}"
            )
            # Update failure count atomically
            update_job_progress(job_id, failed_count_increment=1)
            return False

    # Create all tasks at once (no semaphore, no limits)
    tasks = [dispatch_single_lambda(video) for video in videos]

    # Fire all Lambda dispatches concurrently without waiting for completion
    # Use asyncio.gather with return_exceptions to prevent any single failure from stopping others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful dispatches
    dispatch_count = sum(1 for result in results if result is True)

    logger.info(
        f"Job {job_id}: Dispatched {dispatch_count}/{len(videos)} Lambda functions concurrently"
    )
    return dispatch_count


async def prefetch_and_dispatch_task(job_id: str):
    """
    Background task that:
    1. Pre-fetches metadata in batches
    2. Updates job status
    3. Dispatches Lambda functions
    4. Updates job to 'processing'
    """
    try:
        # Load job data
        job = load_job_from_file(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for pre-fetching")
            return

        videos = job["videos"]
        user_id = job["user_id"]

        logger.info(
            f"Job {job_id}: Starting background pre-fetch for {len(videos)} videos"
        )

        # 1. Pre-fetch metadata in batches (this takes time)
        video_ids = []
        for video in videos:
            try:
                video_id = video.id if hasattr(video, "id") else video["id"]
                video_ids.append(video_id)
            except Exception as e:
                logger.error(f"Error extracting video ID: {str(e)}")

        update_job_progress(job_id, status="prefetching_metadata")

        videos_metadata = await pre_fetch_videos_metadata(video_ids)

        # 2. Update job with metadata
        update_job_progress(
            job_id,
            videos_metadata=videos_metadata,
            prefetch_completed=True,
            status="dispatching_lambda",
        )

        logger.info(
            f"Job {job_id}: Metadata pre-fetch completed, dispatching Lambda functions"
        )

        # 3. Dispatch Lambda functions concurrently with pre-fetched metadata
        dispatched_count = await dispatch_lambdas_concurrently(
            job_id, videos, videos_metadata, user_id, job["formatting_options"]
        )

        # 4. Update job status to processing and start timeout monitoring
        update_job_progress(
            job_id,
            status="processing",
            lambda_dispatched_count=dispatched_count,
            lambda_dispatch_time=time.time(),  # Track when Lambdas were dispatched
        )

        logger.info(
            f"Job {job_id}: Background task completed. Dispatched {dispatched_count} Lambda functions"
        )

        # 5. Start timeout monitoring task
        asyncio.create_task(monitor_job_timeout(job_id, settings.job_timeout_minutes))

    except Exception as e:
        logger.error(f"Job {job_id}: Background pre-fetch task failed: {str(e)}")
        # Update job status to failed
        update_job_progress(job_id, status="failed", error_message=str(e))

        # Refund credits since processing failed
        try:
            from transcript_api import CreditManager

            CreditManager.finalize_credit_usage(
                user_id=job["user_id"],
                reservation_id=job["reservation_id"],
                credits_used=0,  # No credits used since processing failed
                credits_reserved=job["credits_reserved"],
            )
        except Exception as credit_error:
            logger.error(
                f"Failed to refund credits for failed job {job_id}: {str(credit_error)}"
            )


async def monitor_job_timeout(job_id: str, timeout_minutes: int = 10):
    """
    Monitor a job for timeout and mark pending videos as failed.

    Args:
        job_id: The job identifier
        timeout_minutes: Minutes to wait before timing out pending videos
    """
    try:
        logger.info(
            f"Job {job_id}: Starting timeout monitor with {timeout_minutes} minute limit"
        )

        # Wait for the timeout period
        await asyncio.sleep(timeout_minutes * 60)

        # Check if job is still processing
        job = load_job_from_file(job_id)
        if not job:
            logger.warning(f"Job {job_id}: Job not found during timeout check")
            return

        # Only proceed if job is still processing
        if job.get("status") != "processing":
            logger.info(
                f"Job {job_id}: Job no longer processing, timeout monitor exiting"
            )
            return

        # Calculate how many videos are still pending
        total_videos = job["total_videos"]
        processed_count = job["processed_count"]
        pending_count = total_videos - processed_count

        if pending_count > 0:
            logger.warning(
                f"Job {job_id}: {pending_count} videos still pending after {timeout_minutes} minutes, marking as failed"
            )

            # Mark pending videos as failed and complete the job
            update_job_progress(
                job_id,
                status="completed",
                timeout_occurred=True,
                timeout_failed_count=pending_count,
            )

            # Finalize credits for completed job
            try:
                from transcript_api import CreditManager

                updated_job = load_job_from_file(job_id)
                if updated_job and updated_job.get("reservation_id"):
                    CreditManager.finalize_credit_usage(
                        user_id=updated_job["user_id"],
                        reservation_id=updated_job["reservation_id"],
                        credits_used=updated_job["credits_used"],
                        credits_reserved=updated_job["credits_reserved"],
                    )
                    logger.info(f"Job {job_id}: Finalized credits after timeout")
            except Exception as credit_error:
                logger.error(
                    f"Job {job_id}: Failed to finalize credits after timeout: {str(credit_error)}"
                )

            logger.info(
                f"Job {job_id}: Timeout handling completed, job marked as finished"
            )
        else:
            logger.info(
                f"Job {job_id}: All videos processed before timeout, no action needed"
            )

    except Exception as e:
        logger.error(f"Job {job_id}: Error in timeout monitor: {str(e)}")


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
        # channel_download_jobs[job_id] = job
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
        "channel_name": job.get("channel_name"),  # Channel name (if channel job)
        "playlist_id": job.get("playlist_id"),  # Playlist ID (if playlist job)
        "is_playlist": job.get("is_playlist", False),  # Flag to distinguish job type
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
        # New status fields for enhanced workflow
        "prefetch_completed": job.get("prefetch_completed", False),
        "lambda_dispatched_count": job.get("lambda_dispatched_count", 0),
        # Formatting options
        "include_timestamps": job.get("include_timestamps", False),
        "include_video_title": job.get("include_video_title", True),
        "include_video_id": job.get("include_video_id", True),
        "include_video_url": job.get("include_video_url", True),
        "include_view_count": job.get("include_view_count", False),
        "concatenate_all": job.get("concatenate_all", False),
        # Timeout monitoring fields
        "lambda_dispatch_time": job.get("lambda_dispatch_time"),
        "timeout_occurred": job.get("timeout_occurred", False),
        "timeout_failed_count": job.get("timeout_failed_count", 0),
    }

    # Add phase-specific messages
    if job["status"] == "initializing":
        status_info["message"] = "Job created, starting metadata pre-fetch..."
    elif job["status"] == "prefetching_metadata":
        status_info["message"] = (
            f"Pre-fetching metadata for {job['total_videos']} videos..."
        )
    elif job["status"] == "dispatching_lambda":
        status_info["message"] = "Metadata complete, dispatching Lambda functions..."
    elif job["status"] == "processing":
        progress_pct = (job.get("processed_count", 0) / job["total_videos"]) * 100
        elapsed_minutes = (
            time.time() - job.get("lambda_dispatch_time", job["start_time"])
        ) / 60
        timeout_minutes = settings.job_timeout_minutes
        status_info["message"] = (
            f"Processing transcripts... {progress_pct:.1f}% complete "
            f"(elapsed: {elapsed_minutes:.1f} min, timeout at {timeout_minutes} min)"
        )
    elif job["status"] == "completed":
        if job.get("timeout_occurred"):
            status_info["message"] = (
                f"Completed with {job.get('timeout_failed_count', 0)} videos timed out"
            )
        else:
            status_info["message"] = "All transcripts completed successfully"
    elif job["status"] == "completed_with_errors":
        status_info["message"] = f"Completed with {job['failed_count']} errors"
    elif job["status"] == "failed":
        status_info["message"] = (
            f"Job failed: {job.get('error_message', 'Unknown error')}"
        )

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
    # if job_id not in channel_download_jobs:
    #     raise ValueError(f"Job not found with ID: {job_id}")

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
            if not job.get("channel_name"):
                job["channel_name"] = job["source_name"]
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
    # try:
    #     del channel_download_jobs[job_id]
    # except Exception as e:
    #     logger.error(
    #         f"Error deleting job {job_id} from channel_download_jobs: {str(e)}"
    #     )

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
    # if job_id not in channel_download_jobs:
    #     raise ValueError(f"Job not found with ID: {job_id}")

    # job = channel_download_jobs[job_id]
    job = load_job_from_file(job_id)
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
    Get a filesystem-safe version of the source name (channel or playlist) for a job.

    Args:
        job_id: The job identifier

    Returns:
        Safe source name for use in filenames

    Raises:
        ValueError: If job ID not found
    """
    # if job_id not in channel_download_jobs:
    #     raise ValueError(f"Job not found with ID: {job_id}")

    # job_data = channel_download_jobs[job_id]
    job_data = load_job_from_file(job_id)

    # For playlist jobs, use source_name or playlist_id
    if job_data.get("source_type") == "playlist":
        source_name = (
            job_data.get("source_name") or job_data.get("playlist_id") or "playlist"
        )
    else:
        # For channel jobs, use channel_name or source_name
        source_name = (
            job_data.get("channel_name") or job_data.get("source_name") or "channel"
        )

    return sanitize_filename(source_name)


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


# def cleanup_old_jobs(max_age_hours: int = 24) -> None:
#     """
#     Clean up old jobs and their associated files.

#     Args:
#         max_age_hours: Maximum age of jobs to keep in hours
#     """
#     # Track memory before cleanup
#     memory_tracker.log_memory_usage("cleanup_start")

#     current_time = time.time()
#     max_age_seconds = max_age_hours * 3600

#     jobs_to_remove = []

#     # Find old jobs
#     for job_id, job in channel_download_jobs.items():
#         job_start_time = job.get("start_time", 0)
#         job_age = current_time - job_start_time

#         if job_age > max_age_seconds:
#             jobs_to_remove.append(job_id)

#             # Try to remove files
#             for file_path in job.get("files", []):
#                 try:
#                     if os.path.exists(file_path):
#                         os.remove(file_path)
#                         logger.info(f"Removed old file: {file_path}")
#                 except Exception as e:
#                     logger.warning(f"Failed to remove old file {file_path}: {str(e)}")

#             # Try to remove job directory
#             try:
#                 user_id = job.get("user_id")
#                 if user_id:
#                     job_dir = os.path.join(settings.temp_dir, user_id, job_id)
#                     if os.path.exists(job_dir) and os.path.isdir(job_dir):
#                         # Remove all files in the directory
#                         for root, dirs, files in os.walk(job_dir):
#                             for file in files:
#                                 try:
#                                     os.remove(os.path.join(root, file))
#                                 except Exception as e:
#                                     logger.warning(
#                                         f"Failed to remove file in cleanup: {str(e)}"
#                                     )

#                         # Try to remove the directory
#                         try:
#                             os.rmdir(job_dir)
#                             logger.info(f"Removed old job directory: {job_dir}")
#                         except Exception as e:
#                             logger.warning(f"Failed to remove job directory: {str(e)}")
#             except Exception as e:
#                 logger.warning(f"Error during job cleanup for {job_id}: {str(e)}")

#     # Remove old jobs from the dictionary
#     for job_id in jobs_to_remove:
#         try:
#             del channel_download_jobs[job_id]
#             logger.info(f"Removed old job: {job_id}")
#         except Exception as e:
#             logger.warning(f"Error removing job {job_id} from dictionary: {str(e)}")

#     logger.info(f"Cleanup complete. Removed {len(jobs_to_remove)} old jobs.")

#     # Force garbage collection after cleanup and track memory
#     if jobs_to_remove:
#         memory_tracker.force_garbage_collection()
#         memory_tracker.log_memory_usage("cleanup_complete")


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


# =============================================
# S3 ZIP CREATION FUNCTIONS
# =============================================


async def create_transcript_zip_from_s3_concurrent(job_id: str) -> Optional[io.BytesIO]:
    """
    Create ZIP by downloading transcript files from S3 concurrently.
    Faster than sequential downloads with better performance.

    Args:
        job_id: The job identifier

    Returns:
        BytesIO object containing the ZIP file, or None if job not completed

    Raises:
        ValueError: If job ID not found or job not completed
    """
    import boto3
    from config_v2 import settings

    # Load job data
    job = load_job_from_file(job_id)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

    if job["status"] not in ["completed", "completed_with_errors"]:
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job.get("files"):
        raise ValueError("No transcript files found for this job")

    # Initialize S3 client
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=getattr(settings, "aws_access_key_id", None),
            aws_secret_access_key=getattr(settings, "aws_secret_access_key", None),
            region_name=getattr(settings, "aws_default_region", "us-east-1"),
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        # Fallback to environment variables
        s3_client = boto3.client("s3")

    bucket_name = getattr(settings, "s3_bucket_name", None) or os.getenv(
        "S3_BUCKET_NAME"
    )
    if not bucket_name:
        raise ValueError("S3 bucket name not configured")

    # Download all files concurrently (up to 10 at a time)
    async def download_file(file_info):
        """Download a single file from S3"""
        s3_key = file_info["s3_key"]
        video_id = file_info["video_id"]

        def _download():
            try:
                logger.debug(f"Downloading {s3_key} from S3")
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                return {
                    "video_id": video_id,
                    "content": response["Body"].read(),
                    "filename": os.path.basename(s3_key),
                    "success": True,
                    "s3_key": s3_key,
                }
            except Exception as e:
                logger.error(f"Failed to download {s3_key}: {str(e)}")
                return {
                    "video_id": video_id,
                    "content": f"Error downloading transcript for video {video_id}: {str(e)}".encode(),
                    "filename": f"{video_id}_ERROR.txt",
                    "success": False,
                    "s3_key": s3_key,
                }

        # Run S3 download in thread pool
        return await asyncio.to_thread(_download)

    # Download files concurrently (max 10 concurrent downloads)
    semaphore = asyncio.Semaphore(10)

    async def download_with_limit(file_info):
        async with semaphore:
            return await download_file(file_info)

    logger.info(f"Downloading {len(job['files'])} files concurrently from S3")
    download_start = time.time()

    download_tasks = [download_with_limit(file_info) for file_info in job["files"]]
    download_results = await asyncio.gather(*download_tasks)

    download_end = time.time()
    successful_downloads = len([r for r in download_results if r["success"]])
    logger.info(
        f"Downloaded {successful_downloads}/{len(download_results)} files in {download_end - download_start:.2f}s"
    )

    # Create ZIP from downloaded content
    zip_buffer = io.BytesIO()

    if job.get("formatting_options", {}).get("concatenate_all", False):
        # Create concatenated file
        concatenated_content = await create_concatenated_content_from_results(
            download_results, job
        )

        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            source_name = job.get("source_name") or job.get(
                "channel_name", "transcripts"
            )
            safe_source_name = sanitize_filename(source_name)
            filename = f"{safe_source_name}_all_transcripts.txt"
            zip_file.writestr(filename, concatenated_content)

    else:
        # Create individual files in ZIP
        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            for result in download_results:
                zip_file.writestr(result["filename"], result["content"])

    zip_buffer.seek(0)

    logger.info(
        f"Created ZIP archive for job {job_id} with {successful_downloads} files, size: {len(zip_buffer.getvalue())} bytes"
    )

    return zip_buffer


async def create_concatenated_content_from_results(
    download_results: List[Dict[str, Any]], job: Dict[str, Any]
) -> str:
    """
    Create a single concatenated transcript from download results.

    Args:
        download_results: List of download result dictionaries
        job: Job data dictionary

    Returns:
        String containing all transcripts concatenated with separators
    """
    concatenated_parts = []

    # Add header with source information
    source_name = job.get("source_name") or job.get("channel_name", "Unknown")
    source_type = job.get("source_type", "channel")

    concatenated_parts.append(f"{source_type.upper()}: {source_name}")
    concatenated_parts.append(f"TOTAL VIDEOS: {job.get('completed', 0)}")
    concatenated_parts.append(f"GENERATED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    concatenated_parts.append("=" * 80)
    concatenated_parts.append("")

    # Add each transcript from download results
    successful_results = [r for r in download_results if r["success"]]

    for i, result in enumerate(successful_results, 1):
        try:
            # Decode content
            content = result["content"]
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            # Add section separator
            concatenated_parts.append(f"[VIDEO {i}/{len(successful_results)}]")
            concatenated_parts.append(f"Video ID: {result['video_id']}")
            concatenated_parts.append("-" * 60)
            concatenated_parts.append(content)
            concatenated_parts.append("")  # Empty line between videos
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")  # Empty line between sections

        except Exception as e:
            logger.error(
                f"Error processing transcript for video {result['video_id']}: {str(e)}"
            )
            # Add error message to concatenated content
            concatenated_parts.append(
                f"[VIDEO {i}/{len(successful_results)}] - ERROR PROCESSING TRANSCRIPT"
            )
            concatenated_parts.append(f"Video ID: {result['video_id']}")
            concatenated_parts.append(f"S3 Key: {result.get('s3_key', 'unknown')}")
            concatenated_parts.append(f"Error: {str(e)}")
            concatenated_parts.append("")
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")

    # Add failed downloads section if any
    failed_results = [r for r in download_results if not r["success"]]
    if failed_results:
        concatenated_parts.append("=" * 80)
        concatenated_parts.append("FAILED DOWNLOADS")
        concatenated_parts.append("=" * 80)
        concatenated_parts.append("")

        for i, result in enumerate(failed_results, 1):
            concatenated_parts.append(f"[FAILED {i}/{len(failed_results)}]")
            concatenated_parts.append(f"Video ID: {result['video_id']}")
            concatenated_parts.append(f"S3 Key: {result.get('s3_key', 'unknown')}")
            concatenated_parts.append("-" * 40)
            try:
                content = result["content"]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                concatenated_parts.append(content)
            except Exception as e:
                concatenated_parts.append(f"Error displaying failure message: {str(e)}")
            concatenated_parts.append("")

    return "\n".join(concatenated_parts)


async def create_transcript_zip_from_s3_sequential(job_id: str) -> Optional[io.BytesIO]:
    """
    Create ZIP by downloading transcript files from S3 sequentially.
    Fallback option if concurrent downloads cause issues.

    Args:
        job_id: The job identifier

    Returns:
        BytesIO object containing the ZIP file, or None if job not completed

    Raises:
        ValueError: If job ID not found or job not completed
    """
    import boto3
    from config_v2 import settings

    # Load job data
    job = load_job_from_file(job_id)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

    if job["status"] not in ["completed", "completed_with_errors"]:
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job.get("files"):
        raise ValueError("No transcript files found for this job")

    # Initialize S3 client
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=getattr(settings, "aws_access_key_id", None),
            aws_secret_access_key=getattr(settings, "aws_secret_access_key", None),
            region_name=getattr(settings, "aws_default_region", "us-east-1"),
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        # Fallback to environment variables
        s3_client = boto3.client("s3")

    bucket_name = getattr(settings, "s3_bucket_name", None) or os.getenv(
        "S3_BUCKET_NAME"
    )
    if not bucket_name:
        raise ValueError("S3 bucket name not configured")

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = io.BytesIO()

    # Check if we should concatenate all transcripts into a single file
    if job.get("formatting_options", {}).get("concatenate_all", False):
        # Create a single concatenated file from S3 files
        concatenated_content = await create_concatenated_transcript_from_s3_sequential(
            job_id, s3_client, bucket_name
        )

        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            # Use the source name for the concatenated file
            source_name = job.get("source_name") or job.get(
                "channel_name", "transcripts"
            )
            safe_source_name = sanitize_filename(source_name)
            filename = f"{safe_source_name}_all_transcripts.txt"
            zip_file.writestr(filename, concatenated_content)
    else:
        # Create a ZIP file with individual transcript files from S3
        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            for file_info in job["files"]:
                s3_key = file_info["s3_key"]
                video_id = file_info["video_id"]

                try:
                    # Download file content from S3
                    logger.debug(f"Downloading {s3_key} from S3 for ZIP creation")
                    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                    file_content = response["Body"].read()

                    # Create filename from video ID or extract from S3 key
                    filename = f"{video_id}.txt"
                    if "/" in s3_key:
                        filename = os.path.basename(s3_key)

                    # Add to ZIP
                    zip_file.writestr(filename, file_content)
                    logger.debug(f"Added {filename} to ZIP archive")

                except Exception as e:
                    logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
                    # Add error placeholder instead of failing entire ZIP
                    error_content = (
                        f"Error downloading transcript for video {video_id}: {str(e)}"
                    )
                    zip_file.writestr(f"{video_id}_ERROR.txt", error_content)

    # Seek to beginning of buffer for response
    zip_buffer.seek(0)

    logger.info(f"Created ZIP archive for job {job_id} with {len(job['files'])} files")
    return zip_buffer


async def create_concatenated_transcript_from_s3_sequential(
    job_id: str, s3_client, bucket_name: str
) -> str:
    """
    Create a single concatenated transcript by downloading individual files from S3 sequentially.

    Args:
        job_id: The job identifier
        s3_client: Configured boto3 S3 client
        bucket_name: S3 bucket name

    Returns:
        String containing all transcripts concatenated with separators
    """
    job = load_job_from_file(job_id)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

    concatenated_parts = []

    # Add header with source information
    source_name = job.get("source_name") or job.get("channel_name", "Unknown")
    source_type = job.get("source_type", "channel")

    concatenated_parts.append(f"{source_type.upper()}: {source_name}")
    concatenated_parts.append(f"TOTAL VIDEOS: {job.get('completed', 0)}")
    concatenated_parts.append(f"GENERATED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    concatenated_parts.append("=" * 80)
    concatenated_parts.append("")

    # Download and concatenate each transcript file from S3
    for i, file_info in enumerate(job["files"], 1):
        s3_key = file_info["s3_key"]
        video_id = file_info["video_id"]

        try:
            # Download file content from S3
            logger.debug(f"Downloading {s3_key} from S3 for concatenation")
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            content = response["Body"].read().decode("utf-8")

            # Add section separator
            concatenated_parts.append(f"[VIDEO {i}/{len(job['files'])}]")
            concatenated_parts.append(f"Video ID: {video_id}")
            concatenated_parts.append("-" * 60)
            concatenated_parts.append(content)
            concatenated_parts.append("")  # Empty line between videos
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")  # Empty line between sections

        except Exception as e:
            logger.error(f"Error downloading transcript from S3 {s3_key}: {str(e)}")
            # Add error message to concatenated content
            concatenated_parts.append(
                f"[VIDEO {i}/{len(job['files'])}] - ERROR DOWNLOADING TRANSCRIPT"
            )
            concatenated_parts.append(f"Video ID: {video_id}")
            concatenated_parts.append(f"S3 Key: {s3_key}")
            concatenated_parts.append(f"Error: {str(e)}")
            concatenated_parts.append("")
            concatenated_parts.append("=" * 80)
            concatenated_parts.append("")

    return "\n".join(concatenated_parts)


# =============================================
# LEGACY ZIP CREATION (LOCAL FILES)
# =============================================


# ytt_api = YouTubeTranscriptApi()

# asd = ytt_api.fetch("cESaIUWoCJQ")

# bbb = ytt_api.fetch("nCuaNmeVfQY")

# asd
