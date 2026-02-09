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
import zipfile
from urllib.parse import quote
from typing import Dict, List, Optional, Any, Tuple

import boto3
import yt_dlp
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

# Using the Pydantic v2 compatible settings
from config_v2 import settings

# Import hybrid job manager for database operations
from hybrid_job_manager import hybrid_job_manager

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

# =============================================
# S3 CLIENT CONNECTION POOLING
# =============================================

# Global S3 client with connection pooling (reused across requests)
_s3_client = None
_s3_bucket_name = None

# Optional fallback client/bucket for legacy storage (e.g., old US bucket)
_s3_fallback_client = None
_s3_fallback_bucket_name = None


def get_s3_client():
    """
    Get or create a reusable S3 client with optimized connection pooling.
    Reusing the client saves SSL handshake time (~100ms per connection).
    """
    global _s3_client, _s3_bucket_name

    if _s3_client is None:
        # Configure connection pooling for high concurrency
        boto_config = BotoConfig(
            max_pool_connections=200,  # Match our concurrency level
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        )

        try:
            _s3_client = boto3.client(
                "s3",
                aws_access_key_id=getattr(settings, "aws_access_key_id", None),
                aws_secret_access_key=getattr(settings, "aws_secret_access_key", None),
                region_name=getattr(settings, "aws_default_region", "us-east-1"),
                config=boto_config,
            )
        except Exception as e:
            logger.error(f"Failed to create S3 client with settings: {str(e)}")
            _s3_client = boto3.client("s3", config=boto_config)

        _s3_bucket_name = getattr(settings, "s3_bucket_name", None) or os.getenv(
            "S3_BUCKET_NAME"
        )
        logger.info(
            "S3 client initialized with connection pooling (max_pool_connections=200)"
        )

    return _s3_client, _s3_bucket_name


def get_s3_fallback_client():
    """
    Get or create fallback S3 client for legacy bucket.
    Only initialized if S3_BUCKET_NAME_FALLBACK env var is set.
    """
    global _s3_fallback_client, _s3_fallback_bucket_name

    fallback_bucket = os.getenv("S3_BUCKET_NAME_FALLBACK")
    if not fallback_bucket:
        return None, None

    if _s3_fallback_client is None:
        fallback_region = os.getenv("S3_FALLBACK_REGION", "us-east-2")

        boto_config = BotoConfig(
            max_pool_connections=200,
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        )

        try:
            _s3_fallback_client = boto3.client(
                "s3",
                aws_access_key_id=getattr(settings, "aws_access_key_id", None),
                aws_secret_access_key=getattr(settings, "aws_secret_access_key", None),
                region_name=fallback_region,
                config=boto_config,
            )
        except Exception as e:
            logger.error(f"Failed to create fallback S3 client: {str(e)}")
            _s3_fallback_client = boto3.client("s3", config=boto_config)

        _s3_fallback_bucket_name = fallback_bucket
        logger.info(
            f"S3 fallback client initialized: bucket={_s3_fallback_bucket_name}, region={fallback_region}"
        )

    return _s3_fallback_client, _s3_fallback_bucket_name


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


def _get_ydl_opts(base_opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add proxy settings to yt-dlp options if configured.
    Uses rotating residential proxy with random session suffix for IP rotation.
    """
    opts = base_opts.copy()
    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        # Construct Webshare proxy URL with URL-encoded credentials
        # For rotating residential: add random suffix (-1 to -1000) for IP rotation
        base_username = settings.webshare_proxy_username.strip()
        # session_suffix = random.randint(1, 100000)
        # we can use -rotate
        username_with_session = f"{base_username}-rotate"

        username = quote(username_with_session)
        password = quote(settings.webshare_proxy_password.strip())
        proxy_url = f"http://{username}:{password}@p.webshare.io:80"
        opts["proxy"] = proxy_url

        # Log masked proxy URL for debugging (show session number)
        masked_pass = "*" * 5
        logger.info(
            f"Using Webshare proxy: http://{username}:{masked_pass}@p.webshare.io:80"
        )
    return opts


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
# yt-dlp does not need global initialization
logger.info("YouTube Service initialized")


def get_youtube_client():
    """
    Deprecated: No longer used with yt-dlp.
    """
    return None


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
    Get metadata for multiple videos using yt-dlp.
    Note: yt-dlp processes sequentially or we can parallelize here.
    For now, we'll process them in parallel threads to mimic batch speed.

    Args:
        video_ids: List of YouTube video IDs (max 50)

    Returns:
        Dictionary mapping video_id to metadata dict
    """
    if not video_ids:
        return {}

    try:
        # Define a helper to fetch single video metadata
        def _fetch_single_metadata(vid):
            url = f"https://www.youtube.com/watch?v={vid}"
            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                "ignoreerrors": True,
                "no_warnings": True,
            }
            # Add proxy if configured
            ydl_opts = _get_ydl_opts(ydl_opts)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        return None

                    # Map to our structure
                    duration_seconds = info.get("duration", 0)
                    # Format duration to HH:MM:SS
                    hours = duration_seconds // 3600
                    minutes = (duration_seconds % 3600) // 60
                    seconds = duration_seconds % 60
                    if hours > 0:
                        duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = f"{minutes}:{seconds:02d}"

                    return {
                        "id": info.get("id"),
                        "title": info.get("title", "Untitled"),
                        "description": info.get("description", ""),
                        "channelId": info.get("channel_id", ""),
                        "channelTitle": info.get("uploader", ""),
                        "publishedAt": info.get("upload_date", ""),  # YYYYMMDD
                        "thumbnail": info.get("thumbnail", ""),
                        "duration": duration_str,
                        "viewCount": info.get("view_count", 0),
                        "likeCount": info.get("like_count", 0),
                        "commentCount": info.get("comment_count", 0),
                        "url": url,
                        "duration_iso": f"PT{duration_seconds}S",  # Approx ISO
                        "duration_seconds": duration_seconds,
                        "duration_category": _categorize_duration(duration_seconds),
                        "language": "en",  # Default assumption or extract if available
                        "defaultLanguage": "en",
                        "categoryId": 0,
                        "tags": info.get("tags", []),
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for {vid}: {e}")
                return None

        # Run fetches in parallel
        tasks = []
        for vid in video_ids:
            tasks.append(asyncio.to_thread(_fetch_single_metadata, vid))

        results_list = await asyncio.gather(*tasks)

        results = {}
        for res in results_list:
            if res:
                results[res["id"]] = res

        logger.info(
            f"Successfully fetched metadata for {len(results)}/{len(video_ids)} videos"
        )
        return results

    except Exception as e:
        logger.error(f"Error fetching metadata batch: {str(e)}")
        raise


async def get_video_info(video_id: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a YouTube video using yt-dlp.

    Args:
        video_id: The YouTube video ID

    Returns:
        Dictionary with video metadata (title, channel, views, etc.)

    Raises:
        ValueError: If video ID is invalid or video doesn't exist
    """
    try:

        def _fetch_info():
            url = f"https://www.youtube.com/watch?v={video_id}"
            ydl_opts = {
                "quiet": True,
                "skip_download": True,
                "ignoreerrors": True,
                "no_warnings": True,
            }
            # Add proxy if configured
            ydl_opts = _get_ydl_opts(ydl_opts)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info

        info = await asyncio.to_thread(_fetch_info)

        if not info:
            raise ValueError(f"No video found with ID: {video_id}")

        duration_seconds = info.get("duration", 0)
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        if hours > 0:
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_str = f"{minutes}:{seconds:02d}"

        # Build metadata dictionary
        metadata = {
            "id": video_id,
            "title": info.get("title", "Untitled"),
            "description": info.get("description", ""),
            "channelId": info.get("channel_id", ""),
            "channelTitle": info.get("uploader", ""),
            "publishedAt": info.get("upload_date", ""),
            "thumbnail": info.get("thumbnail", ""),
            "duration": duration_str,
            "viewCount": info.get("view_count", 0),
            "likeCount": info.get("like_count", 0),
            "commentCount": info.get("comment_count", 0),
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


def _get_best_thumbnail(info: Dict[str, Any]) -> str:
    """Extract the best available thumbnail from yt-dlp info dict."""
    if info.get("thumbnail"):
        return info["thumbnail"]

    thumbnails = info.get("thumbnails", [])
    if not thumbnails:
        return ""

    # Try to find avatar_uncropped (highest quality avatar)
    for t in thumbnails:
        if t.get("id") == "avatar_uncropped":
            return t.get("url", "")

    # Fallback to the last one (usually highest res)
    return thumbnails[-1].get("url", "")


async def get_channel_info(channel_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a YouTube channel using yt-dlp.

    Args:
        channel_name: Channel name, handle, or channel ID

    Returns:
        Dictionary with channel information (title, description, stats, etc.)

    Raises:
        ValueError: If channel is not found
    """
    try:
        # Construct base URL
        if not channel_name.startswith("http"):
            # Check if it looks like a channel ID (UC...)
            if re.match(r"^UC[\w-]{22}$", channel_name):
                base_url = f"https://www.youtube.com/channel/{channel_name}"
            else:
                # Assume it's a handle if it's not a channel ID
                handle = (
                    channel_name if channel_name.startswith("@") else f"@{channel_name}"
                )
                base_url = f"https://www.youtube.com/{handle}"
        else:
            base_url = channel_name.rstrip("/")
            # Strip any existing tab suffix
            for tab in ["/videos", "/shorts", "/playlists", "/live"]:
                if base_url.endswith(tab):
                    base_url = base_url[: -len(tab)]
                    break

        # Try /videos first, then /shorts if that fails
        tabs_to_try = ["/videos", "/shorts"]
        info = None
        last_error = None

        for tab in tabs_to_try:
            url = base_url + tab
            try:

                def _fetch_channel():
                    ydl_opts = {
                        "quiet": True,
                        "extract_flat": True,
                        "dump_single_json": True,
                        "playlist_items": "0",
                    }
                    # Add proxy if configured
                    ydl_opts = _get_ydl_opts(ydl_opts)
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        return ydl.extract_info(url, download=False)

                info = await asyncio.to_thread(_fetch_channel)

                if info:
                    logger.info(
                        f"Successfully fetched channel info from {tab} tab for {channel_name}"
                    )
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch channel info from {tab} tab: {str(e)}")
                last_error = e
                continue

        if not info:
            error_msg = f"Channel not found: {channel_name}"
            if last_error:
                error_msg += f" (last error: {str(last_error)})"
            raise ValueError(error_msg)

        channel_id = info.get("channel_id") or info.get("id")

        logger.info(
            f"Successfully resolved '{channel_name}' to channel ID '{channel_id}' (title: '{info.get('title')}')"
        )

        # Extract best thumbnail
        thumbnail_url = _get_best_thumbnail(info)

        # Extract subscriber count (yt-dlp uses channel_follower_count for subs sometimes)
        subscriber_count = info.get("subscriber_count") or info.get(
            "channel_follower_count", 0
        )

        return {
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "thumbnail": thumbnail_url,
            "videoCount": info.get("playlist_count", 0),
            "subscriberCount": subscriber_count,
            "viewCount": info.get("view_count", 0),
            "channelId": channel_id,
            "uploadsPlaylistId": (
                f"UU{channel_id[2:]}" if channel_id and len(channel_id) > 2 else None
            ),
        }

    except Exception as e:
        logger.error(f"Error fetching channel info for {channel_name}: {str(e)}")
        raise ValueError(f"Failed to get channel information: {str(e)}")


async def get_playlist_info(playlist_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a YouTube playlist using yt-dlp.

    Args:
        playlist_id: YouTube playlist ID

    Returns:
        Dictionary with playlist information (title, description, video count, etc.)

    Raises:
        ValueError: If playlist is not found
    """
    try:
        url = f"https://www.youtube.com/playlist?list={playlist_id}"

        def _fetch_playlist():
            ydl_opts = {
                "quiet": True,
                "extract_flat": True,
                "dump_single_json": True,
                "playlist_items": "0",  # Don't fetch videos yet
            }
            # Add proxy if configured
            ydl_opts = _get_ydl_opts(ydl_opts)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        info = await asyncio.to_thread(_fetch_playlist)

        if not info:
            raise ValueError(f"Playlist not found: {playlist_id}")

        logger.info(
            f"Successfully fetched playlist '{playlist_id}' (title: '{info.get('title')}')"
        )

        return {
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "thumbnail": info.get(
                "thumbnail", ""
            ),  # Might be empty for playlists in flat mode
            "videoCount": info.get("playlist_count", 0),  # yt-dlp usually provides this
            "channelTitle": info.get("uploader", ""),
            "channelId": info.get("channel_id", ""),
            "playlistId": playlist_id,
            "publishedAt": "",  # yt-dlp might not give this for playlist itself easily
        }

    except Exception as e:
        logger.error(f"Error fetching playlist info for {playlist_id}: {str(e)}")
        raise ValueError(f"Failed to get playlist information: {str(e)}")


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
            transcript_list = await asyncio.to_thread(ytt_api.list, video_id)
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
                logger.info(
                    f"Using pre-fetched metadata for video {video_id} (title: {pre_fetched_metadata.get('title', 'No title')})"
                )
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
        del fetched_transcript
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


def _format_ytdlp_date(date_str: Optional[str], timestamp: Optional[int] = None) -> str:
    """Format YYYYMMDD date string to YYYY-MM-DD.
    Falls back to Unix timestamp if date_str is missing (common with extract_flat mode).
    """
    if date_str and len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    if timestamp:
        try:
            from datetime import datetime as _dt, timezone as _tz

            return _dt.fromtimestamp(timestamp, tz=_tz.utc).strftime("%Y-%m-%d")
        except (OSError, ValueError, OverflowError):
            pass
    return ""


def _fetch_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all videos from a channel using yt-dlp, including Shorts.
    Returns a list of video metadata dicts.
    """
    logger.info(f"Fetching all videos for channel {channel_id} using yt-dlp")

    all_videos_map = {}  # Use dict for deduplication by ID

    # Fetch from both 'videos' (long form) and 'shorts' tabs
    tabs = ["videos", "shorts"]

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "dump_single_json": True,
        "ignoreerrors": True,
    }
    # Add proxy if configured
    ydl_opts = _get_ydl_opts(ydl_opts)

    for tab in tabs:
        try:
            url = f"https://www.youtube.com/channel/{channel_id}/{tab}"
            logger.info(f"Fetching {tab} from {url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if not info:
                logger.warning(f"No info found for {tab} tab of channel {channel_id}")
                continue

            entries = info.get("entries", [])
            logger.info(f"Found {len(entries)} {tab} for channel {channel_id}")

            for entry in entries:
                if not entry:
                    continue

                video_id = entry.get("id")
                if not video_id:
                    continue

                title = entry.get("title", "Untitled")

                # Skip private/deleted
                if title == "[Private video]" or title == "[Deleted video]":
                    continue

                duration_seconds = entry.get("duration")
                if duration_seconds:
                    duration_category = _categorize_duration(duration_seconds)
                else:
                    duration_category = "unknown"

                # Format date (upload_date may be None in extract_flat mode, fall back to timestamp)
                upload_date = entry.get("upload_date")
                timestamp = entry.get("timestamp")
                published_at = _format_ytdlp_date(upload_date, timestamp)

                all_videos_map[video_id] = {
                    "id": video_id,
                    "title": title,
                    "publishedAt": published_at,
                    "url": entry.get("url")
                    or f"https://www.youtube.com/watch?v={video_id}",
                    "duration": duration_category,
                    "duration_seconds": duration_seconds,
                    "viewCount": entry.get("view_count") or 0,
                    "type": "short" if tab == "shorts" else "video",
                }

        except Exception as e:
            logger.error(f"Error fetching {tab} for channel {channel_id}: {str(e)}")
            # Continue to next tab even if one fails

    all_videos = list(all_videos_map.values())

    # Log duration distribution
    duration_counts = {}
    for video in all_videos:
        category = video["duration"]
        duration_counts[category] = duration_counts.get(category, 0) + 1

    logger.info(
        f"Total unique videos found: {len(all_videos)}. Duration distribution: {duration_counts}"
    )

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
    Fetch all videos from a playlist using yt-dlp.
    Returns a list of video metadata dicts.
    """
    logger.info(f"Fetching all videos from playlist {playlist_id} using yt-dlp")

    url = f"https://www.youtube.com/playlist?list={playlist_id}"

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "dump_single_json": True,
        "ignoreerrors": True,
    }
    # Add proxy if configured
    ydl_opts = _get_ydl_opts(ydl_opts)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        raise ValueError(f"Playlist {playlist_id} not found")

    all_videos = []
    entries = info.get("entries", [])

    logger.info(f"Found {len(entries)} videos in playlist {playlist_id}")

    for entry in entries:
        if not entry:
            continue

        video_id = entry.get("id")
        title = entry.get("title", "Untitled")

        if title == "[Private video]" or title == "[Deleted video]":
            continue

        duration_seconds = entry.get("duration")
        if duration_seconds:
            duration_category = _categorize_duration(duration_seconds)
        else:
            duration_category = "unknown"

        # Format date (upload_date may be None in extract_flat mode, fall back to timestamp)
        upload_date = entry.get("upload_date")
        timestamp = entry.get("timestamp")
        published_at = _format_ytdlp_date(upload_date, timestamp)

        all_videos.append(
            {
                "id": video_id,
                "title": title,
                "publishedAt": published_at,
                "url": entry.get("url")
                or f"https://www.youtube.com/watch?v={video_id}",
                "duration": duration_category,
                "duration_seconds": duration_seconds,
                "viewCount": entry.get("view_count") or 0,
            }
        )

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
    logger.info(f"Job {job_id}: Formatting options being sent: {formatting_options}")

    async def dispatch_single_lambda(video):
        """Dispatch a single Lambda function asynchronously"""
        try:
            video_id = video.id if hasattr(video, "id") else video["id"]
            pre_fetched_metadata = videos_metadata.get(video_id, {})

            lambda_payload = {
                "video_id": video_id,
                "job_id": str(job_id),  # Convert UUID to string for JSON serialization
                "user_id": str(
                    user_id
                ),  # Convert UUID to string for JSON serialization
                "pre_fetched_metadata": pre_fetched_metadata,
                "api_base_url": settings.api_base_url,
                # Flatten formatting options to top-level keys for Lambda compatibility
                **formatting_options,
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
            # Note: Failed dispatch count is not tracked as it happens before video processing
            # Only video processing failures are tracked in the database
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

    # Explicitly close the Lambda client to release connection pool
    try:
        lambda_client.close()
    except Exception as e:
        logger.warning(f"Error closing Lambda client: {e}")

    return dispatch_count


async def prefetch_and_dispatch_task(job_id: str):
    """
    Background task that:
    1. Pre-fetches metadata in batches
    2. Updates job status
    3. Dispatches Lambda functions
    4. Updates job to 'processing'
    Now uses database storage via hybrid manager.
    """
    try:
        # Load job data from database
        job = await hybrid_job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for pre-fetching")
            return

        videos = job["videos"]
        user_id = job["user_id"]

        logger.info(
            f"Job {job_id}: Starting background pre-fetch for {len(videos)} videos"
        )

        # 1. Check if we can skip pre-fetch (if metadata is already present)
        video_ids = []
        videos_metadata = {}
        skip_prefetch = False

        # Check first video to see if it has title (and potentially viewCount)
        if videos and len(videos) > 0:
            first_video = videos[0]
            # Handle both dict and object (Pydantic)
            v_title = (
                first_video.get("title")
                if isinstance(first_video, dict)
                else getattr(first_video, "title", None)
            )
            if v_title:
                skip_prefetch = True
                logger.info(
                    f"Job {job_id}: Metadata already present in request, skipping slow pre-fetch"
                )

        if skip_prefetch:
            # Construct metadata from existing video info
            for video in videos:
                is_dict = isinstance(video, dict)
                v_id = video.get("id") if is_dict else video.id
                video_ids.append(v_id)

                # Map fields to what Lambda expects (camelCase format)
                # Lambda expects: title, viewCount (camelCase!), id, webpage_url
                # Handle both camelCase (from API) and snake_case (from DB) for view count
                view_count = 0
                if is_dict:
                    view_count = video.get("viewCount") or video.get("view_count") or 0
                else:
                    view_count = getattr(video, "viewCount", 0) or getattr(
                        video, "view_count", 0
                    )

                videos_metadata[v_id] = {
                    "id": v_id,
                    "title": video.get("title") if is_dict else video.title,
                    "viewCount": view_count,  # Lambda expects camelCase
                    "webpage_url": video.get("url") if is_dict else video.url,
                    "duration": (
                        video.get("duration")
                        if is_dict
                        else getattr(video, "duration", None)
                    ),
                    "upload_date": (
                        video.get("publishedAt")
                        if is_dict
                        else getattr(video, "publishedAt", None)
                    ),
                }
        else:
            # Fallback to original pre-fetch logic
            for video in videos:
                try:
                    video_id = video.get("id") if isinstance(video, dict) else video.id
                    video_ids.append(video_id)
                except Exception as e:
                    logger.error(f"Error extracting video ID: {str(e)}")

            await hybrid_job_manager.update_job(job_id, status="prefetching_metadata")
            videos_metadata = await pre_fetch_videos_metadata(video_ids)

        # 2. Update job with metadata
        await hybrid_job_manager.update_job(
            job_id,
            videos_metadata=videos_metadata,
            prefetch_completed=True,
            status="dispatching",
        )

        # 3. Update video metadata in job_videos table
        await hybrid_job_manager.update_videos_metadata(job_id, videos_metadata)

        logger.info(
            f"Job {job_id}: Metadata pre-fetch completed, dispatching Lambda functions"
        )

        # 3. Dispatch Lambda functions concurrently with pre-fetched metadata
        dispatched_count = await dispatch_lambdas_concurrently(
            job_id, videos, videos_metadata, user_id, job["formatting_options"]
        )

        # 4. Update job status to processing - use safe update to prevent race conditions
        success = await hybrid_job_manager.update_job_status_safe(
            job_id, "processing", expected_current_status="dispatching"
        )

        if not success:
            logger.warning(
                f"Job {job_id}: Status was changed during dispatch, but continuing"
            )

        # Update additional metadata
        await hybrid_job_manager.update_job(
            job_id,
            lambda_dispatched_count=dispatched_count,
            lambda_dispatch_time=time.time(),
        )

        logger.info(
            f"Job {job_id}: Background task completed. Dispatched {dispatched_count} Lambda functions"
        )

        # 5. Start timeout monitoring task
        asyncio.create_task(monitor_job_timeout(job_id, settings.job_timeout_minutes))

    except Exception as e:
        logger.error(f"Job {job_id}: Background pre-fetch task failed: {str(e)}")
        # Update job status to failed
        await hybrid_job_manager.update_job(
            job_id, status="failed", error_message=str(e)
        )

        # Refund credits since processing failed
        try:
            from transcript_api import CreditManager

            await CreditManager.finalize_credit_usage(
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
        job = await hybrid_job_manager.get_job_status(job_id)
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
            await hybrid_job_manager.update_job(
                job_id,
                status="completed",
                timeout_occurred=True,
                timeout_failed_count=pending_count,
            )

            # Finalize credits for completed job
            try:
                from transcript_api import CreditManager

                updated_job = await hybrid_job_manager.get_job_status(job_id)
                if updated_job and updated_job.get("reservation_id"):
                    await CreditManager.finalize_credit_usage(
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
    Now uses hybrid manager for database/file fallback.

    Args:
        job_id: The job identifier

    Returns:
        Dictionary with job status information including credit usage

    Raises:
        ValueError: If job ID is not found
    """
    # This function needs to be async to use hybrid manager, but keeping sync for compatibility
    # We'll create a wrapper function for the API endpoints
    import asyncio

    try:
        # For backwards compatibility, try to run async operation
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to handle this differently
            # For now, fall back to the file-based approach
            job = load_job_from_file(job_id)
        else:
            job = loop.run_until_complete(
                hybrid_job_manager.get_job(job_id, include_videos=False)
            )
    except Exception as e:
        logger.warning(f"Failed to get job {job_id} from hybrid manager: {e}")
        # Fallback to file-based approach
        job = load_job_from_file(job_id)

    if not job:
        logger.error(f"Job {job_id} not found")
        raise ValueError(f"Job {job_id} not found")

    # Calculate progress percentage
    total_videos = job.get("total_videos", 0)
    completed = job.get("completed", 0)
    failed_count = job.get("failed_count", 0)
    processed_count = job.get("processed_count", 0)

    progress_percentage = 0.0
    if total_videos > 0:
        progress_percentage = (processed_count / total_videos) * 100

    # Calculate timing information
    start_time = job.get("start_time")
    end_time = job.get("end_time")
    duration = None

    if isinstance(start_time, (int, float)):
        if end_time and isinstance(end_time, (int, float)):
            duration = end_time - start_time
        else:
            # Job still running, calculate current duration
            duration = time.time() - start_time

    # Build status response
    status_response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "total_videos": total_videos,
        "completed": completed,
        "failed_count": failed_count,
        "processed_count": processed_count,
        "progress_percentage": round(progress_percentage, 2),
        "files": job.get("files", []),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        # Credit tracking
        "credits_reserved": job.get("credits_reserved", 0),
        "credits_used": job.get("credits_used", 0),
        "credits_remaining": job.get("credits_reserved", 0)
        - job.get("credits_used", 0),
        "reservation_id": job.get("reservation_id"),
        # Source information
        "source_type": job.get("source_type", "unknown"),
        "source_name": job.get("source_name", "Unknown Source"),
        "channel_name": job.get("channel_name"),
        "playlist_id": job.get("playlist_id"),
        # Lambda processing
        "lambda_dispatched_count": job.get("lambda_dispatched_count", 0),
        "lambda_dispatch_time": job.get("lambda_dispatch_time"),
        "prefetch_completed": job.get("prefetch_completed", False),
        # Error information
        "error_message": job.get("error_message"),
        "timeout_occurred": job.get("timeout_occurred", False),
    }

    # Add completion details for completed jobs
    if job.get("status") == "completed":
        status_response.update(
            {
                "message": f"Download completed. {completed} videos successful, {failed_count} failed.",
                "download_ready": len(job.get("files", [])) > 0,
            }
        )
    elif job.get("status") == "processing":
        status_response.update(
            {
                "message": f"Processing in progress. {completed}/{total_videos} videos completed.",
                "estimated_time_remaining": None,  # Could add estimation logic here
            }
        )
    elif job.get("status") == "failed":
        status_response.update(
            {
                "message": f"Job failed: {job.get('error_message', 'Unknown error')}",
            }
        )

    return status_response


async def get_job_status_async(job_id: str) -> Dict[str, Any]:
    """
    Async version of get_job_status for use in async contexts.
    Optimized for frequent polling - fetches only essential status fields.
    """
    # Use get_job_status() instead of get_job() to avoid fetching large JSONB fields
    job = await hybrid_job_manager.get_job_status(job_id)

    if not job:
        logger.error(f"Job {job_id} not found")
        raise ValueError(f"Job {job_id} not found")

    # Calculate progress percentage
    total_videos = job.get("total_videos", 0)
    failed_count = job.get("failed_count", 0)
    processed_count = job.get("processed_count", 0)

    progress_percentage = 0.0
    if total_videos > 0:
        progress_percentage = (processed_count / total_videos) * 100

    # Calculate timing information (if lambda_dispatch_time is available)
    lambda_dispatch_time = job.get("lambda_dispatch_time")
    duration = None
    if lambda_dispatch_time:
        if isinstance(lambda_dispatch_time, (int, float)):
            duration = time.time() - lambda_dispatch_time
        else:
            # It's a datetime object from database
            from datetime import datetime

            duration = (datetime.now() - lambda_dispatch_time).total_seconds()

    # Build minimal status response (no large fields)
    status_response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "total_videos": total_videos,
        "failed_count": failed_count,
        "processed_count": processed_count,
        "progress_percentage": round(progress_percentage, 2),
        "duration": duration,
        # Credit tracking
        "credits_reserved": job.get("credits_reserved", 0),
        "credits_used": job.get("credits_used", 0),
        "credits_remaining": job.get("credits_reserved", 0)
        - job.get("credits_used", 0),
        "timeout_occurred": job.get("timeout_occurred", False),
    }

    # Add simple message based on progress
    completed = processed_count - failed_count
    if processed_count >= total_videos:
        status_response["message"] = (
            f"Download completed. {completed} videos successful, {failed_count} failed."
        )
        status_response["download_ready"] = True
        status_response["completed"] = completed
    else:
        status_response["message"] = (
            f"Processing in progress. {processed_count}/{total_videos} videos processed."
        )
        status_response["download_ready"] = False
        status_response["completed"] = completed
    return status_response


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
    # Load job data from database
    job = await hybrid_job_manager.get_job(job_id, include_videos=False)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

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
    # Load job data from database
    job = await hybrid_job_manager.get_job(job_id, include_videos=False)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

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


async def get_safe_channel_name(job_id: str) -> str:
    """
    Get a filesystem-safe version of the source name (channel or playlist) for a job.

    Args:
        job_id: The job identifier

    Returns:
        Safe source name for use in filenames

    Raises:
        ValueError: If job ID not found
    """
    # Load job data from database
    job_data = await hybrid_job_manager.get_job(job_id, include_videos=False)
    if not job_data:
        raise ValueError(f"Job not found with ID: {job_id}")

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
    # Load job data from database
    job = await hybrid_job_manager.get_job(job_id, include_videos=True)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

    if job["status"] not in ["completed", "completed_with_errors"]:
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job.get("files"):
        raise ValueError("No transcript files found for this job")

    # Use pooled S3 client for better performance
    s3_client, bucket_name = get_s3_client()
    if not bucket_name:
        raise ValueError("S3 bucket name not configured")

    # Get fallback client (if configured)
    s3_fallback_client, fallback_bucket_name = get_s3_fallback_client()

    # Determine which bucket to use by checking the first file only
    use_fallback = False
    if s3_fallback_client and fallback_bucket_name and job["files"]:
        first_file = job["files"][0]
        first_key = first_file["s3_key"]

        def _check_first_file():
            try:
                s3_client.head_object(Bucket=bucket_name, Key=first_key)
                return False  # Primary bucket has the file
            except ClientError as e:
                error_code = (
                    e.response.get("Error", {}).get("Code")
                    if hasattr(e, "response")
                    else None
                )
                if error_code in {"NoSuchKey", "404", "NotFound"}:
                    # Check fallback bucket
                    try:
                        s3_fallback_client.head_object(
                            Bucket=fallback_bucket_name, Key=first_key
                        )
                        logger.info(
                            "First file found in fallback bucket; using fallback for all files"
                        )
                        return True  # Use fallback bucket
                    except Exception:
                        pass
                return False  # Stick with primary

        use_fallback = await asyncio.to_thread(_check_first_file)

    # Select the bucket to use for all downloads
    active_client = s3_fallback_client if use_fallback else s3_client
    active_bucket = fallback_bucket_name if use_fallback else bucket_name

    # Download all files concurrently from the determined bucket
    async def download_file(file_info):
        """Download a single file from S3"""
        s3_key = file_info["s3_key"]
        video_id = file_info["video_id"]

        def _download():
            try:
                response = active_client.get_object(Bucket=active_bucket, Key=s3_key)
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

    # Download files concurrently (max 200 concurrent downloads)
    semaphore = asyncio.Semaphore(200)

    async def download_with_limit(file_info):
        async with semaphore:
            return await download_file(file_info)

    logger.info(
        f"Downloading {len(job['files'])} files concurrently from S3 (max 200 concurrent, pooled client)"
    )
    download_start = time.time()

    download_tasks = [download_with_limit(file_info) for file_info in job["files"]]
    download_results = await asyncio.gather(*download_tasks)

    download_end = time.time()
    successful_downloads = len([r for r in download_results if r["success"]])
    logger.info(
        f"Downloaded {successful_downloads}/{len(download_results)} files in {download_end - download_start:.2f}s"
    )

    # Create ZIP from downloaded content
    zip_start = time.time()
    zip_buffer = io.BytesIO()

    try:
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

                # Create ZipInfo with UTF-8 flag to support non-ASCII characters
                zip_info = zipfile.ZipInfo(filename)
                zip_info.flag_bits |= 0x800  # Set UTF-8 flag (bit 11)

                # Ensure content is bytes (encode if string with UTF-8 to support Russian/Unicode)
                content = concatenated_content
                if isinstance(content, str):
                    content = content.encode("utf-8")

                zip_file.writestr(zip_info, content)

        else:
            # Create individual files in ZIP
            with zipfile.ZipFile(
                zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
            ) as zip_file:
                for result in download_results:
                    # Create ZipInfo with UTF-8 flag to support non-ASCII characters (e.g., Russian)
                    zip_info = zipfile.ZipInfo(result["filename"])
                    zip_info.flag_bits |= 0x800  # Set UTF-8 flag (bit 11)

                    # Ensure content is bytes (encode if string with UTF-8 to support Russian/Unicode)
                    content = result["content"]
                    if isinstance(content, str):
                        content = content.encode("utf-8")

                    zip_file.writestr(zip_info, content)

        zip_buffer.seek(0)
        zip_end = time.time()
        zip_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)

        logger.info(
            f"Created ZIP archive for job {job_id} with {successful_downloads} files in {zip_end - zip_start:.2f}s, "
            f"size: {zip_size_mb:.2f} MB (download: {download_end - download_start:.2f}s, zip: {zip_end - zip_start:.2f}s)"
        )

        return zip_buffer

    finally:
        # Explicitly clean up download_results to free memory
        del download_results
        # Note: S3 client is pooled and reused, don't close it


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
    # Load job data from database
    job = await hybrid_job_manager.get_job(job_id, include_videos=False)
    if not job:
        raise ValueError(f"Job not found with ID: {job_id}")

    if job["status"] not in ["completed", "completed_with_errors"]:
        raise ValueError(
            f"Cannot create ZIP: job status is {job['status']}, not completed"
        )

    if not job.get("files"):
        raise ValueError("No transcript files found for this job")

    # Use pooled S3 clients
    s3_client, bucket_name = get_s3_client()
    if not bucket_name:
        raise ValueError("S3 bucket name not configured")

    # Get fallback client (if configured)
    s3_fallback_client, fallback_bucket_name = get_s3_fallback_client()

    # Determine which bucket to use by checking the first file only
    use_fallback = False
    if s3_fallback_client and fallback_bucket_name and job["files"]:
        first_file = job["files"][0]
        first_key = first_file["s3_key"]

        def _check_first_file():
            try:
                s3_client.head_object(Bucket=bucket_name, Key=first_key)
                return False  # Primary bucket has the file
            except ClientError as e:
                error_code = (
                    e.response.get("Error", {}).get("Code")
                    if hasattr(e, "response")
                    else None
                )
                if error_code in {"NoSuchKey", "404", "NotFound"}:
                    # Check fallback bucket
                    try:
                        s3_fallback_client.head_object(
                            Bucket=fallback_bucket_name, Key=first_key
                        )
                        logger.info(
                            "First file found in fallback bucket; using fallback for all files"
                        )
                        return True  # Use fallback bucket
                    except Exception:
                        pass
                return False  # Stick with primary

        use_fallback = await asyncio.to_thread(_check_first_file)

    # Select the bucket to use for all downloads
    active_client = s3_fallback_client if use_fallback else s3_client
    active_bucket = fallback_bucket_name if use_fallback else bucket_name

    if not active_bucket:
        raise ValueError("S3 bucket name not configured")

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = io.BytesIO()

    # Check if we should concatenate all transcripts into a single file
    if job.get("formatting_options", {}).get("concatenate_all", False):
        # Create a single concatenated file from S3 files
        concatenated_content = await create_concatenated_transcript_from_s3_sequential(
            job_id, active_client, active_bucket
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

            # Create ZipInfo with UTF-8 flag to support non-ASCII characters
            zip_info = zipfile.ZipInfo(filename)
            zip_info.flag_bits |= 0x800  # Set UTF-8 flag (bit 11)

            # Ensure content is bytes (encode if string with UTF-8 to support Russian/Unicode)
            content = concatenated_content
            if isinstance(content, str):
                content = content.encode("utf-8")

            zip_file.writestr(zip_info, content)
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
                    response = active_client.get_object(
                        Bucket=active_bucket, Key=s3_key
                    )
                    file_content = response["Body"].read()

                    # Create filename from video ID or extract from S3 key
                    filename = f"{video_id}.txt"
                    if "/" in s3_key:
                        filename = os.path.basename(s3_key)

                    # Ensure content is bytes (encode if string with UTF-8 to support Russian/Unicode)
                    if isinstance(file_content, str):
                        file_content = file_content.encode("utf-8")

                    # Add to ZIP with UTF-8 support
                    zip_info = zipfile.ZipInfo(filename)
                    zip_info.flag_bits |= 0x800  # Set UTF-8 flag (bit 11)
                    zip_file.writestr(zip_info, file_content)
                    logger.debug(f"Added {filename} to ZIP archive")

                except Exception as e:
                    logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
                    # Add error placeholder instead of failing entire ZIP
                    error_content = (
                        f"Error downloading transcript for video {video_id}: {str(e)}"
                    )
                    error_filename = f"{video_id}_ERROR.txt"
                    zip_info = zipfile.ZipInfo(error_filename)
                    zip_info.flag_bits |= 0x800  # Set UTF-8 flag (bit 11)
                    zip_file.writestr(zip_info, error_content.encode("utf-8"))

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
    # Load job data from database
    job = await hybrid_job_manager.get_job(job_id, include_videos=False)
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
