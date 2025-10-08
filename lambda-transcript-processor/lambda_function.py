#!/usr/bin/env python3
"""
AWS Lambda function for processing YouTube video transcripts
"""
import boto3
import logging
import os
import re
import requests
import time
from typing import Dict, Any

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


# Configure logging
logger = logging.getLogger(__name__)


def get_ytt_api() -> YouTubeTranscriptApi:
    """
    Create a new YouTubeTranscriptApi instance with proxy config if needed.
    Thread-safe implementation that creates fresh instances.
    Returns:
        YouTubeTranscriptApi instance
    """
    try:
        proxy_config = WebshareProxyConfig(
            proxy_username=os.getenv("WEBSHARE_PROXY_USERNAME"),
            proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD"),
            retries_when_blocked=1,
        )
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    except Exception as e:
        logger.error(f"Error creating YouTubeTranscriptApi: {str(e)}")
        # Fallback to basic instance
        return YouTubeTranscriptApi()


def get_youtube_client():
    """
    Create a thread-safe YouTube API client for each worker.
    This prevents thread safety issues with the global client.
    """
    try:
        return build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to create YouTube API client: {str(e)}")
        return None  # Fallback to global client


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


def retry_operation(operation, max_retries=3, retry_delay=1.0, *args, **kwargs):
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
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            current_retry += 1

            if current_retry <= max_retries:
                # Log the exception but continue with retry
                logger.warning(
                    f"Operation failed (attempt {current_retry}/{max_retries}): {str(e)}. "
                    f"Retrying in {retry_delay:.1f} seconds..."
                )
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
            else:
                # Log the final failure
                logger.error(f"Operation failed after {max_retries} retries: {str(e)}")
                raise last_exception

    # This should never be reached due to the raise in the else clause
    raise last_exception


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


def get_video_info(video_id: str) -> Dict[str, Any]:
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
        video_response = _fetch_video_info()

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


def call_api_callback(callback_url: str, data: Dict[str, Any], max_retries: int = 3) -> bool:
    """
    Call the FastAPI callback endpoint to report video processing status.
    
    Args:
        callback_url: The full URL to call
        data: The data to send in the request body
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                callback_url,
                json=data,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully called callback: {callback_url}")
                return True
            else:
                logger.warning(f"Callback failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed to call callback {callback_url}: {str(e)}")
            
        # Exponential backoff for retries
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    logger.error(f"Failed to call callback after {max_retries} attempts: {callback_url}")
    return False


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing YouTube video transcripts.
    
    Expected event structure:
    {
        "video_id": "youtube_video_id",
        "job_id": "unique_job_identifier", 
        "user_id": "user_identifier",
        "api_base_url": "https://your-api.com",
        "include_timestamps": true/false,
        "include_video_title": true/false,
        "include_video_id": true/false,
        "include_video_url": true/false,
        "include_view_count": true/false,
        "pre_fetched_metadata": {...} // optional
    }
    
    Returns:
        Success/failure status and calls appropriate callback endpoint
    """
    video_id = event["video_id"]
    job_id = event["job_id"]
    user_id = event["user_id"]
    api_base_url = event.get("api_base_url", "").rstrip("/")
    
    logger.info(f"Starting transcript fetch for video {video_id} in job {job_id}")

    try:
        # Create fresh API instance for thread safety
        ytt_api = get_ytt_api()
        fetch_start = time.time()

        # 1. List available transcripts with better error handling
        try:
            transcript_list = ytt_api.list(video_id)
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
            fetched_transcript = retry_operation(
                lambda: transcript.fetch(),
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

        # Format transcript - optimize by using string builder approach
        format_start = time.time()

        if event.get("include_timestamps", False):
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

        # Get metadata for file headers
        video_title = None
        view_count = None

        if event.get("pre_fetched_metadata"):
            logger.info(
                f"Using pre-fetched metadata for video {video_id} (title: {event['pre_fetched_metadata'].get('title', 'No title')})"
            )
            # Use pre-fetched metadata (much faster)
            video_title = event["pre_fetched_metadata"].get("title", "Untitled_Video")
            view_count = event["pre_fetched_metadata"].get("viewCount", 0)
            logger.debug(
                f"Using pre-fetched metadata for video {video_id}: {video_title}"
            )
        else:
            logger.info("Fetching video metadata - no prefetch")
            # Fallback to fetching metadata (slower, with timeout)
            try:
                metadata_start = time.time()
                video_info = get_video_info(video_id)
                video_title = video_info.get("title", "Untitled_Video")
                view_count = video_info.get("viewCount", 0)
                metadata_end = time.time()
                logger.info(
                    f"Video metadata fetch took {metadata_end - metadata_start:.3f}s for video {video_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get metadata for video {video_id}, using fallback title: {str(e)}"
                )
                video_title = "Untitled_Video"
                view_count = 0

        # Create file content with headers
        file_content = ""

        # Add headers based on formatting options
        if event.get("include_video_title", True):
            file_content += f"Video Title: {video_title}\n"

        if event.get("include_video_id", True):
            file_content += f"Video ID: {video_id}\n"

        if event.get("include_video_url", True):
            file_content += f"URL: https://www.youtube.com/watch?v={video_id}\n"

        if event.get("include_view_count", False):
            file_content += f"View Count: {view_count:,}\n"

        # Add separator if any header was written
        if file_content:
            file_content += "\n"

        file_content += transcript_text

        # Store result in S3
        s3_client = boto3.client("s3")
        bucket_name = os.getenv("S3_BUCKET_NAME")

        # Create S3 key: user_id/job_id/video_id.txt
        s3_key = f"{user_id}/{job_id}/{video_id}.txt"

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content.encode("utf-8"),
            ContentType="text/plain",
            Metadata={
                "video_id": video_id,
                "job_id": job_id,
                "user_id": user_id,
                "transcript_language": metadata.get("transcript_language", "unknown"),
                "transcript_type": metadata.get("transcript_type", "unknown"),
            },
        )

        logger.info(f"Successfully uploaded transcript to S3: {s3_key}")

        # Call success callback
        if api_base_url:
            callback_url = f"{api_base_url}/internal/job/{job_id}/video-complete"
            completion_data = {
                "video_id": video_id,
                "s3_key": s3_key,
                "transcript_length": len(transcript_text),
                "metadata": metadata,
            }
            
            callback_success = call_api_callback(callback_url, completion_data)
            if not callback_success:
                logger.warning(f"Failed to call success callback for video {video_id}")
        else:
            logger.warning("No api_base_url provided, skipping callback")

        # Return success response
        return {
            "statusCode": 200,
            "body": {
                "video_id": video_id,
                "status": "completed",
                "transcript_length": len(transcript_text),
                "s3_key": s3_key,
                "metadata": metadata,
                "callback_called": api_base_url is not None,
            },
        }

    except Exception as e:
        error_message = f"Failed to get transcript for video {video_id}: {str(e)}"
        logger.error(error_message)
        
        # Call failure callback
        if api_base_url:
            callback_url = f"{api_base_url}/internal/job/{job_id}/video-failed"
            failure_data = {
                "video_id": video_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            
            callback_success = call_api_callback(callback_url, failure_data)
            if not callback_success:
                logger.warning(f"Failed to call failure callback for video {video_id}")
        else:
            logger.warning("No api_base_url provided, skipping failure callback")

        # Return error response
        return {
            "statusCode": 500,
            "body": {
                "video_id": video_id,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "callback_called": api_base_url is not None,
            },
        }
