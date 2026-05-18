#!/usr/bin/env python3
"""
AWS Lambda function for processing YouTube video transcripts
"""

import boto3
import logging
import os
import random
import re
import requests
import time
import json
from typing import Dict, Any, List, Optional, Tuple

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


# Configure logging
logger = logging.getLogger(__name__)

RETRIABLE_TRANSCRIPT_ERROR_TYPES = {
    "ChunkedEncodingError",
    "ConnectionError",
    "ConnectTimeout",
    "HTTPError",
    "IncompleteRead",
    "IpBlocked",
    "ProtocolError",
    "ProxyError",
    "ReadTimeout",
    "RequestBlocked",
    "SSLError",
    "Timeout",
    "TooManyRedirects",
    "YouTubeRequestFailed",
}

TERMINAL_TRANSCRIPT_ERROR_TYPES = {
    "AgeRestricted",
    "InvalidVideoId",
    "NoTranscriptFound",
    "PoTokenRequired",
    "TranscriptRetrievalError",
    "TranscriptsDisabled",
    "TranslationLanguageNotAvailable",
    "VideoUnavailable",
    "VideoUnplayable",
}


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


class TranscriptRetrievalError(Exception):
    """Wrap transcript failures with stage and retry metadata."""

    def __init__(
        self,
        video_id: str,
        stage: str,
        attempts: int,
        retriable: bool,
        original_exception: Exception,
    ):
        self.video_id = video_id
        self.stage = stage
        self.attempts = attempts
        self.retriable = retriable
        self.original_exception = original_exception
        self.error_type = type(original_exception).__name__
        message = (
            f"Transcript retrieval failed for video {video_id} at stage {stage} "
            f"after {attempts} attempt(s) "
            f"(retriable={retriable}, error_type={self.error_type}): "
            f"{original_exception}"
        )
        super().__init__(message)


def _iter_exception_messages(error: Exception) -> List[str]:
    """Collect error messages from chained exceptions for classification."""
    messages = []
    seen = set()
    current = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        if message:
            messages.append(message)
        current = current.__cause__ or current.__context__

    return messages


def is_retriable_transcript_error(error: Exception) -> bool:
    """Return True when the error looks like a transient block/network failure."""
    error_type = type(error).__name__
    if error_type in TERMINAL_TRANSCRIPT_ERROR_TYPES:
        return False
    if error_type in RETRIABLE_TRANSCRIPT_ERROR_TYPES:
        return True

    combined_message = " | ".join(_iter_exception_messages(error)).lower()
    retriable_markers = (
        "429",
        "too many requests",
        "sorry/index",
        "unusual traffic",
        "connection broken",
        "request blocked",
        "incomplete read",
        "incompleteread",
        "ip blocked",
        "proxy error",
        "read timed out",
        "connect timeout",
        "connection aborted",
        "temporarily unavailable",
        "bad gateway",
        "service unavailable",
    )
    return any(marker in combined_message for marker in retriable_markers)


def fetch_transcript_with_retries(
    video_id: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> Tuple[Any, Any, int]:
    """Fetch transcript by retrying the entire list/select/fetch flow."""
    last_error: Optional[Exception] = None
    last_attempt = 0
    last_stage = "initialize"
    last_retriable = False

    for attempt in range(1, max_attempts + 1):
        last_attempt = attempt
        ytt_api = None
        stage = "initialize"
        try:
            ytt_api = get_ytt_api()

            stage = "list"
            transcript_list = ytt_api.list(video_id)

            stage = "select"
            try:
                transcript = transcript_list.find_transcript(["en"])
                logger.info(
                    f"Found English transcript for video {video_id} on attempt {attempt}/{max_attempts}"
                )
            except Exception as english_error:
                logger.warning(
                    f"No English transcript found for {video_id} on attempt {attempt}/{max_attempts}. "
                    "Trying first available transcript."
                )
                try:
                    transcript = next(iter(transcript_list))
                    logger.info(
                        f"Using first available transcript ({transcript.language_code}) for video {video_id}"
                    )
                except StopIteration:
                    raise english_error

            stage = "fetch"
            fetched_transcript = transcript.fetch()
            return transcript, fetched_transcript, attempt
        except Exception as error:
            last_error = error
            last_stage = stage
            last_retriable = is_retriable_transcript_error(error)

            logger.warning(
                f"Transcript attempt {attempt}/{max_attempts} failed for video {video_id} "
                f"at stage {stage} (error_type={type(error).__name__}, retriable={last_retriable}): {error}"
            )

            if not last_retriable or attempt >= max_attempts:
                break

            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.info(
                f"Retrying transcript fetch for video {video_id} in {delay:.2f}s "
                f"after transient failure at stage {stage}"
            )
            time.sleep(delay)
        finally:
            if ytt_api is not None:
                del ytt_api

    if last_error is None:
        last_error = RuntimeError(
            "Transcript retrieval failed without a captured error"
        )

    raise TranscriptRetrievalError(
        video_id=video_id,
        stage=last_stage,
        attempts=last_attempt,
        retriable=last_retriable,
        original_exception=last_error,
    )


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


def call_api_callback(
    callback_url: str, data: Dict[str, Any], max_retries: int = 3
) -> bool:
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
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info(f"Successfully called callback: {callback_url}")
                return True
            else:
                logger.warning(
                    f"Callback failed with status {response.status_code}: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1} failed to call callback {callback_url}: {str(e)}"
            )

        # Exponential backoff for retries
        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    logger.error(
        f"Failed to call callback after {max_retries} attempts: {callback_url}"
    )
    return False


def send_result_to_sqs(message: Dict[str, Any]) -> bool:
    """Send a Lambda result message to SQS when configured."""
    queue_url = os.getenv("LAMBDA_RESULTS_QUEUE_URL", "").strip()
    if not queue_url:
        return False

    try:
        sqs_client = boto3.client("sqs")
        sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message))
        logger.info(
            f"Successfully sent {message.get('event_type')} event to SQS for "
            f"video {message.get('video_id')} in job {message.get('job_id')}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Failed to send {message.get('event_type')} event to SQS for "
            f"video {message.get('video_id')} in job {message.get('job_id')}: {str(e)}"
        )
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
    lambda_request_id = getattr(context, "aws_request_id", None)

    try:
        fetch_start = time.time()
        transcript, fetched_transcript, attempt_count = fetch_transcript_with_retries(
            video_id
        )

        fetch_end = time.time()
        logger.info(
            f"API fetch took {fetch_end - fetch_start:.3f}s for video {video_id} "
            f"(attempts={attempt_count})"
        )

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
            logger.warning("No pre-fetched metadata provided, using basic fallback")
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
        result_message = {
            "event_type": "video_completed",
            "job_id": job_id,
            "video_id": video_id,
            "user_id": user_id,
            "lambda_request_id": lambda_request_id,
            "sent_at": int(time.time()),
            "s3_key": s3_key,
            "transcript_length": len(transcript_text),
            "metadata": metadata,
        }
        callback_called = False
        if send_result_to_sqs(result_message):
            callback_called = True
        elif api_base_url:
            callback_url = f"{api_base_url}/internal/job/{job_id}/video-complete"
            completion_data = {
                "video_id": video_id,
                "s3_key": s3_key,
                "transcript_length": len(transcript_text),
                "metadata": metadata,
            }

            callback_called = call_api_callback(callback_url, completion_data)
            if not callback_called:
                logger.warning(f"Failed to call success callback for video {video_id}")
        else:
            logger.warning("No SQS queue or api_base_url provided, skipping callback")

        # Return success response
        return {
            "statusCode": 200,
            "body": {
                "video_id": video_id,
                "status": "completed",
                "transcript_length": len(transcript_text),
                "s3_key": s3_key,
                "metadata": metadata,
                "callback_called": callback_called,
            },
        }

    except Exception as e:
        error_message = f"Failed to get transcript for video {video_id}: {str(e)}"
        error_stage = getattr(e, "stage", "unknown")
        error_attempts = getattr(e, "attempts", 1)
        error_retriable = getattr(e, "retriable", False)
        last_error_type = getattr(e, "error_type", type(e).__name__)

        logger.error(
            f"{error_message} (stage={error_stage}, attempts={error_attempts}, "
            f"retriable={error_retriable}, error_type={last_error_type})"
        )

        # Call failure callback
        result_message = {
            "event_type": "video_failed",
            "job_id": job_id,
            "video_id": video_id,
            "user_id": user_id,
            "lambda_request_id": lambda_request_id,
            "sent_at": int(time.time()),
            "error": str(e),
            "error_type": last_error_type,
            "stage": error_stage,
            "retriable": error_retriable,
            "attempts": error_attempts,
            "last_error_type": last_error_type,
        }
        callback_called = False
        if send_result_to_sqs(result_message):
            callback_called = True
        elif api_base_url:
            callback_url = f"{api_base_url}/internal/job/{job_id}/video-failed"
            failure_data = {
                "video_id": video_id,
                "error": str(e),
                "error_type": last_error_type,
                "stage": error_stage,
                "retriable": error_retriable,
                "attempts": error_attempts,
                "last_error_type": last_error_type,
            }

            callback_called = call_api_callback(callback_url, failure_data)
            if not callback_called:
                logger.warning(f"Failed to call failure callback for video {video_id}")
        else:
            logger.warning(
                "No SQS queue or api_base_url provided, skipping failure callback"
            )

        # Return error response
        return {
            "statusCode": 500,
            "body": {
                "video_id": video_id,
                "status": "failed",
                "error": str(e),
                "error_type": last_error_type,
                "stage": error_stage,
                "retriable": error_retriable,
                "attempts": error_attempts,
                "last_error_type": last_error_type,
                "callback_called": callback_called,
            },
        }
