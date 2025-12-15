#!/usr/bin/env python3
"""
Diagnostic Script: Measure HTTP request count and size per video transcript fetch.

This script intercepts all HTTP requests made by youtube-transcript-api to understand:
1. How many requests are made per video
2. How much data is transferred (request/response sizes)
3. Differences between videos WITH vs WITHOUT transcripts
4. Whether certain channels/video types cost more

USAGE:
1. Add your video IDs to the TEST_VIDEOS dict below
2. Set USE_PROXY = True/False
3. Run: python scripts/diagnose_proxy_usage.py

OUTPUT:
- Per-video metrics (requests, bytes, time, result)
- Summary statistics comparing different video categories
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Set to True to use Webshare proxy (like production), False for direct connection
USE_PROXY = True

# Your Webshare credentials (or set via environment variables)
PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME", "")
PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD", "")

# Video IDs to test - ADD YOUR OWN IDs HERE
# Organize by category to see patterns
TEST_VIDEOS = {
    # Videos that SHOULD have transcripts (manual or auto-generated)
    "with_transcript": [
        "E8Rl7s--lHw",
        "7xAOhOIQQP4",
        "nCuaNmeVfQY",
        # Add video IDs that you know have transcripts
        # Example: "dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
    ],
    # Videos that likely DON'T have transcripts
    "without_transcript": [
        "JaaDYAEQ7d8",
        "rCgwPy5s7Y4",
        "DB-arVJVGXg",
        # Add video IDs from channels with high failure rates
        # These are the ones causing proxy consumption issues
    ],
    # YouTube Shorts (often no transcript)
    "shorts": [
        # Add Short video IDs (usually 60 seconds or less)
    ],
    # Specific channels you want to investigate
    "channel_A": [
        # Add video IDs from a specific channel
    ],
    "channel_B": [
        # Add video IDs from another channel for comparison
    ],
}

# How many retries the library should attempt when blocked (0 = no retries)
# This is the key setting that affects proxy consumption
RETRIES_WHEN_BLOCKED = 1  # Your current production setting

# =============================================================================
# HTTP INTERCEPTION - DO NOT MODIFY
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single HTTP request"""

    method: str
    url: str
    request_size: int
    response_size: int
    status_code: int
    duration_ms: float
    timestamp: str


@dataclass
class VideoMetrics:
    """Aggregated metrics for fetching one video's transcript"""

    video_id: str
    category: str
    success: bool
    error_message: Optional[str] = None
    transcript_found: bool = False
    transcript_language: Optional[str] = None
    transcript_type: Optional[str] = None  # manual or auto-generated
    total_requests: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    requests: List[RequestMetrics] = field(default_factory=list)


class RequestInterceptor:
    """Intercepts and measures all HTTP requests"""

    def __init__(self):
        self.current_video_requests: List[RequestMetrics] = []
        self._original_request = None
        self._installed = False

    def install(self):
        """Install the request interceptor"""
        if self._installed:
            return

        import requests

        self._original_request = requests.Session.request

        interceptor = self

        def intercepted_request(session, method, url, **kwargs):
            start_time = time.time()

            # Calculate request size
            request_size = 0
            if "data" in kwargs and kwargs["data"]:
                request_size += len(str(kwargs["data"]).encode("utf-8"))
            if "json" in kwargs and kwargs["json"]:
                request_size += len(json.dumps(kwargs["json"]).encode("utf-8"))
            if "headers" in kwargs:
                for k, v in kwargs.get("headers", {}).items():
                    request_size += len(f"{k}: {v}".encode("utf-8"))

            # Make the actual request
            response = interceptor._original_request(session, method, url, **kwargs)

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Calculate response size
            response_size = len(response.content) if response.content else 0
            response_size += len(str(response.headers).encode("utf-8"))

            # Record metrics
            metrics = RequestMetrics(
                method=method,
                url=url[:100] + "..." if len(url) > 100 else url,  # Truncate long URLs
                request_size=request_size,
                response_size=response_size,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                timestamp=datetime.now().isoformat(),
            )
            interceptor.current_video_requests.append(metrics)

            return response

        requests.Session.request = intercepted_request
        self._installed = True
        print("✓ HTTP request interceptor installed")

    def uninstall(self):
        """Restore original request method"""
        if self._installed and self._original_request:
            import requests

            requests.Session.request = self._original_request
            self._installed = False

    def start_video(self):
        """Start tracking requests for a new video"""
        self.current_video_requests = []

    def get_video_requests(self) -> List[RequestMetrics]:
        """Get all requests made for current video"""
        return self.current_video_requests.copy()


# =============================================================================
# TRANSCRIPT FETCHING
# =============================================================================


def create_transcript_api(use_proxy: bool, retries: int):
    """Create YouTubeTranscriptApi instance with or without proxy"""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.proxies import WebshareProxyConfig

    if use_proxy and PROXY_USERNAME and PROXY_PASSWORD:
        print(f"✓ Using Webshare proxy (retries_when_blocked={retries})")
        proxy_config = WebshareProxyConfig(
            proxy_username=PROXY_USERNAME,
            proxy_password=PROXY_PASSWORD,
            retries_when_blocked=retries,
        )
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    else:
        print("✓ Using direct connection (no proxy)")
        return YouTubeTranscriptApi()


def fetch_transcript_with_metrics(
    video_id: str,
    category: str,
    api: Any,
    interceptor: RequestInterceptor,
) -> VideoMetrics:
    """Fetch transcript for one video and collect metrics"""

    interceptor.start_video()
    start_time = time.time()

    metrics = VideoMetrics(video_id=video_id, category=category, success=False)

    try:
        # Step 1: List available transcripts
        transcript_list = api.list(video_id)

        # Step 2: Try to find transcript (English first, then any)
        transcript = None
        try:
            transcript = transcript_list.find_transcript(["en"])
        except Exception:
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                pass

        if transcript is None:
            metrics.error_message = "No transcripts available"
            metrics.transcript_found = False
        else:
            # Step 3: Fetch the transcript content
            fetched = transcript.fetch()

            metrics.success = True
            metrics.transcript_found = True
            metrics.transcript_language = transcript.language_code
            metrics.transcript_type = (
                "auto-generated" if transcript.is_generated else "manual"
            )

    except Exception as e:
        error_type = type(e).__name__
        metrics.error_message = f"{error_type}: {str(e)[:100]}"
        metrics.transcript_found = False

    # Collect timing and request metrics
    end_time = time.time()
    metrics.duration_seconds = round(end_time - start_time, 3)

    requests_made = interceptor.get_video_requests()
    metrics.requests = requests_made
    metrics.total_requests = len(requests_made)
    metrics.total_request_bytes = sum(r.request_size for r in requests_made)
    metrics.total_response_bytes = sum(r.response_size for r in requests_made)
    metrics.total_bytes = metrics.total_request_bytes + metrics.total_response_bytes

    return metrics


# =============================================================================
# REPORTING
# =============================================================================


def print_video_result(metrics: VideoMetrics, verbose: bool = False):
    """Print results for one video"""
    status = "✓" if metrics.success else "✗"
    transcript_info = ""
    if metrics.transcript_found:
        transcript_info = f" [{metrics.transcript_language}, {metrics.transcript_type}]"

    print(f"\n{status} {metrics.video_id} ({metrics.category}){transcript_info}")
    print(f"   Requests: {metrics.total_requests}")
    print(
        f"   Data transferred: {metrics.total_bytes:,} bytes ({metrics.total_bytes/1024:.1f} KB)"
    )
    print(f"   Time: {metrics.duration_seconds:.2f}s")

    if metrics.error_message:
        print(f"   Error: {metrics.error_message}")

    if verbose and metrics.requests:
        print("   Request breakdown:")
        for i, req in enumerate(metrics.requests, 1):
            print(f"      {i}. {req.method} {req.url}")
            print(
                f"         Status: {req.status_code}, Size: {req.response_size:,} bytes, Time: {req.duration_ms:.0f}ms"
            )


def print_summary(all_metrics: List[VideoMetrics]):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Group by category
    by_category: Dict[str, List[VideoMetrics]] = {}
    for m in all_metrics:
        if m.category not in by_category:
            by_category[m.category] = []
        by_category[m.category].append(m)

    # Also group by success/failure
    successful = [m for m in all_metrics if m.transcript_found]
    failed = [m for m in all_metrics if not m.transcript_found]

    print(f"\nTotal videos tested: {len(all_metrics)}")
    print(f"Transcripts found: {len(successful)}")
    print(f"Transcripts NOT found: {len(failed)}")

    # Per-category stats
    print("\n--- By Category ---")
    for category, metrics in sorted(by_category.items()):
        if not metrics:
            continue

        found = sum(1 for m in metrics if m.transcript_found)
        avg_requests = sum(m.total_requests for m in metrics) / len(metrics)
        avg_bytes = sum(m.total_bytes for m in metrics) / len(metrics)
        avg_time = sum(m.duration_seconds for m in metrics) / len(metrics)

        print(f"\n{category}:")
        print(
            f"   Videos: {len(metrics)} (found: {found}, not found: {len(metrics) - found})"
        )
        print(f"   Avg requests/video: {avg_requests:.1f}")
        print(f"   Avg bytes/video: {avg_bytes:,.0f} ({avg_bytes/1024:.1f} KB)")
        print(f"   Avg time/video: {avg_time:.2f}s")

    # Compare found vs not found
    print("\n--- Found vs Not Found ---")
    if successful:
        avg_req_found = sum(m.total_requests for m in successful) / len(successful)
        avg_bytes_found = sum(m.total_bytes for m in successful) / len(successful)
        print(
            f"WITH transcript:    {len(successful)} videos, avg {avg_req_found:.1f} requests, avg {avg_bytes_found/1024:.1f} KB"
        )

    if failed:
        avg_req_failed = sum(m.total_requests for m in failed) / len(failed)
        avg_bytes_failed = sum(m.total_bytes for m in failed) / len(failed)
        print(
            f"WITHOUT transcript: {len(failed)} videos, avg {avg_req_failed:.1f} requests, avg {avg_bytes_failed/1024:.1f} KB"
        )

    if successful and failed:
        ratio = avg_req_failed / avg_req_found if avg_req_found > 0 else 0
        print(
            f"\n⚠️  Videos WITHOUT transcripts use {ratio:.1f}x more requests on average"
        )


def save_detailed_report(all_metrics: List[VideoMetrics], filename: str):
    """Save detailed JSON report"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "use_proxy": USE_PROXY,
            "retries_when_blocked": RETRIES_WHEN_BLOCKED,
        },
        "summary": {
            "total_videos": len(all_metrics),
            "transcripts_found": sum(1 for m in all_metrics if m.transcript_found),
            "total_requests": sum(m.total_requests for m in all_metrics),
            "total_bytes": sum(m.total_bytes for m in all_metrics),
        },
        "videos": [asdict(m) for m in all_metrics],
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {filename}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("YouTube Transcript API - Proxy Usage Diagnostic")
    print("=" * 70)

    # Check if any videos are configured
    total_videos = sum(len(ids) for ids in TEST_VIDEOS.values())
    if total_videos == 0:
        print("\n⚠️  No video IDs configured!")
        print("   Edit TEST_VIDEOS in this script to add video IDs to test.")
        print("\n   Example:")
        print("   TEST_VIDEOS = {")
        print('       "with_transcript": ["dQw4w9WgXcQ"],')
        print('       "without_transcript": ["YOUR_VIDEO_ID"],')
        print("   }")
        return

    print(f"\nVideos to test: {total_videos}")
    for category, ids in TEST_VIDEOS.items():
        if ids:
            print(f"   {category}: {len(ids)} videos")

    # Check proxy credentials
    if USE_PROXY and (not PROXY_USERNAME or not PROXY_PASSWORD):
        print("\n⚠️  Proxy enabled but credentials not set!")
        print(
            "   Set WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD environment variables"
        )
        print("   Or edit this script to set PROXY_USERNAME and PROXY_PASSWORD")
        return

    # Setup
    interceptor = RequestInterceptor()
    interceptor.install()

    try:
        api = create_transcript_api(USE_PROXY, RETRIES_WHEN_BLOCKED)

        all_metrics: List[VideoMetrics] = []

        print("\n" + "-" * 70)
        print("TESTING VIDEOS")
        print("-" * 70)

        for category, video_ids in TEST_VIDEOS.items():
            for video_id in video_ids:
                if not video_id:
                    continue

                print(f"\nTesting: {video_id} ({category})...", end="", flush=True)

                # Create fresh API instance for each video (like Lambda does)
                api = create_transcript_api(USE_PROXY, RETRIES_WHEN_BLOCKED)

                metrics = fetch_transcript_with_metrics(
                    video_id, category, api, interceptor
                )
                all_metrics.append(metrics)

                # Quick result indicator
                if metrics.transcript_found:
                    print(f" ✓ ({metrics.total_requests} requests)")
                else:
                    print(f" ✗ ({metrics.total_requests} requests)")

        # Print detailed results
        print("\n" + "-" * 70)
        print("DETAILED RESULTS")
        print("-" * 70)

        for metrics in all_metrics:
            print_video_result(metrics, verbose=True)

        # Print summary
        print_summary(all_metrics)

        # Save detailed report
        report_file = os.path.join(
            PROJECT_ROOT, "scripts", "proxy_diagnostic_report.json"
        )
        save_detailed_report(all_metrics, report_file)

    finally:
        interceptor.uninstall()


if __name__ == "__main__":
    main()
