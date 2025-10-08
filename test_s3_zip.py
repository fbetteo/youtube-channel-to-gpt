#!/usr/bin/env python3
"""
Test script for S3 ZIP creation functionality
"""
import requests
import json
import time

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"


def test_s3_zip_creation():
    """Test the new S3-based ZIP creation endpoint"""

    # You'll need to replace this with a real job ID from your API
    # Get this by creating a job first using the channel download endpoint
    job_id = "YOUR_REAL_JOB_ID_HERE"  # Replace with actual job ID

    print("=== Testing S3 ZIP Creation ===")
    print(f"Job ID: {job_id}")
    print(f"API Base URL: {API_BASE_URL}")
    print()

    # Test job status first
    print("1. Checking job status...")
    try:
        status_url = f"{API_BASE_URL}/channel/download/status/{job_id}"
        status_response = requests.get(status_url)

        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   Status: {status_data.get('status')}")
            print(f"   Completed videos: {status_data.get('completed', 0)}")
            print(f"   Total videos: {status_data.get('total_videos', 0)}")
            print(f"   Files: {len(status_data.get('files', []))}")

            if status_data.get("status") not in ["completed", "completed_with_errors"]:
                print(
                    f"   ❌ Job not ready for download (status: {status_data.get('status')})"
                )
                return

        else:
            print(f"   ❌ Status check failed: {status_response.status_code}")
            print(f"   Response: {status_response.text}")
            return

    except Exception as e:
        print(f"   ❌ Error checking status: {str(e)}")
        return

    # Test ZIP download
    print("\n2. Testing ZIP download from S3...")
    try:
        download_start = time.time()
        download_url = f"{API_BASE_URL}/channel/download/results/{job_id}"

        # You'll need to add your JWT token here
        headers = {
            "Authorization": "Bearer YOUR_JWT_TOKEN_HERE"  # Replace with actual token
        }

        download_response = requests.get(download_url, headers=headers)
        download_end = time.time()

        if download_response.status_code == 200:
            zip_size = len(download_response.content)
            download_time = download_end - download_start

            print(f"   ✅ ZIP download successful!")
            print(f"   Size: {zip_size:,} bytes ({zip_size / 1024 / 1024:.2f} MB)")
            print(f"   Download time: {download_time:.2f} seconds")
            print(f"   Content-Type: {download_response.headers.get('content-type')}")

            # Check custom headers
            custom_headers = [
                "X-Job-ID",
                "X-Files-Count",
                "X-Generation-Time-Seconds",
                "X-Source-Type",
            ]
            for header in custom_headers:
                if header in download_response.headers:
                    print(f"   {header}: {download_response.headers[header]}")

            # Optionally save the ZIP file
            save_zip = input("\n   Save ZIP file to disk? (y/n): ").lower().strip()
            if save_zip == "y":
                filename = f"test_download_{job_id}.zip"
                with open(filename, "wb") as f:
                    f.write(download_response.content)
                print(f"   💾 Saved as: {filename}")

        else:
            print(f"   ❌ ZIP download failed: {download_response.status_code}")
            print(f"   Response: {download_response.text}")

    except Exception as e:
        print(f"   ❌ Error downloading ZIP: {str(e)}")


def test_internal_callbacks():
    """Test the internal callback endpoints"""

    print("\n=== Testing Internal Callback Endpoints ===")

    # Test video completion callback
    job_id = "test-job-" + str(int(time.time()))
    video_id = "test-video-123"

    print(f"Testing with job_id: {job_id}")

    print("\n1. Testing video completion callback...")
    try:
        completion_url = f"{API_BASE_URL}/internal/job/{job_id}/video-complete"
        completion_data = {
            "video_id": video_id,
            "s3_key": f"user123/{job_id}/{video_id}.txt",
            "transcript_length": 1500,
            "metadata": {"language": "en", "type": "manual"},
        }

        response = requests.post(completion_url, json=completion_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

    print("\n2. Testing video failure callback...")
    try:
        failure_url = f"{API_BASE_URL}/internal/job/{job_id}/video-failed"
        failure_data = {
            "video_id": video_id,
            "error": "Test error message",
            "error_type": "TestError",
        }

        response = requests.post(failure_url, json=failure_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")


if __name__ == "__main__":
    print("YouTube Transcript API - S3 ZIP Test Script")
    print("=" * 50)

    print("\nInstructions:")
    print("1. Make sure your API is running on http://127.0.0.1:8000")
    print("2. Update the job_id and JWT token in this script")
    print("3. Ensure you have a completed job with files in S3")
    print()

    choice = input(
        "What would you like to test?\n1. S3 ZIP Creation\n2. Internal Callbacks\n3. Both\nChoice (1/2/3): "
    ).strip()

    if choice in ["1", "3"]:
        test_s3_zip_creation()

    if choice in ["2", "3"]:
        test_internal_callbacks()

    print("\n🏁 Testing completed!")
