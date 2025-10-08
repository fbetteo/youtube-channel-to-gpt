import os
import json
from get_transcript import lambda_handler

# Set environment variables for local testing
os.environ["WEBSHARE_PROXY_USERNAME"] = "koppsqwu"  # Replace with actual
os.environ["WEBSHARE_PROXY_PASSWORD"] = "ivsxc1dfonav"  # Replace with actual
os.environ["YOUTUBE_API_KEY"] = (
    "AIzaSyD0bbn9msto2Y7x__sN1ANfwH71nahCrNo"  # Replace with actual
)
os.environ["S3_BUCKET_NAME"] = "youtube-transcripts-bucket-fbetteo"  # We'll create this
os.environ["API_BASE_URL"] = "https://your-api-domain.com"  # Your FastAPI URL


def test_lambda_locally():
    # Test event (same format Lambda will receive)
    test_event = {
        "video_id": "pGT00gcGcu0",  # Rick Roll video (always available)
        "job_id": "test-job-123",
        "user_id": "test-user-456",
        "include_timestamps": False,
        "include_video_title": True,
        "include_video_id": True,
        "include_video_url": True,
        "include_view_count": False,
        "pre_fetched_metadata": {
            "title": "Rick Astley - Never Gonna Give You Up",
            "viewCount": 1400000000,
        },
    }

    try:
        # Call your lambda handler
        result = lambda_handler(test_event, None)
        print("SUCCESS!")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_lambda_locally()
