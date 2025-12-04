"""
Debug script to download a job's ZIP file without authentication.
Usage: python debug_download_job.py <job_id>
"""

import sys
import asyncio
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hybrid_job_manager import HybridJobManager
import youtube_service


async def download_job_zip(job_id: str):
    """Download ZIP for a job without authentication"""

    # Initialize job manager
    hybrid_job_manager = HybridJobManager()

    # Get job details
    job = await hybrid_job_manager.get_job(job_id, include_videos=True)
    if not job:
        print(f"❌ Job {job_id} not found")
        return

    print(f"✓ Found job {job_id}")
    print(f"  User: {job.get('user_id')}")
    print(f"  Source: {job.get('source_name')} ({job.get('source_type')})")
    print(f"  Status: {job.get('status')}")
    print(f"  Files: {len(job.get('files', []))}")

    # Verify job is completed
    job_status = job.get("status")
    if job_status not in ["completed", "completed_with_errors"]:
        print(f"❌ Job is not ready for download. Current status: {job_status}")
        return

    print(f"\n📦 Creating ZIP from S3 files...")

    try:
        # Use concurrent download method
        zip_buffer = await youtube_service.create_transcript_zip_from_s3_concurrent(
            job_id
        )
    except Exception as e:
        print(f"⚠️  Concurrent download failed, trying sequential: {e}")
        zip_buffer = await youtube_service.create_transcript_zip_from_s3_sequential(
            job_id
        )

    # Generate filename
    source_name = job.get("source_name") or job.get("channel_name", "transcripts")
    safe_source_name = youtube_service.sanitize_filename(source_name)
    is_concatenated = job.get("formatting_options", {}).get("concatenate_all", False)
    filename_suffix = (
        "_concatenated_transcripts.zip" if is_concatenated else "_transcripts.zip"
    )
    filename = f"{safe_source_name}{filename_suffix}"

    # Save to current directory
    output_path = os.path.join(os.getcwd(), filename)

    with open(output_path, "wb") as f:
        f.write(zip_buffer.getvalue())

    file_size_mb = len(zip_buffer.getvalue()) / 1024 / 1024
    print(f"\n✅ Downloaded successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_download_job.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    print(f"🔍 Fetching job {job_id}...\n")

    asyncio.run(download_job_zip(job_id))
