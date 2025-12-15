#!/usr/bin/env python3
"""
Quick script to download a job's ZIP file locally for debugging.
Usage: python download_job_zip.py <job_id>
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from youtube_service import create_transcript_zip_from_s3_concurrent
from hybrid_job_manager import hybrid_job_manager


async def download_job_zip(job_id: str, output_path: str = None):
    """
    Download ZIP file for a job by job_id.

    Args:
        job_id: The job identifier
        output_path: Optional output path, defaults to ./job_{job_id}.zip
    """
    print(f"Fetching job {job_id}...")

    # Verify job exists
    job = await hybrid_job_manager.get_job(job_id, include_videos=False)
    if not job:
        print(f"ERROR: Job {job_id} not found in database")
        return False

    print(f"Job found: {job.get('status')}")
    print(f"Source: {job.get('source_type')} - {job.get('source_name')}")
    print(f"Total videos: {job.get('total_videos')}")
    print(f"Completed: {job.get('processed_count')}")
    print()

    if job.get("status") not in ["completed", "completed_with_errors"]:
        print(f"WARNING: Job status is '{job.get('status')}', not 'completed'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return False

    print("Creating ZIP from S3 files...")
    try:
        zip_buffer = await create_transcript_zip_from_s3_concurrent(job_id)

        if not zip_buffer:
            print("ERROR: Failed to create ZIP (returned None)")
            return False

        # Determine output filename
        if not output_path:
            source_name = job.get("source_name", "unknown")
            safe_name = source_name.replace(" ", "_").replace("/", "_")[:30]
            output_path = f"job_{job_id[:8]}_{safe_name}.zip"

        # Write to file
        with open(output_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ SUCCESS!")
        print(f"ZIP file saved to: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    if len(sys.argv) < 2:
        print("Usage: python download_job_zip.py <job_id> [output_path]")
        print("\nExample:")
        print("  python download_job_zip.py abc123-def456-ghi789")
        print("  python download_job_zip.py abc123-def456-ghi789 my_download.zip")
        sys.exit(1)

    job_id = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = await download_job_zip(job_id, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
