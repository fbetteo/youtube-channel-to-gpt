#!/usr/bin/env python3
"""
Check that downloads rendered from stored segments match the worker's .txt.

Read-only: reads recent completed job_videos and their S3 objects, renders the
.txt from {video}.json with the job's options, and reports any difference.
Run before relying on rendered text (and before the worker stops writing .txt).

Usage:
    python scripts/compare_rendered_txt.py
    python scripts/compare_rendered_txt.py --limit 500 --show 3
"""

import argparse
import asyncio
import difflib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from botocore.exceptions import ClientError

import transcript_formats
import youtube_service
from db_youtube_transcripts.database import close_db_pool, get_db_connection

QUERY = """
    SELECT jv.video_id, jv.title, jv.view_count, jv.s3_key, j.formatting_options
    FROM job_videos jv
    JOIN jobs j ON j.job_id = jv.job_id
    WHERE jv.status = 'completed' AND jv.s3_key IS NOT NULL
    ORDER BY jv.processed_at DESC NULLS LAST
    LIMIT $1
"""


def read_object(s3_client, bucket, key):
    try:
        return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


def compare(row, s3_client, bucket):
    """Return ('match' | 'mismatch' | 'no_segments' | 'no_text', diff lines)."""
    segments = read_object(s3_client, bucket, youtube_service._segments_key_for(row["s3_key"]))
    if segments is None:
        return "no_segments", []
    stored = read_object(s3_client, bucket, row["s3_key"])
    if stored is None:
        return "no_text", []

    rendered = transcript_formats.render_text(
        json.loads(segments),
        youtube_service.parse_formatting_options(row["formatting_options"]),
        row["title"],
        row["view_count"],
    )
    stored_text = stored.decode("utf-8")
    if rendered == stored_text:
        return "match", []
    diff = difflib.unified_diff(
        stored_text.splitlines(), rendered.splitlines(), "stored", "rendered", lineterm="", n=1
    )
    return "mismatch", list(diff)[:20]


async def main(limit: int, show: int) -> int:
    async with get_db_connection() as conn:
        rows = [dict(r) for r in await conn.fetch(QUERY, limit)]
    s3_client, bucket = youtube_service.get_s3_client()

    counts = {"match": 0, "mismatch": 0, "no_segments": 0, "no_text": 0}
    shown = 0
    for row in rows:
        outcome, diff = await asyncio.to_thread(compare, row, s3_client, bucket)
        counts[outcome] += 1
        if outcome == "mismatch" and shown < show:
            shown += 1
            print(f"\nMISMATCH {row['s3_key']}")
            print("\n".join(diff))

    print(f"\nChecked {len(rows)} completed videos: {counts}")
    print("no_segments = created before the worker stored segments (served as stored .txt)")
    await close_db_pool()
    return 1 if counts["mismatch"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--show", type=int, default=5, help="mismatches to print")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.limit, args.show)))
