"""
Backfill the shared transcript cache (src/transcript_cache.py).

1. Postgres video_transcripts rows (the old single-video cache) -> S3
   transcript-cache/{video_id}/{requested_language}.json, unless the object
   already exists. Rows are left in place.
2. Every S3 transcript-cache object without a transcript_cache index row ->
   index row (reads each object once for language and size).

Dry run by default: prints what it would do. Pass --apply to write.
Needs the transcript_cache table (db_youtube_transcripts/migration_add_transcript_cache.py)
and the usual DB_* / AWS_* / S3_BUCKET_NAME settings in .env.

    poetry run python scripts/backfill_transcript_cache.py
    poetry run python scripts/backfill_transcript_cache.py --apply
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import psycopg2
from dotenv import load_dotenv

PREFIX = "transcript-cache/"
SCHEMA_VERSION = 1  # keep in sync with src/transcript_cache.py and the worker

INDEX_SQL = """
    INSERT INTO transcript_cache
        (video_id, requested_language, language, is_generated,
         segment_count, char_count, fetched_at, source)
    VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s), 'backfill')
    ON CONFLICT (video_id, requested_language) DO NOTHING
"""


def index_row(document):
    segments = document["segments"]
    return (
        document["video_id"],
        document["requested_language"],
        document["language"],
        bool(document["is_generated"]),
        len(segments),
        sum(len(str(s.get("text", ""))) for s in segments),
        float(document.get("fetched_at") or 0),
    )


def has_table(conn, name):
    with conn.cursor() as c:
        c.execute("SELECT to_regclass(%s)", (name,))
        return c.fetchone()[0] is not None


def object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def copy_postgres_rows(conn, s3, bucket, apply):
    """Step 1: old video_transcripts rows -> S3 objects + index rows."""
    if not has_table(conn, "video_transcripts"):
        print("Step 1: no video_transcripts table, skipping")
        return
    with conn.cursor() as c:
        c.execute(
            """
            SELECT video_id, requested_language, language, is_generated,
                   segments, extract(epoch FROM created_at)
            FROM video_transcripts
            """
        )
        rows = c.fetchall()

    def handle(row):
        video_id, requested_language, language, is_generated, segments, created = row
        key = f"{PREFIX}{video_id}/{requested_language}.json"
        if object_exists(s3, bucket, key):
            return "exists", None
        document = {
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            "requested_language": requested_language,
            "language": language,
            "is_generated": is_generated,
            "fetched_at": int(created),
            # psycopg2 returns JSONB already decoded
            "segments": segments if isinstance(segments, list) else json.loads(segments),
        }
        if apply:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(document, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
        return "copied", document

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(handle, rows))
    copied = [doc for status, doc in results if status == "copied"]
    print(f"Step 1: {len(rows)} video_transcripts rows, {len(rows) - len(copied)} "
          f"already in S3, {len(copied)} {'copied' if apply else 'to copy'}")
    if apply and copied:
        with conn.cursor() as c:
            c.executemany(INDEX_SQL, [index_row(d) for d in copied])


def index_s3_objects(conn, s3, bucket, apply):
    """Step 2: S3 cache objects without an index row -> index rows."""
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".json")]

    indexed = set()
    if has_table(conn, "transcript_cache"):
        with conn.cursor() as c:
            c.execute("SELECT video_id, requested_language FROM transcript_cache")
            indexed = {f"{PREFIX}{v}/{lang}.json" for v, lang in c.fetchall()}
    missing = [k for k in keys if k not in indexed]
    print(f"Step 2: {len(keys)} S3 cache objects, {len(keys) - len(missing)} indexed, "
          f"{len(missing)} {'to index' if not apply else 'indexing'}")
    if not apply or not missing:
        return

    def read(key):
        try:
            document = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            if document.get("schema_version") == SCHEMA_VERSION:
                return index_row(document)
        except Exception as e:
            print(f"  skip {key}: {e}")
        return None

    with ThreadPoolExecutor(16) as pool:
        rows = [r for r in pool.map(read, missing) if r]
    with conn.cursor() as c:
        c.executemany(INDEX_SQL, rows)
    print(f"Step 2: indexed {len(rows)} objects")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="Write to S3 and Postgres")
    args = parser.parse_args()

    load_dotenv()
    bucket = os.environ["S3_BUCKET_NAME"]
    s3 = boto3.client("s3")  # reads AWS_* from the environment
    conn = psycopg2.connect(
        database=os.getenv("DB_NAME_YOUTUBE_TRANSCRIPTS"),
        host=os.getenv("DB_HOST_YOUTUBE_TRANSCRIPTS"),
        user=os.getenv("DB_USERNAME_YOUTUBE_TRANSCRIPTS"),
        password=os.getenv("DB_PASSWORD_YOUTUBE_TRANSCRIPTS"),
        port=os.getenv("DB_PORT_YOUTUBE_TRANSCRIPTS"),
    )
    conn.autocommit = True
    if args.apply and not has_table(conn, "transcript_cache"):
        raise SystemExit("Run db_youtube_transcripts/migration_add_transcript_cache.py first")
    print(f"Bucket {bucket} | {'APPLY' if args.apply else 'DRY RUN (pass --apply to write)'}")
    try:
        copy_postgres_rows(conn, s3, bucket, args.apply)
        index_s3_objects(conn, s3, bucket, args.apply)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
