"""Recover the two August 2026 file-backed playlist jobs.

Dry-run is the default. Use --apply only after deploying the cancellation schema
migration. The script is deliberately incident-specific and leaves legacy JSON
and S3 objects untouched so the operation remains recoverable.
"""

import argparse
import os

import boto3
import psycopg2
from dotenv import load_dotenv


PLAYLIST_ID = "PLpXs-4paEmUc2Bq1U8-suLYltOW_87F9z"
CONTROL_JOB_ID = "673166d9-3052-4a15-b4e5-fe97942de077"
RECOVER_JOB_ID = "3885d455-dea0-4671-8a0e-f61fe4973150"
CANCEL_JOB_ID = "18e1e655-d210-467e-96a0-51ae57385e19"
UNAVAILABLE_IDS = {"Y2RffTPsFhc", "qSEpMF8GPso"}
DISABLED_ID = "qrJyYBRu26A"
INCIDENT_VIDEO_IDS = [
    "9afTO_dgLMs",
    "onoRB0Wghgk",
    "YesFa8rdCJg",
    "Y2RffTPsFhc",
    "08q_qxrNzG0",
    "JCFqkbdHWWg",
    "OKzjN1chAEM",
    "VE4beTgJI-A",
    "pXrfAeFb-II",
    "g_psnb4tgWc",
    "oT1WrSJjSfA",
    "2bxiARjF4No",
    "bnhYS8bFWBQ",
    "tzXkEygHU50",
    "8NseEepBv1w",
    "naSAPJuQuJw",
    "Ngh-6oRCVZY",
    "1csjIkIDVvM",
    "qSEpMF8GPso",
    "qrJyYBRu26A",
    "y3yZvA2kcWg",
    "Q3zTX-PWPuk",
    "UDabS9GF79A",
    "1jJDojiJBy8",
    "2-WNhhnSn1s",
    "mWyYSXiArUE",
    "ZSq2TUwlEh4",
    "touQQlAS5eY",
    "OTsZtH5gfhI",
    "VGlL1FR8sQE",
]


def connect():
    return psycopg2.connect(
        database=os.environ["DB_NAME_YOUTUBE_TRANSCRIPTS"],
        host=os.environ["DB_HOST_YOUTUBE_TRANSCRIPTS"],
        user=os.environ["DB_USERNAME_YOUTUBE_TRANSCRIPTS"],
        password=os.environ["DB_PASSWORD_YOUTUBE_TRANSCRIPTS"],
        port=os.environ["DB_PORT_YOUTUBE_TRANSCRIPTS"],
    )


def list_job_objects(s3, bucket: str, user_id: str, job_id: str):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{user_id}/{job_id}/")
    objects = {
        item["Key"].rsplit("/", 1)[-1].removesuffix(".txt"): item
        for item in response.get("Contents", [])
        if item["Key"].endswith(".txt")
    }
    for item in objects.values():
        body = s3.get_object(
            Bucket=bucket, Key=item["Key"], Range="bytes=0-1023"
        )["Body"].read()
        first_line = body.decode("utf-8", errors="replace").splitlines()[0]
        item["RecoveredTitle"] = (
            first_line.removeprefix("Video Title: ").strip()
            if first_line.startswith("Video Title: ")
            else None
        )
    return objects


def insert_job(
    cursor,
    *,
    job_id: str,
    user_id: str,
    api_key_id: str,
    objects: dict,
    cancelled: bool,
):
    completed_ids = set(objects)
    unresolved_ids = set(INCIDENT_VIDEO_IDS) - completed_ids
    if len(completed_ids) != 26 or len(unresolved_ids) != 4:
        raise RuntimeError(
            f"Unexpected evidence for {job_id}: completed={len(completed_ids)}, "
            f"unresolved={sorted(unresolved_ids)}"
        )

    start_time = min(item["LastModified"] for item in objects.values())
    end_time = max(item["LastModified"] for item in objects.values())
    status = "cancelled" if cancelled else "completed_with_errors"
    failed_count = 0 if cancelled else len(unresolved_ids)
    skipped_count = len(unresolved_ids) if cancelled else 0
    credits_used = 0 if cancelled else len(INCIDENT_VIDEO_IDS)
    refunded_credits = len(INCIDENT_VIDEO_IDS) if cancelled else 0

    cursor.execute(
        """
        INSERT INTO jobs (
            job_id, user_id, status, source_type, source_id, source_name,
            total_videos, processed_count, completed, failed_count, skipped_count,
            credits_reserved, credits_used, refunded_credits, reservation_id,
            include_timestamps, include_video_title, include_video_id,
            include_video_url, include_view_count, concatenate_all,
            lambda_dispatched_count, lambda_dispatch_time, prefetch_completed,
            formatting_options, playlist_id, api_key_id, start_time, end_time,
            created_at, updated_at, error_message
        ) VALUES (
            %s, %s, %s, 'playlist', %s, 'Testimonies',
            30, 30, 26, %s, %s,
            30, %s, %s, NULL,
            false, true, true, true, false, false,
            30, %s, true,
            %s::jsonb, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            job_id,
            user_id,
            status,
            PLAYLIST_ID,
            failed_count,
            skipped_count,
            credits_used,
            refunded_credits,
            start_time,
            '{"include_timestamps": false, "include_video_title": true, '
            '"include_video_id": true, "include_video_url": true, '
            '"include_view_count": false, "concatenate_all": false}',
            PLAYLIST_ID,
            api_key_id,
            start_time,
            end_time,
            start_time,
            end_time,
            (
                "Recovered duplicate incident job; full goodwill refund applied."
                if cancelled
                else "Recovered from S3 after the durable job insert failed."
            ),
        ),
    )

    rows = []
    for position, video_id in enumerate(INCIDENT_VIDEO_IDS):
        obj = objects.get(video_id)
        if obj:
            video_status = "completed"
            error = None
            processed_at = obj["LastModified"]
            s3_key = obj["Key"]
            file_size = obj["Size"]
        else:
            video_status = "skipped" if cancelled else "failed"
            if video_id in UNAVAILABLE_IDS:
                error = "Unavailable playlist entry"
            elif video_id == DISABLED_ID:
                error = "TranscriptsDisabled: subtitles are disabled"
            else:
                error = "No transcript result was stored during the incident"
            processed_at = end_time
            s3_key = None
            file_size = 0
        rows.append(
            (
                job_id,
                video_id,
                (
                    obj.get("RecoveredTitle")
                    if obj
                    else (
                        "Unavailable video"
                        if video_id in UNAVAILABLE_IDS
                        else f"Recovered video {position + 1}"
                    )
                )
                or f"Recovered video {position + 1}",
                f"https://www.youtube.com/watch?v={video_id}",
                video_status,
                s3_key,
                file_size,
                processed_at,
                error,
            )
        )

    cursor.executemany(
        """
        INSERT INTO job_videos (
            job_id, video_id, title, url, duration_category, view_count,
            status, s3_key, file_size, processed_at, error_message
        ) VALUES (%s, %s, %s, %s, NULL, 0, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_dotenv(".env")

    connection = connect()
    connection.autocommit = False
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT user_id::text FROM jobs WHERE job_id = %s", (CONTROL_JOB_ID,)
        )
        control = cursor.fetchone()
        if not control:
            raise RuntimeError("Control job not found")
        user_id = control[0]
        cursor.execute(
            """
            SELECT key_id::text FROM api_keys
            WHERE user_id = %s
            ORDER BY last_used_at DESC NULLS LAST LIMIT 1
            """,
            (user_id,),
        )
        api_key = cursor.fetchone()
        if not api_key:
            raise RuntimeError("Client API key not found")
        api_key_id = api_key[0]

        cursor.execute(
            "SELECT job_id::text FROM jobs WHERE job_id IN (%s, %s)",
            (RECOVER_JOB_ID, CANCEL_JOB_ID),
        )
        existing = {row[0] for row in cursor.fetchall()}
        if existing:
            print(f"Recovery already present for: {sorted(existing)}")
            connection.rollback()
            return

        s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        bucket = os.environ["S3_BUCKET_NAME"]
        recovered_objects = list_job_objects(s3, bucket, user_id, RECOVER_JOB_ID)
        cancelled_objects = list_job_objects(s3, bucket, user_id, CANCEL_JOB_ID)
        print(
            f"Evidence: recover={len(recovered_objects)} objects, "
            f"cancel={len(cancelled_objects)} objects"
        )

        insert_job(
            cursor,
            job_id=RECOVER_JOB_ID,
            user_id=user_id,
            api_key_id=api_key_id,
            objects=recovered_objects,
            cancelled=False,
        )
        insert_job(
            cursor,
            job_id=CANCEL_JOB_ID,
            user_id=user_id,
            api_key_id=api_key_id,
            objects=cancelled_objects,
            cancelled=True,
        )
        cursor.execute(
            "UPDATE user_credits SET credits = credits + 30 WHERE user_id = %s",
            (user_id,),
        )
        cursor.execute(
            """
            UPDATE api_keys
            SET total_credits_used = GREATEST(total_credits_used - 30, 0)
            WHERE key_id = %s
            """,
            (api_key_id,),
        )

        if args.apply:
            connection.commit()
            print("Recovery committed: one job recovered; duplicate cancelled/refunded.")
        else:
            connection.rollback()
            print("Dry run passed; transaction rolled back. Re-run with --apply to commit.")
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
