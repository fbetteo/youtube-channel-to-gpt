"""
Simple 90-day user and usage analysis.

Run from the repository root:
    ipython scripts/user_segment_analysis.py

Output:
    user_segments_90d.csv

The CSV contains one row per Supabase Auth user with an email address.
Segments ending in "_proxy" are inferred because single-video events,
downloads, pricing views, checkout starts, and purchases are not reliably
recorded in the current database.
"""

import csv
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


DAYS = 90
FREE_CREDITS = 10
OUTPUT_FILE = Path("user_segments_90d.csv")


def connect():
    load_dotenv(".env")
    return psycopg2.connect(
        database=os.environ["DB_NAME_YOUTUBE_TRANSCRIPTS"],
        host=os.environ["DB_HOST_YOUTUBE_TRANSCRIPTS"],
        user=os.environ["DB_USERNAME_YOUTUBE_TRANSCRIPTS"],
        password=os.environ["DB_PASSWORD_YOUTUBE_TRANSCRIPTS"],
        port=os.environ["DB_PORT_YOUTUBE_TRANSCRIPTS"],
    )


def fetch_all(cursor, sql, params=()):
    cursor.execute(sql, params)
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def segment_user(user):
    """
    Assign one primary outreach segment and any additional useful tags.

    The primary segments are mutually exclusive. Tags are not.
    """
    tags = []
    credits = user["current_credits"]
    recent_jobs = user["recent_jobs"]
    recent_successes = user["recent_successes"]
    recent_failures = user["recent_failures"]
    recent_timeouts = user["recent_timeouts"]
    lifetime_jobs = user["lifetime_jobs"]
    lifetime_batch_credits = user["lifetime_batch_credits"]
    purchases = user["purchase_count"]

    if recent_jobs:
        if user["used_channel_recently"]:
            tags.append("channel_user")
        if user["used_playlist_recently"]:
            tags.append("playlist_user")
        if recent_failures or recent_timeouts:
            tags.append("encountered_batch_error_or_timeout")

    if credits == 0:
        tags.append("zero_current_credits")

    if recent_jobs and recent_successes == 0:
        primary = "attempted_failed_batch"
    elif recent_jobs and (recent_failures or recent_timeouts):
        primary = "active_channel_playlist_with_errors"
    elif recent_jobs:
        primary = "active_channel_playlist_user"
    elif (
        lifetime_jobs
        and user["last_job_at"] is not None
        and (lifetime_batch_credits > FREE_CREDITS or (credits or 0) > FREE_CREDITS)
    ):
        primary = "previous_high_usage_inactive_proxy"
    elif lifetime_jobs:
        primary = "previous_batch_user_inactive"
    elif credits == 0 and purchases == 0:
        # A zero balance does not prove that ten free credits were consumed.
        # Older application paths could also create a user with zero credits.
        primary = "zero_balance_no_batch_usage_unknown"
    elif purchases == 0 and credits is not None and 1 <= credits <= 5:
        primary = "used_5_9_free_credits_proxy"
    elif purchases == 0 and credits is not None and 8 <= credits <= 9:
        primary = "used_1_2_free_credits_proxy"
    elif credits == FREE_CREDITS and lifetime_jobs == 0:
        primary = "registered_no_attempt_proxy"
    elif credits is None:
        primary = "registered_no_credit_record"
    else:
        primary = "other_or_unclear"

    if purchases:
        tags.append("buyer")

    return primary, ",".join(tags)


def print_table(headers, rows):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main():
    connection = connect()
    connection.set_session(readonly=True, autocommit=True)
    cursor = connection.cursor()

    clock = fetch_all(
        cursor,
        "SELECT now() AS database_time, now() - %s * interval '1 day' AS cutoff",
        (DAYS,),
    )[0]
    cutoff = clock["cutoff"]

    users = fetch_all(
        cursor,
        """
        WITH job_stats AS (
            SELECT
                user_id,
                count(*) AS lifetime_jobs,
                count(*) FILTER (
                    WHERE created_at >= now() - %s * interval '1 day'
                ) AS recent_jobs,
                coalesce(sum(credits_used), 0) AS lifetime_batch_credits,
                coalesce(sum(credits_used) FILTER (
                    WHERE created_at >= now() - %s * interval '1 day'
                ), 0) AS recent_batch_credits,
                coalesce(sum(completed) FILTER (
                    WHERE created_at >= now() - %s * interval '1 day'
                ), 0) AS recent_successes,
                coalesce(sum(failed_count) FILTER (
                    WHERE created_at >= now() - %s * interval '1 day'
                ), 0) AS recent_failures,
                count(*) FILTER (
                    WHERE created_at >= now() - %s * interval '1 day'
                      AND timeout_occurred
                ) AS recent_timeouts,
                bool_or(
                    source_type = 'channel'
                    AND created_at >= now() - %s * interval '1 day'
                ) AS used_channel_recently,
                bool_or(
                    source_type = 'playlist'
                    AND created_at >= now() - %s * interval '1 day'
                ) AS used_playlist_recently,
                max(created_at) AS last_job_at
            FROM public.jobs
            GROUP BY user_id
        ),
        purchase_stats AS (
            SELECT
                user_id,
                count(*) FILTER (WHERE status = 'completed') AS purchase_count,
                max(created_at) FILTER (WHERE status = 'completed') AS last_purchase_at,
                coalesce(sum(amount_cents) FILTER (
                    WHERE status = 'completed'
                ), 0) AS lifetime_amount_cents
            FROM public.purchases
            GROUP BY user_id
        )
        SELECT
            auth_user.id::text AS user_id,
            auth_user.email,
            auth_user.created_at AS account_created_at,
            auth_user.email_confirmed_at,
            auth_user.last_sign_in_at,
            credits.credits AS current_credits,
            credits.created_at AS credit_record_created_at,
            coalesce(job.lifetime_jobs, 0) AS lifetime_jobs,
            coalesce(job.recent_jobs, 0) AS recent_jobs,
            coalesce(job.lifetime_batch_credits, 0) AS lifetime_batch_credits,
            coalesce(job.recent_batch_credits, 0) AS recent_batch_credits,
            coalesce(job.recent_successes, 0) AS recent_successes,
            coalesce(job.recent_failures, 0) AS recent_failures,
            coalesce(job.recent_timeouts, 0) AS recent_timeouts,
            coalesce(job.used_channel_recently, false) AS used_channel_recently,
            coalesce(job.used_playlist_recently, false) AS used_playlist_recently,
            job.last_job_at,
            coalesce(purchase.purchase_count, 0) AS purchase_count,
            purchase.last_purchase_at,
            coalesce(purchase.lifetime_amount_cents, 0) AS lifetime_amount_cents
        FROM auth.users AS auth_user
        LEFT JOIN public.user_credits AS credits
            ON credits.user_id = auth_user.id
        LEFT JOIN job_stats AS job
            ON job.user_id = auth_user.id
        LEFT JOIN purchase_stats AS purchase
            ON purchase.user_id = auth_user.id
        WHERE auth_user.email IS NOT NULL
        ORDER BY auth_user.created_at DESC
        """,
        (DAYS, DAYS, DAYS, DAYS, DAYS, DAYS, DAYS),
    )

    for user in users:
        primary_segment, segment_tags = segment_user(user)
        user["primary_segment"] = primary_segment
        user["segment_tags"] = segment_tags

    users.sort(key=lambda row: (row["primary_segment"], row["email"].lower()))

    output_columns = [
        "primary_segment",
        "segment_tags",
        "email",
        "user_id",
        "account_created_at",
        "email_confirmed_at",
        "last_sign_in_at",
        "current_credits",
        "recent_jobs",
        "recent_batch_credits",
        "recent_successes",
        "recent_failures",
        "recent_timeouts",
        "used_channel_recently",
        "used_playlist_recently",
        "lifetime_jobs",
        "lifetime_batch_credits",
        "last_job_at",
        "purchase_count",
        "last_purchase_at",
        "lifetime_amount_cents",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        for user in users:
            writer.writerow({column: user[column] for column in output_columns})

    segment_counts = {}
    for user in users:
        segment = user["primary_segment"]
        segment_counts[segment] = segment_counts.get(segment, 0) + 1

    new_users = [user for user in users if user["account_created_at"] >= cutoff]
    new_batch_users = [user for user in new_users if user["recent_jobs"] > 0]
    new_batch_successes = [
        user for user in new_batch_users if user["recent_successes"] > 0
    ]
    likely_new_single_video_users = [
        user
        for user in new_users
        if user["lifetime_jobs"] == 0
        and user["current_credits"] is not None
        and 0 <= user["current_credits"] < FREE_CREDITS
    ]
    likely_new_no_attempt = [
        user
        for user in new_users
        if user["lifetime_jobs"] == 0 and user["current_credits"] == FREE_CREDITS
    ]
    active_batch_users = [user for user in users if user["recent_jobs"] > 0]
    active_batch_error_users = [
        user
        for user in active_batch_users
        if user["recent_failures"] > 0 or user["recent_timeouts"] > 0
    ]

    source_rows = fetch_all(
        cursor,
        """
        SELECT
            source_type,
            count(*) AS jobs,
            count(DISTINCT user_id) AS users,
            sum(completed) AS successful_transcripts,
            sum(failed_count) AS failed_attempts,
            sum(credits_used) AS credits_used
        FROM public.jobs
        WHERE created_at >= now() - %s * interval '1 day'
        GROUP BY source_type
        ORDER BY source_type
        """,
        (DAYS,),
    )

    monthly_rows = fetch_all(
        cursor,
        """
        SELECT
            to_char(date_trunc('month', created_at), 'YYYY-MM') AS month,
            count(*) AS jobs,
            count(DISTINCT user_id) AS users,
            sum(completed) AS successful_transcripts,
            sum(failed_count) AS failed_attempts
        FROM public.jobs
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 6
        """,
    )

    print(f"\nAnalysis window: {cutoff} to {clock['database_time']}")
    print(f"Email/user table written to: {OUTPUT_FILE.resolve()}")
    print(f"Rows written: {len(users)}\n")

    print("90-day funnel")
    print_table(
        ["Metric", "Users"],
        [
            ("New auth accounts", len(new_users)),
            ("Confirmed new channel/playlist users", len(new_batch_users)),
            ("New batch users with >=1 success", len(new_batch_successes)),
            (
                "Likely new single-video users (balance proxy)",
                len(likely_new_single_video_users),
            ),
            ("Likely new registered/no attempt", len(likely_new_no_attempt)),
            ("All active channel/playlist users", len(active_batch_users)),
            ("Active batch users with error/timeout", len(active_batch_error_users)),
        ],
    )

    print("\n90-day channel/playlist usage")
    print_table(
        [
            "Source",
            "Jobs",
            "Users",
            "Successful transcripts",
            "Failed attempts",
            "Credits used",
        ],
        [
            (
                row["source_type"],
                row["jobs"],
                row["users"],
                row["successful_transcripts"],
                row["failed_attempts"],
                row["credits_used"],
            )
            for row in source_rows
        ],
    )

    print("\nPrimary email segments")
    print_table(
        ["Segment", "Users"],
        sorted(segment_counts.items(), key=lambda item: (-item[1], item[0])),
    )

    print("\nRecent batch trend")
    print_table(
        ["Month", "Jobs", "Users", "Successes", "Failures"],
        [
            (
                row["month"],
                row["jobs"],
                row["users"],
                row["successful_transcripts"],
                row["failed_attempts"],
            )
            for row in reversed(monthly_rows)
        ],
    )

    print(
        "\nNot measurable from the current DB: single-video success/failure, "
        "downloads, pricing views, checkout starts, reliable purchases, "
        "landing pages, search queries, and countries."
    )
    print(
        "Important: *_proxy segments are hypotheses for customer interviews, "
        "not exact behavioral facts."
    )

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
