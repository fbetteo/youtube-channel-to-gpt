#!/usr/bin/env python3
"""
Backfill missing job_videos.published_at using YouTube Innertube player API.

Usage:
    python scripts/backfill_published_at_innertube.py
    python scripts/backfill_published_at_innertube.py --days 60 --concurrency 100
    python scripts/backfill_published_at_innertube.py --limit 2000 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx

# from rich.json import args

# Ensure project root imports work when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_db_connection():
    from db_youtube_transcripts.database import get_db_connection

    return get_db_connection


def close_pool():
    from db_youtube_transcripts.database import close_db_pool

    return close_db_pool


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def proxy_url():
    u = os.environ["WEBSHARE_PROXY_USERNAME"]
    p = os.environ["WEBSHARE_PROXY_PASSWORD"]
    return f"http://{quote(u)}-rotate:{quote(p)}@p.webshare.io:80"


async def get_publish_dates(video_ids, concurrency=50, retries=3):
    for attempt in range(1, retries + 1):
        try:
            return await _get_publish_dates(video_ids, concurrency)
        except (httpx.ProxyError, httpx.HTTPStatusError) as e:
            print(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(2 * attempt)
    raise RuntimeError(f"All {retries} attempts failed")


async def _get_publish_dates(video_ids, concurrency=50):
    # Fetch homepage directly (no proxy needed, just extracting API key)
    print("[1/3] Fetching YouTube homepage (direct, no proxy)...")
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=30, follow_redirects=True
    ) as direct:
        home = await direct.get("https://www.youtube.com")
        api_key = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', home.text).group(1)
        client_version = re.search(
            r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', home.text
        ).group(1)
    print(f"[1/3] Got API key={api_key[:8]}... client_version={client_version}")

    # Use per-video proxy sessions (each video gets a different IP)
    print(
        f"[2/3] Fetching {len(video_ids)} videos via proxy (concurrency={concurrency})..."
    )
    url = f"https://www.youtube.com/youtubei/v1/player?key={api_key}"
    sem = asyncio.Semaphore(concurrency)
    done = 0

    def _extract_date(data):
        return (
            data.get("microformat", {})
            .get("playerMicroformatRenderer", {})
            .get("publishDate")
        )

    async def fetch(vid, session_id):
        nonlocal done
        try:
            async with httpx.AsyncClient(
                headers=HEADERS,
                timeout=30,
                proxy=proxy_url(),
                follow_redirects=True,
            ) as client:
                # Try WEB client first
                payload = {
                    "context": {
                        "client": {"clientName": "WEB", "clientVersion": client_version}
                    },
                    "videoId": vid,
                }
                async with sem:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    data = r.json()

            date = _extract_date(data)
            if date:
                return vid, date

            # Fallback for age-restricted / LOGIN_REQUIRED videos
            status = data.get("playabilityStatus", {}).get("status")
            if status == "LOGIN_REQUIRED":
                payload = {
                    "context": {
                        "client": {
                            "clientName": "TVHTML5_SIMPLY_EMBEDDED_PLAYER",
                            "clientVersion": "2.0",
                        },
                        "thirdParty": {"embedUrl": "https://www.youtube.com/"},
                    },
                    "videoId": vid,
                }
                async with sem:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    data = r.json()
                date = _extract_date(data)
                if date:
                    return vid, date
                status = data.get("playabilityStatus", {}).get("status")

            if not date:
                print(f"{vid} -> no publishDate (status={status})")
            done += 1
            if done % 10 == 0 or done == len(video_ids):
                print(f"  [{done}/{len(video_ids)}] fetched")
            return vid, date
        except Exception as e:
            done += 1
            print(f"{vid} -> ERROR (session={session_id}): {type(e).__name__}: {e}")
            return vid, None

    results = await asyncio.gather(*[fetch(v, i) for i, v in enumerate(video_ids)])
    print(f"[3/3] Done. {sum(1 for _, d in results if d)}/{len(results)} resolved")
    return dict(results)


async def fetch_target_video_ids(days, limit=None):
    """Get DISTINCT video IDs with missing published_at from recent rows."""
    async with get_db_connection()() as conn:
        query = """
            SELECT DISTINCT video_id
            FROM job_videos
            WHERE published_at IS NULL
              AND video_id IS NOT NULL
              AND video_id <> ''
              AND COALESCE(processed_at, created_at) >= NOW() - make_interval(days => $1)
            ORDER BY video_id
        """
        if limit and limit > 0:
            query += " LIMIT $2"
            rows = await conn.fetch(query, days, limit)
        else:
            rows = await conn.fetch(query, days)
    return [row["video_id"] for row in rows]


async def update_published_at(video_id, published_at, days):
    """Update recent rows for one video_id where published_at is still NULL."""
    async with get_db_connection()() as conn:
        result = await conn.execute(
            """
            UPDATE job_videos
            SET published_at = $2, updated_at = NOW()
            WHERE video_id = $1
              AND published_at IS NULL
              AND COALESCE(processed_at, created_at) >= NOW() - make_interval(days => $3)
            """,
            video_id,
            published_at,
            days,
        )
    return int(result.split()[-1])


async def run_backfill(days, concurrency, limit, dry_run):
    video_ids = await fetch_target_video_ids(days=days, limit=limit)
    if not video_ids:
        print("No rows with missing published_at found")
        return

    print(f"Found {len(video_ids)} video_ids to backfill")
    dates = await get_publish_dates(video_ids, concurrency=concurrency)

    resolved = {vid: d for vid, d in dates.items() if d}
    # print(resolved)
    print(f"Resolved {len(resolved)}/{len(video_ids)} publish dates")

    if dry_run:
        print("Dry run — no DB updates")
        return

    updated_rows = 0
    for vid, raw_date in resolved.items():
        parsed = datetime.fromisoformat(raw_date).replace(tzinfo=None)
        n = await update_published_at(vid, parsed, days)
        updated_rows += n

    print(f"Updated {updated_rows} rows")


# async def async_main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--days", type=int, default=60)
#     parser.add_argument("--concurrency", type=int, default=100)
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--dry-run", action="store_true")
#     args = parser.parse_args()

#     try:
#         await run_backfill(args.days, args.concurrency, args.limit, args.dry_run)
#     finally:
#         await close_pool()()


# if __name__ == "__main__":
#     asyncio.run(async_main())


try:
    await run_backfill(60, 50, None, False)
finally:
    await close_pool()()
