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
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
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

LOGGER = logging.getLogger("published_at_backfill")


@dataclass
class Progress:
    total_target: int
    pages_done: int = 0
    processed: int = 0
    resolved: int = 0
    unresolved: int = 0
    updated_rows: int = 0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        rate = self.processed / elapsed
        remaining = max(self.total_target - self.processed, 0)
        eta_seconds = (remaining / rate) if rate > 0 else None
        return {
            "elapsed": elapsed,
            "rate": rate,
            "remaining": remaining,
            "eta_seconds": eta_seconds,
        }


@dataclass
class CheckpointState:
    days: int
    limit: Optional[int]
    last_video_id: Optional[str]
    processed_target: int
    total_target: int
    pages_done: int
    processed: int
    resolved: int
    unresolved: int
    updated_rows: int

    @classmethod
    def from_progress(
        cls,
        *,
        days: int,
        limit: Optional[int],
        last_video_id: Optional[str],
        processed_target: int,
        progress: Progress,
    ) -> "CheckpointState":
        return cls(
            days=days,
            limit=limit,
            last_video_id=last_video_id,
            processed_target=processed_target,
            total_target=progress.total_target,
            pages_done=progress.pages_done,
            processed=progress.processed,
            resolved=progress.resolved,
            unresolved=progress.unresolved,
            updated_rows=progress.updated_rows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "limit": self.limit,
            "last_video_id": self.last_video_id,
            "processed_target": self.processed_target,
            "total_target": self.total_target,
            "pages_done": self.pages_done,
            "processed": self.processed,
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "updated_rows": self.updated_rows,
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }


def load_checkpoint(checkpoint_path: Path) -> Optional[CheckpointState]:
    if not checkpoint_path.exists():
        return None

    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return CheckpointState(
        days=int(raw["days"]),
        limit=(int(raw["limit"]) if raw.get("limit") is not None else None),
        last_video_id=raw.get("last_video_id"),
        processed_target=int(raw.get("processed_target", 0)),
        total_target=int(raw.get("total_target", 0)),
        pages_done=int(raw.get("pages_done", 0)),
        processed=int(raw.get("processed", 0)),
        resolved=int(raw.get("resolved", 0)),
        unresolved=int(raw.get("unresolved", 0)),
        updated_rows=int(raw.get("updated_rows", 0)),
    )


def save_checkpoint(checkpoint_path: Path, state: CheckpointState) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def proxy_url() -> str:
    u = os.environ["WEBSHARE_PROXY_USERNAME"]
    p = os.environ["WEBSHARE_PROXY_PASSWORD"]
    return f"http://{quote(u)}-rotate:{quote(p)}@p.webshare.io:80"


async def get_innertube_config() -> tuple[str, str]:
    """Fetch Innertube API key and WEB client version from homepage."""
    LOGGER.info("Fetching YouTube homepage (direct, no proxy) for Innertube config...")
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=30, follow_redirects=True
    ) as client:
        home = await client.get("https://www.youtube.com")
        home.raise_for_status()

    api_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', home.text)
    version_match = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', home.text)
    if not api_match or not version_match:
        raise RuntimeError(
            "Failed to extract INNERTUBE_API_KEY or INNERTUBE_CLIENT_VERSION"
        )

    api_key = api_match.group(1)
    client_version = version_match.group(1)
    LOGGER.info(
        "Innertube config ready: api_key=%s... client_version=%s",
        api_key[:8],
        client_version,
    )
    return api_key, client_version


def _extract_publish_date(payload: dict[str, Any]) -> Optional[str]:
    return (
        payload.get("microformat", {})
        .get("playerMicroformatRenderer", {})
        .get("publishDate")
    )


async def _post_player(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.ProxyError, httpx.HTTPStatusError, httpx.ReadTimeout) as exc:
            if attempt >= retries:
                raise
            await asyncio.sleep(0.75 * attempt)
            LOGGER.debug(
                "retrying post_player attempt=%s/%s due to %s", attempt, retries, exc
            )

    raise RuntimeError("Unreachable retry loop")


async def fetch_publish_date_for_video(
    video_id: str,
    client: httpx.AsyncClient,
    player_url: str,
    client_version: str,
    retries: int,
) -> Optional[str]:
    payload_web = {
        "context": {
            "client": {"clientName": "WEB", "clientVersion": client_version},
        },
        "videoId": video_id,
    }
    data = await _post_player(client, player_url, payload_web, retries=retries)
    publish_date = _extract_publish_date(data)
    if publish_date:
        return publish_date

    status = data.get("playabilityStatus", {}).get("status")
    if status != "LOGIN_REQUIRED":
        return None

    payload_embed = {
        "context": {
            "client": {
                "clientName": "TVHTML5_SIMPLY_EMBEDDED_PLAYER",
                "clientVersion": "2.0",
            },
            "thirdParty": {"embedUrl": "https://www.youtube.com/"},
        },
        "videoId": video_id,
    }
    data = await _post_player(client, player_url, payload_embed, retries=retries)
    return _extract_publish_date(data)


async def count_target_video_ids(days: int, limit: Optional[int] = None) -> int:
    """Count DISTINCT video IDs with missing published_at from recent rows."""
    async with get_db_connection()() as conn:
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT video_id)
            FROM job_videos
            WHERE published_at IS NULL
              AND video_id IS NOT NULL
              AND video_id <> ''
              AND COALESCE(processed_at, created_at) >= NOW() - make_interval(days => $1)
            """,
            days,
        )
    total_int = int(total or 0)
    if limit and limit > 0:
        return min(total_int, limit)
    return total_int


async def fetch_target_video_ids_page(
    days: int,
    last_video_id: Optional[str],
    page_size: int,
    remaining_limit: Optional[int],
) -> list[str]:
    """Fetch one page of distinct target video IDs using keyset pagination."""
    fetch_size = page_size
    if remaining_limit is not None:
        fetch_size = min(fetch_size, remaining_limit)

    if fetch_size <= 0:
        return []

    async with get_db_connection()() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT video_id
            FROM job_videos
            WHERE published_at IS NULL
              AND video_id IS NOT NULL
              AND video_id <> ''
              AND COALESCE(processed_at, created_at) >= NOW() - make_interval(days => $3)
              AND ($2::TEXT IS NULL OR video_id > $2)
            ORDER BY video_id
            LIMIT $1
            """,
            fetch_size,
            last_video_id,
            days,
        )
    return [row["video_id"] for row in rows]


async def update_published_at_batch(
    updates: list[tuple[str, datetime]],
    days: int,
) -> int:
    """Batch update published_at for multiple video IDs; returns rows updated."""
    if not updates:
        return 0

    video_ids = [item[0] for item in updates]
    published_ats = [item[1] for item in updates]

    async with get_db_connection()() as conn:
        updated = await conn.fetchval(
            """
            WITH payload AS (
                SELECT *
                FROM unnest($1::TEXT[], $2::TIMESTAMP[]) AS x(video_id, published_at)
            ), upd AS (
                UPDATE job_videos j
                SET published_at = p.published_at,
                    updated_at = NOW()
                FROM payload p
                WHERE j.video_id = p.video_id
                  AND j.published_at IS NULL
                  AND COALESCE(j.processed_at, j.created_at) >= NOW() - make_interval(days => $3)
                RETURNING 1
            )
            SELECT COUNT(*) FROM upd
            """,
            video_ids,
            published_ats,
            days,
        )
    return int(updated or 0)


async def process_video_page(
    video_ids: list[str],
    player_url: str,
    client_version: str,
    concurrency: int,
    retries: int,
    progress: Progress,
) -> list[tuple[str, datetime]]:
    """Resolve publish dates for one page using bounded worker concurrency."""
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    for video_id in video_ids:
        queue.put_nowait(video_id)
    for _ in range(concurrency):
        queue.put_nowait(None)

    resolved_updates: list[tuple[str, datetime]] = []
    lock = asyncio.Lock()

    timeout = httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=30.0)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 20), max_keepalive_connections=concurrency
    )

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=timeout,
        proxy=proxy_url(),
        follow_redirects=True,
        limits=limits,
    ) as client:

        async def worker() -> None:
            while True:
                video_id = await queue.get()
                if video_id is None:
                    queue.task_done()
                    return

                try:
                    raw_date = await fetch_publish_date_for_video(
                        video_id=video_id,
                        client=client,
                        player_url=player_url,
                        client_version=client_version,
                        retries=retries,
                    )
                    async with lock:
                        progress.processed += 1
                        if raw_date:
                            progress.resolved += 1
                            resolved_updates.append(
                                (
                                    video_id,
                                    datetime.fromisoformat(raw_date).replace(
                                        tzinfo=None
                                    ),
                                )
                            )
                        else:
                            progress.unresolved += 1
                except Exception as exc:
                    async with lock:
                        progress.processed += 1
                        progress.unresolved += 1
                    LOGGER.debug("%s -> error: %s", video_id, exc)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        await asyncio.gather(*workers)

    return resolved_updates


def _fmt_duration(seconds: float) -> str:
    seconds_int = max(int(seconds), 0)
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


async def progress_reporter(
    progress: Progress, interval_seconds: float, stop_event: asyncio.Event
) -> None:
    while not stop_event.is_set():
        snap = progress.snapshot()
        eta = (
            _fmt_duration(snap["eta_seconds"])
            if snap["eta_seconds"] is not None
            else "n/a"
        )
        LOGGER.info(
            "progress processed=%s/%s resolved=%s unresolved=%s updated_rows=%s pages=%s rate=%.1f vid/s elapsed=%s eta=%s",
            progress.processed,
            progress.total_target,
            progress.resolved,
            progress.unresolved,
            progress.updated_rows,
            progress.pages_done,
            snap["rate"],
            _fmt_duration(snap["elapsed"]),
            eta,
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


async def run_backfill(
    days: int,
    concurrency: int,
    limit: Optional[int],
    db_page_size: int,
    update_batch_size: int,
    retries: int,
    progress_interval: float,
    checkpoint_path: Path,
    resume: bool,
    checkpoint_every_pages: int,
    dry_run: bool,
) -> None:
    checkpoint_state: Optional[CheckpointState] = None
    if resume:
        checkpoint_state = load_checkpoint(checkpoint_path)
        if checkpoint_state:
            if checkpoint_state.days != days or checkpoint_state.limit != limit:
                raise ValueError(
                    "Checkpoint parameters mismatch. Use the same --days/--limit as the checkpoint run."
                )
            LOGGER.info("Resuming from checkpoint: %s", checkpoint_path)

    total_target = (
        checkpoint_state.total_target
        if checkpoint_state and checkpoint_state.total_target > 0
        else await count_target_video_ids(days=days, limit=limit)
    )
    if total_target == 0:
        LOGGER.info("No rows with missing published_at found")
        return

    LOGGER.info("Starting backfill for %s distinct video_ids", total_target)
    api_key, client_version = await get_innertube_config()
    player_url = f"https://www.youtube.com/youtubei/v1/player?key={api_key}"

    progress = Progress(total_target=total_target)
    processed_target = 0
    last_video_id: Optional[str] = None

    if checkpoint_state:
        progress.pages_done = checkpoint_state.pages_done
        progress.processed = checkpoint_state.processed
        progress.resolved = checkpoint_state.resolved
        progress.unresolved = checkpoint_state.unresolved
        progress.updated_rows = checkpoint_state.updated_rows
        processed_target = checkpoint_state.processed_target
        last_video_id = checkpoint_state.last_video_id

    stop_event = asyncio.Event()
    reporter_task = asyncio.create_task(
        progress_reporter(
            progress, interval_seconds=progress_interval, stop_event=stop_event
        )
    )

    try:
        while True:
            remaining_limit = None
            if limit and limit > 0:
                remaining_limit = max(limit - processed_target, 0)
                if remaining_limit == 0:
                    break

            page_ids = await fetch_target_video_ids_page(
                days=days,
                last_video_id=last_video_id,
                page_size=db_page_size,
                remaining_limit=remaining_limit,
            )
            if not page_ids:
                break

            last_video_id = page_ids[-1]
            processed_target += len(page_ids)

            resolved_updates = await process_video_page(
                video_ids=page_ids,
                player_url=player_url,
                client_version=client_version,
                concurrency=concurrency,
                retries=retries,
                progress=progress,
            )

            progress.pages_done += 1
            if dry_run:
                if checkpoint_every_pages > 0 and (
                    progress.pages_done % checkpoint_every_pages == 0
                ):
                    save_checkpoint(
                        checkpoint_path,
                        CheckpointState.from_progress(
                            days=days,
                            limit=limit,
                            last_video_id=last_video_id,
                            processed_target=processed_target,
                            progress=progress,
                        ),
                    )
                continue

            for idx in range(0, len(resolved_updates), update_batch_size):
                chunk = resolved_updates[idx : idx + update_batch_size]
                updated = await update_published_at_batch(chunk, days=days)
                progress.updated_rows += updated

            if checkpoint_every_pages > 0 and (
                progress.pages_done % checkpoint_every_pages == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    CheckpointState.from_progress(
                        days=days,
                        limit=limit,
                        last_video_id=last_video_id,
                        processed_target=processed_target,
                        progress=progress,
                    ),
                )

        save_checkpoint(
            checkpoint_path,
            CheckpointState.from_progress(
                days=days,
                limit=limit,
                last_video_id=last_video_id,
                processed_target=processed_target,
                progress=progress,
            ),
        )
    finally:
        stop_event.set()
        await reporter_task

    summary = progress.snapshot()
    LOGGER.info(
        "done processed=%s resolved=%s unresolved=%s updated_rows=%s pages=%s elapsed=%s avg_rate=%.1f vid/s",
        progress.processed,
        progress.resolved,
        progress.unresolved,
        progress.updated_rows,
        progress.pages_done,
        _fmt_duration(summary["elapsed"]),
        summary["rate"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db-page-size", type=int, default=1000)
    parser.add_argument("--update-batch-size", type=int, default=500)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("build/transcripts/backfill_published_at_checkpoint.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-pages", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        await run_backfill(
            days=args.days,
            concurrency=args.concurrency,
            limit=args.limit,
            db_page_size=args.db_page_size,
            update_batch_size=args.update_batch_size,
            retries=args.retries,
            progress_interval=args.progress_interval,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
            checkpoint_every_pages=args.checkpoint_every_pages,
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()()


if __name__ == "__main__":
    asyncio.run(async_main())
