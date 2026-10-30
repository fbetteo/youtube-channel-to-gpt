"""
Manage creator chat creators and their transcript index (src/creator_chat.py).

    poetry run python scripts/creators.py add @marc-lou --category indie-hacking
    poetry run python scripts/creators.py index marc-lou [--limit 5] [--reindex]
    poetry run python scripts/creators.py search marc-lou "how do you find ideas?"
    poetry run python scripts/creators.py eval [marc-lou] [-v]

add      Look up a channel and upsert its creators row (enabled stays false).
index    Chunk and embed every cached transcript of the creator's long-form
         videos and livestreams (Shorts are skipped: mostly clips of longer
         videos). Videos already indexed with the current model are skipped,
         so a stopped run resumes. Only reads the transcript cache: videos
         without a cached transcript are reported, never fetched.
search   Embed a question and print the best chunks, to check retrieval.
eval     Score search against scripts/creator_chat_eval.json: questions with
         the videos (judged from titles) a good answer should come from.
         Run it before and after any change to chunking or search.

Needs the DB_*, AWS/S3 and SUMMARY_OPENAI_API_KEY settings in .env.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import creator_chat  # noqa: E402
import transcript_cache  # noqa: E402
import youtube_service  # noqa: E402
from db_youtube_transcripts.database import close_db_pool, get_db_connection  # noqa: E402

# text-embedding-3-small, USD per 1M tokens
EMBEDDING_PRICE_PER_MILLION = 0.02
INDEX_CONCURRENCY = 5
EVAL_FILE = Path(__file__).with_name("creator_chat_eval.json")


async def get_creator(conn, slug):
    creator = await conn.fetchrow("SELECT * FROM creators WHERE slug = $1", slug)
    if not creator:
        sys.exit(f"No creator with slug '{slug}'. Run: creators.py add <handle> --category ...")
    return creator


async def cmd_add(args):
    info = await youtube_service.get_channel_info(args.handle)
    handle = args.handle if args.handle.startswith("@") else f"@{args.handle}"
    slug = args.slug or handle.lstrip("@").lower()
    async with get_db_connection() as conn:
        await conn.execute(
            """
            INSERT INTO creators (channel_id, slug, name, handle, category, avatar_url, tagline)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (channel_id) DO UPDATE SET
                slug = EXCLUDED.slug, name = EXCLUDED.name, handle = EXCLUDED.handle,
                category = EXCLUDED.category, avatar_url = EXCLUDED.avatar_url,
                tagline = COALESCE(EXCLUDED.tagline, creators.tagline)
            """,
            info["channelId"],
            slug,
            info["title"],
            handle,
            args.category,
            info["thumbnail"] or None,
            args.tagline,
        )
    print(f"Saved {info['title']} ({info['channelId']}) as '{slug}' in {args.category}")
    print(f"Avatar: {info['thumbnail']}")


def published_at(video):
    try:
        return datetime.fromisoformat(video["publishedAt"])
    except (KeyError, TypeError, ValueError):
        return None


async def index_video(video, language_key, channel_id):
    """Chunk, embed and store one video. Returns (chunks, tokens), (0, 0) if unusable."""
    document = await transcript_cache.load(
        video["id"], None if language_key == "auto" else language_key
    )
    if not document:
        return 0, 0
    chunks = creator_chat.build_chunks(document["segments"])
    if not chunks:
        return 0, 0
    vectors, tokens = await creator_chat.embed(
        [creator_chat.embedding_input(video["title"], c["text"]) for c in chunks]
    )
    rows = [
        (
            video["id"],
            c["index"],
            channel_id,
            video["title"],
            published_at(video),
            c["start"],
            c["end"],
            c["text"],
            creator_chat.to_pgvector(vector),
            creator_chat.EMBEDDING_MODEL,
        )
        for c, vector in zip(chunks, vectors)
    ]
    async with get_db_connection() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM transcript_chunks WHERE video_id = $1", video["id"])
            await conn.executemany(
                """
                INSERT INTO transcript_chunks
                    (video_id, chunk_index, channel_id, video_title, published_at,
                     start_seconds, end_seconds, text, embedding, embedding_model)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10)
                """,
                rows,
            )
    return len(chunks), tokens


async def cmd_index(args):
    async with get_db_connection() as conn:
        creator = await get_creator(conn, args.slug)
    channel_id = creator["channel_id"]

    videos = await youtube_service.get_all_channel_videos(channel_id)
    videos = [v for v in videos if v["type"] != "short"]
    ids = [v["id"] for v in videos]

    async with get_db_connection() as conn:
        # Prefer the 'auto' entry when a video was cached under several languages.
        cached = {
            r["video_id"]: r["requested_language"]
            for r in await conn.fetch(
                """
                SELECT DISTINCT ON (video_id) video_id, requested_language
                FROM transcript_cache WHERE video_id = ANY($1)
                ORDER BY video_id, (requested_language = 'auto') DESC
                """,
                ids,
            )
        }
        indexed = {
            r["video_id"]
            for r in await conn.fetch(
                """
                SELECT DISTINCT video_id FROM transcript_chunks
                WHERE channel_id = $1 AND embedding_model = $2
                """,
                channel_id,
                creator_chat.EMBEDDING_MODEL,
            )
        }

    todo = [v for v in videos if v["id"] in cached and (args.reindex or v["id"] not in indexed)]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{creator['name']}: {len(videos)} long-form videos, {len(cached)} cached, "
        f"{len(indexed)} already indexed, {len(todo)} to index"
    )

    semaphore = asyncio.Semaphore(INDEX_CONCURRENCY)
    totals = {"videos": 0, "chunks": 0, "tokens": 0, "failed": 0}

    async def run(video):
        async with semaphore:
            try:
                chunks, tokens = await index_video(video, cached[video["id"]], channel_id)
            except Exception as e:
                totals["failed"] += 1
                print(f"  FAILED {video['id']}: {e}")
                return
            if chunks:
                totals["videos"] += 1
                totals["chunks"] += chunks
                totals["tokens"] += tokens
            done = totals["videos"] + totals["failed"]
            print(f"  [{done}/{len(todo)}] {video['id']} {chunks} chunks  {video['title'][:60]}")

    await asyncio.gather(*(run(v) for v in todo))
    cost = totals["tokens"] / 1_000_000 * EMBEDDING_PRICE_PER_MILLION
    print(
        f"Indexed {totals['videos']} videos, {totals['chunks']} chunks, "
        f"{totals['tokens']:,} tokens (~${cost:.4f}), {totals['failed']} failed"
    )


def timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


async def cmd_search(args):
    async with get_db_connection() as conn:
        creator = await get_creator(conn, args.slug)
        [vector], _ = await creator_chat.embed([args.question])
        results = await creator_chat.search(
            conn, creator["channel_id"], args.question, vector, args.k
        )
    for r in results:
        print(
            f"\n{r['score']:.3f}  {r['video_title']}  @ {timestamp(r['start_seconds'])}\n"
            f"       https://www.youtube.com/watch?v={r['video_id']}&t={int(r['start_seconds'])}s\n"
            f"       {r['text'][:300]}"
        )


async def cmd_eval(args):
    """Hit rate and MRR of the top-k chunks against scripts/creator_chat_eval.json."""
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    slugs = [args.slug] if args.slug else list(cases)
    hits, reciprocal_ranks = [], []
    for slug in slugs:
        async with get_db_connection() as conn:
            creator = await get_creator(conn, slug)
            questions = [case["question"] for case in cases[slug]]
            vectors, _ = await creator_chat.embed(questions)
            for case, vector in zip(cases[slug], vectors):
                results = await creator_chat.search(
                    conn, creator["channel_id"], case["question"], vector, args.k
                )
                rank = next(
                    (i for i, r in enumerate(results, 1) if r["video_id"] in case["relevant"]),
                    None,
                )
                hits.append(rank is not None)
                reciprocal_ranks.append(1 / rank if rank else 0)
                if args.verbose:
                    print(f"  {slug:<10} {str(rank or '-'):>2}  {case['question']}")
    n = len(hits)
    print(f"hit@{args.k} {sum(hits)}/{n} = {sum(hits) / n:.2f}   MRR {sum(reciprocal_ranks) / n:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)

    add = sub.add_parser("add")
    add.add_argument("handle", help="@handle or channel ID")
    add.add_argument("--category", required=True)
    add.add_argument("--slug", help="defaults to the handle without @")
    add.add_argument("--tagline")
    add.set_defaults(fn=cmd_add)

    index = sub.add_parser("index")
    index.add_argument("slug")
    index.add_argument("--limit", type=int, help="index at most N videos, newest first")
    index.add_argument("--reindex", action="store_true", help="redo already indexed videos")
    index.set_defaults(fn=cmd_index)

    search = sub.add_parser("search")
    search.add_argument("slug")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=8)
    search.set_defaults(fn=cmd_search)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("slug", nargs="?", help="defaults to every creator in the eval file")
    evaluate.add_argument("-k", type=int, default=8)
    evaluate.add_argument("-v", "--verbose", action="store_true", help="rank per question")
    evaluate.set_defaults(fn=cmd_eval)

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    async def run():
        try:
            await args.fn(args)
        finally:
            await close_db_pool()

    asyncio.run(run())


if __name__ == "__main__":
    main()
