"""
Creator chat: retrieval over one creator's video transcripts (RAG).

Captions from the shared transcript cache are grouped into ~90 s chunks that
keep their start time, embedded with OpenAI, and stored in transcript_chunks
(db_youtube_transcripts/migration_add_creator_chat.py). A question is matched
to one creator's chunks by meaning and by words (hybrid search).

Indexing (scripts/creators.py) and querying must use the same EMBEDDING_MODEL
and EMBEDDING_DIMENSIONS; changing either means re-indexing every creator.

A chat turn: rewrite a follow-up into a standalone question, search, then
stream an answer that cites the numbered sources. The model cites numbers only; the server owns the sources'
titles and timestamps, so links cannot be invented.
"""

import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config_v2 import settings
from transcript_formats import normalize_caption_text

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512  # must match transcript_chunks.embedding vector(512)
CHUNK_SECONDS = 90
EMBEDDING_BATCH_SIZE = 256

CHAT_SOURCES = 8
HISTORY_MESSAGES = 6
ANSWER_MAX_TOKENS = 800
OPENAI_TIMEOUT_SECONDS = 30

_client: Optional[AsyncOpenAI] = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.summary_openai_api_key, timeout=OPENAI_TIMEOUT_SECONDS
        )
    return _client


def is_configured() -> bool:
    return bool(settings.summary_openai_api_key)


def build_chunks(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group caption segments into chunks of about CHUNK_SECONDS."""
    chunks: List[Dict[str, Any]] = []
    for segment in segments:
        text = normalize_caption_text(segment.get("text"))
        if not text:
            continue
        start = float(segment.get("start") or 0)
        end = start + float(segment.get("duration") or 0)
        if not chunks or start >= chunks[-1]["start"] + CHUNK_SECONDS:
            chunks.append({"start": start, "end": end, "parts": [text]})
        else:
            chunks[-1]["parts"].append(text)
            chunks[-1]["end"] = max(chunks[-1]["end"], end)
    return [
        {"index": i, "start": c["start"], "end": c["end"], "text": " ".join(c["parts"])}
        for i, c in enumerate(chunks)
    ]


def embedding_input(title: Optional[str], text: str) -> str:
    """The title gives a chunk context its own words may lack."""
    return f"{title}\n\n{text}" if title else text


async def embed(texts: List[str]) -> Tuple[List[List[float]], int]:
    """Embed texts in order. Returns (vectors, total tokens billed)."""
    vectors: List[List[float]] = []
    tokens = 0
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        response = await _openai().embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[i : i + EMBEDDING_BATCH_SIZE],
            dimensions=EMBEDDING_DIMENSIONS,
        )
        vectors.extend(item.embedding for item in response.data)
        tokens += response.usage.total_tokens
    return vectors, tokens


def to_pgvector(vector: List[float]) -> str:
    """pgvector's text input format; the query casts it with ::vector."""
    return "[" + ",".join(str(x) for x in vector) + "]"


# Hybrid search: the CANDIDATES best chunks by meaning (vector) and by words
# (keyword) are merged with reciprocal rank fusion, so a chunk ranked well by
# either, or decently by both, comes first. Keyword search ORs the question's
# words: requiring all of them matches almost nothing for a natural question.
RRF_K = 60
CANDIDATES = 50

_SEARCH_SQL = f"""
    WITH q AS (
        SELECT replace(plainto_tsquery('english', $3)::text, '&', '|')::tsquery AS query
    ),
    vec AS (
        SELECT video_id, chunk_index,
               row_number() OVER (ORDER BY embedding <=> $2::vector) AS r
        FROM transcript_chunks WHERE channel_id = $1
        ORDER BY embedding <=> $2::vector LIMIT {CANDIDATES}
    ),
    kw AS (
        SELECT video_id, chunk_index,
               row_number() OVER (ORDER BY ts_rank(search_text, q.query) DESC) AS r
        FROM transcript_chunks CROSS JOIN q
        WHERE channel_id = $1 AND search_text @@ q.query
        ORDER BY ts_rank(search_text, q.query) DESC LIMIT {CANDIDATES}
    ),
    fused AS (
        SELECT video_id, chunk_index, sum(1.0 / ({RRF_K} + r)) AS score
        FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) ranked
        GROUP BY video_id, chunk_index
    )
    SELECT c.video_id, c.video_title, c.start_seconds, c.end_seconds, c.text, f.score
    FROM fused f JOIN transcript_chunks c USING (video_id, chunk_index)
    ORDER BY f.score DESC
    LIMIT $4
"""


async def search(
    conn, channel_id: str, question: str, query_vector: List[float], limit: int = 8
) -> List[Dict[str, Any]]:
    """Best chunks of one creator for a question, best first."""
    rows = await conn.fetch(
        _SEARCH_SQL, channel_id, to_pgvector(query_vector), question, limit
    )
    return [dict(row) for row in rows]


async def list_creators(slug: Optional[str] = None) -> List[Dict[str, Any]]:
    """Enabled creators as the public API shows them; one when slug is given."""
    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT c.slug, c.name, c.handle, c.category, c.avatar_url, c.tagline,
                   count(DISTINCT t.video_id) AS video_count
            FROM creators c LEFT JOIN transcript_chunks t USING (channel_id)
            WHERE c.enabled AND ($1::text IS NULL OR c.slug = $1)
            GROUP BY c.channel_id
            ORDER BY c.category, c.name
            """,
            slug,
        )
    return [dict(row) for row in rows]


async def get_enabled_creator(slug: str) -> Optional[Dict[str, Any]]:
    from db_youtube_transcripts.database import get_db_connection

    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            "SELECT channel_id, name FROM creators WHERE slug = $1 AND enabled", slug
        )
    return dict(row) if row else None


def clean_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Last HISTORY_MESSAGES turns. Old [n] citations point at old sources, so drop them."""
    return [
        {"role": turn["role"], "content": re.sub(r"\s*\[\d+\]", "", turn["content"])}
        for turn in history[-HISTORY_MESSAGES:]
    ]


async def rewrite_query(
    creator_name: str, history: List[Dict[str, str]], message: str
) -> str:
    """
    The latest message as a standalone question, so a follow-up like "what
    about the second one?" searches for its topic. Generic words (the
    creator's name, "strategies", "tips") match every chunk and drowned the
    topic in the eval, so the prompt forbids adding them.
    """
    conversation = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    response = await _openai().chat.completions.create(
        model=settings.creator_chat_model,
        temperature=0,
        max_tokens=60,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Rewrite the user's latest message as a standalone question for "
                    f"searching {creator_name}'s video transcripts. Replace references "
                    f"to the conversation with what they refer to. Keep the user's own "
                    f"words otherwise. Do not add the creator's name or generic words "
                    f"like strategy, framework, tips, video or transcript. Reply with "
                    f"only the question."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation:\n{conversation or '(none)'}\n\nLatest message: {message}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip() or message


def to_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the client shows for each [n]: numbered from 1 in prompt order."""
    return [
        {
            "n": n,
            "video_id": c["video_id"],
            "title": c["video_title"],
            "start_seconds": int(c["start_seconds"]),
            "url": f"https://www.youtube.com/watch?v={c['video_id']}&t={int(c['start_seconds'])}s",
        }
        for n, c in enumerate(chunks, 1)
    ]


def build_messages(
    creator_name: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
    message: str,
) -> List[Dict[str, str]]:
    excerpts = "\n\n".join(
        f"[{n}] {c['video_title']}\n{c['text']}" for n, c in enumerate(chunks, 1)
    )
    system = (
        f"You answer questions about what {creator_name} says in their YouTube videos, "
        f"using only the numbered transcript excerpts below.\n"
        f"- Talk about {creator_name} in the third person; never speak as them.\n"
        f"- After each claim, cite the excerpts it comes from, like [2] or [1][4].\n"
        f"- If the excerpts don't answer the question, say so in one sentence and "
        f"mention what they do cover. Never fill gaps from general knowledge.\n"
        f"- Be concise: a short answer or a few bullets, unless asked for detail. "
        f"Use Markdown.\n"
        f"- Captions are auto-generated, so names and numbers may be misspelled.\n\n"
        f"Excerpts:\n\n{excerpts}"
    )
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": message}]


async def chat_events(
    creator: Dict[str, Any], message: str, history: List[Dict[str, str]]
) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
    """One chat turn as (event, data) pairs: sources, delta..., done."""
    from db_youtube_transcripts.database import get_db_connection

    history = clean_history(history)
    # A first message is already standalone; rewriting it only adds latency.
    query = await rewrite_query(creator["name"], history, message) if history else message
    [vector], _ = await embed([query])
    async with get_db_connection() as conn:
        chunks = await search(conn, creator["channel_id"], query, vector, CHAT_SOURCES)
    yield "sources", {"query": query, "sources": to_sources(chunks)}

    stream = await _openai().chat.completions.create(
        model=settings.creator_chat_model,
        max_tokens=ANSWER_MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        messages=build_messages(creator["name"], chunks, history, message),
    )
    usage = None
    async for part in stream:
        if part.usage:
            usage = {
                "input_tokens": part.usage.prompt_tokens,
                "output_tokens": part.usage.completion_tokens,
            }
        if part.choices and part.choices[0].delta.content:
            yield "delta", {"text": part.choices[0].delta.content}
    yield "done", {"usage": usage}
