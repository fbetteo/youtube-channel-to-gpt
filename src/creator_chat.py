"""
Creator chat: retrieval over one creator's video transcripts (RAG).

Captions from the shared transcript cache are grouped into ~90 s chunks that
keep their start time, embedded with OpenAI, and stored in transcript_chunks
(db_youtube_transcripts/migration_add_creator_chat.py). A question is matched
to one creator's chunks by meaning and by words (hybrid search).

Indexing (scripts/creators.py) and querying must use the same EMBEDDING_MODEL
and EMBEDDING_DIMENSIONS; changing either means re-indexing every creator.
"""

from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config_v2 import settings
from transcript_formats import normalize_caption_text

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512  # must match transcript_chunks.embedding vector(512)
CHUNK_SECONDS = 90
EMBEDDING_BATCH_SIZE = 256

_client: Optional[AsyncOpenAI] = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.summary_openai_api_key)
    return _client


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
