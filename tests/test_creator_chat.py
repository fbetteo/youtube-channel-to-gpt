import asyncio
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import creator_chat
import transcript_api
from rate_limiter import RateLimiter


def test_build_chunks_groups_by_time_and_keeps_timestamps():
    segments = [
        {"text": "Welcome back.", "start": 0.0, "duration": 3},
        {"text": "Today:\nideas", "start": 40.0, "duration": 5},
        {"text": "  ", "start": 60.0, "duration": 2},
        {"text": "First idea.", "start": 95.0, "duration": 4},
        {"text": "Second idea.", "start": 150.0, "duration": 6},
        {"text": "Thanks.", "start": 200.0, "duration": 2},
    ]

    assert creator_chat.build_chunks(segments) == [
        {"index": 0, "start": 0.0, "end": 45.0, "text": "Welcome back. Today: ideas"},
        {"index": 1, "start": 95.0, "end": 156.0, "text": "First idea. Second idea."},
        {"index": 2, "start": 200.0, "end": 202.0, "text": "Thanks."},
    ]


def test_build_chunks_of_empty_transcript():
    assert creator_chat.build_chunks([{"text": "", "start": 0, "duration": 1}]) == []


def test_embedding_input_prefixes_title():
    assert creator_chat.embedding_input("My video", "chunk") == "My video\n\nchunk"
    assert creator_chat.embedding_input(None, "chunk") == "chunk"


def test_to_pgvector():
    assert creator_chat.to_pgvector([0.5, -1.0, 0.0]) == "[0.5,-1.0,0.0]"


# --- chat turn ---------------------------------------------------------------

CHUNKS = [
    {"video_id": "vid1", "video_title": "Pricing", "start_seconds": 75.4, "end_seconds": 160, "text": "Charge more.", "score": 0.03},
    {"video_id": "vid2", "video_title": "Offers", "start_seconds": 0.0, "end_seconds": 90, "text": "Add value.", "score": 0.02},
]


def _fake_openai(answer_parts):
    async def stream():
        for part in answer_parts:
            yield types.SimpleNamespace(
                usage=None,
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=part))],
            )
        yield types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=7), choices=[]
        )

    create = AsyncMock(side_effect=lambda **kwargs: stream())
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))), create


def _patch_turn(monkeypatch, answer_parts=("Charge ", "more [1].")):
    client, create = _fake_openai(answer_parts)
    monkeypatch.setattr(creator_chat, "_openai", lambda: client)
    monkeypatch.setattr(creator_chat, "rewrite_query", AsyncMock(return_value="pricing services"))
    monkeypatch.setattr(creator_chat, "embed", AsyncMock(return_value=([[0.1]], 3)))
    search = AsyncMock(return_value=CHUNKS)
    monkeypatch.setattr(creator_chat, "search", search)

    @asynccontextmanager
    async def no_db():
        yield None

    import db_youtube_transcripts.database as database
    monkeypatch.setattr(database, "get_db_connection", no_db)
    return create, search


def _run_turn(history):
    async def collect():
        creator = {"channel_id": "UC1", "name": "Alex"}
        return [e async for e in creator_chat.chat_events(creator, "and for services?", history)]

    return asyncio.run(collect())


def test_first_message_searches_as_typed_and_streams_cited_answer(monkeypatch):
    create, search = _patch_turn(monkeypatch)

    events = _run_turn([])

    assert search.await_args.args[2] == "and for services?"
    creator_chat.rewrite_query.assert_not_awaited()
    assert events == [
        ("sources", {"query": "and for services?", "sources": [
            {"n": 1, "video_id": "vid1", "title": "Pricing", "start_seconds": 75,
             "url": "https://www.youtube.com/watch?v=vid1&t=75s"},
            {"n": 2, "video_id": "vid2", "title": "Offers", "start_seconds": 0,
             "url": "https://www.youtube.com/watch?v=vid2&t=0s"},
        ]}),
        ("delta", {"text": "Charge "}),
        ("delta", {"text": "more [1]."}),
        ("done", {"usage": {"input_tokens": 100, "output_tokens": 7}}),
    ]
    system = create.await_args.kwargs["messages"][0]["content"]
    assert "[1] Pricing\nCharge more.\n\n[2] Offers\nAdd value." in system


def test_follow_up_searches_with_rewrite_and_drops_old_citations(monkeypatch):
    create, search = _patch_turn(monkeypatch)
    history = [
        {"role": "user", "content": "How do I price?"},
        {"role": "assistant", "content": "Raise prices [1][2]."},
    ]

    events = _run_turn(history)

    assert search.await_args.args[2] == "pricing services"
    assert events[0][1]["query"] == "pricing services"
    messages = create.await_args.kwargs["messages"]
    assert messages[1:] == [
        {"role": "user", "content": "How do I price?"},
        {"role": "assistant", "content": "Raise prices."},
        {"role": "user", "content": "and for services?"},
    ]


def test_history_keeps_only_recent_turns():
    history = [{"role": "user", "content": str(i)} for i in range(10)]

    assert [t["content"] for t in creator_chat.clean_history(history)] == ["4", "5", "6", "7", "8", "9"]


# --- HTTP endpoint ------------------------------------------------------------


def _post_chat(monkeypatch, n_requests=1, user=None, creator={"channel_id": "UC1", "name": "Alex"}, fail=False):
    monkeypatch.setattr(creator_chat, "is_configured", lambda: True)
    monkeypatch.setattr(creator_chat, "get_enabled_creator", AsyncMock(return_value=creator))
    monkeypatch.setattr(transcript_api, "creator_chat_limiter", RateLimiter(window_size=86400))
    monkeypatch.setattr(transcript_api.settings, "creator_chat_anonymous_daily", 2)
    monkeypatch.setattr(transcript_api.settings, "creator_chat_user_daily", 3)

    async def events(creator, message, history):
        yield "sources", {"query": message, "sources": []}
        if fail:
            raise RuntimeError("provider down")
        yield "delta", {"text": "Hi"}
        yield "done", {"usage": None}

    monkeypatch.setattr(creator_chat, "chat_events", events)
    if user:
        transcript_api.app.dependency_overrides[transcript_api.get_user_or_anonymous] = lambda: user

    async def send():
        transport = httpx.ASGITransport(app=transcript_api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return [
                await client.post("/creators/alex/chat", json={"message": "hello"})
                for _ in range(n_requests)
            ]

    try:
        return asyncio.run(send())
    finally:
        transcript_api.app.dependency_overrides.clear()


def test_chat_streams_sse_events_with_remaining_count(monkeypatch):
    [response] = _post_chat(monkeypatch)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: sources\ndata: {"query": "hello", "sources": []}\n\n'
        'event: delta\ndata: {"text": "Hi"}\n\n'
        'event: done\ndata: {"usage": null, "remaining": 1}\n\n'
    )


def test_anonymous_visitors_must_sign_in_after_the_free_messages(monkeypatch):
    responses = _post_chat(monkeypatch, n_requests=3)

    assert [r.status_code for r in responses] == [200, 200, 401]
    assert responses[2].json() == {"detail": "Sign in to keep chatting."}


def test_signed_in_users_get_a_higher_daily_cap(monkeypatch):
    user = {"is_authenticated": True, "user_id": "user-1"}
    responses = _post_chat(monkeypatch, n_requests=4, user=user)

    assert [r.status_code for r in responses] == [200, 200, 200, 429]


def test_unknown_creator_is_404(monkeypatch):
    responses = _post_chat(monkeypatch, n_requests=3, creator=None)

    assert [r.status_code for r in responses] == [404, 404, 404]


def test_failure_mid_answer_ends_with_error_event(monkeypatch):
    [response] = _post_chat(monkeypatch, fail=True)

    assert response.text.endswith(
        'event: error\ndata: {"detail": "Chat failed. Please try again."}\n\n'
    )
