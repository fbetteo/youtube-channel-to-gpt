import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault(
    "yt_dlp",
    types.SimpleNamespace(
        YoutubeDL=object, utils=types.SimpleNamespace(DownloadError=RuntimeError)
    ),
)

import summary_service
import transcript_cache
from routers import developer_api
from test_transcript_segments_cache_and_formats import FakeS3, load_lambda_module

SEGMENTS = [{"text": "hello world", "start": 0.0, "duration": 2.0}]
METADATA = {"video_id": "vid", "transcript_language": "en", "transcript_type": "manual"}
API_KEY = {"user_id": "user-1", "key_id": "key-1"}


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(
        transcript_cache.youtube_service, "get_s3_client", lambda: (fake, "bucket")
    )
    index = AsyncMock()
    monkeypatch.setattr(transcript_cache, "_index", index)
    fake.index = index
    return fake


def test_cache_format_matches_the_worker():
    lambda_module = load_lambda_module()
    for language in (None, "en", "EN", "pt-br"):
        assert transcript_cache.cache_key("vid", language) == (
            lambda_module.transcript_cache_key(
                "vid", lambda_module.normalize_preferred_language(language)
            )
        )
    ours = transcript_cache.build_document("vid", "EN", "en", False, SEGMENTS, 123)
    worker = lambda_module.build_segments_document(
        "vid", lambda_module.normalize_preferred_language("EN"), "en", False, SEGMENTS, 123
    )
    assert ours == worker


def test_load_reads_worker_written_objects_and_indexes_the_hit(s3):
    document = transcript_cache.build_document("vid", "en", "en", True, SEGMENTS, 123)
    s3.objects["transcript-cache/vid/en.json"] = json.dumps(document).encode()

    assert asyncio.run(transcript_cache.load("vid", "en")) == document
    s3.index.assert_awaited_once_with(document, hit=True)
    assert asyncio.run(transcript_cache.load("vid", None)) is None  # 'auto' is another key


def test_load_transcript_fetches_once_then_serves_from_cache(s3, monkeypatch):
    fetch = AsyncMock(return_value=(SEGMENTS, METADATA))
    monkeypatch.setattr(summary_service.youtube_service, "get_transcript_data", fetch)

    first = asyncio.run(summary_service.load_transcript("vid", None))
    second = asyncio.run(summary_service.load_transcript("vid", None))

    assert first == second == (SEGMENTS, METADATA)
    fetch.assert_awaited_once()
    assert "transcript-cache/vid/auto.json" in s3.objects
    s3.index.assert_any_await(
        json.loads(s3.objects["transcript-cache/vid/auto.json"]), source="single"
    )


def test_timed_out_fetch_keeps_running_and_fills_the_cache(s3, monkeypatch):
    async def slow_fetch(video_id, language):
        await asyncio.sleep(0.2)
        return SEGMENTS, METADATA

    monkeypatch.setattr(summary_service.youtube_service, "get_transcript_data", slow_fetch)

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            await summary_service.load_transcript_within("vid", "en", timeout=0.05)
        await asyncio.sleep(0.3)  # the background fetch finishes meanwhile

    asyncio.run(scenario())
    assert "transcript-cache/vid/en.json" in s3.objects


def _developer_mocks(monkeypatch, load):
    mocks = types.SimpleNamespace(
        reserve=AsyncMock(return_value="r1"),
        finalize=AsyncMock(),
        access=AsyncMock(),
    )
    monkeypatch.setattr(developer_api, "get_user_credits", AsyncMock(return_value=5))
    monkeypatch.setattr(developer_api, "reserve_credits", mocks.reserve)
    monkeypatch.setattr(developer_api, "finalize_credits", mocks.finalize)
    monkeypatch.setattr(developer_api, "increment_api_key_credits_used", AsyncMock())
    monkeypatch.setattr(summary_service, "record_paid_access", mocks.access)
    monkeypatch.setattr(summary_service, "load_transcript_within", load)
    return mocks


def test_developer_single_uses_shared_cache_and_charges_once(monkeypatch):
    load = AsyncMock(return_value=(SEGMENTS, METADATA))
    mocks = _developer_mocks(monkeypatch, load)
    body = developer_api.SingleTranscriptRequest(video_url="dQw4w9WgXcQ")

    response = asyncio.run(developer_api.get_single_transcript(body, API_KEY))

    assert response.transcript == "hello world"
    assert response.language == "en"
    assert response.title is None
    load.assert_awaited_once_with("dQw4w9WgXcQ", "en", 45)
    mocks.reserve.assert_awaited_once_with("user-1", 1)
    mocks.finalize.assert_not_awaited()
    mocks.access.assert_awaited_once_with("user-1", "dQw4w9WgXcQ")


def test_developer_single_timeout_returns_504_and_refunds(monkeypatch):
    mocks = _developer_mocks(monkeypatch, AsyncMock(side_effect=asyncio.TimeoutError))
    body = developer_api.SingleTranscriptRequest(video_url="dQw4w9WgXcQ")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(developer_api.get_single_transcript(body, API_KEY))

    assert exc_info.value.status_code == 504
    mocks.finalize.assert_awaited_once_with("user-1", "", 0, 1)
    mocks.access.assert_not_awaited()


def test_developer_single_invalid_url_charges_nothing(monkeypatch):
    mocks = _developer_mocks(monkeypatch, AsyncMock())
    body = developer_api.SingleTranscriptRequest(video_url="not a video")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(developer_api.get_single_transcript(body, API_KEY))

    assert exc_info.value.status_code == 400
    mocks.reserve.assert_not_awaited()
    mocks.finalize.assert_not_awaited()


def test_language_request_falls_back_to_matching_auto_entry(s3):
    english = transcript_cache.build_document("vid", None, "en", True, SEGMENTS, 123)
    s3.objects["transcript-cache/vid/auto.json"] = json.dumps(english).encode()

    assert asyncio.run(transcript_cache.load("vid", "en")) == english
    assert asyncio.run(transcript_cache.load("vid", "es")) is None  # track is English


def test_youtube_client_requests_have_a_default_timeout(monkeypatch):
    seen = {}

    def fake_request(self, method, url, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(transcript_cache.youtube_service.requests.Session, "request", fake_request)
    session = transcript_cache.youtube_service._TimeoutSession()

    session.request("GET", "https://www.youtube.com")
    assert seen["timeout"] == transcript_cache.youtube_service.YOUTUBE_HTTP_TIMEOUT
    session.request("GET", "https://www.youtube.com", timeout=3)
    assert seen["timeout"] == 3
