import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from openai import AsyncOpenAI


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault(
    "yt_dlp",
    types.SimpleNamespace(
        YoutubeDL=object,
        utils=types.SimpleNamespace(DownloadError=RuntimeError),
    ),
)

import summary_service
import transcript_api


SEGMENTS = [
    {"text": "Welcome to the show.", "start": 0.0, "duration": 4},
    {"text": "Today: pgvector.", "start": 10.0, "duration": 4},
    {"text": "First, embeddings.", "start": 35.0, "duration": 4},
    {"text": "", "start": 50.0, "duration": 1},
    {"text": "Then indexes.", "start": 70.0, "duration": 4},
]
RAW_SUMMARY = {
    "tldr": "A talk about pgvector.",
    "takeaways": [
        {"text": "Embeddings come first.", "source": 1},
        {"text": "Made-up point.", "source": 99},
    ],
    "sections": [],
}
METADATA = {
    "video_id": "dQw4w9WgXcQ",
    "transcript_language": "en",
    "transcript_type": "manual",
}


def test_blocks_group_captions_by_time_and_skip_empty_text():
    blocks = summary_service.build_blocks(SEGMENTS)

    assert [block["start"] for block in blocks] == [0.0, 35.0, 70.0]
    assert blocks[0]["text"] == "Welcome to the show. Today: pgvector."


def test_sources_map_to_real_times_and_invalid_ones_get_no_timestamp():
    blocks = summary_service.build_blocks(SEGMENTS)

    summary = summary_service.resolve_sources(RAW_SUMMARY, blocks, "abc")

    first, invented = summary["takeaways"]
    assert first["start_seconds"] == 35
    assert first["timestamp"] == "0:35"
    assert first["url"] == "https://www.youtube.com/watch?v=abc&t=35s"
    assert invented["text"] == "Made-up point."
    assert invented["start_seconds"] is None and invented["url"] is None


def test_sections_only_requested_for_long_videos():
    short = summary_service.build_messages([{"start": 0, "text": "hi"}], "en", "concise")
    long = summary_service.build_messages(
        [{"start": 0, "text": "hi"}, {"start": 900, "text": "bye"}], "es", "detailed"
    )

    assert "empty sections list" in short[1]["content"]
    assert "topic sections" in long[1]["content"]
    assert "Output language (BCP-47): es" in long[1]["content"]
    assert "[1] bye" in long[1]["content"]


def test_hour_long_timestamps_include_hours():
    assert summary_service.format_timestamp(3725) == "1:02:05"


def _client_returning(payload, status_code=200):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(status_code, json=payload)

    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, seen


def _completion(content, finish_reason="stop"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


def test_generate_summary_sends_strict_schema_through_the_real_sdk(monkeypatch):
    client, seen = _client_returning(_completion(json.dumps(RAW_SUMMARY)))
    monkeypatch.setattr(summary_service, "_get_client", lambda: client)

    raw, usage = asyncio.run(
        summary_service.generate_summary([{"start": 0, "text": "hi"}], "en", "concise")
    )

    assert raw == RAW_SUMMARY
    assert usage == {"input_tokens": 120, "output_tokens": 40}
    response_format = seen["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert seen["body"]["max_completion_tokens"] > 0


@pytest.mark.parametrize(
    "payload, status_code",
    [
        (_completion(json.dumps(RAW_SUMMARY), finish_reason="length"), 200),
        (_completion("not json"), 200),
        ({"error": {"message": "overloaded"}}, 500),
    ],
)
def test_generate_summary_failures_become_502(monkeypatch, payload, status_code):
    client, _ = _client_returning(payload, status_code)
    monkeypatch.setattr(summary_service, "_get_client", lambda: client)

    with pytest.raises(summary_service.SummaryError) as exc_info:
        asyncio.run(
            summary_service.generate_summary([{"start": 0, "text": "hi"}], "en", "concise")
        )

    assert exc_info.value.status_code == 502


def _no_cache(monkeypatch, cached_summary=None):
    monkeypatch.setattr(
        summary_service, "_load_cached_transcript", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(summary_service, "_save_cached_transcript", AsyncMock())
    monkeypatch.setattr(
        summary_service, "_load_cached_summary", AsyncMock(return_value=cached_summary)
    )
    save_summary = AsyncMock()
    monkeypatch.setattr(summary_service, "_save_cached_summary", save_summary)
    return save_summary


def test_summarize_video_fetches_generates_and_caches(monkeypatch):
    save_summary = _no_cache(monkeypatch)
    monkeypatch.setattr(
        summary_service.youtube_service,
        "get_transcript_data",
        AsyncMock(return_value=(SEGMENTS, METADATA)),
    )
    generate = AsyncMock(return_value=(RAW_SUMMARY, {"input_tokens": 1, "output_tokens": 2}))
    monkeypatch.setattr(summary_service, "generate_summary", generate)

    result = asyncio.run(
        summary_service.summarize_video("dQw4w9WgXcQ", None, None, "concise")
    )

    assert result["summary_language"] == "en"  # defaults to the transcript's
    assert result["cached"] is False
    assert result["transcript"].startswith("Welcome to the show.")
    assert result["summary"]["takeaways"][0]["start_seconds"] == 35
    save_summary.assert_awaited_once()


def test_cached_summary_skips_the_model(monkeypatch):
    _no_cache(
        monkeypatch,
        cached_summary={"raw": RAW_SUMMARY, "usage": {"input_tokens": 1, "output_tokens": 2}},
    )
    monkeypatch.setattr(
        summary_service.youtube_service,
        "get_transcript_data",
        AsyncMock(return_value=(SEGMENTS, METADATA)),
    )
    generate = AsyncMock()
    monkeypatch.setattr(summary_service, "generate_summary", generate)

    result = asyncio.run(
        summary_service.summarize_video("dQw4w9WgXcQ", None, "es", "concise")
    )

    assert result["cached"] is True
    generate.assert_not_awaited()


def test_missing_captions_and_oversized_videos(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        summary_service.youtube_service,
        "get_transcript_data",
        AsyncMock(side_effect=ValueError("TranscriptsDisabled")),
    )
    with pytest.raises(summary_service.TranscriptUnavailable):
        asyncio.run(summary_service.summarize_video("vid", None, None, "concise"))

    monkeypatch.setattr(
        summary_service.youtube_service,
        "get_transcript_data",
        AsyncMock(return_value=(SEGMENTS, METADATA)),
    )
    monkeypatch.setattr(summary_service.settings, "summary_max_input_chars", 10)
    with pytest.raises(summary_service.SummaryError) as exc_info:
        asyncio.run(summary_service.summarize_video("vid", None, None, "concise"))
    assert exc_info.value.status_code == 422


# ---- Endpoint: credits, refunds, anonymous limits ----


class _Request:
    pass


def _call_endpoint(monkeypatch, user_info, outcome, already_paid=False):
    monkeypatch.setattr(summary_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        summary_service, "has_recent_paid_access", AsyncMock(return_value=already_paid)
    )
    monkeypatch.setattr(summary_service, "record_paid_access", AsyncMock())
    deduct = AsyncMock(return_value=True)
    refund = AsyncMock()
    monkeypatch.setattr(transcript_api.CreditManager, "deduct_credit", deduct)
    monkeypatch.setattr(transcript_api.CreditManager, "add_credits", refund)
    rate_limit = lambda request: {"client_ip": "1.2.3.4", "remaining_requests": 9}
    limiter_calls = []
    monkeypatch.setattr(
        transcript_api,
        "check_anonymous_rate_limit",
        lambda request: limiter_calls.append(request) or rate_limit(request),
    )
    monkeypatch.setattr(summary_service, "summarize_video", AsyncMock(**outcome))
    body = transcript_api.SummaryRequest(youtube_url="https://youtu.be/dQw4w9WgXcQ")

    try:
        result = asyncio.run(
            transcript_api.summarize_single_video(body, _Request(), user_info)
        )
        error = None
    except HTTPException as e:
        result, error = None, e
    return result, error, deduct, refund, limiter_calls


SIGNED_IN = {"is_authenticated": True, "user_id": "user-1"}
ANONYMOUS = {"is_authenticated": False, "user_id": None}


def test_signed_in_success_costs_one_credit(monkeypatch):
    result, error, deduct, refund, _ = _call_endpoint(
        monkeypatch, SIGNED_IN, {"return_value": {"ok": True}}
    )

    assert error is None and result == {"ok": True}
    deduct.assert_awaited_once_with("user-1")
    refund.assert_not_awaited()


def test_our_failures_are_refunded(monkeypatch):
    _, error, _, refund, _ = _call_endpoint(
        monkeypatch,
        SIGNED_IN,
        {"side_effect": summary_service.SummaryError(502, "provider down")},
    )

    assert error.status_code == 502
    refund.assert_awaited_once_with("user-1", 1)


def test_videos_without_captions_are_charged(monkeypatch):
    _, error, _, refund, _ = _call_endpoint(
        monkeypatch,
        SIGNED_IN,
        {"side_effect": summary_service.TranscriptUnavailable("no captions")},
    )

    assert error.status_code == 400
    refund.assert_not_awaited()


def test_anonymous_users_use_the_shared_rate_limit(monkeypatch):
    result, error, deduct, _, limiter_calls = _call_endpoint(
        monkeypatch, ANONYMOUS, {"return_value": {"ok": True}}
    )

    assert error is None
    assert len(limiter_calls) == 1
    deduct.assert_not_awaited()


def test_unconfigured_provider_returns_503_before_charging(monkeypatch):
    deduct = AsyncMock()
    monkeypatch.setattr(summary_service, "is_configured", lambda: False)
    monkeypatch.setattr(transcript_api.CreditManager, "deduct_credit", deduct)
    body = transcript_api.SummaryRequest(youtube_url="https://youtu.be/dQw4w9WgXcQ")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(transcript_api.summarize_single_video(body, _Request(), SIGNED_IN))

    assert exc_info.value.status_code == 503
    deduct.assert_not_awaited()


def test_request_validates_languages_and_length():
    with pytest.raises(ValueError):
        transcript_api.SummaryRequest(youtube_url="x", summary_language="not a code!")
    with pytest.raises(ValueError):
        transcript_api.SummaryRequest(youtube_url="x", length="essay")
    assert (
        transcript_api.SummaryRequest(youtube_url="x", summary_language="PT-br").summary_language
        == "pt-BR"
    )


def test_summary_after_a_paid_transcript_is_free(monkeypatch):
    result, error, deduct, refund, _ = _call_endpoint(
        monkeypatch, SIGNED_IN, {"return_value": {"ok": True}}, already_paid=True
    )

    assert error is None
    deduct.assert_not_awaited()
    summary_service.record_paid_access.assert_not_awaited()


def test_free_summary_failures_are_not_refunded(monkeypatch):
    _, error, _, refund, _ = _call_endpoint(
        monkeypatch,
        SIGNED_IN,
        {"side_effect": summary_service.SummaryError(502, "provider down")},
        already_paid=True,
    )

    assert error.status_code == 502
    refund.assert_not_awaited()


def test_paid_summary_records_access(monkeypatch):
    _call_endpoint(monkeypatch, SIGNED_IN, {"return_value": {"ok": True}})

    summary_service.record_paid_access.assert_awaited_once_with("user-1", "dQw4w9WgXcQ")


def test_raw_download_uses_shared_cache_and_records_access(monkeypatch):
    monkeypatch.setattr(
        transcript_api.CreditManager, "deduct_credit", AsyncMock(return_value=True)
    )
    record = AsyncMock()
    monkeypatch.setattr(summary_service, "record_paid_access", record)
    load = AsyncMock(return_value=(SEGMENTS, METADATA))
    monkeypatch.setattr(summary_service, "load_transcript", load)
    body = transcript_api.DownloadURLRequest(
        youtube_url="https://youtu.be/dQw4w9WgXcQ", include_timestamps=True
    )

    response = asyncio.run(
        transcript_api.download_transcript_raw(body, _Request(), SIGNED_IN, {})
    )

    assert response.body.decode().startswith("[00:00] Welcome to the show.")
    assert response.headers["X-Transcript-Language"] == "en"
    load.assert_awaited_once_with("dQw4w9WgXcQ", None)
    record.assert_awaited_once_with("user-1", "dQw4w9WgXcQ")


class _HeaderRequest:
    def __init__(self, headers, host="10.0.0.1"):
        self.headers = headers
        self.client = types.SimpleNamespace(host=host)


@pytest.mark.parametrize(
    "secret, headers, expected",
    [
        # Trusted proxy: secret matches, so X-Client-IP wins.
        ("s3", {"X-Proxy-Secret": "s3", "X-Client-IP": "9.9.9.9", "X-Forwarded-For": "1.1.1.1"}, "9.9.9.9"),
        # Wrong or missing secret: X-Client-IP is ignored.
        ("s3", {"X-Proxy-Secret": "no", "X-Client-IP": "9.9.9.9", "X-Forwarded-For": "1.1.1.1"}, "1.1.1.1"),
        ("", {"X-Proxy-Secret": "", "X-Client-IP": "9.9.9.9"}, "10.0.0.1"),
        # A spoofed first hop does not help: the hop our nginx appends is last.
        ("", {"X-Forwarded-For": "6.6.6.6, 2.2.2.2"}, "2.2.2.2"),
        ("", {}, "10.0.0.1"),
    ],
)
def test_client_ip_for_anonymous_limits(monkeypatch, secret, headers, expected):
    monkeypatch.setattr(transcript_api.settings, "proxy_shared_secret", secret)

    assert transcript_api.get_client_ip(_HeaderRequest(headers)) == expected
