import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi import HTTPException


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
from rate_limiter import RateLimiter


USER = {"is_authenticated": True, "user_id": "user-1"}
ANONYMOUS = {"is_authenticated": False}
IP = "1.2.3.4"


def _call_raw(monkeypatch, user_info, outcome, url="https://youtu.be/dQw4w9WgXcQ"):
    deduct = AsyncMock(return_value=True)
    refund = AsyncMock()
    record = AsyncMock()
    limiter = RateLimiter()
    monkeypatch.setattr(transcript_api.CreditManager, "deduct_credit", deduct)
    monkeypatch.setattr(transcript_api.CreditManager, "add_credits", refund)
    monkeypatch.setattr(summary_service, "record_paid_access", record)
    monkeypatch.setattr(summary_service, "load_transcript_within", AsyncMock(**outcome))
    monkeypatch.setattr(transcript_api, "transcript_limiter", limiter)
    monkeypatch.setattr(
        transcript_api,
        "check_anonymous_rate_limit",
        lambda request: limiter.add_request(IP)
        or {"client_ip": IP, "remaining_requests": 9},
    )
    body = transcript_api.DownloadURLRequest(youtube_url=url, include_timestamps=False)

    try:
        result = asyncio.run(
            transcript_api.download_transcript_raw(body, object(), user_info, {})
        )
        error = None
    except HTTPException as e:
        result, error = None, e
    return result, error, refund, record, limiter


TRANSCRIPT = {
    "return_value": (
        [{"text": "Hello there.", "start": 0.0, "duration": 2}],
        {"transcript_language": "en", "transcript_type": "manual"},
    )
}


def test_success_is_charged_and_unlocks_the_summary(monkeypatch):
    result, error, refund, record, _ = _call_raw(monkeypatch, USER, TRANSCRIPT)

    assert error is None
    assert result.body == b"Hello there."
    refund.assert_not_awaited()
    record.assert_awaited_once_with("user-1", "dQw4w9WgXcQ")


def test_unavailable_video_is_refunded(monkeypatch):
    _, error, refund, record, _ = _call_raw(
        monkeypatch, USER, {"side_effect": ValueError("No transcript available")}
    )

    assert error.status_code == 400
    refund.assert_awaited_once_with("user-1", 1)
    record.assert_not_awaited()


def test_invalid_url_is_refunded(monkeypatch):
    _, error, refund, _, _ = _call_raw(monkeypatch, USER, TRANSCRIPT, url="not a url")

    assert error.status_code == 400
    refund.assert_awaited_once_with("user-1", 1)


def test_timeout_is_a_refunded_504(monkeypatch):
    _, error, refund, _, _ = _call_raw(
        monkeypatch, USER, {"side_effect": asyncio.TimeoutError()}
    )

    assert error.status_code == 504
    refund.assert_awaited_once_with("user-1", 1)


def test_anonymous_failure_gives_the_limiter_slot_back(monkeypatch):
    _, error, _, _, limiter = _call_raw(
        monkeypatch, ANONYMOUS, {"side_effect": ValueError("No transcript available")}
    )

    assert error.status_code == 400
    assert limiter.get_request_count(IP) == 0


def test_anonymous_success_keeps_the_limiter_slot(monkeypatch):
    _, error, _, _, limiter = _call_raw(monkeypatch, ANONYMOUS, TRANSCRIPT)

    assert error is None
    assert limiter.get_request_count(IP) == 1
