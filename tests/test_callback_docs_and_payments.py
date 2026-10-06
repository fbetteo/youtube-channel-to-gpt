import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault(
    "yt_dlp",
    types.SimpleNamespace(
        YoutubeDL=object,
        utils=types.SimpleNamespace(DownloadError=RuntimeError),
    ),
)

import job_result_processor
import transcript_api


def test_completion_ignores_reported_s3_key_and_uses_job_owner(monkeypatch):
    monkeypatch.setattr(
        job_result_processor,
        "_get_job_status",
        AsyncMock(return_value={"status": "processing", "user_id": "owner"}),
    )
    mark_completed = AsyncMock(return_value=True)
    monkeypatch.setattr(
        job_result_processor.JobManager, "mark_video_completed", mark_completed
    )
    monkeypatch.setattr(
        job_result_processor,
        "_finalize_job_if_needed",
        AsyncMock(
            return_value={"status": "processing", "processed_count": 1, "total_videos": 2}
        ),
    )

    asyncio.run(
        job_result_processor.process_video_completion(
            "job-1", {"video_id": "vid", "s3_key": "victim/other-job/secret.txt"}
        )
    )

    file_info = mark_completed.await_args.kwargs["file_info"]
    assert file_info["s3_key"] == "owner/job-1/vid.txt"


@pytest.mark.parametrize(
    "configured, supplied, allowed",
    [
        ("", "", False),
        ("", "change_me_in_production", False),
        ("s3cret", "wrong", False),
        ("s3cret", "ñ", False),
        ("s3cret", "s3cret", True),
    ],
)
def test_docs_secret_fails_closed(monkeypatch, configured, supplied, allowed):
    monkeypatch.setattr(transcript_api, "DOCS_SECRET", configured)

    if allowed:
        transcript_api.require_docs_secret(supplied)
    else:
        with pytest.raises(HTTPException) as exc_info:
            transcript_api.require_docs_secret(supplied)
        assert exc_info.value.status_code == 404


class _WebhookRequest:
    headers = {"stripe-signature": "sig"}

    async def body(self):
        return b"{}"


def _checkout_event(payment_status="paid"):
    return {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_1",
                "payment_status": payment_status,
                "metadata": {
                    "project": "transcript-api",
                    "user_id": "user-1",
                    "credits": "100",
                },
            }
        },
    }


def _run_webhook(monkeypatch, event, granted=True):
    monkeypatch.setattr(
        transcript_api.stripe.Webhook, "construct_event", lambda *args: event
    )
    grant = AsyncMock(return_value=granted)
    monkeypatch.setattr(transcript_api.CreditManager, "grant_checkout_credits", grant)
    result = asyncio.run(transcript_api.stripe_webhook(_WebhookRequest()))
    return result, grant


def test_webhook_grants_credits_once_per_session(monkeypatch):
    result, grant = _run_webhook(monkeypatch, _checkout_event())

    assert result == {"status": "success"}
    grant.assert_awaited_once_with("cs_test_1", "user-1", 100)


def test_webhook_redelivery_does_not_grant_again(monkeypatch):
    result, _ = _run_webhook(monkeypatch, _checkout_event(), granted=False)

    assert result == {"status": "duplicate"}


def test_webhook_ignores_unpaid_sessions(monkeypatch):
    result, grant = _run_webhook(monkeypatch, _checkout_event("unpaid"))

    assert result == {"status": "ignored"}
    grant.assert_not_awaited()


def test_webhook_accepts_full_discount_sessions(monkeypatch):
    result, grant = _run_webhook(monkeypatch, _checkout_event("no_payment_required"))

    assert result == {"status": "success"}
    grant.assert_awaited_once()
