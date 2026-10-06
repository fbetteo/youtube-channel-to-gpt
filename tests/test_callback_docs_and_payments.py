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


def test_http_result_callbacks_can_be_disabled(monkeypatch):
    process = AsyncMock()
    monkeypatch.setattr(transcript_api, "process_video_completion", process)
    monkeypatch.setattr(
        transcript_api.settings, "enable_http_result_callbacks", False
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(transcript_api.video_completed("job-1", {"video_id": "vid"}))

    assert exc_info.value.status_code == 404
    process.assert_not_awaited()


def test_http_result_callbacks_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_HTTP_RESULT_CALLBACKS", raising=False)
    assert transcript_api.settings.__class__().enable_http_result_callbacks is True


class _CreditsConnection:
    def __init__(self):
        self.balance = None

    async def fetchrow(self, query, user_id):
        return None if self.balance is None else {"credits": self.balance}

    async def fetchval(self, query, user_id):
        return self.balance


def test_new_user_sees_welcome_credits_on_first_check(monkeypatch):
    from contextlib import asynccontextmanager
    from db_youtube_transcripts import database

    conn = _CreditsConnection()

    @asynccontextmanager
    async def fake_connection():
        yield conn

    async def fake_create(user_id, credits=0):
        conn.balance = credits

    monkeypatch.setattr(database, "get_db_connection", fake_connection)
    monkeypatch.setattr(
        transcript_api.CreditManager, "create_user_credits", fake_create
    )

    assert asyncio.run(transcript_api.CreditManager.get_user_credits("new-user")) == 25


def test_timed_out_videos_are_not_charged(monkeypatch):
    from contextlib import asynccontextmanager
    from db_youtube_transcripts import job_manager

    executed = []

    class _Tx:
        async def fetchrow(self, query, *args):
            return {"status": "processing"}

        async def fetchval(self, query, *args):
            return 2 if "'processing'" in query else 1

        async def execute(self, query, *args):
            executed.append(query)

    @asynccontextmanager
    async def fake_transaction():
        yield _Tx()

    monkeypatch.setattr(job_manager, "get_db_transaction", fake_transaction)

    result = asyncio.run(
        job_manager.JobManager.fail_unresolved_videos("job-1", "no result")
    )

    assert result == {"failed": 2, "skipped": 1}
    jobs_update = next(q for q in executed if "UPDATE jobs" in q)
    assert "credits_used" not in jobs_update


def test_stale_job_sweep_refunds_unresolved_work(monkeypatch):
    import youtube_service
    from db_youtube_transcripts.job_manager import JobManager

    monkeypatch.setattr(
        JobManager, "find_stale_active_jobs", AsyncMock(return_value=["job-1"])
    )
    fail = AsyncMock(return_value={"failed": 1, "skipped": 3})
    finalize = AsyncMock(return_value={})
    monkeypatch.setattr(JobManager, "fail_unresolved_videos", fail)
    monkeypatch.setattr(JobManager, "finalize_job_if_complete", finalize)

    assert asyncio.run(youtube_service.reconcile_stale_jobs(15)) == 1
    fail.assert_awaited_once_with("job-1", "No progress for 15 minutes")
    finalize.assert_awaited_once_with("job-1")


@pytest.mark.parametrize(
    "failure, charged",
    [
        ({"retriable": False, "error_type": "TranscriptsDisabled"}, True),
        ({"retriable": True, "error_type": "RequestBlocked"}, False),
        ({"retriable": True, "error_type": "DeadlineExceeded"}, False),
        ({}, True),  # older workers without the flag keep the previous charge
    ],
)
def test_only_content_failures_are_charged(monkeypatch, failure, charged):
    monkeypatch.setattr(
        job_result_processor,
        "_get_job_status",
        AsyncMock(return_value={"status": "processing", "user_id": "owner"}),
    )
    mark_failed = AsyncMock(return_value=True)
    monkeypatch.setattr(
        job_result_processor.JobManager, "mark_video_failed", mark_failed
    )
    monkeypatch.setattr(
        job_result_processor,
        "_finalize_job_if_needed",
        AsyncMock(
            return_value={"status": "processing", "processed_count": 1, "total_videos": 2}
        ),
    )

    asyncio.run(
        job_result_processor.process_video_failure(
            "job-1", {"video_id": "vid", "error": "boom", **failure}
        )
    )

    assert mark_failed.await_args.kwargs["charge"] is charged


def test_lambda_reports_retriable_failure_before_its_own_timeout(monkeypatch):
    import importlib.util
    import time

    # Load by path: the Lambda folder bundles its own boto3/requests copies,
    # which must not shadow the project's on sys.path.
    spec = importlib.util.spec_from_file_location(
        "lambda_function",
        Path(__file__).resolve().parents[1]
        / "lambda-transcript-processor"
        / "lambda_function.py",
    )
    lambda_function = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lambda_function)

    monkeypatch.setattr(
        lambda_function,
        "fetch_transcript_with_retries",
        lambda *args, **kwargs: time.sleep(3),
    )
    context = types.SimpleNamespace(
        get_remaining_time_in_millis=lambda: (lambda_function.DEADLINE_SAFETY_SECONDS + 1)
        * 1000
    )

    with pytest.raises(lambda_function.TranscriptRetrievalError) as exc_info:
        lambda_function.fetch_transcript_before_deadline("vid", None, context)

    assert exc_info.value.retriable is True
    assert exc_info.value.stage == "deadline"
    assert exc_info.value.error_type == "DeadlineExceeded"
