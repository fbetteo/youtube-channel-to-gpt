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

import hybrid_job_manager as hybrid_module
import job_result_processor
import transcript_api
import youtube_service
from routers import developer_api, mcp


def test_playlist_job_normalization_filters_unavailable_and_deduplicates():
    videos = [
        {"id": "valid-1", "title": "First", "duration": "short"},
        {"id": "hidden-1", "title": None, "duration": "unknown"},
        {"id": "hidden-2", "title": "[Unavailable video]", "duration": "unknown"},
        {"id": "valid-1", "title": "Duplicate", "duration": "short"},
        {"id": "valid-2", "title": "Second", "duration": "unknown"},
    ]

    accepted, excluded = youtube_service.normalize_videos_for_job(videos)

    assert [video["id"] for video in accepted] == ["valid-1", "valid-2"]
    assert accepted[1]["duration"] is None
    assert [item["reason"] for item in excluded] == [
        "unavailable",
        "unavailable",
        "duplicate",
    ]


def test_durable_job_creation_never_falls_back_to_file(monkeypatch):
    manager = hybrid_module.HybridJobManager()
    manager._db_initialized = True
    create_db = AsyncMock(side_effect=RuntimeError("database rejected job"))
    save_file = AsyncMock()
    monkeypatch.setattr(hybrid_module.JobManager, "create_job_with_videos", create_db)
    monkeypatch.setattr(youtube_service, "save_job_to_file", save_file)

    with pytest.raises(RuntimeError, match="database rejected job"):
        asyncio.run(
            manager.create_job(
                "job-id",
                {
                    "user_id": "user-id",
                    "source_id": "playlist-id",
                    "source_name": "Playlist",
                    "playlist_id": "playlist-id",
                    "credits_reserved": 1,
                    "reservation_id": "reservation-id",
                },
                [{"id": "video-id", "title": "Video", "duration": "short"}],
                reserve_credits=True,
            )
        )

    save_file.assert_not_awaited()


def test_developer_discovery_placeholder_is_durable(monkeypatch):
    create_job = AsyncMock(return_value="job-id")
    save_file = AsyncMock()
    monkeypatch.setattr(developer_api.hybrid_job_manager, "create_job", create_job)
    monkeypatch.setattr(youtube_service, "save_job_to_file", save_file)

    asyncio.run(
        developer_api._create_developer_discovery_placeholder(
            job_id="job-id",
            user_id="user-id",
            source_type="playlist",
            source_input="playlist-id",
            api_key_id="key-id",
        )
    )

    create_job.assert_awaited_once()
    args, kwargs = create_job.await_args
    assert args[0] == "job-id"
    assert args[2] == []
    assert kwargs["allow_empty"] is True
    save_file.assert_not_awaited()


def test_cancelled_job_ignores_late_lambda_result(monkeypatch):
    monkeypatch.setattr(
        job_result_processor,
        "_get_job_status",
        AsyncMock(
            return_value={
                "status": "cancelled",
                "processed_count": 3,
                "total_videos": 3,
            }
        ),
    )
    mark_completed = AsyncMock()
    monkeypatch.setattr(
        job_result_processor.JobManager, "mark_video_completed", mark_completed
    )

    result = asyncio.run(
        job_result_processor.process_video_completion(
            "job-id", {"video_id": "video-id", "s3_key": "key"}
        )
    )

    assert result["status"] == "ignored"
    assert result["job_status"] == "cancelled"
    mark_completed.assert_not_awaited()


def test_website_video_status_preserves_not_found(monkeypatch):
    monkeypatch.setattr(transcript_api, "load_video_job_from_file", lambda job_id: None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(transcript_api.get_videos_fetch_status("missing"))

    assert exc_info.value.status_code == 404


def test_developer_cancel_response_keeps_partial_download(monkeypatch):
    monkeypatch.setattr(
        developer_api.JobManager,
        "cancel_job",
        AsyncMock(
            return_value={
                "job_id": "job-id",
                "status": "cancelled",
                "total_videos": 5,
                "processed_count": 5,
                "completed": 2,
                "failed_count": 0,
                "skipped_count": 3,
                "credits_reserved": 5,
                "credits_used": 2,
                "refunded_credits": 3,
                "refunded_now": 3,
            }
        ),
    )

    response = asyncio.run(
        developer_api.cancel_job("job-id", {"user_id": "user-id"})
    )

    assert response.status == "cancelled"
    assert response.refunded_now == 3
    assert response.download_ready is True
    assert response.download_url == "/api/v1/jobs/job-id/download"


def test_hosted_mcp_exposes_cancel_job():
    tools_by_name = {tool["name"]: tool for tool in mcp.TOOLS}

    assert "cancel_job" in tools_by_name
    assert tools_by_name["cancel_job"]["inputSchema"]["required"] == ["job_id"]
