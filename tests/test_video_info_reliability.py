import asyncio
import logging
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _StubDownloadError(Exception):
    pass


yt_dlp_module = sys.modules.setdefault(
    "yt_dlp",
    types.SimpleNamespace(
        YoutubeDL=object,
        utils=types.SimpleNamespace(DownloadError=_StubDownloadError),
    ),
)
if not hasattr(yt_dlp_module, "utils"):
    yt_dlp_module.utils = types.SimpleNamespace(DownloadError=_StubDownloadError)

import youtube_service
from src import transcript_api


VIDEO_ID = "qZLE0yZAHAg"
VIDEO_INFO = {
    "id": VIDEO_ID,
    "title": "Test video",
    "description": "Description",
    "channel_id": "channel-id",
    "uploader": "Channel",
    "upload_date": "20260101",
    "thumbnail": "https://example.com/thumb.jpg",
    "duration": 125,
    "view_count": 10,
    "like_count": 2,
    "comment_count": 1,
}


@pytest.fixture(autouse=True)
def reset_video_info_executor():
    youtube_service.shutdown_video_info_executor()
    yield
    youtube_service.shutdown_video_info_executor()


def _install_fake_youtube_dl(monkeypatch, outcomes):
    options = []
    thread_names = []
    instances = []

    class FakeYoutubeDL:
        def __init__(self, ydl_opts):
            self.ydl_opts = ydl_opts
            options.append(ydl_opts)
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, download=False):
            thread_names.append(threading.current_thread().name)
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome()
            return outcome

    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return options, thread_names, instances


def _download_error(message):
    return youtube_service.yt_dlp.utils.DownloadError(message)


def test_video_info_uses_bounded_options_and_dedicated_executor(monkeypatch):
    options, thread_names, instances = _install_fake_youtube_dl(
        monkeypatch, [VIDEO_INFO.copy()]
    )

    metadata = asyncio.run(youtube_service.get_video_info(VIDEO_ID))

    assert metadata == {
        "id": VIDEO_ID,
        "title": "Test video",
        "description": "Description",
        "channelId": "channel-id",
        "channelTitle": "Channel",
        "publishedAt": "20260101",
        "thumbnail": "https://example.com/thumb.jpg",
        "duration": "2:05",
        "viewCount": 10,
        "likeCount": 2,
        "commentCount": 1,
        "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
    }
    assert len(instances) == 1
    assert options[0]["socket_timeout"] == 6
    assert options[0]["retries"] == 0
    assert options[0]["extractor_retries"] == 0
    assert options[0]["ignoreerrors"] is False
    assert options[0]["skip_download"] is True
    assert youtube_service.VIDEO_INFO_EXECUTOR_MAX_WORKERS == 4
    assert thread_names[0].startswith("video-info")


def test_proxy_log_does_not_include_credentials(monkeypatch, caplog):
    monkeypatch.setattr(
        youtube_service.settings, "webshare_proxy_username", "secret-user"
    )
    monkeypatch.setattr(
        youtube_service.settings, "webshare_proxy_password", "secret-password"
    )
    caplog.set_level(logging.INFO, logger=youtube_service.__name__)

    options = youtube_service._get_ydl_opts({})

    assert "proxy" in options
    assert "secret-user" not in caplog.text
    assert "secret-password" not in caplog.text
    assert "Using Webshare rotating proxy for yt-dlp" in caplog.text


def test_video_info_retries_transient_download_error_with_new_instance(monkeypatch):
    monkeypatch.setattr(youtube_service, "VIDEO_INFO_RETRY_DELAY_SECONDS", 0)
    _, _, instances = _install_fake_youtube_dl(
        monkeypatch,
        [_download_error("proxy connection reset"), VIDEO_INFO.copy()],
    )

    metadata = asyncio.run(youtube_service.get_video_info(VIDEO_ID))

    assert metadata["title"] == "Test video"
    assert len(instances) == 2


def test_video_info_does_not_retry_terminal_failure(monkeypatch):
    _, _, instances = _install_fake_youtube_dl(
        monkeypatch,
        [_download_error("ERROR: [youtube] Private video. Sign in if granted access")],
    )

    with pytest.raises(youtube_service.VideoMetadataNotAccessible):
        asyncio.run(youtube_service.get_video_info(VIDEO_ID))

    assert len(instances) == 1


def test_video_info_exhausts_two_transient_attempts(monkeypatch):
    monkeypatch.setattr(youtube_service, "VIDEO_INFO_RETRY_DELAY_SECONDS", 0)
    _, _, instances = _install_fake_youtube_dl(
        monkeypatch,
        [
            _download_error("proxy connection reset"),
            _download_error("proxy connection reset"),
        ],
    )

    with pytest.raises(youtube_service.VideoMetadataUnavailable) as exc_info:
        asyncio.run(youtube_service.get_video_info(VIDEO_ID))

    assert exc_info.value.attempts == 2
    assert len(instances) == 2


def test_video_info_timeout_is_bounded_and_attempted_twice(monkeypatch):
    monkeypatch.setattr(youtube_service, "VIDEO_INFO_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(youtube_service, "VIDEO_INFO_TOTAL_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(youtube_service, "VIDEO_INFO_RETRY_DELAY_SECONDS", 0)

    def slow_result():
        time.sleep(0.05)
        return VIDEO_INFO.copy()

    _, _, instances = _install_fake_youtube_dl(
        monkeypatch, [slow_result, slow_result]
    )
    started_at = time.perf_counter()

    with pytest.raises(youtube_service.VideoMetadataUnavailable):
        asyncio.run(youtube_service.get_video_info(VIDEO_ID))

    assert time.perf_counter() - started_at < 0.1
    assert len(instances) == 2


@pytest.mark.parametrize(
    ("service_error", "expected_status"),
    [
        (
            youtube_service.VideoMetadataNotAccessible(
                VIDEO_ID, _download_error("Private video")
            ),
            404,
        ),
        (
            youtube_service.VideoMetadataUnavailable(
                VIDEO_ID, 2, _download_error("proxy error")
            ),
            503,
        ),
        (RuntimeError("unexpected defect"), 500),
        (ValueError("unexpected service value error"), 500),
    ],
)
def test_video_info_endpoint_maps_service_errors(
    monkeypatch, service_error, expected_status
):
    async def raise_service_error(video_id):
        raise service_error

    monkeypatch.setattr(
        transcript_api.youtube_service, "get_video_info", raise_service_error
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(transcript_api.get_video_info(VIDEO_ID, session={}))

    assert exc_info.value.status_code == expected_status


def test_video_info_endpoint_rejects_invalid_url_before_service(monkeypatch):
    get_video_info_mock = AsyncMock()
    monkeypatch.setattr(
        transcript_api.youtube_service, "get_video_info", get_video_info_mock
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(transcript_api.get_video_info("bad", session={}))

    assert exc_info.value.status_code == 400
    get_video_info_mock.assert_not_awaited()
