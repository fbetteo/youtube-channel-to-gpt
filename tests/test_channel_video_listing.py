import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("yt_dlp", types.SimpleNamespace(YoutubeDL=object))

from src.transcript_api import app
from routers import developer_api


client = TestClient(app)

CHANNEL_INFO = {
    "title": "Marques Brownlee",
    "channelId": "UCBJycsmduvYEL83R_U4JriQ",
    "thumbnail": "https://example.com/t.jpg",
    "subscriberCount": 21400000,
    "videoCount": None,
    "description": "",
}


def _video(i, kind="video"):
    return {
        "id": f"vid{i}",
        "title": f"Video {i}",
        "url": f"https://www.youtube.com/watch?v=vid{i}",
        "duration": "medium",
        "duration_seconds": 600,
        "viewCount": 1000 + i,
        "publishedAt": "2026-10-01T00:00:00Z",
        "type": kind,
    }


def _setup(monkeypatch, videos, calls):
    async def fake_info(channel):
        return CHANNEL_INFO

    async def fake_videos(channel_id, preferred_language=None, max_per_tab=None):
        calls.append(max_per_tab)
        return videos

    monkeypatch.setattr(developer_api.youtube_service, "get_channel_info", fake_info)
    monkeypatch.setattr(developer_api.youtube_service, "get_all_channel_videos", fake_videos)
    app.dependency_overrides[developer_api.validate_api_key] = lambda: {"user_id": "u1"}


def teardown_function():
    app.dependency_overrides.clear()


def test_listing_caps_tabs_and_reports_has_more(monkeypatch):
    calls = []
    _setup(monkeypatch, [_video(i) for i in range(4)], calls)

    response = client.get("/api/v1/channels/%40mkbhd/videos?limit=3")

    assert response.status_code == 200
    body = response.json()
    assert calls == [4]  # limit + 1 per tab
    assert body["total_videos"] == 3
    assert body["limit"] == 3
    assert body["has_more"] is True
    assert [v["id"] for v in body["videos"]] == ["vid0", "vid1", "vid2"]


def test_listing_maps_service_fields(monkeypatch):
    _setup(monkeypatch, [_video(1, "short")], [])

    body = client.get("/api/v1/channels/%40mkbhd/videos").json()

    assert body["limit"] == 100
    assert body["has_more"] is False
    video = body["videos"][0]
    assert video["duration_category"] == "medium"
    assert video["view_count"] == 1001
    assert video["published_at"] == "2026-10-01T00:00:00Z"
    assert video["type"] == "short"
    assert body["duration_breakdown"]["medium"] == 1


def test_listing_rejects_out_of_range_limit(monkeypatch):
    _setup(monkeypatch, [], [])

    assert client.get("/api/v1/channels/%40mkbhd/videos?limit=0").status_code == 422
    assert client.get("/api/v1/channels/%40mkbhd/videos?limit=2001").status_code == 422


def test_channel_info_maps_service_fields(monkeypatch):
    _setup(monkeypatch, [], [])

    body = client.get("/api/v1/channels/%40mkbhd/info").json()

    assert body["channel_id"] == "UCBJycsmduvYEL83R_U4JriQ"
    assert body["title"] == "Marques Brownlee"
    assert body["thumbnail_url"] == "https://example.com/t.jpg"
    assert body["video_count"] is None
