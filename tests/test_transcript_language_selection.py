import importlib.util
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import yt_dlp  # noqa: F401
except ModuleNotFoundError:
    yt_dlp_stub = ModuleType("yt_dlp")
    yt_dlp_stub.__spec__ = importlib.util.spec_from_loader("yt_dlp", loader=None)
    yt_dlp_stub.utils = SimpleNamespace(DownloadError=RuntimeError)
    sys.modules["yt_dlp"] = yt_dlp_stub

import youtube_service


class FakeFetchedTranscript:
    def to_raw_data(self):
        return [{"text": "hello", "start": 0.0, "duration": 1.0}]


class FakeTranscript:
    def __init__(self, language_code: str, *, generated: bool = False):
        self.language_code = language_code
        self.is_generated = generated
        self.fetch_calls = 0

    def fetch(self):
        self.fetch_calls += 1
        return FakeFetchedTranscript()


class FakeTranscriptApi:
    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.list_calls = 0

    def list(self, video_id):
        self.list_calls += 1
        return self.transcripts


def load_lambda_module():
    module_path = PROJECT_ROOT / "lambda-transcript-processor" / "lambda_function.py"
    spec = importlib.util.spec_from_file_location("transcript_lambda", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalizes_bcp47_style_language_codes():
    assert youtube_service.normalize_preferred_language(" IT ") == "it"
    assert youtube_service.normalize_preferred_language("pt-br") == "pt-BR"
    assert youtube_service.normalize_preferred_language("ZH-hans") == "zh-Hans"
    assert youtube_service.normalize_preferred_language(None) is None


@pytest.mark.parametrize("value", ["", "i", "italian", "en_US", "en--US"])
def test_rejects_invalid_language_codes(value):
    with pytest.raises(ValueError, match="preferred_language"):
        youtube_service.normalize_preferred_language(value)


def test_automatic_uses_first_available_transcript():
    english = FakeTranscript("en")
    italian = FakeTranscript("it")

    selected, fallback_used = youtube_service.select_transcript(
        [english, italian], None
    )

    assert selected is english
    assert fallback_used is False


def test_exact_language_prefers_manual_caption():
    generated_italian = FakeTranscript("it", generated=True)
    manual_italian = FakeTranscript("it", generated=False)

    selected, fallback_used = youtube_service.select_transcript(
        [generated_italian, manual_italian], "it"
    )

    assert selected is manual_italian
    assert fallback_used is False


def test_primary_language_matches_regional_variant():
    english = FakeTranscript("en")
    regional_italian = FakeTranscript("it-IT")

    selected, fallback_used = youtube_service.select_transcript(
        [english, regional_italian], "it"
    )

    assert selected is regional_italian
    assert fallback_used is False


def test_missing_preference_falls_back_to_first_available():
    english = FakeTranscript("en")
    spanish = FakeTranscript("es")

    selected, fallback_used = youtube_service.select_transcript(
        [english, spanish], "it"
    )

    assert selected is english
    assert fallback_used is True


def test_empty_transcript_list_fails_cleanly():
    with pytest.raises(ValueError, match="No transcripts"):
        youtube_service.select_transcript([], "it")


def test_fetch_uses_one_list_and_one_fetch_call(monkeypatch):
    english = FakeTranscript("en")
    italian = FakeTranscript("it")
    api = FakeTranscriptApi([english, italian])
    monkeypatch.setattr(youtube_service, "get_ytt_api", lambda: api)

    selected, _, attempts = youtube_service._fetch_transcript_with_retries(
        "video-id", preferred_language="it"
    )

    assert selected is italian
    assert attempts == 1
    assert api.list_calls == 1
    assert italian.fetch_calls == 1
    assert english.fetch_calls == 0


def test_lambda_legacy_default_is_english_but_explicit_none_is_automatic(monkeypatch):
    lambda_module = load_lambda_module()
    italian = FakeTranscript("it")
    english = FakeTranscript("en")

    legacy_api = FakeTranscriptApi([italian, english])
    monkeypatch.setattr(lambda_module, "get_ytt_api", lambda: legacy_api)
    selected, _, _ = lambda_module.fetch_transcript_with_retries("video-id")
    assert selected is english

    automatic_api = FakeTranscriptApi([italian, english])
    monkeypatch.setattr(lambda_module, "get_ytt_api", lambda: automatic_api)
    selected, _, _ = lambda_module.fetch_transcript_with_retries(
        "video-id", preferred_language=None
    )
    assert selected is italian


def test_web_request_models_normalize_and_default_language_preferences():
    import transcript_api

    automatic = transcript_api.DownloadURLRequest(youtube_url="video-id")
    italian = transcript_api.DownloadURLRequest(
        youtube_url="video-id", preferred_language="IT"
    )
    batch = transcript_api.BatchPlaylistDownloadRequest(
        playlists=[{"playlist_id": "playlist-id"}],
        preferred_language="pt-br",
    )

    assert automatic.preferred_language is None
    assert italian.preferred_language == "it"
    assert batch.preferred_language == "pt-BR"


def test_selected_channel_job_persists_language_for_lambda_dispatch(monkeypatch):
    import transcript_api

    captured = {}

    async def get_credits(user_id):
        return 10

    async def get_channel_info(channel_name):
        return {"id": "channel-id", "title": "Channel"}

    async def create_job(job_id, job_data, videos, reserve_credits):
        captured["job_data"] = job_data

    async def prefetch_and_dispatch(job_id):
        return None

    monkeypatch.setattr(
        transcript_api.CreditManager, "get_user_credits", get_credits
    )
    monkeypatch.setattr(
        transcript_api.youtube_service, "get_channel_info", get_channel_info
    )
    monkeypatch.setattr(
        transcript_api.hybrid_job_manager, "create_job", create_job
    )
    monkeypatch.setattr(
        transcript_api.youtube_service,
        "prefetch_and_dispatch_task",
        prefetch_and_dispatch,
    )

    request = transcript_api.SelectedVideosRequest(
        channel_name="channel",
        preferred_language="it",
        videos=[
            {
                "id": "video-id",
                "title": "Video",
                "url": "https://www.youtube.com/watch?v=video-id",
                "duration": "medium",
            }
        ],
    )
    asyncio.run(
        transcript_api.download_selected_videos(
            request=request,
            payload={"sub": "user-id"},
            session={},
        )
    )

    assert captured["job_data"]["formatting_options"]["preferred_language"] == "it"


def test_lambda_dispatch_forwards_explicit_automatic_preference(monkeypatch):
    payloads = []

    class FakeLambdaClient:
        def invoke(self, **kwargs):
            payloads.append(json.loads(kwargs["Payload"]))

    monkeypatch.setattr(
        youtube_service.boto3, "client", lambda service_name: FakeLambdaClient()
    )

    result = asyncio.run(
        youtube_service.dispatch_lambdas_concurrently(
            job_id="job-id",
            videos=[{"id": "video-id"}],
            videos_metadata={},
            user_id="user-id",
            formatting_options={"preferred_language": None},
        )
    )

    assert result["dispatched_count"] == 1
    assert payloads[0]["preferred_language"] is None
