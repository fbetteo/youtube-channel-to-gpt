import asyncio
import importlib.util
import sys
import types
from pathlib import Path

from fastapi import BackgroundTasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.setdefault(
    "yt_dlp",
    types.SimpleNamespace(
        YoutubeDL=object,
        utils=types.SimpleNamespace(DownloadError=RuntimeError),
    ),
)

import transcript_api
import youtube_service


def load_lambda_module():
    module_path = PROJECT_ROOT / "lambda-transcript-processor" / "lambda_function.py"
    spec = importlib.util.spec_from_file_location("transcript_lambda", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transcript_formatting_collapses_internal_whitespace():
    segments = [
        {"text": "Prima  riga\nseconda\triga", "start": 0.0},
        {"text": "  prossimo   segmento ", "start": 65.2},
        {"text": " \n\t ", "start": 70.0},
    ]
    expected_plain = "Prima riga seconda riga prossimo segmento"
    expected_timestamped = (
        "[00:00] Prima riga seconda riga\n[01:05] prossimo segmento"
    )

    assert youtube_service.format_transcript_segments(segments, False) == expected_plain
    assert (
        youtube_service.format_transcript_segments(segments, True)
        == expected_timestamped
    )

    lambda_module = load_lambda_module()
    assert lambda_module.format_transcript_segments(segments, False) == expected_plain
    assert (
        lambda_module.format_transcript_segments(segments, True)
        == expected_timestamped
    )


def test_localized_options_preserve_existing_extractor_arguments(monkeypatch):
    monkeypatch.setattr(youtube_service, "_get_ydl_opts", lambda options: options)
    base_options = {
        "extract_flat": True,
        "extractor_args": {"youtubetab": {"approximate_date": [""]}},
    }

    options = youtube_service._get_localized_ydl_opts(base_options, "IT")

    assert options["extractor_args"] == {
        "youtubetab": {"approximate_date": [""]},
        "youtube": {"lang": ["it"]},
    }
    assert "youtube" not in base_options["extractor_args"]


def test_localized_flat_extraction_retries_without_language(monkeypatch):
    observed_options = []
    outcomes = [RuntimeError("unsupported language"), {"entries": []}]

    class FakeYoutubeDL:
        def __init__(self, options):
            observed_options.append(options)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, download=False):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(youtube_service, "_get_ydl_opts", lambda options: options)

    result = youtube_service._extract_flat_info(
        "https://www.youtube.com/playlist?list=test",
        {
            "extract_flat": True,
            "extractor_args": {"youtubetab": {"approximate_date": [""]}},
        },
        "it",
    )

    assert result == {"entries": []}
    assert observed_options[0]["extractor_args"]["youtube"] == {"lang": ["it"]}
    assert "youtube" not in observed_options[1]["extractor_args"]


def test_channel_discovery_persists_and_forwards_language(monkeypatch):
    created_jobs = []

    async def create_job(job_id, job_data, **kwargs):
        created_jobs.append(job_data)

    monkeypatch.setattr(transcript_api, "create_discovery_job", create_job)
    background_tasks = BackgroundTasks()

    response = asyncio.run(
        transcript_api.list_all_channel_videos(
            "channel-id", background_tasks, preferred_language="IT"
        )
    )

    assert response["preferred_language"] == "it"
    assert created_jobs[0]["preferred_language"] == "it"
    assert background_tasks.tasks[0].args == (response["job_id"], "channel-id", "it")


def test_playlist_discovery_persists_and_forwards_language(monkeypatch):
    created_jobs = []

    async def create_job(job_id, job_data, **kwargs):
        created_jobs.append(job_data)

    monkeypatch.setattr(transcript_api, "create_discovery_job", create_job)
    background_tasks = BackgroundTasks()

    response = asyncio.run(
        transcript_api.list_all_playlist_videos(
            "PL1234567890", background_tasks, preferred_language="it"
        )
    )

    assert response["preferred_language"] == "it"
    assert created_jobs[0]["preferred_language"] == "it"
    assert background_tasks.tasks[0].args == (
        response["job_id"],
        "PL1234567890",
        "it",
    )


def test_discovery_openapi_exposes_optional_language_query():
    schema = transcript_api.app.openapi()
    for path in (
        "/channel/{channel_name}/all-videos",
        "/playlist/{playlist_id}/all-videos",
    ):
        parameters = schema["paths"][path]["get"]["parameters"]
        language = next(item for item in parameters if item["name"] == "preferred_language")
        assert language["in"] == "query"
        assert language["required"] is False
