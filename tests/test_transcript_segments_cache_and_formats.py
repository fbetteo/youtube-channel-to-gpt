import importlib.util
import io
import json
import sys
import types
from pathlib import Path

from botocore.exceptions import ClientError


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

import transcript_formats
import youtube_service


def load_lambda_module():
    module_path = PROJECT_ROOT / "lambda-transcript-processor" / "lambda_function.py"
    spec = importlib.util.spec_from_file_location("transcript_lambda", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeS3:
    """In-memory get_object/put_object with S3's NoSuchKey error shape."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body


SEGMENTS = [
    {"text": "first  line", "start": 0.0, "duration": 2.5},
    {"text": " \n ", "start": 1.0, "duration": 1.0},
    {"text": "second", "start": 2.0, "duration": 3.0},
    {"text": "late", "start": 3725.5, "duration": 1.25},
]


def segments_document(video_id="vid"):
    return {
        "schema_version": 1,
        "video_id": video_id,
        "requested_language": "en",
        "language": "en",
        "is_generated": True,
        "fetched_at": 1_000_000,
        "segments": SEGMENTS,
    }


# ---- Rendering ----


def test_srt_trims_overlaps_skips_empty_cues_and_formats_hours():
    assert transcript_formats.render_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:02,000\nfirst line\n\n"
        "2\n00:00:02,000 --> 00:00:05,000\nsecond\n\n"
        "3\n01:02:05,500 --> 01:02:06,750\nlate\n"
    )


def test_vtt_has_header_and_dot_milliseconds():
    rendered = transcript_formats.render_vtt(SEGMENTS)
    assert rendered.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nfirst line")


def test_json_without_timestamps_is_one_text_with_default_fields():
    rendered = json.loads(
        transcript_formats.render_document(segments_document(), "json", {}, "A title")
    )
    assert rendered == {
        "title": "A title",
        "video_id": "vid",
        "url": "https://www.youtube.com/watch?v=vid",
        "language": "en",
        "is_generated": True,
        "text": "first line second late",
    }


def test_json_with_timestamps_has_clean_segments_and_chosen_fields():
    rendered = json.loads(
        transcript_formats.render_document(
            segments_document(),
            "json",
            {
                "include_timestamps": True,
                "include_video_title": False,
                "include_video_url": False,
                "include_view_count": True,
            },
            "A title",
            1234,
        )
    )
    assert "title" not in rendered and "url" not in rendered and "text" not in rendered
    assert rendered["video_id"] == "vid"
    assert rendered["view_count"] == 1234
    assert rendered["segments"] == [
        {"start": 0.0, "duration": 2.5, "text": "first line"},
        {"start": 2.0, "duration": 3.0, "text": "second"},
        {"start": 3725.5, "duration": 1.25, "text": "late"},
    ]


def test_subtitle_formats_ignore_text_options():
    options = {"include_timestamps": False, "include_video_title": True}
    assert transcript_formats.render_document(
        segments_document(), "srt", options, "A title"
    ) == transcript_formats.render_srt(SEGMENTS)


# ---- Job downloads ----


def test_read_job_transcript_renders_from_segments_file():
    s3 = FakeS3(
        {
            "u/j/vid.txt": b"plain",
            "u/j/vid.json": json.dumps(segments_document()).encode(),
        }
    )
    content, extension = youtube_service.read_job_transcript(
        s3, "bucket", "u/j/vid.txt", "srt"
    )
    assert extension == "srt"
    assert content.decode().startswith("1\n00:00:00,000 --> 00:00:02,000")


def test_read_job_transcript_falls_back_to_txt_for_older_jobs():
    s3 = FakeS3({"u/j/vid.txt": b"plain"})
    assert youtube_service.read_job_transcript(
        s3, "bucket", "u/j/vid.txt", "vtt"
    ) == (b"plain", "txt")


def test_txt_download_renders_from_segments_with_job_options():
    s3 = FakeS3(
        {
            "u/j/vid.txt": b"stale stored text",
            "u/j/vid.json": json.dumps(segments_document()).encode(),
        }
    )
    content, extension = youtube_service.read_job_transcript(
        s3,
        "bucket",
        "u/j/vid.txt",
        title="Title",
        view_count=1234,
        formatting_options={"include_video_url": False, "include_view_count": True},
    )
    assert extension == "txt"
    assert content.decode() == (
        "Video Title: Title\nVideo ID: vid\nView Count: 1,234\n\n"
        "first line second late"
    )


def test_txt_download_uses_stored_text_without_segments():
    s3 = FakeS3({"u/j/vid.txt": b"plain"})
    assert youtube_service.read_job_transcript(s3, "bucket", "u/j/vid.txt") == (
        b"plain",
        "txt",
    )


class BrokenSegmentsS3(FakeS3):
    def get_object(self, Bucket, Key):
        if Key.endswith(".json"):
            raise ClientError(
                {"Error": {"Code": "SlowDown", "Message": "busy"}}, "GetObject"
            )
        return super().get_object(Bucket, Key)


def test_txt_download_falls_back_to_stored_text_when_segments_fail():
    s3 = BrokenSegmentsS3({"u/j/vid.txt": b"plain"})
    assert youtube_service.read_job_transcript(s3, "bucket", "u/j/vid.txt") == (
        b"plain",
        "txt",
    )


def test_subtitle_download_surfaces_segment_read_errors():
    s3 = BrokenSegmentsS3({"u/j/vid.txt": b"plain"})
    try:
        youtube_service.read_job_transcript(s3, "bucket", "u/j/vid.txt", "srt")
    except ClientError:
        return
    raise AssertionError("expected the segments read error")


def test_concatenation_only_applies_to_txt():
    job = {"formatting_options": {"concatenate_all": True}}
    assert youtube_service.should_concatenate(job, "txt")
    assert not youtube_service.should_concatenate(job, "srt")
    assert not youtube_service.should_concatenate({}, "txt")


def test_transcript_filename_uses_format_extension():
    assert (
        youtube_service.build_transcript_filename("vid", "Title", extension="srt")
        == "Title_vid.srt"
    )


def test_overrides_replace_only_given_job_options():
    merged = transcript_formats.merge_formatting_options(
        {"include_timestamps": False, "include_video_url": True, "other": 1},
        {"include_timestamps": True, "include_video_url": None, "unknown": True},
    )
    assert merged == {"include_timestamps": True, "include_video_url": True, "other": 1}


def _zip_entries(zip_buffer):
    import zipfile

    with zipfile.ZipFile(zip_buffer) as archive:
        return {name: archive.read(name).decode() for name in archive.namelist()}


def _patch_job(monkeypatch, s3, formatting_options):
    job = {
        "status": "completed",
        "formatting_options": json.dumps(formatting_options),
        "source_name": "Source",
        "completed": 2,
        "files": [
            {"video_id": "new", "title": "New", "s3_key": "u/j/new.txt", "view_count": 5},
            {"video_id": "old", "title": "Old", "s3_key": "u/j/old.txt", "view_count": 7},
        ],
    }

    async def get_job(job_id, include_videos=False):
        return json.loads(json.dumps(job))

    monkeypatch.setattr(youtube_service.hybrid_job_manager, "get_job", get_job)
    monkeypatch.setattr(youtube_service, "get_s3_client", lambda: (s3, "bucket"))
    monkeypatch.setattr(youtube_service, "get_s3_fallback_client", lambda: (None, None))


def test_zip_download_applies_overrides_to_rendered_videos_only(monkeypatch):
    import asyncio

    s3 = FakeS3(
        {
            "u/j/new.txt": b"stored new",
            "u/j/new.json": json.dumps(segments_document("new")).encode(),
            "u/j/old.txt": b"stored old",
        }
    )
    _patch_job(monkeypatch, s3, {"include_timestamps": False})

    as_saved = _zip_entries(
        asyncio.run(youtube_service.create_transcript_zip_from_s3_concurrent("j"))
    )
    overridden = _zip_entries(
        asyncio.run(
            youtube_service.create_transcript_zip_from_s3_concurrent(
                "j",
                option_overrides={
                    "include_timestamps": True,
                    "include_video_url": False,
                    "include_view_count": None,
                },
            )
        )
    )

    assert as_saved["New_new.txt"] == (
        "Video Title: New\nVideo ID: new\n"
        "URL: https://www.youtube.com/watch?v=new\n\nfirst line second late"
    )
    assert overridden["New_new.txt"] == (
        "Video Title: New\nVideo ID: new\n\n"
        "[00:00] first line\n[00:02] second\n[62:05] late"
    )
    # Created before segments were stored: the text cannot be re-rendered.
    assert as_saved["Old_old.txt"] == overridden["Old_old.txt"] == "stored old"


def test_zip_download_can_override_concatenation(monkeypatch):
    import asyncio

    s3 = FakeS3({"u/j/new.txt": b"stored new", "u/j/old.txt": b"stored old"})
    _patch_job(monkeypatch, s3, {})

    entries = _zip_entries(
        asyncio.run(
            youtube_service.create_transcript_zip_from_s3_concurrent(
                "j", option_overrides={"concatenate_all": True}
            )
        )
    )

    assert list(entries) == ["Source_all_transcripts.txt"]
    assert "stored new" in entries["Source_all_transcripts.txt"]


# ---- Worker cache ----


class FakeTranscript:
    language_code = "en"
    is_generated = False


class FakeFetched:
    def to_raw_data(self):
        return SEGMENTS


def run_handler(lambda_module, monkeypatch, s3, fetch, **event_fields):
    monkeypatch.setattr(lambda_module.boto3, "client", lambda service: s3)
    monkeypatch.setattr(lambda_module, "fetch_transcript_before_deadline", fetch)
    monkeypatch.setattr(lambda_module, "send_result_to_sqs", lambda message: True)
    event = {
        "video_id": "vid",
        "job_id": "job",
        "user_id": "user",
        "preferred_language": "EN",
        "pre_fetched_metadata": {"title": "Title"},
    }
    event.update(event_fields)
    return lambda_module.lambda_handler(
        event, types.SimpleNamespace(aws_request_id="req")
    )


def fetch_fake(video_id, language, context):
    return FakeTranscript(), FakeFetched(), 1


def test_rendered_text_matches_the_worker_text_for_every_option(monkeypatch):
    lambda_module = load_lambda_module()
    for options in (
        {},
        {"include_timestamps": True, "include_view_count": True},
        {
            "include_video_title": False,
            "include_video_id": False,
            "include_video_url": False,
        },
    ):
        s3 = FakeS3()
        run_handler(
            lambda_module,
            monkeypatch,
            s3,
            fetch_fake,
            pre_fetched_metadata={"title": "Título", "viewCount": 98765},
            **options,
        )
        rendered = transcript_formats.render_text(
            json.loads(s3.objects["user/job/vid.json"]), options, "Título", 98765
        )
        assert rendered.encode("utf-8") == s3.objects["user/job/vid.txt"], options


def test_worker_cache_miss_fetches_and_stores_cache_job_segments_and_text(
    monkeypatch,
):
    lambda_module = load_lambda_module()
    s3 = FakeS3()
    calls = []

    def fetch(video_id, language, context):
        calls.append(language)
        return FakeTranscript(), FakeFetched(), 1

    result = run_handler(lambda_module, monkeypatch, s3, fetch)

    assert result["statusCode"] == 200
    assert calls == ["en"]
    assert result["body"]["metadata"]["transcript_source"] == "youtube"
    cached = json.loads(s3.objects["transcript-cache/vid/en.json"])
    assert cached["segments"] == SEGMENTS
    assert cached["is_generated"] is False
    assert json.loads(s3.objects["user/job/vid.json"]) == cached
    assert s3.objects["user/job/vid.txt"].decode().startswith("Video Title: Title")


def test_worker_cache_hit_skips_youtube(monkeypatch):
    lambda_module = load_lambda_module()
    document = segments_document()
    document["fetched_at"] = int(lambda_module.time.time())
    s3 = FakeS3({"transcript-cache/vid/en.json": json.dumps(document).encode()})

    def fetch(*args):
        raise AssertionError("cache hit must not call YouTube")

    result = run_handler(lambda_module, monkeypatch, s3, fetch)

    assert result["statusCode"] == 200
    assert result["body"]["metadata"]["transcript_source"] == "cache"
    assert result["body"]["metadata"]["transcript_type"] == "auto-generated"
    assert "user/job/vid.txt" in s3.objects
    assert "user/job/vid.json" in s3.objects


def test_worker_cache_never_expires_by_default(monkeypatch):
    monkeypatch.delenv("TRANSCRIPT_CACHE_MAX_AGE_DAYS", raising=False)
    lambda_module = load_lambda_module()
    s3 = FakeS3(
        {"transcript-cache/vid/en.json": json.dumps(segments_document()).encode()}
    )
    key = "transcript-cache/vid/en.json"
    ten_years = 10 * 365 * 86400

    assert lambda_module.TRANSCRIPT_CACHE_MAX_AGE_DAYS is None
    assert lambda_module.load_cached_transcript(
        s3, "b", key, now=1_000_000 + ten_years
    )
    assert lambda_module.load_cached_transcript(s3, "b", "missing") is None


def test_worker_cache_max_age_and_disable(monkeypatch):
    key = "transcript-cache/vid/en.json"
    s3 = FakeS3({key: json.dumps(segments_document()).encode()})

    monkeypatch.setenv("TRANSCRIPT_CACHE_MAX_AGE_DAYS", "30")
    lambda_module = load_lambda_module()
    thirty_days = 30 * 86400
    assert lambda_module.load_cached_transcript(s3, "b", key, now=1_000_060)
    assert (
        lambda_module.load_cached_transcript(
            s3, "b", key, now=1_000_000 + thirty_days + 1
        )
        is None
    )

    monkeypatch.setenv("TRANSCRIPT_CACHE_MAX_AGE_DAYS", "0")
    lambda_module = load_lambda_module()
    assert lambda_module.load_cached_transcript(s3, "b", key, now=1_000_060) is None


def test_worker_cache_key_defaults_to_auto():
    lambda_module = load_lambda_module()
    assert lambda_module.transcript_cache_key("vid", None) == (
        "transcript-cache/vid/auto.json"
    )


# ---- Channel discovery ----


def test_channel_discovery_includes_streams_and_survives_a_failing_tab(monkeypatch):
    def fake_extract(url, base_opts, preferred_language):
        tab = url.rsplit("/", 1)[-1]
        if tab == "shorts":
            raise RuntimeError("tab missing")
        entry_id = {"videos": "v1", "streams": "s1"}[tab]
        return {"entries": [{"id": entry_id, "title": f"{tab} title", "duration": 900}]}

    monkeypatch.setattr(youtube_service, "_extract_flat_info", fake_extract)

    videos = youtube_service._fetch_all_channel_videos("UCchannel")

    assert {(v["id"], v["type"]) for v in videos} == {("v1", "video"), ("s1", "live")}
