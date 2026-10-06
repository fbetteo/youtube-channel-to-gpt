"""
Render a stored segments document (written by the Lambda worker next to each
transcript .txt) as a download format: text with the job's headers, SRT, VTT,
or JSON.

render_text must stay byte-identical to the worker's .txt (lambda_function.py)
while the worker still writes both; scripts/compare_rendered_txt.py checks it.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import Query

# Download formats; FastAPI validates request values against this type.
OutputFormat = Literal["txt", "srt", "vtt", "json"]

# Job formatting options a download may override (None keeps the job's value).
OVERRIDABLE_OPTIONS = (
    "include_timestamps",
    "include_video_title",
    "include_video_id",
    "include_video_url",
    "include_view_count",
    "concatenate_all",
)


def merge_formatting_options(
    job_options: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """The job's saved options with any non-None download overrides applied."""
    merged = dict(job_options or {})
    for key, value in (overrides or {}).items():
        if key in OVERRIDABLE_OPTIONS and value is not None:
            merged[key] = value
    return merged


def download_option_overrides(
    include_timestamps: Optional[bool] = Query(
        None, description="Override the job's timestamp setting for this download"
    ),
    include_video_title: Optional[bool] = Query(
        None, description="Override the job's 'Video Title:' header setting"
    ),
    include_video_id: Optional[bool] = Query(
        None, description="Override the job's 'Video ID:' header setting"
    ),
    include_video_url: Optional[bool] = Query(
        None, description="Override the job's 'URL:' header setting"
    ),
    include_view_count: Optional[bool] = Query(
        None, description="Override the job's 'View Count:' header setting"
    ),
    concatenate_all: Optional[bool] = Query(
        None, description="Override the job's single-file setting (txt only)"
    ),
) -> Dict[str, Optional[bool]]:
    """FastAPI dependency: txt download overrides given as query parameters."""
    return {
        "include_timestamps": include_timestamps,
        "include_video_title": include_video_title,
        "include_video_id": include_video_id,
        "include_video_url": include_video_url,
        "include_view_count": include_view_count,
        "concatenate_all": concatenate_all,
    }


def normalize_caption_text(text: Any) -> str:
    """Collapse subtitle-internal line breaks and repeated whitespace."""
    return " ".join(str(text or "").split())


def format_transcript_segments(
    transcript_data: List[Dict[str, Any]], include_timestamps: bool
) -> str:
    """Build transcript text after normalizing each caption segment."""
    if not include_timestamps:
        return " ".join(
            cleaned_text
            for segment in transcript_data
            if (cleaned_text := normalize_caption_text(segment.get("text")))
        )

    transcript_lines = []
    for segment in transcript_data:
        cleaned_text = normalize_caption_text(segment.get("text"))
        if not cleaned_text:
            continue
        start_time_sec = segment["start"]
        minutes = int(start_time_sec // 60)
        seconds = int(start_time_sec % 60)
        transcript_lines.append(f"[{minutes:02d}:{seconds:02d}] {cleaned_text}")
    return "\n".join(transcript_lines)


def render_text(
    document: Dict[str, Any],
    formatting_options: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    view_count: Optional[int] = None,
) -> str:
    """The job's .txt: optional headers, a blank line, then the transcript."""
    options = formatting_options or {}
    video_id = document.get("video_id")

    header = ""
    if options.get("include_video_title", True):
        header += f"Video Title: {title or 'Untitled_Video'}\n"
    if options.get("include_video_id", True):
        header += f"Video ID: {video_id}\n"
    if options.get("include_video_url", True):
        header += f"URL: https://www.youtube.com/watch?v={video_id}\n"
    if options.get("include_view_count", False):
        header += f"View Count: {int(view_count or 0):,}\n"
    if header:
        header += "\n"

    return header + format_transcript_segments(
        document.get("segments", []),
        include_timestamps=options.get("include_timestamps", False),
    )


def _cue_text(segment: Dict[str, Any]) -> str:
    return normalize_caption_text(segment.get("text"))


def _cues(segments: List[Dict[str, Any]]) -> List[tuple]:
    """(start, end, text) per non-empty segment, with overlaps trimmed."""
    cues = []
    usable = [s for s in segments if _cue_text(s)]
    for index, segment in enumerate(usable):
        start = float(segment.get("start") or 0)
        end = start + float(segment.get("duration") or 0)
        # Auto-generated captions overlap; end each cue where the next begins.
        if index + 1 < len(usable):
            next_start = float(usable[index + 1].get("start") or 0)
            if start < next_start < end:
                end = next_start
        cues.append((start, end, _cue_text(segment)))
    return cues


def _timestamp(seconds: float, separator: str) -> str:
    total_ms = int(round(max(seconds, 0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def render_srt(segments: List[Dict[str, Any]]) -> str:
    blocks = [
        f"{i}\n{_timestamp(start, ',')} --> {_timestamp(end, ',')}\n{text}"
        for i, (start, end, text) in enumerate(_cues(segments), 1)
    ]
    return "\n\n".join(blocks) + "\n"


def render_vtt(segments: List[Dict[str, Any]]) -> str:
    blocks = [
        f"{_timestamp(start, '.')} --> {_timestamp(end, '.')}\n{text}"
        for start, end, text in _cues(segments)
    ]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def render_json(
    document: Dict[str, Any],
    formatting_options: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    view_count: Optional[int] = None,
) -> str:
    """
    One JSON object per video. The header options pick the metadata fields, as
    in render_text; include_timestamps picks timed `segments` over one `text`.
    """
    options = formatting_options or {}
    video_id = document.get("video_id")
    segments = document.get("segments", [])

    record: Dict[str, Any] = {}
    if options.get("include_video_title", True):
        record["title"] = title
    if options.get("include_video_id", True):
        record["video_id"] = video_id
    if options.get("include_video_url", True):
        record["url"] = f"https://www.youtube.com/watch?v={video_id}"
    if options.get("include_view_count", False):
        record["view_count"] = int(view_count or 0)
    record["language"] = document.get("language")
    record["is_generated"] = document.get("is_generated")

    if options.get("include_timestamps", False):
        record["segments"] = [
            {
                "start": round(float(segment.get("start") or 0), 3),
                "duration": round(float(segment.get("duration") or 0), 3),
                "text": text,
            }
            for segment in segments
            if (text := normalize_caption_text(segment.get("text")))
        ]
    else:
        record["text"] = format_transcript_segments(segments, include_timestamps=False)

    return json.dumps(record, ensure_ascii=False, indent=2)


def render_document(
    document: Dict[str, Any],
    output_format: str,
    formatting_options: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    view_count: Optional[int] = None,
) -> str:
    """Render segments as srt, vtt, or json (subtitle formats ignore the options)."""
    segments = document.get("segments", [])
    if output_format == "srt":
        return render_srt(segments)
    if output_format == "vtt":
        return render_vtt(segments)
    if output_format == "json":
        return render_json(document, formatting_options, title, view_count)
    raise ValueError(f"Cannot render format {output_format!r} from segments")
