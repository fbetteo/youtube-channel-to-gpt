import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import creator_chat


def test_build_chunks_groups_by_time_and_keeps_timestamps():
    segments = [
        {"text": "Welcome back.", "start": 0.0, "duration": 3},
        {"text": "Today:\nideas", "start": 40.0, "duration": 5},
        {"text": "  ", "start": 60.0, "duration": 2},
        {"text": "First idea.", "start": 95.0, "duration": 4},
        {"text": "Second idea.", "start": 150.0, "duration": 6},
        {"text": "Thanks.", "start": 200.0, "duration": 2},
    ]

    assert creator_chat.build_chunks(segments) == [
        {"index": 0, "start": 0.0, "end": 45.0, "text": "Welcome back. Today: ideas"},
        {"index": 1, "start": 95.0, "end": 156.0, "text": "First idea. Second idea."},
        {"index": 2, "start": 200.0, "end": 202.0, "text": "Thanks."},
    ]


def test_build_chunks_of_empty_transcript():
    assert creator_chat.build_chunks([{"text": "", "start": 0, "duration": 1}]) == []


def test_embedding_input_prefixes_title():
    assert creator_chat.embedding_input("My video", "chunk") == "My video\n\nchunk"
    assert creator_chat.embedding_input(None, "chunk") == "chunk"


def test_to_pgvector():
    assert creator_chat.to_pgvector([0.5, -1.0, 0.0]) == "[0.5,-1.0,0.0]"
