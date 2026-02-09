import os
import sys

# Add src directory to Python path
workspace_root = r"c:\Users\franb\projects\youtube-ai\youtube-channel-to-gpt"
src_path = os.path.join(workspace_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now imports will work
from config_v2 import settings
from hybrid_job_manager import hybrid_job_manager
from youtube_service import _get_ydl_opts
import yt_dlp

ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": False,
    "dump_single_json": True,
    # 'extractor_args': {'youtubetab': {'approximate_date': ['']}}
}

url = "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ"
# Add proxy if configured
ydl_opts = _get_ydl_opts(ydl_opts)
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    respones = ydl.extract_info(url, download=False)

respones


from youtube_service import _fetch_all_playlist_videos

asd = _fetch_all_playlist_videos("PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ")


asd
