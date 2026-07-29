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

### test list of videos in a playlist
url = f"https://www.youtube.com/channel/UCXUPKJO5MZQN11PqgIvyuvQ/playlists"

ydl_opts = {
    "quiet": True,
    "extract_flat": True,
    "dump_single_json": True,
    "ignoreerrors": True,
}
ydl_opts = _get_ydl_opts(ydl_opts)

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

info
info.keys()

info["entries"]

# poetry run python -m yt_dlp --flat-playlist --playlist-end 5 --print "%(id)s | views=%(view_count)s | %(title)s" "https://www.youtube.com/@andrejkarpathy/videos"


# extract in playlist
url = f"https://www.youtube.com/channel/UCXUPKJO5MZQN11PqgIvyuvQ/playlists"

ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": "in_playlist",
    # "dump_single_json": True,
    "ignoreerrors": True,
}
ydl_opts = _get_ydl_opts(ydl_opts)

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

info
info.keys()

info["entries"]


## Channel test

url = "https://www.youtube.com/@andrejkarpathy/videos"

ydl_opts = {
    "quiet": True,
    "extract_flat": True,
    "playlistend": 5,
    "dump_single_json": True,
    "ignoreerrors": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

for entry in info["entries"]:
    print(
        entry.get("id"),
        entry.get("view_count"),
        entry.get("duration"),
        entry.get("title"),
    )


##

url = f"https://www.youtube.com/channel/UCXUPKJO5MZQN11PqgIvyuvQ/videos"

ydl_opts = {
    "quiet": True,
    "extract_flat": True,
    "dump_single_json": True,
    "ignoreerrors": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

for entry in info["entries"]:
    print(
        entry.get("id"),
        entry.get("view_count"),
        entry.get("duration"),
        entry.get("title"),
    )

info
