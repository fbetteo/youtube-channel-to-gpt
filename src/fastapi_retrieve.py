import os
import json
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Load environment variables
load_dotenv()

# API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME")
WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD")

youtube = build("youtube", "v3", developerKey=API_KEY)


ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=WEBSHARE_PROXY_USERNAME, proxy_password=WEBSHARE_PROXY_PASSWORD
    )
)


def sanitize_filename(filename):
    """Removes characters that are invalid in Windows and Linux filenames."""
    # Remove characters invalid in Windows/Linux: < > : " / \ | ? * and control chars (0-31)
    # Also remove leading/trailing spaces/dots which can cause issues on Windows.
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
    # Replace multiple spaces with a single space
    sanitized = re.sub(r"\s+", " ", sanitized)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    # Prevent empty filenames
    if not sanitized:
        sanitized = "_"
    # Optionally, limit the length (common filesystem limit is 255 bytes, but keep it shorter)
    max_len = 30  # Adjust as needed
    return sanitized[:max_len]


class VideoRetrieval:
    def __init__(self, channel_name, max_results, user_id):
        self.channel_name = channel_name
        # self.channel_id = channel_id
        self.max_results = max_results
        self.transcript_files = []
        self.build_dir = f"../build/{user_id}"
        self.video_metadata = {}  # Store video metadata (ID -> {title, link})
        # Create build directory if it doesn't exist
        os.makedirs(self.build_dir, exist_ok=True)

    def get_channel_id(self):
        request = youtube.search().list(
            part="snippet", type="channel", q=self.channel_name, maxResults=1
        )
        response = request.execute()

        if response["items"]:
            # Assuming the first search result is the desired channel
            self.channel_id = response["items"][0]["id"]["channelId"]
            return self.channel_id
        else:
            return "Channel not found"

    def get_medium_videos(self):
        medium_videos = (
            youtube.search()
            .list(
                part="snippet",
                channelId=self.channel_id,
                type="video",
                videoDuration="medium",
                maxResults=self.max_results,
            )
            .execute()
        )
        return medium_videos

    def get_long_videos(self):
        long_videos = (
            youtube.search()
            .list(
                part="snippet",
                channelId=self.channel_id,
                type="video",
                videoDuration="long",
                maxResults=self.max_results,
            )
            .execute()
        )
        return long_videos

    def get_video_ids(self):
        medium_videos = self.get_medium_videos()
        long_videos = self.get_long_videos()
        self.video_ids = []
        for video in medium_videos["items"] + long_videos["items"]:
            if video["id"]["kind"] == "youtube#video":
                self.video_ids.append(video["id"]["videoId"])
        return self.video_ids

    def get_transcripts(self, video_ids=None):
        print("Starting transcription")

        if video_ids is None:
            video_ids = self.video_ids

        self.all_transcripts = ""
        self.transcript_files = []

        def fetch_transcript(youtubeId):
            try:
                retrievedTranscript = ytt_api.fetch(youtubeId)
                print(f"Retrieved transcript for {youtubeId}")
                transcribedText = ""

                # Get video details
                video_request = youtube.videos().list(part="snippet", id=youtubeId)
                video_response = video_request.execute()
                video_title = video_response["items"][0]["snippet"]["title"]
                sanitized_title = sanitize_filename(video_title)
                video_link = f"https://youtu.be/{youtubeId}"

                # Save metadata
                self.video_metadata[youtubeId] = {
                    "title": video_title,
                    "link": video_link,
                }

                # Iterate through the transcript and add each section to a string
                for transcribedSection in retrievedTranscript:
                    transcribedText += transcribedSection.text + " "

                # Save individual transcript file with video title
                video_folder_path = os.path.join(self.build_dir, youtubeId)
                os.makedirs(video_folder_path, exist_ok=True)
                file_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_{youtubeId}.txt"
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Video Title: {video_title}\nVideo ID: {youtubeId}\n\n")
                    f.write(transcribedText)

                return file_path, transcribedText
            except Exception as e:
                print(f"Could not retrieve transcript for {youtubeId}")
                print(f"Error: {e}")
                return None, None

        with ThreadPoolExecutor(
            max_workers=5
        ) as executor:  # Adjust max_workers as needed
            future_to_video = {
                executor.submit(fetch_transcript, vid): vid for vid in video_ids
            }

            for future in as_completed(future_to_video):
                file_path, transcribedText = future.result()
                if file_path and transcribedText:
                    self.transcript_files.append(file_path)
                    self.all_transcripts += transcribedText

        if not self.transcript_files:
            raise ValueError("No transcripts found or could not be retrieved")

        return self.transcript_files


### TEST CODE ###
# from fastapi import FastAPI, HTTPException
# CHANNEL_NAME = "marc-lou"

# video_retrieval = VideoRetrieval(CHANNEL_NAME, 5)
# try:
#     video_retrieval.get_channel_id()
# except Exception as e:
#     raise HTTPException(
#         status_code=400, detail="Error in get_channel_id()" + str(e)
#     )

# try:
#     video_retrieval.get_video_ids()
# except Exception as e:
#     raise HTTPException(status_code=400, detail="Error in get_video_ids()" + str(e))

# try:
#     video_retrieval.get_transcripts()
# except Exception as e:
#     raise HTTPException(
#         status_code=400, detail="Error in get_transcripts()" + str(e)
#     )


# retrievedTranscript = YouTubeTranscriptApi.get_transcript("40zozi-rGQM")


# retrievedTranscript

# aa = YouTubeTranscriptApi()
# aa.fetch()
# new_retrievedTranscript = aa.fetch("40zozi-rGQM")
# new_retrievedTranscript

# transcribedText = ""
# for transcribedSection in new_retrievedTranscript:
#                 transcribedText += transcribedSection["text"] + " "
