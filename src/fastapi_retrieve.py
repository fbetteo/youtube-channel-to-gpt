import os
import json

from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)


class VideoRetrieval:
    def __init__(self, channel_id, max_results):
        self.channel_id = channel_id
        self.max_results = max_results

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
        # Iterate through the array of YouTube IDs
        for youtubeId in video_ids:
            # Retrieve the transcript for the video
            try:
                retrievedTranscript = YouTubeTranscriptApi.get_transcript(youtubeId)
                print("Retrieved transcript for " + youtubeId)
                transcribedText = ""
            except:
                print("Could not retrieve transcript for " + youtubeId)
                continue
            finally:
                print("Continuing to next video")

            # Iterate through the transcript and add each section to a string
            for transcribedSection in retrievedTranscript:
                transcribedText += transcribedSection["text"] + " "

            self.all_transcripts += transcribedText

            # Write the transcribed text to a transcript file
            print("Writing transcript for " + youtubeId + " to file")
            transcriptionFile = open(
                f"../build/transcript_{self.channel_id}.txt", "a", encoding="utf-8"
            )
            transcriptionFile.write(transcribedText)
            transcriptionFile.close()

        print("Finished transcribing")
