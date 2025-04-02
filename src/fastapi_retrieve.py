import os
import json
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)


class VideoRetrieval:
    def __init__(self, channel_name, max_results):
        self.channel_name = channel_name
        # self.channel_id = channel_id
        self.max_results = max_results

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
        # Iterate through the array of YouTube IDs
        for youtubeId in video_ids:
            # Retrieve the transcript for the video
            try:
                retrievedTranscript = YouTubeTranscriptApi.get_transcript(youtubeId)
                print("Retrieved transcript for " + youtubeId)
                transcribedText = ""
                time.sleep(1)
            except Exception as e:
                print("Could not retrieve transcript for " + youtubeId)
                print(f"Error: {e}")
                time.sleep(1)
                continue
            finally:
                print("Continuing to next video")

            # Iterate through the transcript and add each section to a string
            for transcribedSection in retrievedTranscript:
                transcribedText += transcribedSection["text"] + " "

            self.all_transcripts += transcribedText

            if self.all_transcripts == "":
                raise ValueError("No transcripts found or could not be retrieved")

            # Write the transcribed text to a transcript file
            print("Writing transcript for " + youtubeId + " to file")
            transcriptionFile = open(
                f"../build/transcript_{self.channel_id}.txt", "a", encoding="utf-8"
            )
            transcriptionFile.write(transcribedText)
            transcriptionFile.close()

        print("Finished transcribing")


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
