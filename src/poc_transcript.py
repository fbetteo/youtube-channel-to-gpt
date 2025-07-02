from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()

video_id = "ebGUMEcL9MA"
# ebGUMEcL9MA

transcript_list = ytt_api.list(video_id)

transcript_list
transcript = ytt_api.fetch(video_id)


c = list(transcript_list)

c[0].language_code

c[0].fetch()
