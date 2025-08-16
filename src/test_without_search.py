from googleapiclient.discovery import build

from src.config_v2 import settings
# Replace with your API key and channel ID
API_KEY = settings.youtube_api_key
CHANNEL_ID = "UC_x5XG1OV2P6uZZ5FSM9Ttw" # Example: Google Developers
CHANNEL_ID = "UC8T50lSOidYquOjjVT07mZA" 

youtube = build("youtube", "v3", developerKey=API_KEY)

# Step 1: Get the uploads playlist ID
channel_response = youtube.channels().list(
    part="contentDetails",
    id=CHANNEL_ID
).execute()

uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

# Step 2: Get all video IDs from the uploads playlist
video_ids = []
next_page_token = None

while True:
    playlist_response = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=50,
        pageToken=next_page_token
    ).execute()

    for item in playlist_response["items"]:
        video_ids.append(item["contentDetails"]["videoId"])

    next_page_token = playlist_response.get("nextPageToken")
    if not next_page_token:
        break

print(video_ids)