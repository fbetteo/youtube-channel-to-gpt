def extract_youtube_id(url: str) -> str:
    """Extract video ID from various forms of YouTube URLs.

    Args:
        url: A YouTube URL in various formats

    Returns:
        The extracted video ID
    """
    # Common YouTube URL patterns
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|\/watch\?v=)([^#\&\?\/\s]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matches, perhaps the URL is already just the ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    raise ValueError(f"Could not extract YouTube video ID from URL: {url}")


def get_channel_info(self):
    """Get detailed information about the channel including logo, video count, etc."""
    request = youtube.channels().list(part="snippet,statistics", id=self.channel_id)
    response = request.execute()

    if not response["items"]:
        raise ValueError(f"No channel found with ID: {self.channel_id}")

    channel_info = response["items"][0]
    return {
        "title": channel_info["snippet"]["title"],
        "description": channel_info["snippet"].get("description", ""),
        "thumbnail": channel_info["snippet"]["thumbnails"]["high"]["url"],
        "videoCount": int(channel_info["statistics"]["videoCount"]),
        "subscriberCount": int(channel_info["statistics"].get("subscriberCount", 0)),
        "viewCount": int(channel_info["statistics"].get("viewCount", 0)),
        "channelId": self.channel_id,
    }


def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a specific YouTube video.

    Args:
        video_id: The YouTube video ID

    Returns:
        Dictionary containing video metadata like title, author, views, etc.

    Raises:
        ValueError: If the video ID is invalid or the video doesn't exist
    """
    try:
        # Get video details from YouTube API
        video_request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id,
        )
        video_response = video_request.execute()

        if not video_response.get("items"):
            raise ValueError(f"No video found with ID: {video_id}")

        video_data = video_response["items"][0]
        snippet = video_data["snippet"]
        statistics = video_data["statistics"]
        content_details = video_data["contentDetails"]

        # Format duration string (PT1H2M3S -> 1:02:03)
        duration = content_details.get("duration", "PT0S")
        duration_str = self._format_duration(duration)

        # Extract thumbnail URLs
        thumbnails = snippet.get("thumbnails", {})
        best_thumbnail = (
            thumbnails.get("maxres")
            or thumbnails.get("high")
            or thumbnails.get("medium")
            or thumbnails.get("default")
            or {}
        )

        # Build metadata dictionary
        metadata = {
            "id": video_id,
            "title": snippet.get("title", "Untitled"),
            "description": snippet.get("description", ""),
            "channelId": snippet.get("channelId", ""),
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt": snippet.get("publishedAt", ""),
            "thumbnail": best_thumbnail.get("url", ""),
            "duration": duration_str,
            "viewCount": int(statistics.get("viewCount", 0)),
            "likeCount": int(statistics.get("likeCount", 0)),
            "commentCount": int(statistics.get("commentCount", 0)),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        return metadata

    except Exception as e:
        raise ValueError(f"Failed to get metadata for video {video_id}: {str(e)}")


def _format_duration(self, duration_str: str) -> str:
    """
    Format ISO 8601 duration string (PT1H2M3S) to readable format (1:02:03)

    Args:
        duration_str: ISO 8601 duration string from YouTube API

    Returns:
        Formatted duration string
    """
    match_hours = re.search(r"(\d+)H", duration_str)
    match_minutes = re.search(r"(\d+)M", duration_str)
    match_seconds = re.search(r"(\d+)S", duration_str)

    hours = int(match_hours.group(1)) if match_hours else 0
    minutes = int(match_minutes.group(1)) if match_minutes else 0
    seconds = int(match_seconds.group(1)) if match_seconds else 0

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"
