# YouTube Transcript API

A dedicated API for downloading YouTube video transcripts, with features for both individual videos and entire channels.

## Features

- **Free Tier (No Authentication Required):**
  - Download transcripts for individual YouTube videos
  - Rate limited to 3 downloads per hour per IP address

- **Authenticated Tier:**
  - Download all transcripts from a YouTube channel as a ZIP file
  - Get channel information (title, logo, video count, etc.)
  - List videos from a channel with metadata


## Setup

### Prerequisites

- Python 3.8+
- YouTube Data API key
- (Optional) WebShare proxy credentials for YouTube Transcript API

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube-channel-to-gpt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Option 1: Install with pip
pip install -r requirements.txt
pip install pydantic-settings==2.1.0

# Option 2: Run the update script
python update_dependencies.py
```

4. Create a `.env` file with your configuration:
```
YOUTUBE_API_KEY=your_youtube_api_key
TRANSCRIPT_API_KEY=your_api_secret_key
WEBSHARE_PROXY_USERNAME=your_proxy_username  # Optional
WEBSHARE_PROXY_PASSWORD=your_proxy_password  # Optional
```

### Running the API

Start the API server:

```bash
python src/transcript_api.py
```

The API will be available at http://localhost:8000
