# YouTube Transcript & GPT Assistant Services

## Project Overview

This repository contains **two independent FastAPI projects** that share infrastructure:

### **PRIMARY FOCUS: Transcript API** (`src/transcript_api.py`)
- **Purpose**: Standalone YouTube transcript download service with credit-based billing
- **Status**: Active development focus
- **Key Features**: Channel/video transcript extraction, Stripe payments, user credit management

### **Legacy: YouTube Chat Assistant** (`src/fastapi_main.py`) 
- **Purpose**: Creates OpenAI assistants from YouTube transcripts for chat interactions
- **Status**: Functional but not current development focus
- **Future**: Will eventually depend on transcript API service

---

## **TRANSCRIPT API SERVICE** - Main Development Focus

### Core Architecture
The transcript API (`src/transcript_api.py`) is a **standalone service** for downloading YouTube transcripts with:
- **Credit-based access control** (authenticated users consume credits)
- **Anonymous rate limiting** (for unauthenticated users)
- **Stripe payment integration** for credit purchases
- **Channel/video download support** with progress tracking

### Key Data Flow
1. **Authentication**: Supabase JWT validation or anonymous rate limiting
2. **Credit Management**: Check/deduct credits before transcript attempts
3. **Download Processing**: `youtube_service.py` handles YouTube API calls and transcript extraction
4. **File Management**: Organized storage in `build/transcripts/` by user/session

### Authentication & Access Control
```python
# Supabase JWT validation for authenticated users
async def validate_jwt(credentials: HTTPAuthorizationCredentials = Depends(security))

# Anonymous users get rate limiting
@app.post("/download/transcript")
async def download_transcript(user_info: Dict = Depends(get_user_or_anonymous))
```

### Database Schema (Transcript API)
- **Primary DB**: `db_youtube_transcripts/database.py` - dedicated connection
- **user_credits table**: Tracks credit balances by Supabase user ID
- **Credit deduction**: Happens before download attempts (even on failure)

### File Organization (Transcript API)
```
build/transcripts/
├── anonymous-{ip}/          # Anonymous user sessions
│   └── {session_id}/
└── {user_id}/              # Authenticated users
    └── {job_id}/           # Download job results
```

## Development Workflows (Transcript API)

### Running the Service
```bash
# Transcript API (primary focus)
uvicorn src.transcript_api:app --reload --port 8001

# Alternative startup method
cd src && python transcript_api.py
```

### Environment Setup (Transcript API)
Required variables for transcript service:
```bash
# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key

# Authentication
SUPABASE_SECRET_YOUTUBE_TRANSCRIPTS=your_supabase_secret

# Database (dedicated connection)
DB_HOST_YOUTUBE_TRANSCRIPTS=your_db_host
DB_NAME_YOUTUBE_TRANSCRIPTS=your_db_name
DB_USERNAME_YOUTUBE_TRANSCRIPTS=your_db_user
DB_PASSWORD_YOUTUBE_TRANSCRIPTS=your_db_pass

# Stripe payments
STRIPE_SECRET_KEY_LIVE=your_stripe_secret
STRIPE_WEBHOOK_SECRET_TRANSCRIPTS=your_webhook_secret

# Optional: Proxy for rate limit bypass
WEBSHARE_PROXY_USERNAME=your_proxy_user
WEBSHARE_PROXY_PASSWORD=your_proxy_pass
```

### Current Active API Endpoints (Post-Cleanup)
**Core Transcript Functionality** (quota-optimized):
- `GET /channel/{channel_name}` - Channel info with video listing
- `GET /channel/{channel_name}/all-videos` - Complete video list with duration categorization
- `POST /channel/download/selected` - Download transcripts for selected videos  
- `GET /channel/download/status/{job_id}` - Check download progress
- `GET /channel/download/results/{job_id}` - Download completed transcripts
- `POST /download/transcript/raw` - Individual video transcript download
- `GET /video-info` - Single video metadata

**User Management & Billing**:
- `GET /user/credits` - Check credit balance
- `GET /user/profile` - User profile information
- `POST /payments/create-checkout-session` - Stripe checkout
- `POST /payments/webhook` - Stripe webhook handler

**Notes**: Redundant endpoints have been removed to streamline the API surface while preserving all core functionality. The remaining endpoints provide quota-efficient operations with 99%+ API quota savings.

### Testing Transcript API Endpoints
```bash
# Download single video transcript (existing)
curl -X POST "http://localhost:8001/download/transcript" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/VIDEO_ID"}'

# Download selected videos with formatting options (CURRENT)
curl -X POST "http://localhost:8001/channel/download/selected" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "channel_name": "channelname",
    "videos": [{"id": "video1", "title": "Video 1"}],
    "include_timestamps": true,
    "include_video_title": false,
    "concatenate_all": true
  }'

# Get all videos for a channel for selection (CURRENT)
curl -X GET "http://localhost:8001/channel/channelname/all-videos"

# Check download progress (CURRENT)
curl -X GET "http://localhost:8001/channel/download/status/your-job-id" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Download completed transcripts (CURRENT)
curl -X GET "http://localhost:8001/channel/download/results/your-job-id" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Check user credits (authenticated)
curl -X GET "http://localhost:8001/user/credits" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Transcript API Specific Patterns

### Advanced Formatting Options (CURRENT)
```python
# Frontend can now control transcript formatting via request body:
{
  "channel_name": "example_channel",
  "videos": [...],
  "include_timestamps": false,      // Toggle timestamps in transcripts
  "include_video_title": true,      // Toggle video title in header
  "include_video_id": true,         // Toggle video ID in header  
  "include_video_url": true,        // Toggle video URL in header
  "concatenate_all": false          // Single file vs individual files
}
```

### Streamlined API Workflow (CURRENT)
The API has been streamlined to focus on core functionality with quota-optimized operations:

1. **Video Discovery**: `GET /channel/{channel_name}/all-videos`
   - Returns complete list with duration categorization (short/medium/long)
   - Uses efficient uploads playlist approach (99%+ quota savings)
   - Includes metadata for user selection

2. **Selected Download**: `POST /channel/download/selected`
   - Downloads only user-selected videos
   - Supports all formatting options
   - Returns job ID for progress tracking

3. **Progress Monitoring**: `GET /channel/download/status/{job_id}`
   - Real-time status updates
   - Credit usage tracking
   - Error reporting

4. **Result Retrieval**: `GET /channel/download/results/{job_id}`
   - Download completed transcript ZIP
   - Supports both individual files and concatenated formats

### Single File Concatenation
- **Individual Files** (default): ZIP contains separate .txt files per video
- **Concatenated Mode**: ZIP contains single `{channel}_all_transcripts.txt` with:
  - Channel header with metadata
  - Each video separated by section markers
  - Consistent formatting across all videos

### Credit Management System
```python
class CreditManager:
    async def deduct_credits(self, user_id: str, amount: int = 1):
        # Credits deducted BEFORE download attempt
        # Even failed downloads consume credits
```

### Rate Limiting Strategy
- **Anonymous users**: IP-based rate limiting via `rate_limiter.py`
- **Authenticated users**: Credit-based access (bypass rate limits)
- **Fallback**: When credits exhausted, fall back to anonymous rate limiting

### Download Job Management
```python
# Async job tracking for channel downloads
channel_download_jobs = {}  # In-memory job status tracking

# File cleanup with configurable retention
async def cleanup_job():
    youtube_service.cleanup_old_jobs(max_age_hours=24)
```

### Stripe Integration (Transcript API)
```python
# Price tiers for credit packages
PRICE_CREDITS_MAP = {
    "price_1Rg9SQCakpeOUC7BAhOPV6BB": 400,   # 400 credits
    "price_1Rg9SQCakpeOUC7BeHCiO38e": 1000,  # 1000 credits  
    "price_1Rg9SQCakpeOUC7BooNktUiI": 3000,  # 3000 credits
}

# Webhook distinguishes services via metadata
metadata = {"project": "transcript-api", "user_uuid": user_id}
```

## Integration Points (Transcript API)

### YouTube API Usage (`youtube_service.py`) - QUOTA OPTIMIZED
```python
# Channel/video ID extraction and validation
async def extract_youtube_id(url: str) -> Tuple[str, str]

# Video metadata fetching with retry logic
async def get_video_info(video_id: str) -> Dict

# QUOTA EFFICIENT: Channel video discovery using uploads playlist
async def get_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]
# Uses playlistItems.list (1 unit) + batch videos.list (1 unit per 50 videos)
# Replaces expensive search.list calls (100 units each)
# Achieves 99%+ quota reduction for video discovery

# Duration categorization with efficient batch processing
def _categorize_duration(duration_seconds: int) -> str:
    # Returns 'short' (≤60s), 'medium' (61s-20min), 'long' (>20min)

# Transcript extraction with proxy support
def get_ytt_api() -> YouTubeTranscriptApi:
    # Uses WebshareProxyConfig for rate limit bypass
```

### Database Operations
```python
# Dedicated database connection for transcript service
from db_youtube_transcripts.database import get_db_youtube_transcripts

# Credit management queries
async def get_user_credits(user_id: str) -> int
async def deduct_user_credits(user_id: str, amount: int) -> bool
```

### File Management System
```python
# Organized storage with cleanup
temp_dir = settings.temp_dir  # "../build/transcripts"

# Automatic cleanup of old files
youtube_service.cleanup_old_jobs(max_age_hours=24)
```

---

## **LEGACY: YouTube Chat Assistant Service** 

### Overview (Background Context)
The original service (`src/fastapi_main.py`) creates OpenAI assistants from YouTube transcripts:
- **Database**: `db/database.py` (different from transcript API)
- **File Storage**: `build/{user_id}/` and pickle files
- **Features**: Thread management, assistant creation, source attribution

### Key Differences from Transcript API
- Uses OpenAI for chat functionality vs. pure transcript download
- Different database schema and connection
- Pickle-based caching vs. session-based storage
- Assistant/thread management vs. job-based downloads

### Future Integration Plan
The chat assistant will eventually consume the transcript API service instead of handling YouTube downloads directly.

---

## Common Debugging & Maintenance

### Transcript API Debugging
```bash
# Check service status
curl http://localhost:8001/

# Monitor background cleanup job
# Runs every hour, cleans files older than 24h

# Database connection test
python db_youtube_transcripts/database.py

# YouTube API quota monitoring
# Check console logs for rate limit errors
```

### File System Management
```bash
# Check transcript storage
ls -la build/transcripts/

# Manual cleanup if needed
find build/transcripts/ -type f -mtime +1 -delete
```

## Key Configuration Files

### Transcript API Specific
- `src/config_v2.py`: Pydantic settings for transcript service
- `db_youtube_transcripts/database.py`: Dedicated database connection
- `src/rate_limiter.py`: Anonymous user rate limiting
- `src/youtube_service.py`: Core YouTube interaction logic
