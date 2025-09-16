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

**Playlist Functionality** (NEW - mirrors channel workflow):
- `GET /playlist/{playlist_id}` - Playlist info with metadata
- `GET /playlist/{playlist_id}/all-videos` - Complete playlist video list with duration categorization
- `POST /playlist/download/selected` - Download transcripts for selected playlist videos

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

# PLAYLIST ENDPOINTS (NEW)

# Download selected playlist videos with formatting options
curl -X POST "http://localhost:8001/playlist/download/selected" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "playlist_name": "PLxxxxxx",
    "videos": [{"id": "video1", "title": "Video 1"}],
    "include_timestamps": true,
    "include_video_title": false,
    "concatenate_all": true
  }'

# Get all videos for a playlist for selection
curl -X GET "http://localhost:8001/playlist/PLxxxxxx/all-videos"

# Get playlist info
curl -X GET "http://localhost:8001/playlist/PLxxxxxx"
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

## Playlist Support Architecture (NEW)

### Overview
The playlist functionality extends the existing channel-based workflow to support YouTube playlists. The implementation leverages the fact that channels internally use "uploads playlists" - allowing for minimal code duplication while providing identical functionality.

### Key Insight: Playlist-Native Architecture
The existing channel implementation **already works with playlists** under the hood:
1. Channel videos are fetched via the channel's "uploads playlist"
2. The same `playlistItems.list()` API calls are used
3. All processing logic (duration categorization, metadata extraction, job management) is playlist-agnostic

### Playlist Implementation Components

#### Backend Functions (`youtube_service.py`)
```python
# URL/ID parsing and validation
def extract_playlist_id(url_or_id: str) -> str:
    # Extracts playlist ID from URLs like:
    # https://youtube.com/playlist?list=PLxxxxxx
    # https://youtube.com/watch?v=VIDEO&list=PLxxxxxx
    # PLxxxxxx (direct ID)

# Playlist metadata fetching
async def get_playlist_info(playlist_id: str) -> Dict[str, Any]:
    # Returns: title, description, video count, channel info, thumbnail

# Video listing (mirrors channel logic exactly)
def _fetch_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]:
async def get_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]:
    # Same pagination, duration categorization, and metadata as channels

# Unified download function (handles both channels and playlists)
async def start_selected_videos_transcript_download(
    channel_name: str = None,
    playlist_name: str = None,
    videos: List[Dict[str, Any]] = None,
    user_id: str = None,
    is_playlist: bool = False,
    # ... other parameters
) -> str:
```

#### API Models (`transcript_api.py`)
```python
# Playlist request models (mirror channel models)
class PlaylistRequest(BaseModel):
    playlist_name: str = Field(..., description="YouTube playlist ID or URL")
    max_results: int = Field(30, description="Maximum number of videos to fetch")
    # Same formatting options as ChannelRequest

class SelectedPlaylistVideosRequest(BaseModel):
    playlist_name: str = Field(..., description="YouTube playlist ID or URL")
    videos: List[VideoInfo] = Field(...)
    # Identical formatting options as SelectedVideosRequest
```

#### API Endpoints (Mirror Channel Pattern)
```python
# Playlist info endpoint
@app.get("/playlist/{playlist_id}")
async def get_playlist_info(playlist_id: str):
    # Validates playlist, returns metadata

# Playlist video listing  
@app.get("/playlist/{playlist_id}/all-videos")
async def list_all_playlist_videos(playlist_id: str):
    # Returns all videos with duration categories for selection

# Playlist download endpoint
@app.post("/playlist/download/selected")
async def download_selected_playlist_videos(
    request: SelectedPlaylistVideosRequest,
    payload: dict = Depends(validate_jwt),
    session: Dict = Depends(get_user_session),
):
    # Same credit management, job creation, and processing as channels
```

### Unified Job Management
Playlist jobs are tracked using the same infrastructure as channel jobs with additional fields:

```python
# Job data structure (supports both channels and playlists)
job_data = {
    "status": "processing",
    "channel_name": channel_name if not is_playlist else None,
    "playlist_id": playlist_name if is_playlist else None,
    "source_id": source_id,        # Channel ID or Playlist ID
    "source_name": source_name,    # Channel title or Playlist title  
    "source_type": source_type,    # "channel" or "playlist"
    "total_videos": num_videos,
    # ... rest identical to channel jobs
}
```

### Shared Infrastructure Benefits
- **Credit System**: Same credit deduction and reservation logic
- **Job Tracking**: Same progress monitoring and status endpoints
- **File Management**: Same ZIP generation and download handling
- **Error Handling**: Same retry logic and failure reporting
- **Metadata Optimization**: Same pre-fetching and batch processing

### URL Support Patterns
The system handles various playlist URL formats:
```python
# Supported playlist URL patterns:
"https://youtube.com/playlist?list=PLxxxxxx"
"https://youtube.com/watch?v=VIDEO&list=PLxxxxxx"  # Playlist from video page
"https://www.youtube.com/playlist?list=PLxxxxxx"
"PLxxxxxx"  # Direct playlist ID
```

### Frontend Integration Pattern
The frontend can use **identical UI components** for both channels and playlists:

```javascript
// Unified workflow pattern
function handleInputType(inputUrl) {
    const type = detectInputType(inputUrl); // 'channel' or 'playlist'
    
    if (type === 'channel') {
        // Use /channel/* endpoints
        return fetchChannelWorkflow(inputUrl);
    } else if (type === 'playlist') {
        // Use /playlist/* endpoints  
        return fetchPlaylistWorkflow(inputUrl);
    }
}

// Same video selection UI, same job monitoring, same download handling
```

### Implementation Statistics
- **~150 lines of new code** (mostly function adaptations)
- **Zero changes** to credit system, job management, or file handling
- **100% code reuse** for video processing, transcript extraction, and ZIP generation
- **Identical API patterns** enabling frontend component reuse

### Credit Management System (CURRENT - Enhanced with Persistence)
```python
class CreditManager:
    # Phase 1: Upfront credit reservation (prevents overselling)
    @staticmethod
    def reserve_credits(user_id: str, amount: int) -> str:
        # Returns reservation_id for tracking
    
    # Phase 2: Credit finalization with refunds for unused credits
    @staticmethod 
    def finalize_credit_usage(user_id: str, reservation_id: str, 
                            credits_used: int, credits_reserved: int):
        # Refunds unused credits (credits_reserved - credits_used)
        
# Job-level credit tracking with persistence
async def download_channel_transcripts_task(job_id: str):
    # 1. Reserve credits upfront for all videos
    reservation_id = CreditManager.reserve_credits(user_id, total_videos)
    
    # 2. PERSIST reservation_id to disk immediately
    update_job_progress(job_id, reservation_id=reservation_id)
    
    # 3. Track actual usage per video (atomic increments)
    # Each video: credits_used_increment=1
    
    # 4. Finalize with refund of unused credits
    CreditManager.finalize_credit_usage(
        user_id=user_id,
        reservation_id=job["reservation_id"],  # Loaded from disk
        credits_used=job["credits_used"],
        credits_reserved=job["credits_reserved"]
    )
```

### Rate Limiting Strategy
- **Anonymous users**: IP-based rate limiting via `rate_limiter.py`
- **Authenticated users**: Credit-based access (bypass rate limits)
- **Fallback**: When credits exhausted, fall back to anonymous rate limiting

### Download Job Management (CURRENT - Persistent Storage Architecture)
```python
# Hybrid job tracking: In-memory + persistent disk storage
channel_download_jobs = {}  # In-memory job status tracking (for active jobs)

# NEW: Persistent job storage with atomic operations
JOBS_STORAGE_DIR = os.path.join(settings.temp_dir, "jobs")

# Job persistence functions with atomic file operations
def save_job_to_file(job_id: str, job_data: Dict[str, Any]) -> None:
    # Converts Pydantic objects to serializable format
    # Handles video objects, metadata, and all job state

def load_job_from_file(job_id: str) -> Optional[Dict[str, Any]]:
    # Loads job from persistent storage
    # Used for recovery and concurrent access

def update_job_progress(job_id: str, **updates):
    # ATOMIC operations with file locking (Unix) and retry logic
    # Supports special operations:
    # - completed_increment=1: atomic counter increment
    # - files_append=file_info: atomic list append
    # - failed_count_increment=1: atomic failure tracking
```

### Persistent Storage Benefits (CURRENT)
- **Service Restart Recovery**: Jobs survive service restarts
- **Credit Reservation Persistence**: `reservation_id` saved to disk for proper cleanup
- **Race Condition Prevention**: Atomic file operations prevent concurrent update conflicts
- **Progress Accuracy**: Real-time progress tracking across multiple concurrent video downloads
- **Audit Trail**: Complete job history preserved on disk

### Metadata Pre-fetching Optimization (CURRENT)
```python
# NEW: Efficient metadata pre-fetching at job creation
async def start_selected_videos_transcript_download():
    # Pre-fetch ALL video metadata in batches (50 videos per API call)
    videos_metadata = await pre_fetch_videos_metadata(video_ids)
    
    # Store pre-fetched metadata in job for reuse
    job_data["videos_metadata"] = videos_metadata
    
    # Each video download uses pre-fetched data (no additional API calls)
    pre_fetched_metadata = job.get("videos_metadata", {}).get(video_id, {})
```

### Atomic Progress Updates (CURRENT)
```python
# Individual video completion with atomic operations
async def process_single_video(job_id: str, video_id: str, output_dir: str):
    # Success case: atomic multi-field update
    updated_job = update_job_progress(
        job_id,
        files_append=file_info,           # Add file to list atomically
        completed_increment=1,            # Increment success counter
        credits_used_increment=1,         # Track credit usage
    )
    
    # Failure case: atomic failure tracking
    update_job_progress(
        job_id, 
        failed_count_increment=1,         # Increment failure counter
        credits_used_increment=1          # Still count failed attempts
    )
    
    # Always: atomic processed tracking
    update_job_progress(job_id, processed_count_increment=1)
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
async def extract_playlist_id(url_or_id: str) -> str  # NEW: Playlist URL parsing

# Video metadata fetching with retry logic
async def get_video_info(video_id: str) -> Dict

# QUOTA EFFICIENT: Channel video discovery using uploads playlist
async def get_all_channel_videos(channel_id: str) -> List[Dict[str, Any]]
# NEW: Playlist video discovery (identical efficiency)
async def get_all_playlist_videos(playlist_id: str) -> List[Dict[str, Any]]
# Uses playlistItems.list (1 unit) + batch videos.list (1 unit per 50 videos)
# Replaces expensive search.list calls (100 units each)
# Achieves 99%+ quota reduction for video discovery

# Duration categorization with efficient batch processing
def _categorize_duration(duration_seconds: int) -> str:
    # Returns 'short' (≤60s), 'medium' (61s-20min), 'long' (>20min)

# Transcript extraction with proxy support
def get_ytt_api() -> YouTubeTranscriptApi:
    # Uses WebshareProxyConfig for rate limit bypass

# NEW: Playlist metadata fetching
async def get_playlist_info(playlist_id: str) -> Dict[str, Any]:
    # Uses playlists.list (1 unit) for playlist metadata
```

### Database Operations
```python
# Dedicated database connection for transcript service
from db_youtube_transcripts.database import get_db_youtube_transcripts

# Credit management queries
async def get_user_credits(user_id: str) -> int
async def deduct_user_credits(user_id: str, amount: int) -> bool
```

### File Management System (CURRENT - Persistent + Atomic)
```python
# Organized storage with persistent job tracking
temp_dir = settings.temp_dir  # "../build/transcripts"

# NEW: Persistent job storage directory
JOBS_STORAGE_DIR = os.path.join(settings.temp_dir, "jobs")

# Each job stored as: {JOBS_STORAGE_DIR}/{job_id}.json
# Contains: progress, metadata, credit tracking, file lists

# Automatic cleanup with job state preservation
youtube_service.cleanup_old_jobs(max_age_hours=24)
# Cleans transcript files but preserves job metadata for audit

# Atomic file operations prevent corruption
def update_job_progress(job_id: str, **updates):
    # File locking (Unix) + retry logic for concurrent safety
    # JSON serialization handles Pydantic objects properly
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

# Test playlist functionality (NEW)
curl http://localhost:8001/playlist/PLxxxxxx
curl http://localhost:8001/playlist/PLxxxxxx/all-videos

# Monitor background cleanup job
# Runs every hour, cleans files older than 24h

# Database connection test
python db_youtube_transcripts/database.py

# YouTube API quota monitoring
# Check console logs for rate limit errors
```

### Playlist-Specific Debugging (NEW)
```bash
# Test playlist URL parsing
python -c "
from src.youtube_service import extract_playlist_id
print(extract_playlist_id('https://youtube.com/playlist?list=PLxxxxxx'))
print(extract_playlist_id('PLxxxxxx'))
"

# Verify playlist access permissions
# Public playlists: Work immediately
# Private playlists: Return 404 error
# Unlisted playlists: Work with direct playlist ID

# Common playlist error patterns:
# - 404: Playlist not found or private
# - 422: Invalid playlist_name in request body
# - 500: Job filename generation error (check get_safe_channel_name)
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
